//! The whole-program pipeline: lift every function, resolve what each call calls,
//! infer signatures to a fixpoint, rewrite each function to its signature, and
//! emit. Signatures need the callees' signatures, so this can't be done one
//! function at a time; every per-function step runs in parallel with rayon.
//!
//! Call targets, in order of preference:
//! * a relocation at the call site (relocatable objects, where the call's bytes
//!   don't hold the target yet) naming a symbol;
//! * a function in the binary starting at the target address;
//! * an ELF PLT stub, or a GOT slot for `call [rip+x]`, naming an import;
//! * anything else is an indirect call through a function pointer.
//!
//! A callee that is emitted is called by its Rust name. Imports, and functions
//! that aren't emitted (not selected, or not lifted), are declared in an
//! `extern "C"` block in `mod ffi` (`prelude`) and called through it.
use crate::abi::{self, CallShape, Sig, Site};
use crate::emit::{emit_function_in, CallInfo, EmitStats, Env, Mode};
use crate::ir::*;
use crate::lift::Lifter;
use crate::opt::clean;
use crate::verify::verify;
use object::{Object, ObjectKind, ObjectSection, ObjectSymbol, ObjectSymbolTable, RelocationTarget, SectionKind};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

/// Signature inference stops after this many rounds even if a cycle of
/// signatures keeps changing (it is monotone in practice, so this is a backstop).
const MAX_ROUNDS: usize = 64;

/// A function to decompile.
pub struct Input<'a> {
    /// Symbol name.
    pub name: String,
    /// Rust identifier to emit it as (unique among selected functions).
    pub ident: String,
    pub addr: u64,
    pub bytes: &'a [u8],
    /// Emitted. Functions that aren't are still lifted for their signatures.
    pub selected: bool,
}

/// What a call site calls.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    /// A function in the program, by index.
    Func(usize),
    /// A symbol the program doesn't define (an import).
    Import(String),
    /// A function pointer, or an address that isn't a known function.
    Indirect,
}

pub struct Func {
    pub name: String,
    pub ident: String,
    pub addr: u64,
    pub selected: bool,
    /// The lifted, cleaned IR, rewritten to its signature; or why it didn't lift.
    pub ir: Result<Function, String>,
    /// The IR straight out of the lifter, if `build` was asked to keep it.
    pub raw_ir: Option<String>,
    pub sig: Sig,
    sites: Vec<Site>,
    targets: Vec<Target>,
    guesses: Vec<u8>,
}

/// An `extern "C"` declaration in `mod ffi`.
#[derive(Clone, Debug)]
pub struct Extern {
    pub name: String,
    pub ident: String,
    pub sig: Sig,
}

pub struct Program {
    pub funcs: Vec<Func>,
    pub externs: Vec<Extern>,
    extern_of: HashMap<String, usize>,
}

/// Names for call targets, from the object file.
#[derive(Default)]
struct Symbols {
    /// Relocation at an address in code: the symbol it refers to, or the address it
    /// resolves to (section-relative relocations for static functions).
    relocs: BTreeMap<u64, Reloc>,
    /// PLT stub address -> imported symbol.
    plt: HashMap<u64, String>,
    /// GOT slot address -> symbol.
    got: HashMap<u64, String>,
}

#[derive(Clone, Debug)]
enum Reloc {
    Sym(String),
    Addr(u64),
}

impl Symbols {
    fn parse(data: &[u8]) -> Symbols {
        let Ok(file) = object::File::parse(data) else { return Symbols::default() };
        let mut s = Symbols::default();
        if file.kind() == ObjectKind::Relocatable {
            for sec in file.sections().filter(|x| x.kind() == SectionKind::Text) {
                for (off, r) in sec.relocations() {
                    let RelocationTarget::Symbol(i) = r.target() else { continue };
                    let Ok(sym) = file.symbol_by_index(i) else { continue };
                    let at = sec.address() + off;
                    let reloc = if sym.kind() == object::SymbolKind::Section {
                        // S + A - P relative to the field; the target is S + A + 4
                        if sym.section_index() != Some(sec.index()) {
                            continue;
                        }
                        Reloc::Addr(sym.address().wrapping_add(r.addend() as u64).wrapping_add(4))
                    } else {
                        match sym.name() {
                            Ok(n) if !n.is_empty() => Reloc::Sym(n.to_string()),
                            _ => continue,
                        }
                    };
                    s.relocs.insert(at, reloc);
                }
            }
            return s;
        }
        // GOT slots filled by the dynamic loader, by symbol.
        let dynsyms = file.dynamic_symbol_table();
        for (at, r) in file.dynamic_relocations().into_iter().flatten() {
            let RelocationTarget::Symbol(i) = r.target() else { continue };
            let Some(sym) = dynsyms.as_ref().and_then(|t| t.symbol_by_index(i).ok()) else { continue };
            if let Ok(n) = sym.name() {
                if !n.is_empty() {
                    s.got.insert(at, n.to_string());
                }
            }
        }
        // PLT stubs: `[endbr64;] jmp [rip+slot]`, one per entry.
        for sec in file.sections() {
            let name = sec.name().unwrap_or("");
            if !matches!(name, ".plt" | ".plt.sec" | ".plt.got") {
                continue;
            }
            let Ok(code) = sec.data() else { continue };
            let mut dec = iced_x86::Decoder::with_ip(64, code, sec.address(), iced_x86::DecoderOptions::NONE);
            let mut entry = None;
            for i in &mut dec {
                if i.mnemonic() == iced_x86::Mnemonic::Endbr64 {
                    entry = Some(i.ip());
                    continue;
                }
                if i.flow_control() == iced_x86::FlowControl::IndirectBranch && i.is_ip_rel_memory_operand() {
                    if let Some(n) = s.got.get(&i.ip_rel_memory_address()) {
                        s.plt.insert(entry.unwrap_or(i.ip()), n.clone());
                    }
                }
                entry = None;
            }
        }
        s
    }

    /// The relocation inside the instruction at `ip` (calls and jumps put it in
    /// their first few bytes).
    fn reloc_at(&self, ip: u64) -> Option<&Reloc> {
        self.relocs.range(ip + 1..=ip + 4).next().map(|(_, r)| r)
    }
}

fn konst(f: &Function, v: ValueId) -> Option<u64> {
    match f.insts[v].kind {
        InstKind::Const(c) => Some(f.consts[c.index()] as u64),
        _ => None,
    }
}

impl Program {
    /// Lift, resolve, infer signatures and rewrite every function. `file` is the
    /// whole binary, for relocations and import names (`None` for raw bytes).
    pub fn build(inputs: Vec<Input>, file: Option<&[u8]>, keep_raw_ir: bool) -> Program {
        let syms = file.map(Symbols::parse).unwrap_or_default();
        let by_addr: HashMap<u64, usize> = inputs.iter().enumerate().map(|(i, x)| (x.addr, i)).rev().collect();
        let mut by_name: HashMap<String, usize> = HashMap::new();
        for (i, x) in inputs.iter().enumerate() {
            by_name.entry(x.name.clone()).or_insert(i);
        }

        // 1. Lift and clean.
        let lifted: Vec<(Result<Function, String>, Option<String>)> = inputs
            .par_iter()
            .map_init(
                || {
                    let mut l = Lifter::new();
                    l.track_exits = true;
                    l
                },
                |lifter, x| {
                let mut f = Function::with_capacity(256, 16);
                let mut raw = None;
                let r = lifter
                    .lift(x.bytes, x.addr, &mut f)
                    .map_err(|e| describe(&e))
                    .and_then(|()| verify(&f).map_err(|e| format!("lifted IR failed verification: {e:?}")))
                    .and_then(|()| {
                        if keep_raw_ir && x.selected {
                            raw = Some(crate::dump::dump(&f));
                        }
                        clean(&mut f);
                        verify(&f).map_err(|e| format!("cleaned IR failed verification: {e:?}"))
                    });
                (r.map(|()| f), raw)
            })
            .collect();

        // 2. Call sites and their targets.
        let resolve = |f: &Function, site: Site| -> Target {
            let (callee, _) = site.parts(f);
            let ip = site.ip(f);
            let by_symbol = |n: &str| match by_name.get(n) {
                Some(&i) => Target::Func(i),
                None => Target::Import(n.to_string()),
            };
            let InstKind::IntToPtr(a) = f.insts[callee].kind else { return Target::Indirect };
            if let Some(r) = syms.reloc_at(ip) {
                return match r {
                    Reloc::Sym(n) => by_symbol(n),
                    Reloc::Addr(t) => by_addr.get(t).map_or(Target::Indirect, |&i| Target::Func(i)),
                };
            }
            match f.insts[a].kind {
                InstKind::Const(_) => {
                    let t = konst(f, a).unwrap();
                    if let Some(&i) = by_addr.get(&t) {
                        return Target::Func(i);
                    }
                    syms.plt.get(&t).map_or(Target::Indirect, |n| by_symbol(n))
                }
                // call [rip+slot]
                InstKind::Load { ptr, .. } => match f.insts[ptr].kind {
                    InstKind::IntToPtr(s) => konst(f, s).and_then(|s| syms.got.get(&s)).map_or(Target::Indirect, |n| by_symbol(n)),
                    _ => Target::Indirect,
                },
                _ => Target::Indirect,
            }
        };
        let mut funcs: Vec<Func> = inputs
            .into_par_iter()
            .zip(lifted)
            .map(|(x, (ir, raw_ir))| {
                let (sites, targets, guesses) = match &ir {
                    Ok(f) => {
                        let sites = abi::sites(f);
                        let targets = sites.iter().map(|&s| resolve(f, s)).collect();
                        let guesses = sites.iter().map(|&s| abi::guess_args(f, s)).collect();
                        (sites, targets, guesses)
                    }
                    Err(_) => Default::default(),
                };
                Func { name: x.name, ident: x.ident, addr: x.addr, selected: x.selected, ir, raw_ir, sig: Sig::default(), sites, targets, guesses }
            })
            .collect();

        // Callees whose signature can't be inferred (imports outside the libc
        // table, functions that didn't lift) take, at every call, the most
        // arguments any call site sets up, so one declaration fits all calls.
        let mut guessed: HashMap<Target, Sig> = HashMap::new();
        for f in &funcs {
            for (t, &g) in f.targets.iter().zip(&f.guesses) {
                let key = match t {
                    Target::Func(i) if funcs[*i].ir.is_err() => t.clone(),
                    Target::Import(n) if crate::libc::lookup(n).is_none() => t.clone(),
                    _ => continue,
                };
                let e = guessed.entry(key).or_insert(Sig { ret: true, ..Sig::default() });
                e.args = e.args.max(g);
            }
        }
        let stack_args: Vec<u8> = funcs.par_iter().map(|f| f.ir.as_ref().map_or(0, abi::stack_args)).collect();

        // 3. Signatures, to a fixpoint (Jacobi rounds: every function from the
        //    previous round's callee signatures). A function returns rdx too
        //    (`ret2`) only if some caller reads rdx after calling it.
        let mut sigs: Vec<Sig> = stack_args.iter().map(|&s| Sig { stack_args: s, ..Sig::default() }).collect();
        let mut rdx_wanted = vec![false; funcs.len()];
        for _round in 0..MAX_ROUNDS {
            let next: Vec<(Sig, Vec<bool>)> = funcs
                .par_iter()
                .enumerate()
                .map(|(i, f)| match &f.ir {
                    Ok(ir) => {
                        let callee = |k: usize| site_sig(&f.targets[k], f.guesses[k], &sigs, &guessed);
                        let r = abi::infer(ir, &f.sites, &callee, sigs[i], rdx_wanted[i]);
                        (Sig { stack_args: stack_args[i], ..r.sig }, r.rdx_read)
                    }
                    Err(_) => (sigs[i], Vec::new()),
                })
                .collect();
            let mut wanted = vec![false; funcs.len()];
            for (f, (_, read)) in funcs.iter().zip(&next) {
                for (t, &r) in f.targets.iter().zip(read) {
                    if let (Target::Func(j), true) = (t, r) {
                        wanted[*j] = true;
                    }
                }
            }
            let next: Vec<Sig> = next.into_iter().map(|(s, _)| s).collect();
            if next == sigs && wanted == rdx_wanted {
                break;
            }
            sigs = next;
            rdx_wanted = wanted;
        }

        // 4. Rewrite each function to its signature.
        funcs.par_iter_mut().enumerate().for_each(|(i, f)| {
            f.sig = sigs[i];
            let Ok(ir) = &mut f.ir else { return };
            let shapes: Vec<CallShape> = (0..f.sites.len())
                .map(|k| {
                    let s = site_sig(&f.targets[k], f.guesses[k], &sigs, &guessed);
                    let args = if s.variadic { s.args.max(f.guesses[k]) } else { s.args };
                    CallShape { args, ..s }
                })
                .collect();
            f.sites = abi::apply(ir, sigs[i], &f.sites, &|k| shapes[k]);
            debug_assert!(verify(ir).is_ok(), "{}: {:?}", f.name, verify(ir));
        });
        // `apply` must leave valid IR; a failure is a bug, reported like a lift error.
        for f in &mut funcs {
            if let Ok(ir) = &f.ir {
                if let Err(e) = verify(ir) {
                    f.ir = Err(format!("IR failed verification after signature recovery: {e:?}"));
                }
            }
        }

        // 5. Extern declarations for what emitted code calls but doesn't define.
        let mut externs: Vec<Extern> = Vec::new();
        let mut extern_of: HashMap<String, usize> = HashMap::new();
        let mut idents: HashMap<String, usize> = HashMap::new();
        for fi in 0..funcs.len() {
            if !funcs[fi].selected || funcs[fi].ir.is_err() {
                continue;
            }
            for k in 0..funcs[fi].sites.len() {
                let (name, sig) = match &funcs[fi].targets[k] {
                    Target::Func(j) if !funcs[*j].selected || funcs[*j].ir.is_err() => {
                        let s = if funcs[*j].ir.is_ok() { sigs[*j] } else { guessed[&Target::Func(*j)] };
                        (funcs[*j].name.clone(), s)
                    }
                    Target::Import(n) => (n.clone(), site_sig(&funcs[fi].targets[k], 0, &sigs, &guessed)),
                    _ => continue,
                };
                if extern_of.contains_key(&name) {
                    continue;
                }
                let mut ident = crate::names::sanitize(&name);
                let n = idents.entry(ident.clone()).or_insert(0);
                *n += 1;
                if *n > 1 {
                    ident = format!("{ident}_{}", *n - 1);
                }
                extern_of.insert(name.clone(), externs.len());
                externs.push(Extern { name, ident, sig });
            }
        }
        Program { funcs, externs, extern_of }
    }

    /// The callee of each call site in function `i`, for the emitter.
    fn call_info(&self, i: usize, site: Site) -> Option<CallInfo> {
        let f = &self.funcs[i];
        let k = f.sites.iter().position(|&s| s == site)?;
        let ext = |name: &str| {
            let e = &self.externs[*self.extern_of.get(name)?];
            Some(CallInfo { path: Some(format!("ffi::{}", e.ident)), ret: e.sig.ret, ret2: e.sig.ret2, foreign: true })
        };
        match &f.targets[k] {
            Target::Func(j) => {
                let g = &self.funcs[*j];
                if g.selected && g.ir.is_ok() {
                    Some(CallInfo { path: Some(g.ident.clone()), ret: g.sig.ret, ret2: g.sig.ret2, foreign: false })
                } else {
                    ext(&g.name)
                }
            }
            Target::Import(n) => ext(n),
            Target::Indirect => None,
        }
    }

    /// Emit every selected function that lifted, in parallel: `(source, stats)`,
    /// or `None` for the others. A function is `unsafe` if it does something
    /// unsafe itself or calls a decompiled function that is.
    pub fn emit_all(&self, mode: Mode, global_of: &(dyn Fn(u64) -> Option<String> + Sync)) -> Vec<Option<(String, EmitStats)>> {
        // In safe mode a decompiled caller can only pass addresses, so functions
        // that are called take integers.
        let mut called = vec![false; self.funcs.len()];
        for f in self.funcs.iter().filter(|f| f.selected && f.ir.is_ok()) {
            for t in &f.targets {
                if let Target::Func(j) = t {
                    called[*j] = true;
                }
            }
        }
        let mut out: Vec<Option<(String, EmitStats)>> = self
            .funcs
            .par_iter()
            .enumerate()
            .map(|(i, f)| {
                let ir = f.ir.as_ref().ok().filter(|_| f.selected)?;
                let call = |s: Site| self.call_info(i, s);
                let env = Env { sig: Some(f.sig), call: &call, demote: called[i], structure: true, global_of };
                let mut src = String::new();
                let stats = emit_function_in(ir, &f.ident, mode, &env, &mut src);
                Some((src, stats))
            })
            .collect();
        // unsafe propagates from callee to caller
        let mut unsafe_fn: Vec<bool> = out.iter().map(|o| o.as_ref().is_some_and(|(s, _)| s.starts_with("pub unsafe fn"))).collect();
        let mut changed = true;
        while changed {
            changed = false;
            for (i, f) in self.funcs.iter().enumerate() {
                if out[i].is_none() || unsafe_fn[i] {
                    continue;
                }
                if f.targets.iter().any(|t| matches!(t, Target::Func(j) if out[*j].is_some() && unsafe_fn[*j])) {
                    unsafe_fn[i] = true;
                    changed = true;
                }
            }
        }
        for (i, o) in out.iter_mut().enumerate() {
            if let Some((src, _)) = o {
                if unsafe_fn[i] && src.starts_with("pub fn") {
                    src.replace_range(..6, "pub unsafe fn");
                }
            }
        }
        out
    }

    /// `mod ffi`, declaring every extern the emitted code calls; empty if none.
    pub fn prelude(&self) -> String {
        if self.externs.is_empty() {
            return String::new();
        }
        let mut s = String::from("\n/// Functions the decompiled code calls but doesn't define.\npub mod ffi {\n");
        if self.externs.iter().any(|e| e.sig.ret2) {
            s.push_str("    /// A 16-byte result, returned in rax:rdx.\n    #[repr(C)]\n    pub struct Pair(pub u64, pub u64);\n\n");
        }
        s.push_str("    extern \"C\" {\n");
        let mut ext: Vec<&Extern> = self.externs.iter().collect();
        ext.sort_by(|a, b| a.ident.cmp(&b.ident));
        for e in ext {
            let mut params: Vec<String> = (0..e.sig.args).map(|k| format!("a{k}: u64")).collect();
            params.extend((0..e.sig.stack_args).map(|k| format!("a{}: u64", 6 + k as usize)));
            if e.sig.variadic {
                params.push("...".into());
            }
            if e.ident != e.name {
                let _ = writeln!(s, "        #[link_name = {:?}]", e.name);
            }
            let ret = match (e.sig.ret, e.sig.ret2) {
                (_, true) => " -> Pair",
                (true, _) => " -> u64",
                _ => "",
            };
            let _ = writeln!(s, "        pub fn {}({}){ret};", e.ident, params.join(", "));
        }
        s.push_str("    }\n}\n");
        s
    }
}

fn site_sig(t: &Target, guess: u8, sigs: &[Sig], guessed: &HashMap<Target, Sig>) -> Sig {
    match t {
        Target::Func(j) => guessed.get(t).copied().unwrap_or(sigs[*j]),
        Target::Import(n) => crate::libc::lookup(n).or_else(|| guessed.get(t).copied()).unwrap_or_default(),
        Target::Indirect => Sig { args: guess, ret: true, ..Sig::default() },
    }
}

/// One line for a lift error.
pub fn describe(e: &crate::lift::LiftError) -> String {
    use crate::lift::LiftError;
    match e {
        LiftError::Unsupported { ip, mnemonic } => format!("unsupported instruction {mnemonic:?} at {ip:#x}"),
        LiftError::FlagsNotInBlock { ip } => format!("branch at {ip:#x} reads flags set in another block"),
        LiftError::BranchOutOfRange { ip, target } => format!("branch at {ip:#x} leaves the function (to {target:#x})"),
        LiftError::TargetInsideInstruction { target } => format!("branch into the middle of an instruction at {target:#x}"),
    }
}
