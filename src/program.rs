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
use crate::borrow::{analyze_with, Analysis, Callee, Class, Ctx, Pass};
use crate::emit::{emit_function_in, CallInfo, EmitStats, Env, Mode};
use crate::ir::*;
use crate::lift::Lifter;
use crate::opt::clean;
use crate::types::{FnTypes, TypeModel, TypeStats};
use crate::verify::verify;
use object::{Object, ObjectKind, ObjectSection, ObjectSymbol, RelocationTarget, SectionKind};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};
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
    /// Recovered types (`types.rs`); `None` if it didn't lift.
    pub types: Option<FnTypes>,
    sites: Vec<Site>,
    targets: Vec<Target>,
    guesses: Vec<(u8, u8)>,
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
    /// The types `Func::types` refer to, structs included.
    pub tys: TyTable,
    pub type_stats: TypeStats,
    /// GOT slot -> the address in the binary the loader fills it with.
    got_addr: HashMap<u64, u64>,
}

/// How `Program::build_with` recovers types.
#[derive(Copy, Clone)]
pub struct BuildOptions<'a> {
    /// Use DWARF debug info when the file has it.
    pub dwarf: bool,
    /// Proposes argument and return types; the facts accept or reject them.
    pub model: Option<&'a dyn TypeModel>,
}

impl Default for BuildOptions<'_> {
    fn default() -> Self {
        BuildOptions { dwarf: true, model: None }
    }
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
    /// GOT slot address -> the address the loader puts there (`R_X86_64_RELATIVE`):
    /// a function or object in the binary itself.
    got_addr: HashMap<u64, u64>,
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
        // GOT slots filled by the dynamic loader, and the PLT stubs that jump through them.
        s.got = crate::discover::got_names(&file);
        s.plt = crate::discover::plt_names(&file, &s.got);
        // GOT slots the loader fills with an address in the binary itself
        // (`R_X86_64_RELATIVE`): calls through them reach a decompiled function.
        for (at, r) in file.dynamic_relocations().into_iter().flatten() {
            if let RelocationTarget::Absolute = r.target() {
                if r.size() == 64 || r.size() == 0 {
                    s.got_addr.insert(at, r.addend() as u64);
                }
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

/// The sections a linked binary loads, as (address, bytes), for reading jump
/// tables. Empty for relocatable objects, whose tables are relocations.
fn loaded_sections(data: &[u8]) -> Vec<(u64, &[u8])> {
    let Ok(file) = object::File::parse(data) else { return Vec::new() };
    if file.kind() == ObjectKind::Relocatable {
        return Vec::new();
    }
    file.sections()
        .filter(|s| s.address() != 0 && s.kind() != SectionKind::UninitializedData)
        .filter_map(|s| Some((s.address(), s.data().ok().filter(|d| !d.is_empty())?)))
        .collect()
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
        Self::build_with(inputs, file, keep_raw_ir, BuildOptions::default())
    }

    /// `build`, choosing where types come from.
    pub fn build_with(inputs: Vec<Input>, file: Option<&[u8]>, keep_raw_ir: bool, opts: BuildOptions) -> Program {
        let syms = file.map(Symbols::parse).unwrap_or_default();
        let sections = file.map(loaded_sections).unwrap_or_default();
        let thread_pointer = file.and_then(|d| crate::load::tls(&object::File::parse(d).ok()?)).map(|t| t.thread_pointer);
        let by_addr: HashMap<u64, usize> = inputs.iter().enumerate().map(|(i, x)| (x.addr, i)).rev().collect();
        let mut by_name: HashMap<String, usize> = HashMap::new();
        for (i, x) in inputs.iter().enumerate() {
            by_name.entry(x.name.clone()).or_insert(i);
        }
        // A call that never returns ends its block: to an import like `abort`
        // (by relocation in an object file, through the PLT or the GOT), or to a
        // function that doesn't return, by name or by its code, directly or
        // through a GOT slot.
        let named: Vec<(u64, &[u8], &str)> = inputs.iter().map(|x| (x.addr, x.bytes, x.name.as_str())).collect();
        let noreturn_at: HashSet<u64> = match file.and_then(|d| object::File::parse(d).ok()) {
            Some(obj) if obj.kind() != ObjectKind::Relocatable => crate::discover::noreturn_calls(&obj, &named, &syms.got_addr),
            _ => named.iter().filter(|x| crate::discover::noreturn(x.2)).map(|x| x.0).collect(),
        };
        let noreturn = |ip: u64, target: Option<u64>| match syms.reloc_at(ip) {
            Some(Reloc::Sym(n)) => crate::discover::noreturn(n) || by_name.get(n).is_some_and(|&i| noreturn_at.contains(&inputs[i].addr)),
            Some(Reloc::Addr(a)) => noreturn_at.contains(a),
            None => target.is_some_and(|t| noreturn_at.contains(&t)),
        };

        // 1. Lift and clean, and meanwhile read the debug info (one thread).
        let mut tys = TyTable::new();
        let addrs: std::collections::HashSet<u64> = inputs.iter().map(|x| x.addr).collect();
        let read_debug = || match (opts.dwarf, file) {
            (true, Some(d)) => crate::dwarf::read(d, &mut tys, &|a| addrs.contains(&a)),
            _ => None,
        };
        let lift = || -> Vec<(Result<Function, String>, Option<String>)> {
            inputs
                .par_iter()
                .map_init(
                    || {
                        let mut l = Lifter::new();
                        l.track_exits = true;
                        l.thread_pointer = thread_pointer;
                        l
                    },
                    |lifter, x| {
                        let mut f = Function::with_capacity(256, 16);
                        let mut raw = None;
                        let own = [(x.addr, x.bytes)];
                        let data: &[(u64, &[u8])] = if sections.is_empty() { &own } else { &sections };
                        let r = lifter.lift_full(x.bytes, x.addr, data, &noreturn, &mut f);
                        let r = r
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
                    },
                )
                .collect()
        };
        let (lifted, debug) = rayon::join(lift, read_debug);

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
                    InstKind::IntToPtr(s) => match konst(f, s) {
                        Some(s) if syms.got.contains_key(&s) => by_symbol(&syms.got[&s]),
                        Some(s) => syms.got_addr.get(&s).and_then(|t| by_addr.get(t)).map_or(Target::Indirect, |&i| Target::Func(i)),
                        None => Target::Indirect,
                    },
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
                        let guesses = sites.iter().map(|&s| (abi::guess_args(f, s), abi::guess_fargs(f, s))).collect();
                        (sites, targets, guesses)
                    }
                    Err(_) => Default::default(),
                };
                Func {
                    name: x.name,
                    ident: x.ident,
                    addr: x.addr,
                    selected: x.selected,
                    ir,
                    raw_ir,
                    sig: Sig::default(),
                    types: None,
                    sites,
                    targets,
                    guesses,
                }
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
                e.args = e.args.max(g.0);
                e.fargs = e.fargs.max(g.1);
            }
        }
        let stack_args: Vec<u8> = funcs.par_iter().map(|f| f.ir.as_ref().map_or(0, abi::stack_args)).collect();

        // 3. Signatures, to a fixpoint (Jacobi rounds: every function from the
        //    previous round's callee signatures). A function returns rdx too
        //    (`ret2`) only if some caller reads rdx after calling it; a caller
        //    reading xmm0 says it returns a float. A function whose inputs (its
        //    own signature, its callees', and what its callers read) didn't
        //    change last round gets the same result again, so only the others
        //    are re-inferred.
        let mut sigs: Vec<Sig> = stack_args.iter().map(|&s| Sig { stack_args: s, ..Sig::default() }).collect();
        let mut wanted_by = vec![0u8; funcs.len()];
        let mut callers: Vec<Vec<usize>> = vec![Vec::new(); funcs.len()];
        for (i, f) in funcs.iter().enumerate() {
            for t in &f.targets {
                if let Target::Func(j) = t {
                    if callers[*j].last() != Some(&i) {
                        callers[*j].push(i);
                    }
                }
            }
        }
        let mut results: Vec<(Sig, Vec<u8>)> = sigs.iter().map(|&s| (s, Vec::new())).collect();
        let mut dirty = vec![true; funcs.len()];
        for _round in 0..MAX_ROUNDS {
            let next: Vec<Option<(Sig, Vec<u8>)>> = funcs
                .par_iter()
                .enumerate()
                .map(|(i, f)| match &f.ir {
                    Ok(ir) if dirty[i] => {
                        let callee = |k: usize| site_sig(&f.targets[k], f.guesses[k], &sigs, &guessed);
                        let r = abi::infer(ir, &f.sites, &callee, sigs[i], wanted_by[i]);
                        Some((Sig { stack_args: stack_args[i], ..r.sig }, r.reads))
                    }
                    _ => None,
                })
                .collect();
            for (i, r) in next.into_iter().enumerate() {
                if let Some(r) = r {
                    results[i] = r;
                }
            }
            let mut wanted = vec![0u8; funcs.len()];
            for (f, (_, read)) in funcs.iter().zip(&results) {
                for (t, &r) in f.targets.iter().zip(read) {
                    if let Target::Func(j) = t {
                        wanted[*j] |= r;
                    }
                }
            }
            dirty.iter_mut().for_each(|d| *d = false);
            let mut changed = false;
            for i in 0..funcs.len() {
                if results[i].0 != sigs[i] {
                    changed = true;
                    dirty[i] = true;
                    for &c in &callers[i] {
                        dirty[c] = true;
                    }
                }
                if wanted[i] != wanted_by[i] {
                    changed = true;
                    dirty[i] = true;
                }
            }
            if !changed {
                break;
            }
            sigs = results.iter().map(|r| r.0).collect();
            wanted_by = wanted;
        }

        // 4. Rewrite each function to its signature.
        funcs.par_iter_mut().enumerate().for_each(|(i, f)| {
            f.sig = sigs[i];
            let Ok(ir) = &mut f.ir else { return };
            let shapes: Vec<CallShape> = (0..f.sites.len())
                .map(|k| {
                    let s = site_sig(&f.targets[k], f.guesses[k], &sigs, &guessed);
                    let (args, fargs) = if s.variadic { (s.args.max(f.guesses[k].0), s.fargs.max(f.guesses[k].1)) } else { (s.args, s.fargs) };
                    CallShape { args, fargs, ..s }
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

        // 5. Types: from debug info, a model, and how values are used.
        let type_inputs: Vec<Option<crate::types::Input>> = funcs
            .iter()
            .map(|f| {
                let ir = f.ir.as_ref().ok()?;
                let calls = f
                    .sites
                    .iter()
                    .zip(&f.targets)
                    .filter_map(|(&s, t)| {
                        let &Target::Func(j) = t else { return None };
                        let g = &funcs[j];
                        g.ir.as_ref().ok()?;
                        let regs = abi::SYSV_ARGS[..(g.sig.args as usize).min(6)].iter().copied().chain((0..g.sig.stack_args).map(|k| abi::STACK_ARG_BASE + k));
                        let args = regs.zip(s.parts(ir).1.get(&ir.value_pool).iter().copied()).collect();
                        let ret = match s {
                            Site::Call(id) if g.sig.rax() => Some(id),
                            _ => None,
                        };
                        Some(crate::types::CallEdge { callee: j, args, ret })
                    })
                    .collect();
                Some(crate::types::Input { f: ir, sig: f.sig, addr: f.addr, name: &f.name, calls })
            })
            .collect();
        let (fn_types, type_stats) = crate::types::recover(&type_inputs, debug.as_ref(), opts.model, &mut tys);
        drop(type_inputs);
        for (f, t) in funcs.iter_mut().zip(fn_types) {
            f.types = t;
        }

        // 6. Extern declarations for what emitted code calls but doesn't define.
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
                        // (a function that failed only after signature recovery has its signature)
                        let s = guessed.get(&Target::Func(*j)).copied().unwrap_or(sigs[*j]);
                        (funcs[*j].name.clone(), s)
                    }
                    Target::Import(n) => (n.clone(), site_sig(&funcs[fi].targets[k], (0, 0), &sigs, &guessed)),
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
        Program { funcs, externs, extern_of, tys, type_stats, got_addr: syms.got_addr }
    }

    /// The callee of each call site in function `i`, for the emitter. With
    /// `safe`, in safe mode: how the callee takes each argument, or its raw twin
    /// if this call can't lend it a slice. With `safe` and `fast`, for code
    /// emitted in fast mode inside a safe program (a twin, or a function that
    /// `--check` sent back): callees that take slices are called through their
    /// twins.
    fn call_info(&self, i: usize, site: Site, safe: Option<&Summaries>, fast: bool) -> Option<CallInfo> {
        let f = &self.funcs[i];
        let k = f.sites.iter().position(|&s| s == site)?;
        let ext = |name: &str| {
            let e = &self.externs[*self.extern_of.get(name)?];
            Some(CallInfo {
                path: Some(format!("ffi::{}", e.ident)),
                ret: e.sig.ret,
                ret2: e.sig.ret2,
                foreign: true,
                arg_tys: arg_tys(e.sig, &[]),
                ret_ty: e.sig.fret.then_some(TyId::F64),
                ..CallInfo::default()
            })
        };
        let plain = match &f.targets[k] {
            Target::Func(j) => {
                let g = &self.funcs[*j];
                if g.selected && g.ir.is_ok() {
                    let t = g.types.as_ref();
                    Some(CallInfo {
                        path: Some(g.ident.clone()),
                        ret: g.sig.ret,
                        ret2: g.sig.ret2,
                        foreign: false,
                        arg_tys: arg_tys(g.sig, &t.map(|t| t.args.iter().map(|a| a.ty).collect::<Vec<_>>()).unwrap_or_default()),
                        ret_ty: if g.sig.fret { Some(TyId::F64) } else { t.and_then(|t| t.ret) },
                        ..CallInfo::default()
                    })
                } else {
                    ext(&g.name)
                }
            }
            Target::Import(n) => ext(n),
            // through a pointer: integers and the float arguments the call sets up
            Target::Indirect => {
                let (args, fargs) = f.guesses[k];
                let sig = Sig { args, fargs, ..Sig::default() };
                Some(CallInfo { path: None, ret: true, foreign: true, arg_tys: arg_tys(sig, &[]), ..CallInfo::default() })
            }
        };
        let Some(s) = safe else { return plain };
        let plain = plain?;
        // a call to a raw twin: integers, and as unsafe as an FFI call
        let twin = |j: usize| CallInfo { path: Some(s.twin_ident[j].clone()), raw: true, ..plain.clone() };
        if let Target::Func(j) = f.targets[k] {
            if s.twin[j] && (fast || s.raw_sites.contains(&(i, site))) {
                return Some(twin(j));
            }
        }
        if fast {
            return Some(plain);
        }
        let name = match &f.targets[k] {
            Target::Func(j) => Some(self.funcs[*j].name.as_str()),
            Target::Import(n) => Some(n.as_str()),
            Target::Indirect => None,
        };
        Some(match s.callee(self, i, k) {
            Some(sum) => CallInfo {
                args: sum.args.clone(),
                alloc: sum.alloc,
                free: sum.args.first() == Some(&Pass::Free),
                builtin: name.and_then(crate::libc::builtin),
                ..plain
            },
            None => plain,
        })
    }

    /// Emit every selected function that lifted, in parallel: `(source, stats)`,
    /// or `None` for the others. A function is `unsafe` if it does something
    /// unsafe itself or calls a decompiled function that is.
    pub fn emit_all(&self, mode: Mode, global_of: &(dyn Fn(u64) -> Option<String> + Sync)) -> Vec<Option<(String, EmitStats)>> {
        self.emit_all_with(mode, &Options { global_of, ..Options::default() })
    }

    /// `emit_all`, with safe mode's view of the binary's globals and the
    /// functions that `--check` sent back to fast mode. In safe mode a function
    /// with a raw twin has the twin's source right after its own.
    pub fn emit_all_with(&self, mode: Mode, opts: &Options) -> Vec<Option<(String, EmitStats)>> {
        let safe = (mode == Mode::Safe).then(|| self.summaries(opts));
        // Decompiled callers lend byte slices, not structs: only functions no
        // decompiled code calls take `&S` / `&mut S`.
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
                let fast = mode == Mode::Fast || opts.fast.get(i).copied().unwrap_or(false);
                let emit = |ident: &str, fast: bool, src: &mut String| {
                    let call = |site: Site| self.call_info(i, site, safe.as_ref(), fast);
                    let analysis = safe.as_ref().filter(|_| !fast).and_then(|s| s.analyses[i].as_ref());
                    let env = Env {
                        sig: Some(f.sig),
                        call: &call,
                        demote: false,
                        structure: true,
                        global_of: opts.global_of,
                        global_slice: opts.global_slice,
                        analysis,
                        types: f.types.as_ref().map(|t| (t, &self.tys)),
                        struct_args: !called[i],
                    };
                    emit_function_in(ir, ident, if fast { Mode::Fast } else { Mode::Safe }, &env, src)
                };
                let mut src = String::new();
                let mut stats = emit(&f.ident, fast, &mut src);
                if let Some(s) = safe.as_ref().filter(|s| s.twin[i]) {
                    src.push_str(&format!("\n/// `{}` taking integers, for callers that can't lend it a slice.\n", f.ident));
                    let t = emit(&s.twin_ident[i], true, &mut src);
                    stats.twins += 1;
                    stats.twin_raw += t.raw_by.iter().sum::<usize>();
                }
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

    /// Every function's safe-mode borrow analysis, with call summaries (`None`
    /// for those not emitted in safe mode).
    pub fn analyses(&self, opts: &Options) -> Vec<Option<Analysis>> {
        self.summaries(opts).analyses
    }

    /// The identifier each function is called by from data (function pointers in
    /// statics): its raw twin, if safe mode gives it one, since code that calls
    /// through a pointer passes integers.
    pub fn pointer_idents(&self, opts: &Options) -> Vec<String> {
        let s = self.summaries(opts);
        (0..self.funcs.len()).map(|i| if s.twin[i] { s.twin_ident[i].clone() } else { self.funcs[i].ident.clone() }).collect()
    }

    /// Safe mode's whole-program pass (docs/ownership.md, stage 5): the borrow
    /// analysis of every function, with each call seeing its callee's summary,
    /// iterated until no summary changes.
    ///
    /// Summaries start optimistic (a callee ignores every argument) and only get
    /// worse, so this terminates. A callee takes an argument as a slice if its
    /// own code allows it. A call that can't lend one (the pointer comes from
    /// several objects, or from one that isn't safe, or two arguments would
    /// borrow the same bytes mutably) calls the callee's *raw twin* instead: the
    /// same function emitted in fast mode, taking integers. So one caller that
    /// can't lend a slice doesn't take the slice away from the others. Twins are
    /// also made for functions that data points to (called through pointers,
    /// with integers) and for the slice-taking callees of every twin and of every
    /// function emitted in fast mode.
    fn summaries<'o>(&self, opts: &Options<'o>) -> Summaries<'o> {
        let n = self.funcs.len();
        let fast = |i: usize| opts.fast.get(i).copied().unwrap_or(false);
        let emitted = |i: usize| self.funcs[i].selected && self.funcs[i].ir.is_ok();
        let safe_fn: Vec<bool> = (0..n).map(|i| emitted(i) && !fast(i)).collect();
        let by_addr: HashMap<u64, usize> = self.funcs.iter().enumerate().map(|(i, f)| (f.addr, i)).collect();
        let mut taken = vec![false; n];
        for f in &self.funcs {
            if let Ok(ir) = &f.ir {
                for a in address_taken(ir, &self.got_addr) {
                    if let Some(&j) = by_addr.get(&a) {
                        taken[j] = true;
                    }
                }
            }
        }
        for a in opts.address_taken {
            if let Some(&j) = by_addr.get(a) {
                taken[j] = true;
            }
        }
        let mut demoted: Vec<Vec<bool>> =
            self.funcs.iter().map(|f| vec![false; f.ir.as_ref().map_or(0, |ir| ir.blocks[ir.entry].params.len as usize)]).collect();
        let mut callers: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, f) in self.funcs.iter().enumerate() {
            for t in &f.targets {
                if let Target::Func(j) = t {
                    if !callers[*j].contains(&i) {
                        callers[*j].push(i);
                    }
                }
            }
        }
        let mut used: std::collections::HashSet<String> = self.funcs.iter().map(|f| f.ident.clone()).collect();
        let twin_ident = self
            .funcs
            .iter()
            .map(|f| {
                let mut t = format!("{}_raw", f.ident);
                while !used.insert(t.clone()) {
                    t.push('_');
                }
                t
            })
            .collect();
        let mut s = Summaries {
            analyses: (0..n).map(|_| None).collect(),
            summary: (0..n)
                .map(|i| safe_fn[i].then(|| Callee { args: vec![Pass::Ignore; 6 + self.funcs[i].sig.stack_args as usize], ..Callee::default() }))
                .collect(),
            safe_fn,
            global_slice: opts.global_slice,
            raw_sites: std::collections::HashSet::new(),
            twin: vec![false; n],
            twin_ident,
        };
        let mut dirty: Vec<bool> = s.safe_fn.clone();
        for round in 0.. {
            if !dirty.iter().any(|&d| d) {
                break;
            }
            // Give up on precision for whatever still changes after many rounds:
            // every argument an integer is a fixpoint.
            if round == MAX_ROUNDS {
                for i in (0..n).filter(|&i| dirty[i]) {
                    demoted[i].iter_mut().for_each(|d| *d = true);
                }
            }
            let fresh: Vec<(usize, Analysis)> = (0..n)
                .into_par_iter()
                .filter(|&i| dirty[i])
                .map(|i| {
                    let ir = self.funcs[i].ir.as_ref().unwrap();
                    let callee = |site: Site| {
                        if s.raw_sites.contains(&(i, site)) {
                            return None;
                        }
                        let k = self.funcs[i].sites.iter().position(|&x| x == site)?;
                        s.callee(self, i, k)
                    };
                    let dem = |k: usize| demoted[i].get(k).copied().unwrap_or(false);
                    let global_ok = |c: u64| (s.global_slice)(c).is_some();
                    (i, analyze_with(ir, &Ctx { callee: &callee, demoted: &dem, global_ok: &global_ok }))
                })
                .collect();
            dirty = vec![false; n];
            for (i, a) in fresh {
                let f = &self.funcs[i];
                let ir = f.ir.as_ref().unwrap();
                // calls that can't lend a slice go to the callee's raw twin
                for &(site, _) in &a.unprovable {
                    if s.raw_sites.insert((i, site)) {
                        dirty[i] = true;
                    }
                }
                let new = summarize(ir, f.sig, &a);
                if s.summary[i].as_ref() != Some(&new) {
                    s.summary[i] = Some(new);
                    for &c in &callers[i] {
                        dirty[c] |= s.safe_fn[c];
                    }
                }
                s.analyses[i] = Some(a);
            }
        }

        // Raw twins, and the slice-taking callees of every twin.
        let lends = |s: &Summaries, j: usize| s.summary[j].as_ref().is_some_and(|c| c.args.iter().any(|p| matches!(p, Pass::Borrow { .. })));
        let mut work: Vec<usize> = (0..n).filter(|&j| taken[j] && s.safe_fn[j] && lends(&s, j)).collect();
        for &(i, site) in &s.raw_sites {
            if let Some(k) = self.funcs[i].sites.iter().position(|&x| x == site) {
                if let Target::Func(j) = self.funcs[i].targets[k] {
                    work.push(j);
                }
            }
        }
        // what fast-mode code calls
        work.extend((0..n).filter(|&i| emitted(i) && fast(i)));
        while let Some(j) = work.pop() {
            if s.safe_fn[j] && lends(&s, j) {
                if s.twin[j] {
                    continue;
                }
                s.twin[j] = true;
            }
            // a twin (or a fast function) calls its callees in fast mode too
            for t in &self.funcs[j].targets {
                if let Target::Func(h) = *t {
                    if s.safe_fn[h] && lends(&s, h) && !s.twin[h] {
                        work.push(h);
                    }
                }
            }
        }
        s
    }

    /// The recovered structs the emitted functions use, the `simd` module if
    /// they use SSE lane ops, and `mod ffi`, declaring every extern the emitted
    /// code calls; empty if none of these.
    pub fn prelude(&self) -> String {
        let mut roots = Vec::new();
        for f in self.funcs.iter().filter(|f| f.selected && f.ir.is_ok()) {
            let Some(t) = &f.types else { continue };
            roots.extend(t.args.iter().filter_map(|a| a.ty));
            roots.extend(t.ret);
            roots.extend(t.pointee.iter().flatten());
        }
        roots.sort_unstable_by_key(|t| t.index());
        roots.dedup();
        let structs = crate::types::render_structs(&self.tys, roots);
        let mut s = String::new();
        if !structs.is_empty() {
            s.push('\n');
            s.push_str(&structs);
        }
        if self.funcs.iter().any(|f| f.selected && f.ir.as_ref().is_ok_and(crate::simd::uses)) {
            s.push_str(crate::simd::PRELUDE);
        }
        if self.externs.is_empty() {
            return s;
        }
        s.push_str("\n/// Functions the decompiled code calls but doesn't define.\npub mod ffi {\n");
        if self.externs.iter().any(|e| e.sig.ret2) {
            s.push_str("    /// A 16-byte result, returned in rax:rdx.\n    #[repr(C)]\n    pub struct Pair(pub u64, pub u64);\n\n");
        }
        s.push_str("    extern \"C\" {\n");
        let mut ext: Vec<&Extern> = self.externs.iter().collect();
        ext.sort_by(|a, b| a.ident.cmp(&b.ident));
        for e in ext {
            let mut params: Vec<String> = (0..e.sig.args).map(|k| format!("a{k}: u64")).collect();
            params.extend((0..e.sig.stack_args).map(|k| format!("a{}: u64", 6 + k as usize)));
            // an f32 argument or result travels in the same register as an f64
            params.extend((0..e.sig.fargs).map(|j| format!("x{j}: f64")));
            if e.sig.variadic {
                params.push("...".into());
            }
            if e.ident != e.name {
                let _ = writeln!(s, "        #[link_name = {:?}]", e.name);
            }
            let ret = match (e.sig.ret, e.sig.ret2, e.sig.fret) {
                (_, true, _) => " -> Pair",
                (true, _, true) => " -> f64",
                (true, _, _) => " -> u64",
                _ => "",
            };
            let _ = writeln!(s, "        pub fn {}({}){ret};", e.ident, params.join(", "));
        }
        s.push_str("    }\n}\n");
        s
    }
}

/// Safe mode's options for `emit_all_with`.
pub struct Options<'a> {
    /// Rust expression for a constant address into the binary's data.
    pub global_of: &'a (dyn Fn(u64) -> Option<String> + Sync),
    /// The read-only `Bytes` static an address is in, which safe code can index.
    pub global_slice: &'a (dyn Fn(u64) -> Option<String> + Sync),
    /// Functions to emit in fast mode even in safe mode (by index), because
    /// their safe output didn't borrow-check (`--check`).
    pub fast: &'a [bool],
    /// Addresses of functions that data points to (vtables, callbacks): callable
    /// from anywhere, so their arguments stay integers.
    pub address_taken: &'a [u64],
}

impl Default for Options<'_> {
    fn default() -> Self {
        Options { global_of: &|_| None, global_slice: &|_| None, fast: &[], address_taken: &[] }
    }
}

/// The result of `Program::summaries`.
struct Summaries<'a> {
    /// Each safe-mode function's borrow analysis, with call summaries.
    analyses: Vec<Option<Analysis>>,
    /// What each safe-mode function does with its arguments.
    summary: Vec<Option<Callee>>,
    /// Emitted in safe mode.
    safe_fn: Vec<bool>,
    global_slice: &'a (dyn Fn(u64) -> Option<String> + Sync),
    /// Calls (caller, site) that can't lend the callee a slice: they call its
    /// raw twin.
    raw_sites: std::collections::HashSet<(usize, Site)>,
    /// Functions emitted a second time in fast mode, as `twin_ident`.
    twin: Vec<bool>,
    twin_ident: Vec<String>,
}

impl Summaries<'_> {
    /// The summary of the callee of call site `k` in function `i`.
    fn callee(&self, p: &Program, i: usize, k: usize) -> Option<Callee> {
        match &p.funcs[i].targets[k] {
            Target::Func(j) => crate::libc::summary(&p.funcs[*j].name).or_else(|| self.summary[*j].clone()),
            Target::Import(name) => crate::libc::summary(name),
            Target::Indirect => None,
        }
    }
}

/// The entry parameter index holding argument position `k`, if the function uses it.
fn entry_index(f: &Function, k: usize) -> Option<usize> {
    let p = abi::arg_param(f, k)?;
    f.blocks[f.entry].params.get(&f.value_pool).iter().position(|&x| x == p)
}

/// What a function does with each argument, from its borrow analysis.
fn summarize(f: &Function, sig: Sig, a: &Analysis) -> Callee {
    let n = sig.args as usize + sig.stack_args as usize;
    let mut c = Callee { args: Vec::with_capacity(n), ..Callee::default() };
    for k in 0..n {
        let pos = if k < sig.args as usize { k } else { 6 + k - sig.args as usize };
        let pass = match entry_index(f, pos) {
            None => Pass::Ignore,
            Some(e) => {
                let p = &a.params[e];
                if p.returned && pos < 64 {
                    c.ret_from |= 1 << pos;
                }
                match p.class {
                    Class::Shared | Class::Mut if p.reg != crate::borrow::RSP => {
                        Pass::Borrow { mutbl: p.class == Class::Mut, nullable: p.nullable }
                    }
                    Class::NotPointer if !a.escaped.get(e).copied().unwrap_or(true) => Pass::Ignore,
                    _ => Pass::Escape,
                }
            }
        };
        if pos >= c.args.len() {
            c.args.resize(pos, Pass::Ignore);
            c.args.push(pass);
        } else {
            c.args[pos] = pass;
        }
    }
    c
}

/// Addresses used as values other than to call them (function pointers),
/// directly or loaded from a GOT slot.
fn address_taken(f: &Function, got: &HashMap<u64, u64>) -> Vec<u64> {
    let mut callees = std::collections::HashSet::new();
    for (_, blk) in f.blocks.iter() {
        for &id in blk.insts.get(&f.value_pool) {
            if let InstKind::Call { callee, .. } = f.insts[id].kind {
                callees.insert(callee);
            }
        }
        if let Terminator::TailCall { callee, .. } = blk.term {
            callees.insert(callee);
        }
    }
    let mut out = Vec::new();
    for (_, blk) in f.blocks.iter() {
        for &id in blk.insts.get(&f.value_pool) {
            match f.insts[id].kind {
                InstKind::IntToPtr(v) => {
                    if let (Some(a), false) = (konst(f, v), callees.contains(&id)) {
                        out.push(a);
                    }
                }
                // `mov rax, [rip+slot]`, then something other than `call rax`
                InstKind::Load { ptr, .. } if !used_only_as_callee(f, id, &callees) => {
                    if let InstKind::IntToPtr(s) = f.insts[ptr].kind {
                        if let Some(&a) = konst(f, s).and_then(|s| got.get(&s)) {
                            out.push(a);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    out
}

/// Is `v` used only as the callee of calls (directly or through `IntToPtr`)?
fn used_only_as_callee(f: &Function, v: ValueId, callees: &std::collections::HashSet<ValueId>) -> bool {
    let mut ok = true;
    for (_, blk) in f.blocks.iter() {
        for &id in blk.insts.get(&f.value_pool) {
            let k = f.insts[id].kind;
            let mut uses = false;
            crate::verify::for_each_operand(k, f, |o| uses |= o == v);
            if !uses {
                continue;
            }
            let as_callee = match k {
                InstKind::Call { callee, args } => callee == v && !args.get(&f.value_pool).contains(&v),
                InstKind::IntToPtr(_) => callees.contains(&id),
                _ => false,
            };
            ok &= as_callee;
        }
        let mut in_term = false;
        crate::emit::term_operands(f, blk.term, |o| in_term |= o == v);
        if in_term {
            ok &= matches!(blk.term, Terminator::TailCall { callee, args } if callee == v && !args.get(&f.value_pool).contains(&v));
        }
    }
    ok
}

/// The Rust types of a call's arguments, as `emit` wants them: `int` for the
/// integer ones (register, then stack; `None` for `u64`), then `f64` for each
/// float argument.
fn arg_tys(sig: Sig, int: &[Option<TyId>]) -> Vec<Option<TyId>> {
    let n = sig.args as usize + sig.stack_args as usize;
    let mut v: Vec<Option<TyId>> = int.iter().copied().chain(std::iter::repeat(None)).take(n).collect();
    v.extend(std::iter::repeat_n(Some(TyId::F64), sig.fargs as usize));
    v
}

fn site_sig(t: &Target, (args, fargs): (u8, u8), sigs: &[Sig], guessed: &HashMap<Target, Sig>) -> Sig {
    match t {
        Target::Func(j) => guessed.get(t).copied().unwrap_or(sigs[*j]),
        Target::Import(n) => crate::libc::lookup(n).or_else(|| guessed.get(t).copied()).unwrap_or_default(),
        Target::Indirect => Sig { args, fargs, ret: true, ..Sig::default() },
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
