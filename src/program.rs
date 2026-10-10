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
use crate::opt::{clean, merge_straight, split_returns};
use crate::types::{FnTypes, TypeModel, TypeStats};
use crate::verify::verify;
use object::{Object, ObjectKind, ObjectSection, ObjectSymbol, RelocationTarget, SectionKind};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;

/// Signature inference stops after this many rounds even if a cycle of
/// signatures keeps changing (it is monotone in practice, so this is a backstop).
const MAX_ROUNDS: usize = 64;
/// The round from which summaries only grow.
const WIDEN_ROUND: usize = 16;

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
    /// Per site, for a callee without an inferred signature: the integer,
    /// float and stack arguments it sets up, and whether it reads a float
    /// result (xmm0, not rax) after the call.
    guesses: Vec<(u8, u8, u8, bool)>,
    /// Per site, the argument registers passed on unchanged from entry
    /// (`abi::passed_through`).
    passes: Vec<u8>,
    /// Per site, the site whose xmm0 it passes on (`abi::passes_xmm0`).
    xmm0_from: Vec<Option<usize>>,
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
    /// Training examples for a type model from the debug info's prototypes, if
    /// `BuildOptions::dataset` asked for them; `func` indexes `funcs`.
    pub examples: Vec<crate::types::Example>,
    /// GOT slot -> the address in the binary the loader fills it with.
    got_addr: HashMap<u64, u64>,
}

/// How `Program::build_with` lifts functions and recovers types.
#[derive(Copy, Clone)]
pub struct BuildOptions<'a> {
    /// Keep an instruction the lifter has no model of as inline assembly
    /// (`Lifter::asm`) instead of failing its function.
    pub asm: bool,
    /// Use DWARF debug info when the file has it.
    pub dwarf: bool,
    /// Proposes argument and return types; the facts accept or reject them.
    pub model: Option<&'a dyn TypeModel>,
    /// Collect `Program::examples` (`--emit dataset`).
    pub dataset: bool,
}

impl Default for BuildOptions<'_> {
    fn default() -> Self {
        BuildOptions { asm: true, dwarf: true, model: None, dataset: false }
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
/// tables. Empty for relocatable objects, whose tables are relocations. In a
/// PIE the dynamic loader fills in a table of absolute addresses (a computed
/// `goto`'s labels in `.data.rel.ro`); those slots hold their targets here.
fn loaded_sections(data: &[u8]) -> Vec<(u64, Vec<u8>)> {
    use object::elf::R_X86_64_RELATIVE;
    use object::RelocationFlags;
    let Ok(file) = object::File::parse(data) else { return Vec::new() };
    if file.kind() == ObjectKind::Relocatable {
        return Vec::new();
    }
    let mut out: Vec<(u64, Vec<u8>)> = file
        .sections()
        .filter(|s| s.address() != 0 && s.kind() != SectionKind::UninitializedData)
        .filter_map(|s| Some((s.address(), s.data().ok().filter(|d| !d.is_empty())?.to_vec())))
        .collect();
    for (at, r) in file.dynamic_relocations().into_iter().flatten() {
        if r.flags() != (RelocationFlags::Elf { r_type: R_X86_64_RELATIVE }) {
            continue;
        }
        if let Some((a, b)) = out.iter_mut().find(|(a, b)| *a <= at && at + 8 <= *a + b.len() as u64) {
            let k = (at - *a) as usize;
            b[k..k + 8].copy_from_slice(&(r.addend() as u64).to_le_bytes());
        }
    }
    out
}

/// The strongly connected components of the graph `succ` (Tarjan's,
/// without recursion): each node's component, numbered from 0.
fn components(succ: &[Vec<usize>]) -> Vec<usize> {
    let n = succ.len();
    let (mut index, mut low) = (vec![usize::MAX; n], vec![0usize; n]);
    let (mut on, mut stack, mut comp) = (vec![false; n], Vec::new(), vec![0usize; n]);
    let (mut next, mut ncomp) = (0, 0);
    for root in 0..n {
        if index[root] != usize::MAX {
            continue;
        }
        let mut work = vec![(root, 0usize)];
        while let Some(&mut (v, ref mut k)) = work.last_mut() {
            if *k == 0 && index[v] == usize::MAX {
                index[v] = next;
                low[v] = next;
                next += 1;
                stack.push(v);
                on[v] = true;
            }
            if let Some(&w) = succ[v].get(*k) {
                *k += 1;
                if index[w] == usize::MAX {
                    work.push((w, 0));
                } else if on[w] {
                    low[v] = low[v].min(index[w]);
                }
                continue;
            }
            work.pop();
            if let Some(&(u, _)) = work.last() {
                low[u] = low[u].min(low[v]);
            }
            if low[v] == index[v] {
                let at = stack.iter().rposition(|&w| w == v).unwrap();
                for w in stack.drain(at..) {
                    on[w] = false;
                    comp[w] = ncomp;
                }
                ncomp += 1;
            }
        }
    }
    comp
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
        let loaded = file.map(loaded_sections).unwrap_or_default();
        let sections: Vec<(u64, &[u8])> = loaded.iter().map(|(a, b)| (*a, b.as_slice())).collect();
        // GOT-relative data (`opt::fold_got_offsets`): the GOT's extent, and the
        // allocated sections a GOT-relative address may land in.
        let alloc: Vec<(u64, u64, String)> = file
            .and_then(|d| object::File::parse(d).ok())
            .filter(|o| o.kind() != ObjectKind::Relocatable)
            .map(|o| {
                o.sections()
                    .filter(|s| s.address() != 0 && s.size() != 0)
                    .filter(|s| matches!(s.kind(), SectionKind::Data | SectionKind::UninitializedData | SectionKind::ReadOnlyData | SectionKind::ReadOnlyString))
                    .map(|s| (s.address(), s.address() + s.size(), s.name().unwrap_or("").to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let got = alloc
            .iter()
            .filter(|s| matches!(s.2.as_str(), ".got" | ".got.plt"))
            .map(|s| (s.0, s.1))
            .reduce(|a, b| (a.0.min(b.0), a.1.max(b.1)));
        let lands = |a: u64| alloc.iter().any(|s| s.0 <= a && a < s.1 && !matches!(s.2.as_str(), ".got" | ".got.plt"));
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
                        l.asm = opts.asm;
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
                                // Code that hides from a linear sweep may also hide behind branches
                                // that always go one way. (Compiled code has them too, Rust's
                                // overflow checks on constants, but its panic paths are evidence
                                // the register summaries rely on, so it is left as it is.)
                                if lifter.followed_flow() {
                                    crate::opt::fold_branches(&mut f);
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
                let (sites, targets, guesses, passes, xmm0_from) = match &ir {
                    Ok(f) => {
                        let sites = abi::sites(f);
                        let targets = sites.iter().map(|&s| resolve(f, s)).collect();
                        let args: Vec<u8> = sites.iter().map(|&s| abi::guess_args(f, s)).collect();
                        let stack = abi::guess_stack(f, &sites, &args);
                        let guesses = sites.iter().zip(args).zip(stack).map(|((&s, a), st)| (a, abi::guess_fargs(f, s), st, false)).collect();
                        let passes = sites.iter().map(|&s| abi::passed_through(f, s)).collect();
                        let xmm0_from = sites.iter().map(|&s| abi::passes_xmm0(f, &sites, s)).collect();
                        (sites, targets, guesses, passes, xmm0_from)
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
                    passes,
                    xmm0_from,
                }
            })
            .collect();

        // What callers set up and whether there are any, for functions with
        // calls that don't return (`abi::infer`). A tail call passes on
        // whatever the callee returns: its callers read what the caller's own
        // callers read (rax if it has none), each round below.
        let mut set_args = vec![0u8; funcs.len()];
        let mut called = vec![0u8; funcs.len()];
        for f in &funcs {
            for ((t, g), s) in f.targets.iter().zip(&f.guesses).zip(&f.sites) {
                if let Target::Func(j) = t {
                    set_args[*j] = set_args[*j].max(g.0);
                    called[*j] |= abi::CALLED | if matches!(s, Site::Tail(_)) { abi::TAIL_CALLED } else { 0 };
                }
            }
        }
        for (a, c) in set_args.iter_mut().zip(&called) {
            if *c == 0 {
                *a = 6;
            }
        }
        // A call to code whose signature isn't known (through a pointer, an
        // import outside the libc table, a function that didn't lift) also
        // passes the registers the function forwards from its own entry, as
        // far as its callers set them up: `malloc(n) { return hooks.malloc(n); }`.
        let failed: Vec<bool> = funcs.iter().map(|f| f.ir.is_err()).collect();
        for (f, &set) in funcs.iter_mut().zip(&set_args) {
            for k in 0..f.sites.len() {
                let unknown = match &f.targets[k] {
                    Target::Indirect => true,
                    Target::Import(n) => crate::libc::lookup(n).is_none(),
                    Target::Func(j) => failed[*j],
                };
                let pass = f.passes[k] & ((1u16 << set.min(6)) - 1) as u8;
                if unknown && pass != 0 {
                    let n = 8 - pass.leading_zeros() as u8;
                    f.guesses[k].0 = f.guesses[k].0.max(n);
                }
            }
        }

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
        let mut stack_args: Vec<u8> = funcs.par_iter().map(|f| f.ir.as_ref().map_or(0, abi::stack_args)).collect();
        // A variadic function of the program reads its stack arguments through
        // a `va_list` (the address of the first one), not one by one: it takes
        // as many as any call passes.
        let va: Vec<bool> = funcs.par_iter().map(|f| f.ir.as_ref().is_ok_and(|ir| ir.variadic || abi::stores_stack_area(ir))).collect();
        for f in &funcs {
            for (t, g) in f.targets.iter().zip(&f.guesses) {
                if let Target::Func(j) = t {
                    if va[*j] {
                        stack_args[*j] = stack_args[*j].max(g.2);
                    }
                }
            }
        }
        // A tail call passes on the caller's own stack arguments, where the
        // callee reads its own: the caller takes as many as the callee does.
        loop {
            let mut changed = false;
            for (i, f) in funcs.iter().enumerate() {
                for (t, s) in f.targets.iter().zip(&f.sites) {
                    if let (Target::Func(j), Site::Tail(_)) = (t, s) {
                        if stack_args[*j] > stack_args[i] {
                            stack_args[i] = stack_args[*j];
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // 3. Signatures, to a fixpoint (Jacobi rounds: every function from the
        //    previous round's callee signatures). A function returns rdx too
        //    (`ret2`) only if some caller reads rdx after calling it; a caller
        //    reading xmm0 says it returns a float. A function whose inputs (its
        //    own signature, its callees', and what its callers read) didn't
        //    change last round gets the same result again, so only the others
        //    are re-inferred.
        let mut sigs: Vec<Sig> = stack_args.iter().map(|&s| Sig { stack_args: s, ..Sig::default() }).collect();
        // Functions that call each other, directly or in a cycle (`next()`
        // tail-calling `step()`, which tail-calls `next()`), are inferred
        // together: they start from keeping every register and returning a
        // value across the calls among them, and drop what doesn't hold, to
        // a fixpoint (a tree walk keeping `r8` for its caller; a matcher
        // returning `match(s + 1, p)`). Starting from nothing, a cycle could
        // never show that it returns what each member returns.
        let succ: Vec<Vec<usize>> = funcs
            .iter()
            .map(|f| f.targets.iter().filter_map(|t| if let Target::Func(j) = t { Some(*j) } else { None }).collect())
            .collect();
        let comp = components(&succ);
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for (i, &c) in comp.iter().enumerate() {
            if groups.len() <= c {
                groups.resize(c + 1, Vec::new());
            }
            groups[c].push(i);
        }
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
        wanted_by.clone_from(&called);
        let mut results: Vec<(Sig, Vec<u8>)> = sigs.iter().map(|&s| (s, Vec::new())).collect();
        let mut dirty = vec![true; funcs.len()];
        for _round in 0..MAX_ROUNDS {
            let next: Vec<Vec<(usize, Sig, Vec<u8>)>> = groups
                .par_iter()
                .map(|g| {
                    let g: Vec<usize> = g.iter().copied().filter(|&i| funcs[i].ir.is_ok()).collect();
                    if !g.iter().any(|&i| dirty[i]) {
                        return Vec::new();
                    }
                    let cyclic = g.len() > 1 || funcs[g[0]].targets.iter().any(|t| *t == Target::Func(g[0]));
                    // what each member keeps and returns, for the calls among them
                    let mut own: Vec<(u16, u32, bool)> = g
                        .iter()
                        .map(|&i| if cyclic { (abi::CALLER_SAVED, u32::MAX, true) } else { (sigs[i].preserves, sigs[i].xpreserves, sigs[i].ret) })
                        .collect();
                    loop {
                        let out: Vec<(usize, Sig, Vec<u8>)> = g
                            .iter()
                            .map(|&i| {
                                let f = &funcs[i];
                                let callee = |k: usize| match f.targets[k] {
                                    Target::Func(j) if cyclic && comp[j] == comp[i] => {
                                        let (preserves, xpreserves, ret) = own[g.iter().position(|&m| m == j).unwrap()];
                                        Sig { preserves, xpreserves, ret, ..site_sig(&f.targets[k], f.guesses[k], &sigs, &guessed) }
                                    }
                                    _ => site_sig(&f.targets[k], f.guesses[k], &sigs, &guessed),
                                };
                                let r = abi::infer(f.ir.as_ref().unwrap(), &f.sites, &callee, sigs[i], wanted_by[i], set_args[i]);
                                (i, Sig { stack_args: stack_args[i], variadic: r.sig.variadic || va[i], ..r.sig }, r.reads)
                            })
                            .collect();
                        let kept: Vec<(u16, u32, bool)> = own
                            .iter()
                            .zip(&out)
                            .map(|(o, (_, s, _))| (o.0 & s.preserves, o.1 & s.xpreserves, o.2 && s.ret))
                            .collect();
                        if !cyclic || kept == own {
                            break out;
                        }
                        own = kept;
                    }
                })
                .collect();
            for (i, sig, reads) in next.into_iter().flatten() {
                results[i] = (sig, reads);
            }
            let mut wanted = called.clone();
            for (f, w) in funcs.iter().zip(&wanted_by) {
                if w & abi::CALLED == 0 {
                    for (t, s) in f.targets.iter().zip(&f.sites) {
                        if let (Target::Func(j), Site::Tail(_)) = (t, s) {
                            wanted[*j] |= abi::READ_RAX;
                        }
                    }
                }
            }
            for (i, (f, (_, read))) in funcs.iter().zip(&results).enumerate() {
                for ((t, &r), s) in f.targets.iter().zip(read).zip(&f.sites) {
                    if let Target::Func(j) = t {
                        wanted[*j] |= r;
                        // a tail call's callers read what this one's callers read
                        if matches!(s, Site::Tail(_)) && wanted_by[i] & abi::CALLED != 0 {
                            wanted[*j] |= wanted_by[i] & (abi::READ_RAX | abi::READ_RDX | abi::READ_XMM0);
                        }
                    }
                }
            }
            dirty.iter_mut().for_each(|d| *d = false);
            let mut changed = false;
            // A callee without a signature returns a float if callers read
            // xmm0 after calling it and none reads rax (`strtod`, `pow`).
            let mut float_ret: HashMap<Target, (bool, bool)> = HashMap::new();
            for (i, f) in funcs.iter_mut().enumerate() {
                for (k, &r) in results[i].1.iter().enumerate() {
                    let (x, a) = (r & abi::READ_XMM0 != 0, r & abi::READ_RAX != 0);
                    match &f.targets[k] {
                        Target::Indirect if f.guesses[k].3 != (x && !a) => {
                            f.guesses[k].3 = x && !a;
                            changed = true;
                            dirty[i] = true;
                        }
                        t if guessed.contains_key(t) => {
                            let e = float_ret.entry(t.clone()).or_default();
                            (e.0, e.1) = (e.0 || x, e.1 || a);
                        }
                        _ => {}
                    }
                }
            }
            let mut raised = Vec::new();
            // ... and passes a float on to code without a signature when it
            // got it from a call that returns one
            for (i, f) in funcs.iter_mut().enumerate() {
                for k in 0..f.sites.len() {
                    // (or a variadic one, which takes what the call sets up)
                    let unknown = match &f.targets[k] {
                        Target::Indirect => true,
                        Target::Import(n) if crate::libc::lookup(n).is_some_and(|s| s.variadic) => true,
                        t => guessed.contains_key(t),
                    };
                    let Some(j) = f.xmm0_from[k].filter(|_| unknown && f.guesses[k].1 == 0) else { continue };
                    if site_sig(&f.targets[j], f.guesses[j], &sigs, &guessed).fret {
                        f.guesses[k].1 = 1;
                        raised.push(f.targets[k].clone());
                        changed = true;
                        dirty[i] = true;
                    }
                }
            }
            for t in raised.drain(..) {
                if let Some(g) = guessed.get_mut(&t) {
                    g.fargs = g.fargs.max(1);
                }
            }
            for (t, (x, a)) in float_ret {
                let g = guessed.get_mut(&t).unwrap();
                if g.fret != (x && !a) {
                    g.fret = x && !a;
                    changed = true;
                    for (i, f) in funcs.iter().enumerate() {
                        if f.targets.contains(&t) {
                            dirty[i] = true;
                        }
                    }
                }
            }
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
                .map(|k| site_sig(&f.targets[k], f.guesses[k], &sigs, &guessed))
                .collect();
            f.sites = abi::apply(ir, sigs[i], &f.sites, &|k| shapes[k]);
            // one `return` per path; the calls it copies are new sites
            let (made, copied) = split_returns(ir);
            // one entry per loop, so the loop's own edges skip the `match bb`
            let entries = crate::dispatch::single_entry(ir);
            // (after `apply`, which gives the GOT's address back to a register a call keeps)
            let folded = got.map_or(0, |g| crate::opt::fold_got_offsets(ir, g, &lands));
            if made + entries + folded > 0 {
                clean(ir);
            }
            merge_straight(ir);
            for (old, new) in copied {
                let k = f.sites.iter().position(|&s| s == old).expect("a call site");
                f.sites.push(new);
                f.targets.push(f.targets[k].clone());
                f.guesses.push(f.guesses[k]);
                f.passes.push(f.passes[k]);
                f.xmm0_from.push(None);
            }
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
        let examples = match (&debug, opts.dataset) {
            (Some(d), true) => crate::types::examples(&type_inputs, d, &tys),
            _ => Vec::new(),
        };
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
                    Target::Import(n) => (n.clone(), site_sig(&funcs[fi].targets[k], (0, 0, 0, false), &sigs, &guessed)),
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
        Program { funcs, externs, extern_of, tys, type_stats, examples, got_addr: syms.got_addr }
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
            // a variadic one's extra arguments: as many as this call sets up
            // (the floats among them go in xmm registers, as `f64`s)
            let (args, fargs, stack, _) = f.guesses[k];
            let sig = match e.sig.variadic {
                true => Sig { args: e.sig.args.max(args), fargs: e.sig.fargs.max(fargs), stack_args: e.sig.stack_args.max(stack), ..e.sig },
                false => e.sig,
            };
            Some(CallInfo {
                path: Some(format!("ffi::{}", e.ident)),
                ret: e.sig.ret,
                ret2: e.sig.ret2,
                foreign: true,
                arg_tys: arg_tys(sig, &[]),
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
                let (args, fargs, _, fret) = f.guesses[k];
                let sig = Sig { args, fargs, ..Sig::default() };
                Some(CallInfo { path: None, ret: true, foreign: true, arg_tys: arg_tys(sig, &[]), ret_ty: fret.then_some(TyId::F64), ..CallInfo::default() })
            }
        };
        let Some(s) = safe else { return plain };
        let plain = plain?;
        // a call to a raw twin: integers, and as unsafe as an FFI call
        // (with the callee's summary: the arguments it borrows are passed as
        // pointers made from the caller's slices)
        let twin = |j: usize| CallInfo {
            path: Some(s.twin_ident[j].clone()),
            raw: true,
            args: s.callee(self, i, k).map(|c| c.args).unwrap_or_default(),
            ..plain.clone()
        };
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
        // The address of a function in the output (`lea rdi, [rip+f]`, a callback)
        // is that function, or its raw twin: code calling through a pointer
        // passes integers.
        let by_addr: HashMap<u64, usize> = self.funcs.iter().enumerate().map(|(i, f)| (f.addr, i)).collect();
        let fn_ptr = |addr: u64| {
            let &j = by_addr.get(&addr)?;
            let f = &self.funcs[j];
            if !f.selected || f.ir.is_err() {
                return None;
            }
            let ident = match safe.as_ref() {
                Some(s) if s.twin[j] => &s.twin_ident[j],
                _ => &f.ident,
            };
            Some(format!("({ident} as u64)"))
        };
        let global_of = |addr: u64| fn_ptr(addr).or_else(|| (opts.global_of)(addr));
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
                        global_of: &global_of,
                        global_end: opts.global_end,
                        global_before: opts.global_before,
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
                // (a function that isn't selected has no identifier, and no twin)
                if f.ident.is_empty() {
                    return String::new();
                }
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
                        let k = self.funcs[i].sites.iter().position(|&x| x == site)?;
                        if s.raw_sites.contains(&(i, site)) {
                            // the raw twin: what it borrows, it still only accesses
                            let c = s.callee(self, i, k)?;
                            return Some(Callee { args: c.args, raw: true, stores: c.stores, ..Callee::default() });
                        }
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
                let mut new = summarize(ir, f.sig, &a);
                // Past the first rounds, a summary only grows, so cycles whose
                // summaries feed back on each other settle.
                if round >= WIDEN_ROUND {
                    if let Some(old) = &s.summary[i] {
                        new = widen(old, new);
                    }
                }
                // An argument it takes as a slice that callers don't lend one for
                // (it escapes through what it is stored into, or the widened
                // summary says so) takes an integer, so the two agree.
                for (pos, pass) in new.args.iter().enumerate() {
                    let Some(e) = entry_index(ir, pos) else { continue };
                    let p = &a.params[e];
                    let slice = matches!(p.class, Class::Shared | Class::Mut) && p.reg != crate::borrow::RSP;
                    if slice && !matches!(pass, Pass::Borrow { .. }) && !demoted[i][e] {
                        demoted[i][e] = true;
                        dirty[i] = true;
                    }
                }
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
        let p = self.prelude_parts(false);
        let mut s = p.structs;
        if p.simd {
            s.push_str(crate::simd::PRELUDE);
        }
        if !p.ffi.is_empty() {
            s.push_str("\n/// Functions the decompiled code calls but doesn't define.\npub mod ffi {\n");
            s.push_str(&p.ffi);
            s.push_str("}\n");
        }
        s
    }

    /// `prelude`'s pieces, for output split into files (`project.rs`). A
    /// function of the binary that failed to lift is a `todo!()` in `mod ffi`
    /// instead of an extern, so the output links; with `stub_own`, so is one
    /// that isn't selected.
    pub fn prelude_parts(&self, stub_own: bool) -> PreludeParts {
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
        let mut p = PreludeParts {
            structs: if structs.is_empty() { String::new() } else { format!("\n{structs}") },
            simd: self.funcs.iter().any(|f| f.selected && f.ir.as_ref().is_ok_and(crate::simd::uses)),
            ffi: String::new(),
        };
        if self.externs.is_empty() {
            return p;
        }
        let s = &mut p.ffi;
        if self.externs.iter().any(|e| e.sig.ret2) {
            s.push_str("    /// A 16-byte result, returned in rax:rdx.\n    #[repr(C)]\n    pub struct Pair(pub u64, pub u64);\n\n");
        }
        // A selected function that didn't lift is stubbed either way: an
        // extern for it would never resolve.
        let own: std::collections::HashSet<&str> = self
            .funcs
            .iter()
            .filter(|f| stub_own || (f.selected && f.ir.is_err()))
            .map(|f| f.name.as_str())
            .collect();
        let mut ext: Vec<&Extern> = self.externs.iter().collect();
        ext.sort_by(|a, b| a.ident.cmp(&b.ident));
        let (stubs, ext): (Vec<&Extern>, Vec<&Extern>) =
            ext.into_iter().partition(|e| !e.sig.variadic && own.contains(e.name.as_str()));
        let decl = |e: &Extern| {
            let mut params: Vec<String> = (0..e.sig.args).map(|k| format!("a{k}: u64")).collect();
            params.extend((0..e.sig.stack_args).map(|k| format!("a{}: u64", 6 + k as usize)));
            // an f32 argument or result travels in the same register as an f64
            params.extend((0..e.sig.fargs).map(|j| format!("x{j}: f64")));
            if e.sig.variadic {
                params.push("...".into());
            }
            let ret = match (e.sig.ret, e.sig.ret2, e.sig.fret) {
                (_, true, _) => " -> Pair",
                (true, _, true) => " -> f64",
                (true, _, _) => " -> u64",
                _ => "",
            };
            format!("fn {}({}){ret}", e.ident, params.join(", "))
        };
        if !ext.is_empty() {
            s.push_str("    extern \"C\" {\n");
            for e in ext {
                if e.ident != e.name {
                    let _ = writeln!(s, "        #[link_name = {:?}]", e.name);
                }
                let _ = writeln!(s, "        pub {};", decl(e));
            }
            s.push_str("    }\n");
        }
        if !stubs.is_empty() {
            s.push_str("\n    // Functions of the binary that aren't in the output.\n");
        }
        // a function that didn't lift panics with the reason, as its own stub does
        let failed: HashMap<&str, &str> =
            self.funcs.iter().filter_map(|f| Some((f.name.as_str(), f.ir.as_ref().err()?.as_str()))).collect();
        for e in stubs {
            let why = match failed.get(e.name.as_str()) {
                Some(err) => format!("not lifted: {err}"),
                None => format!("{} isn't decompiled", e.name),
            };
            let _ = writeln!(s, "    pub unsafe {} {{\n        todo!({why:?})\n    }}", decl(e));
        }
        p
    }
}

/// What `Program::prelude` is made of.
pub struct PreludeParts {
    /// The recovered structs, or empty.
    pub structs: String,
    /// Whether to include `simd::PRELUDE`.
    pub simd: bool,
    /// The body of `mod ffi`, indented one level, or empty if nothing is extern.
    pub ffi: String,
}

/// Safe mode's options for `emit_all_with`.
pub struct Options<'a> {
    /// Rust expression for a constant address into the binary's data.
    pub global_of: &'a (dyn Fn(u64) -> Option<String> + Sync),
    /// The end of the static holding an address, as another address (`Env::global_end`).
    pub global_end: &'a (dyn Fn(u64, u64) -> Option<String> + Sync),
    /// An address as an offset back from the static after it (`Env::global_before`).
    pub global_before: &'a (dyn Fn(u64) -> Option<String> + Sync),
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
        Options { global_of: &|_| None, global_end: &|_, _| None, global_before: &|_| None, global_slice: &|_| None, fast: &[], address_taken: &[] }
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

/// A summary at least as conservative as both.
fn widen(old: &Callee, new: Callee) -> Callee {
    // (`old`, `new`): the function's own analysis gave `new`, and an argument
    // it doesn't take as a slice can't be lent one
    let pass = |a: Pass, b: Pass| match (a, b) {
        (Pass::Borrow { mutbl, keeps, .. }, Pass::Ignore) => Pass::Raw { mutbl, keeps },
        (Pass::Ignore, x) | (x, Pass::Ignore) => x,
        (
            Pass::Borrow { mutbl: m1, nullable: n1, len: l1, keeps: k1 },
            Pass::Borrow { mutbl: m2, nullable: n2, len: l2, keeps: k2 },
        ) => Pass::Borrow { mutbl: m1 | m2, nullable: n1 | n2, len: l1.zip(l2).map(|(a, b)| a.max(b)), keeps: k1 | k2 },
        (Pass::Borrow { mutbl: m1, keeps: k1, .. } | Pass::Raw { mutbl: m1, keeps: k1 }, Pass::Borrow { mutbl: m2, keeps: k2, .. } | Pass::Raw { mutbl: m2, keeps: k2 }) => {
            Pass::Raw { mutbl: m1 | m2, keeps: k1 | k2 }
        }
        (a, b) if a == b => a,
        _ => Pass::Escape,
    };
    let n = old.args.len().max(new.args.len());
    let at = |c: &Callee, k: usize| c.args.get(k).copied().unwrap_or(Pass::Escape);
    let mut stores = old.stores.clone();
    stores.extend_from_slice(&new.stores);
    stores.sort();
    stores.dedup();
    Callee {
        args: (0..n).map(|k| pass(at(old, k), at(&new, k))).collect(),
        ret_from: old.ret_from | new.ret_from,
        ret2_from: old.ret2_from | new.ret2_from,
        ret_contents: old.ret_contents | new.ret_contents,
        ret2_contents: old.ret2_contents | new.ret2_contents,
        stores,
        ..new
    }
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
                if p.returned2 && pos < 64 {
                    c.ret2_from |= 1 << pos;
                }
                if p.returns_contents.0 && pos < 64 {
                    c.ret_contents |= 1 << pos;
                }
                if p.returns_contents.1 && pos < 64 {
                    c.ret2_contents |= 1 << pos;
                }
                // where it stores the argument, or what it loads from its
                // object, by position; into a parameter that isn't an argument,
                // or at an offset it doesn't know, the argument escapes or keeps.
                // (An argument it never dereferences may be an integer: one stored
                // as it is escapes, so the caller doesn't take it for a pointer.)
                let (mut keeps, mut escapes) = (p.keeps, false);
                if p.reg != crate::borrow::RSP {
                    for &(j, off, deref) in &p.into {
                        let into = (0..n).map(|q| if q < sig.args as usize { q } else { 6 + q - sig.args as usize }).find(|&q| entry_index(f, q) == Some(j as usize));
                        match (into, off) {
                            (Some(q), Some(off)) if q <= u8::MAX as usize && pos <= u8::MAX as usize && (deref || p.class != Class::NotPointer) => {
                                c.stores.push(crate::borrow::Store { from: pos as u8, into: q as u8, off, deref })
                            }
                            _ if deref => keeps = true,
                            _ => escapes = true,
                        }
                    }
                }
                match p.class {
                    _ if escapes => Pass::Escape,
                    Class::Shared | Class::Mut if p.reg != crate::borrow::RSP => {
                        Pass::Borrow { mutbl: p.class == Class::Mut, nullable: p.nullable, len: p.extent, keeps }
                    }
                    Class::NotPointer if !a.escaped.get(e).copied().unwrap_or(true) => Pass::Ignore,
                    Class::Raw if p.during_call => Pass::Raw { mutbl: a.written.get(e).copied().unwrap_or(true), keeps },
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
    c.stores.sort();
    c.stores.dedup();
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

/// What a call site passes: the callee's signature, with a variadic one's
/// extra arguments as far as the site sets registers up, and the stack
/// arguments it stores after all six registers.
fn site_sig(t: &Target, (args, fargs, stack, fret): (u8, u8, u8, bool), sigs: &[Sig], guessed: &HashMap<Target, Sig>) -> Sig {
    let s = match t {
        Target::Func(j) => guessed.get(t).copied().unwrap_or(sigs[*j]),
        Target::Import(n) => crate::libc::lookup(n).or_else(|| guessed.get(t).copied()).unwrap_or_default(),
        Target::Indirect => Sig { args, fargs, ret: true, fret, ..Sig::default() },
    };
    match (s.variadic, t) {
        // one of the program's own: it takes (and reads its register save area
        // from) every register, but only those this call sets up mean anything
        (true, Target::Func(_)) => Sig { unset: (s.args.saturating_sub(args), s.fargs.saturating_sub(fargs), s.stack_args.saturating_sub(stack)), ..s },
        (true, _) => Sig { args: s.args.max(args), fargs: s.fargs.max(fargs), stack_args: s.stack_args.max(stack), ..s },
        (false, _) => s,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn with(args: Vec<Pass>) -> Callee {
        Callee { args, ..Callee::default() }
    }

    #[test]
    fn widening_never_lends_a_slice_the_function_doesnt_take() {
        let borrow = Pass::Borrow { mutbl: true, nullable: false, len: Some(8), keeps: false };
        // callers lent a slice before, and now the function takes an integer
        let w = widen(&with(vec![borrow]), with(vec![Pass::Ignore]));
        assert_eq!(w.args, vec![Pass::Raw { mutbl: true, keeps: false }]);
        // growing from the optimistic start is unchanged
        let w = widen(&with(vec![Pass::Ignore]), with(vec![borrow]));
        assert_eq!(w.args, vec![borrow]);
    }
}
