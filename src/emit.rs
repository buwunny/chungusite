//! IR -> Rust source. One emitter for both modes (docs/ir.md).
//!
//! Every IR value becomes a Rust integer (`u8`..`u64`, or `bool` for comparisons).
//! Pointers are `u64` addresses, so the output always type-checks no matter how a
//! register is used, and each memory access becomes an explicit read or write.
//!
//! With recovered types (`Env::types`, from `types.rs`) integers get their
//! signedness (`i32` ...), the signature gets real argument and return types
//! (`n: i32`, `p: *mut Node`, `-> bool`), and a load or store at a known offset
//! from a pointer to a struct becomes a field access (`(*p).count`). Arguments
//! are converted to the `u64` registers the body works on in a short prologue.
//!
//! * **fast**: every load and store is a raw, unaligned access inside `unsafe { }`,
//!   and the function is an `unsafe fn` taking every argument as `u64`.
//! * **safe**: arguments that `borrow::analyze` classifies as `&T` / `&mut T`
//!   arrive as byte slices (`&[u8]`, `&mut [u8]`, or `Option<..>` when nullable).
//!   An access whose pointer derives from exactly one such argument becomes a
//!   bounds-checked slice read or write; everything else falls back to the
//!   fast-mode raw access. The function is only `unsafe` if a raw access remains.
//!
//! Control flow is structured (`structure.rs`): `if`/`else`, `while`, `loop` with
//! `break` and `continue`, and early `return`, with block parameters as mutable
//! variables. Only an irreducible part of the CFG becomes a
//! `loop { match bb { .. } }` state machine.
//!
//! A value used once, in the block that defines it, is written into its use
//! instead of getting a `let` (a load only if nothing between them writes
//! memory), and constants are always written as literals.
//!
//! The input must be clean SSA (`opt::clean`), which is what the borrow analysis
//! expects too.
use crate::abi::{Sig, Site, STACK_ARG_BASE, SYSV_ARGS};
use crate::lift::XMM_PARAM;
use crate::borrow::{analyze_with, Analysis, Class, Ctx, Off, ParamBorrow, Pass, Root, RSP};
use crate::cfg::Cfg;
use crate::expr::{self, lit};
use crate::structure::{declare_in_place, print, structure, Node, Source, DISPATCH_VAR};
use crate::ir::*;
use crate::sources::bucket;
use crate::types::{base_of, leaf, render, width, FnTypes};
use crate::verify::for_each_operand;
use std::fmt::Write;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Mode {
    Fast,
    Safe,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmitStats {
    /// Loads and stores emitted safely: bounds-checked slice accesses, or fields
    /// of a `&S` / `&mut S` argument.
    pub checked: usize,
    /// Loads, stores and copies emitted as raw pointer accesses inside `unsafe`,
    /// plus FFI and indirect calls. Zero means the body has no raw pointer.
    pub raw: usize,
    /// Raw loads, stores and copies by where the pointer comes from
    /// (`sources::SOURCES`: frame, global, argument, other).
    pub raw_by: [usize; 4],
    /// Instructions or terminators the emitter can't express yet (`todo!()`).
    pub todo: usize,
    /// Instructions kept as inline assembly (`asm!`).
    pub asm: usize,
    /// Raw twins emitted after this function (`program.rs`), and their raw
    /// loads, stores and copies (not counted in `raw` or `raw_by`).
    pub twins: usize,
    pub twin_raw: usize,
    /// Functions with irreducible control flow, which is emitted as a
    /// `loop { match bb { .. } }` state machine (just the irreducible part,
    /// unless structuring is off or fails).
    pub state_machines: usize,
}

const REG: [&str; 16] = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
];


/// How a safe-mode argument arrives.
#[derive(Copy, Clone, PartialEq, Eq)]
enum ArgKind {
    Int,
    /// `&[u8]`, or `&[T]` for an element type `elem` that every access reads or
    /// writes whole, at an offset that is a multiple of its size.
    Slice { mutbl: bool, nullable: bool, elem: Option<TyId> },
    /// `&S` / `&mut S` for a recovered struct `S`: every access through it is a
    /// field access.
    Struct { mutbl: bool, nullable: bool, ty: TyId },
}

/// What a call site calls, as `program.rs` resolved it.
#[derive(Clone, Debug, Default)]
pub struct CallInfo {
    /// The Rust path to call: a decompiled function's name, or `ffi::name` for an
    /// extern. `None` calls the callee value as a C function pointer.
    pub path: Option<String>,
    /// The callee returns a value in rax.
    pub ret: bool,
    /// ... and another in rdx: the call is a `(u64, u64)`.
    pub ret2: bool,
    /// Calling it is an FFI call (`unsafe` in any mode).
    pub foreign: bool,
    /// Safe mode: how the callee takes each argument (`Pass::Borrow`: a slice).
    /// Missing arguments are integers.
    pub args: Vec<Pass>,
    /// Safe mode: it allocates `args[a]` (times `args[b]`) bytes, like `malloc`.
    pub alloc: Option<(u8, Option<u8>)>,
    /// Safe mode: it frees its first argument, like `free`.
    pub free: bool,
    /// Safe mode: it can be written as slice operations (`memcpy`, `memset`).
    pub builtin: Option<crate::libc::Builtin>,
    /// A call to a function's raw twin (fast-mode code): counted as raw, like
    /// an FFI call.
    pub raw: bool,
    /// Recovered argument types, in signature order (`None`: `u64`). Empty if
    /// the callee takes every argument as `u64`.
    pub arg_tys: Vec<Option<TyId>>,
    /// Recovered return type (`None`: `u64`).
    pub ret_ty: Option<TyId>,
}

/// The whole-program context a function is emitted in (`program.rs`).
pub struct Env<'a> {
    /// The function's signature after `abi::apply`; `None` for IR straight from
    /// the lifter, where every register read on entry is an argument.
    pub sig: Option<Sig>,
    /// Each call site's callee.
    pub call: &'a dyn Fn(Site) -> Option<CallInfo>,
    /// Safe mode only: take every argument as an integer, because decompiled
    /// callers pass addresses, not slices.
    pub demote: bool,
    /// Try structured control flow before the state machine.
    pub structure: bool,
    /// Rust expression for a constant address that points into the binary's data
    /// (the address of a `static`), or `None` to keep the raw address.
    pub global_of: &'a dyn Fn(u64) -> Option<String>,
    /// `global_end(base, at)`: the end of the `static` holding `base`, if
    /// another one starts at `at` (`Globals::end_expr`).
    pub global_end: &'a dyn Fn(u64, u64) -> Option<String>,
    /// `global_before(addr)`: `addr` as an offset back from the `static` that
    /// starts just after it (`Globals::before_expr`), for a base the code only
    /// indexes from.
    pub global_before: &'a dyn Fn(u64) -> Option<String>,
    /// Safe mode: the read-only `Bytes` static containing an address, which
    /// reads can index as a slice; `None` keeps reads through it raw.
    pub global_slice: &'a dyn Fn(u64) -> Option<String>,
    /// Safe mode: the borrow analysis, if the caller ran it with call summaries
    /// (`program.rs`); otherwise it runs here, knowing nothing about calls.
    pub analysis: Option<&'a Analysis>,
    /// Recovered types (`types.rs`), and the table they refer to. `None` prints
    /// every integer unsigned and every argument as `u64`.
    pub types: Option<(&'a FnTypes, &'a TyTable)>,
    /// Safe mode: struct arguments can be `&S` / `&mut S`. Only for functions
    /// that decompiled code doesn't call, since callers lend byte slices.
    pub struct_args: bool,
}

/// Where safe-mode code indexes a root's bytes.
struct Place {
    /// The slice to read: `rdi_ref`, `frame.0`, `heap12`, `NAME.b`.
    read: String,
    /// The slice to write, if it can be written.
    write: Option<String>,
    /// The root's address, as a `u64`.
    base: String,
    /// A slice of this element type rather than bytes (`ArgKind::Slice`).
    elem: Option<TyId>,
}

struct Emitter<'a> {
    f: &'a Function,
    sig: Option<Sig>,
    call: &'a dyn Fn(Site) -> Option<CallInfo>,
    /// Values something uses (a void call's value usually isn't).
    used: Vec<bool>,
    /// Values only computed to call a callee that is called by name instead.
    skip: Vec<bool>,
    /// Try structured control flow before the state machine.
    structure: bool,
    /// Borrow analysis (safe mode only).
    borrow: Option<&'a Analysis>,
    /// How each root of the analysis is indexed, if it is safe.
    places: Vec<Option<Place>>,
    args: Vec<ArgKind>,
    /// Entry parameter index of each value, if it is one.
    entry_param: Vec<Option<usize>>,
    /// Values that need a variable declared up front (block params, and values
    /// used outside the block that defines them).
    hoisted: Vec<bool>,
    /// Values written into their single use instead of getting a `let`.
    inline: Vec<bool>,
    /// The expression of each inlined value, once its block has been emitted.
    exprs: Vec<Option<String>>,
    /// Rust expression for a constant address that points into the binary's data.
    global_of: &'a dyn Fn(u64) -> Option<String>,
    global_end: &'a dyn Fn(u64, u64) -> Option<String>,
    global_before: &'a dyn Fn(u64) -> Option<String>,
    /// Values used only as the base of an indexed address (`index_only`).
    index_only: Vec<bool>,
    /// Where each value points (`sources.rs`), to count raw accesses by source.
    src: Vec<u8>,
    stats: EmitStats,
    table: &'a TyTable,
    types: Option<&'a FnTypes>,
    /// Rust type of each value.
    vt: Vec<TyId>,
    /// The name each entry parameter has in the signature, when it isn't the
    /// register name (a slice, a reference, a pointer, a name from debug info).
    bind: Vec<Option<String>>,
    /// What each block parameter always equals (`types::aliases`).
    alias: Vec<ValueId>,
    u64_ty: TyId,
}

/// Emit `f`, straight from the lifter (no signature), as a Rust function named
/// `name` into `out`. Calls go through the target's address as a function pointer.
/// `name_of` is unused: calls are named by `program.rs`, which knows the callees.
pub fn emit_function(
    f: &Function,
    name: &str,
    mode: Mode,
    name_of: &dyn Fn(u64) -> Option<String>,
    out: &mut String,
) -> EmitStats {
    emit_function_with(f, name, mode, true, name_of, &|_| None, out)
}

/// `emit_function`, plus `global_of`, which turns a constant address used as a
/// pointer into a Rust expression (the address of a `static`), or `None` to keep
/// the raw address.
pub fn emit_function_with_globals(
    f: &Function,
    name: &str,
    mode: Mode,
    name_of: &dyn Fn(u64) -> Option<String>,
    global_of: &dyn Fn(u64) -> Option<String>,
    out: &mut String,
) -> EmitStats {
    emit_function_with(f, name, mode, true, name_of, global_of, out)
}

/// `emit_function_with_globals`, choosing whether to structure control flow. With
/// `structure: false` every function with a branch is a state machine.
pub fn emit_function_with(
    f: &Function,
    name: &str,
    mode: Mode,
    structure: bool,
    name_of: &dyn Fn(u64) -> Option<String>,
    global_of: &dyn Fn(u64) -> Option<String>,
    out: &mut String,
) -> EmitStats {
    let _ = name_of;
    let call = |_: Site| None;
    let env = Env { sig: None, call: &call, demote: false, structure, global_of, global_end: &|_, _| None, global_before: &|_| None, global_slice: &|_| None, analysis: None, types: None, struct_args: false };
    emit_function_in(f, name, mode, &env, out)
}

/// Emit `f` as a Rust function named `name`, with its signature and callees from
/// `env`.
pub fn emit_function_in(f: &Function, name: &str, mode: Mode, env: &Env, out: &mut String) -> EmitStats {
    let cfg = Cfg::new(f);
    let entry_params = f.blocks[f.entry].params.get(&f.value_pool);
    let mut entry_param = vec![None; f.insts.len()];
    for (k, &p) in entry_params.iter().enumerate() {
        entry_param[p.index()] = Some(k);
    }

    let owned;
    let borrow = match (mode, env.analysis) {
        (Mode::Fast, _) => None,
        (Mode::Safe, Some(a)) => Some(a),
        (Mode::Safe, None) => {
            let global_ok = |c: u64| (env.global_slice)(c).is_some();
            owned = analyze_with(f, &Ctx { callee: &|_| None, demoted: &|_| false, global_ok: &global_ok });
            Some(&owned)
        }
    };
    let mut own = TyTable::new();
    let (types, table) = match env.types {
        Some((t, table)) => (Some(t), table),
        None => {
            for b in [1, 2, 4, 8] {
                own.int(b, false);
            }
            (None, &own)
        }
    };
    let int = |w: u8| table.get(&Ty::Int { bits: w * 8, signed: false }).expect("integer types are interned");
    let mut vt: Vec<TyId> = match types {
        Some(t) => t.vals.clone(),
        None => f.insts.iter().map(|(_, i)| width(i.ty).map_or(i.ty, int)).collect(),
    };
    if let Some(t) = types {
        pointer_locals(f, &cfg, t, table, &mut vt);
    }
    let alias = crate::types::aliases(f);

    let args: Vec<ArgKind> = entry_params
        .iter()
        .enumerate()
        .map(|(k, &p)| match borrow {
            Some(a) if !env.demote => {
                let kind = arg_kind(&a.params[k]);
                let pointee = types.and_then(|t| t.pointee[p.index()]).filter(|_| env.struct_args);
                match (kind, pointee) {
                    (ArgKind::Slice { mutbl, nullable, .. }, Some(s))
                        if matches!(table.tys[s], Ty::Struct(_)) && struct_arg(f, &cfg, a, env.call, &alias, table, k, p, s) =>
                    {
                        ArgKind::Struct { mutbl, nullable, ty: s }
                    }
                    (ArgKind::Slice { mutbl, nullable, .. }, Some(s))
                        if matches!(table.tys[s], Ty::Int { bits: 16 | 32 | 64, .. }) && elems_arg(f, &cfg, a, k, p, table.size_of(s) as u8) =>
                    {
                        ArgKind::Slice { mutbl, nullable, elem: Some(s) }
                    }
                    _ => kind,
                }
            }
            _ => ArgKind::Int,
        })
        .collect();

    let hoisted = hoisted(f, &cfg);
    let skip = callee_only(f, &cfg, env.call);
    let inline = inlined(f, &cfg, &hoisted, &skip, mode == Mode::Safe);
    let mut e = Emitter {
        f,
        sig: env.sig,
        call: env.call,
        used: used(f, &cfg),
        skip,
        structure: env.structure,
        borrow,
        places: Vec::new(),
        args,
        entry_param,
        hoisted,
        exprs: vec![None; inline.len()],
        inline,
        global_of: env.global_of,
        global_end: env.global_end,
        global_before: env.global_before,
        index_only: index_only(f),
        src: crate::sources::sources(f),
        stats: EmitStats::default(),
        table,
        types,
        vt,
        bind: vec![None; entry_params.len()],
        alias,
        u64_ty: int(8),
    };
    e.bind_names();

    e.places = e.places(env.global_slice);
    let mut body = String::new();
    e.body(&cfg, &mut body);
    e.signature(name, out);
    out.push_str(&body);
    out.push_str("}\n");
    e.stats
}

/// Can safe-mode argument `k` (entry parameter `p`) be a `&S` for struct `s`?
/// Only if its root is safe, every access through it is a field access of `s`
/// at its own address, and no call borrows it as a slice.
#[allow(clippy::too_many_arguments)]
fn struct_arg(
    f: &Function,
    cfg: &Cfg,
    a: &Analysis,
    call: &dyn Fn(Site) -> Option<CallInfo>,
    alias: &[ValueId],
    table: &TyTable,
    k: usize,
    p: ValueId,
    s: TyId,
) -> bool {
    let Some(r) = a.roots.iter().position(|&x| x == Root::Param(k as u8)) else { return false };
    if !a.safe[r] {
        return false;
    }
    let through = |v: ValueId| a.origin[v.index()].roots & (1u128 << r) != 0;
    let mut any = false;
    for &b in &cfg.rpo {
        for &id in f.blocks[b].insts.get(&f.value_pool) {
            let (ptr, bytes) = match f.insts[id].kind {
                InstKind::Load { ptr, .. } => (ptr, width(f.insts[id].ty)),
                InstKind::Store { ptr, val, .. } => (ptr, width(f.insts[val].ty)),
                InstKind::MemCopy { dst, src, .. } if through(dst) || through(src) => return false,
                InstKind::MemFill { dst, .. } if through(dst) => return false,
                InstKind::Call { args, .. } => {
                    let pass = call(Site::Call(id)).map(|c| c.args).unwrap_or_default();
                    let lent = args.get(&f.value_pool).iter().zip(&pass).any(|(&v, x)| matches!(x, Pass::Borrow { .. }) && through(v));
                    if lent {
                        return false;
                    }
                    continue;
                }
                _ => continue,
            };
            if !through(ptr) {
                continue;
            }
            let (base, d) = base_of(f, alias, ptr);
            let ok = base == p && bytes.is_some_and(|w| leaf(table, s, d, w as u32).is_some());
            if !ok {
                return false;
            }
            any = true;
        }
    }
    // a tail call can lend it too
    for &b in &cfg.rpo {
        if let Terminator::TailCall { args, .. } = f.blocks[b].term {
            let pass = call(Site::Tail(b)).map(|c| c.args).unwrap_or_default();
            if args.get(&f.value_pool).iter().zip(&pass).any(|(&v, x)| matches!(x, Pass::Borrow { .. }) && through(v)) {
                return false;
            }
        }
    }
    any
}

/// Can safe-mode argument `k` (entry parameter `p`), a slice, be a `&[T]` for an
/// element of `w` bytes? Only if its root is safe, every access through it is
/// `w` bytes at an offset from `p` that is a multiple of `w`, and nothing copies
/// or fills it or is passed it (callees take byte slices, `memcpy` copies bytes).
fn elems_arg(f: &Function, cfg: &Cfg, a: &Analysis, k: usize, p: ValueId, w: u8) -> bool {
    let Some(r) = a.roots.iter().position(|&x| x == Root::Param(k as u8)) else { return false };
    if !a.safe[r] {
        return false;
    }
    let through = |v: ValueId| a.origin[v.index()].roots & (1u128 << r) != 0;
    let res = residues(f, cfg, p, w as u64);
    let lent = |args: ListRef| args.get(&f.value_pool).iter().any(|&v| through(v));
    let mut any = false;
    for &b in &cfg.rpo {
        for &id in f.blocks[b].insts.get(&f.value_pool) {
            let (ptr, bytes) = match f.insts[id].kind {
                InstKind::Load { ptr, .. } => (ptr, width(f.insts[id].ty)),
                InstKind::Store { ptr, val, .. } => (ptr, width(f.insts[val].ty)),
                InstKind::MemCopy { dst, src, .. } if through(dst) || through(src) => return false,
                InstKind::MemFill { dst, .. } if through(dst) => return false,
                InstKind::Call { args, .. } if lent(args) => return false,
                _ => continue,
            };
            if !through(ptr) {
                continue;
            }
            if bytes != Some(w) || res[ptr.index()] != Some(0) {
                return false;
            }
            any = true;
        }
        if let Terminator::TailCall { args, .. } = f.blocks[b].term {
            if lent(args) {
                return false;
            }
        }
    }
    any
}

/// Each value modulo `w` (a power of two), counting `p` as 0: a pointer with
/// residue 0 is a multiple of `w` bytes away from `p`. `None` where that isn't
/// known.
fn residues(f: &Function, cfg: &Cfg, p: ValueId, w: u64) -> Vec<Option<u64>> {
    #[derive(Copy, Clone, PartialEq)]
    enum R {
        Top,
        K(u64),
        Bot,
    }
    let mask = w - 1;
    let n = f.insts.len();
    let mut r = vec![R::Top; n];
    let meet = |a: R, b: R| match (a, b) {
        (R::Top, x) | (x, R::Top) => x,
        (R::K(x), R::K(y)) if x == y => R::K(x),
        _ => R::Bot,
    };
    // (block parameter, value passed to it) on every edge
    let mut incoming: Vec<(ValueId, ValueId)> = Vec::new();
    for &b in &cfg.rpo {
        let pool = &f.value_pool;
        match f.blocks[b].term {
            Terminator::Jump { to, args } => incoming.extend(f.blocks[to].params.get(pool).iter().copied().zip(args.get(pool).iter().copied())),
            Terminator::Branch { t, f: e, args, .. } => {
                let tp = f.blocks[t].params.get(pool);
                let ep = f.blocks[e].params.get(pool);
                let a = args.get(pool);
                incoming.extend(tp.iter().copied().zip(a.iter().copied()));
                incoming.extend(ep.iter().copied().zip(a[tp.len().min(a.len())..].iter().copied()));
            }
            _ => {}
        }
    }
    for &q in f.blocks[f.entry].params.get(&f.value_pool) {
        r[q.index()] = if q == p { R::K(0) } else { R::Bot };
    }
    let two = |a: R, b: R, op: &dyn Fn(u64, u64) -> u64| match (a, b) {
        (R::Bot, _) | (_, R::Bot) => R::Bot,
        (R::Top, _) | (_, R::Top) => R::Top,
        (R::K(x), R::K(y)) => R::K(op(x, y) & mask),
    };
    loop {
        let mut changed = false;
        let mut set = |r: &mut Vec<R>, v: ValueId, x: R| {
            if r[v.index()] != x {
                r[v.index()] = x;
                changed = true;
            }
        };
        for &(q, a) in &incoming {
            let x = meet(r[q.index()], r[a.index()]);
            set(&mut r, q, x);
        }
        for &b in &cfg.rpo {
            for &id in f.blocks[b].insts.get(&f.value_pool) {
                use InstKind::*;
                let x = match f.insts[id].kind {
                    Const(c) => R::K(f.consts[c.index()] as u64 & mask),
                    Bin { op, lhs, rhs } => {
                        let (a, c) = (r[lhs.index()], r[rhs.index()]);
                        match op {
                            BinOp::Add => two(a, c, &|x, y| x.wrapping_add(y)),
                            BinOp::Sub => two(a, c, &|x, y| x.wrapping_sub(y)),
                            BinOp::Mul if a == R::K(0) || c == R::K(0) => R::K(0),
                            BinOp::Mul => two(a, c, &|x, y| x.wrapping_mul(y)),
                            BinOp::And if a == R::K(0) || c == R::K(0) => R::K(0),
                            BinOp::And => two(a, c, &|x, y| x & y),
                            BinOp::Or => two(a, c, &|x, y| x | y),
                            BinOp::Xor => two(a, c, &|x, y| x ^ y),
                            BinOp::Shl => match f.insts[rhs].kind {
                                Const(k) => {
                                    let k = f.consts[k.index()] as u64 & 63;
                                    if (1u64 << k) & mask == 0 { R::K(0) } else { two(a, R::K(0), &|x, _| x << k) }
                                }
                                _ => R::Bot,
                            },
                            _ => R::Bot,
                        }
                    }
                    Cast { kind: CastKind::Trunc | CastKind::ZExt | CastKind::SExt | CastKind::Bitcast, v } | IntToPtr(v) | PtrToInt(v) => r[v.index()],
                    PtrOffset { base, index, scale, disp } => {
                        let i = match index {
                            None => R::K(0),
                            Some(_) if scale as u64 & mask == 0 => R::K(0),
                            Some(i) => two(r[i.index()], R::K(scale as u64), &|x, y| x.wrapping_mul(y)),
                        };
                        two(two(r[base.index()], i, &|x, y| x.wrapping_add(y)), R::K(disp as i64 as u64), &|x, y| x.wrapping_add(y))
                    }
                    Select { t, f: e, .. } => meet(r[t.index()], r[e.index()]),
                    _ => R::Bot,
                };
                set(&mut r, id, x);
            }
        }
        if !changed {
            break;
        }
    }
    r.into_iter().map(|x| if let R::K(k) = x { Some(k) } else { None }).collect()
}

/// Types the values in the body that point at a struct `*mut S` instead of
/// `u64`: values loaded from memory or carried by block parameters, whose every
/// use the emitter converts (an access through it, an address computed from it,
/// a compare, a call argument, an edge copy, a return).
fn pointer_locals(f: &Function, cfg: &Cfg, t: &FnTypes, table: &TyTable, vt: &mut [TyId]) {
    use InstKind::*;
    let n = f.insts.len();
    let mut ok = vec![false; n];
    let entry: Vec<ValueId> = f.blocks[f.entry].params.get(&f.value_pool).to_vec();
    for &b in &cfg.rpo {
        let blk = &f.blocks[b];
        for &v in blk.params.get(&f.value_pool).iter().chain(blk.insts.get(&f.value_pool)) {
            let def = matches!(f.insts[v].kind, Load { .. }) || (matches!(f.insts[v].kind, BlockParam(_)) && !entry.contains(&v));
            let s = t.pointee[v.index()].filter(|&s| matches!(table.tys[s], Ty::Struct(_)));
            ok[v.index()] = def && width(f.insts[v].ty) == Some(8) && s.is_some_and(|s| table.get(&Ty::RawPtr { pointee: s, mutbl: Mutbl::Mut }).is_some());
        }
    }
    let bad = |ok: &mut Vec<bool>, v: ValueId| ok[v.index()] = false;
    for &b in &cfg.rpo {
        let blk = &f.blocks[b];
        for &id in blk.insts.get(&f.value_pool) {
            match f.insts[id].kind {
                Load { .. } | Bin { op: BinOp::Add | BinOp::Sub, .. } | PtrOffset { .. } => {}
                Call { callee, .. } => bad(&mut ok, callee),
                Store { val, .. } if width(f.insts[val].ty) != Some(8) => bad(&mut ok, val),
                Store { .. } => {}
                Cmp { lhs, rhs, .. } if width(f.insts[lhs].ty) == Some(8) && width(f.insts[rhs].ty) == Some(8) => {}
                k => for_each_operand(k, f, |v| bad(&mut ok, v)),
            }
            if let Bin { op: BinOp::Add | BinOp::Sub, rhs, .. } = f.insts[id].kind {
                bad(&mut ok, rhs);
            }
            if let PtrOffset { index: Some(i), .. } = f.insts[id].kind {
                bad(&mut ok, i);
            }
        }
        if let Terminator::Switch { v, .. } | Terminator::Branch { c: v, .. } | Terminator::TailCall { callee: v, .. } = blk.term {
            bad(&mut ok, v);
        }
    }
    for v in 0..n {
        if ok[v] {
            let s = t.pointee[v].expect("checked above");
            vt[v] = table.get(&Ty::RawPtr { pointee: s, mutbl: Mutbl::Mut }).expect("checked above");
        }
    }
}

fn arg_kind(p: &ParamBorrow) -> ArgKind {
    match p.class {
        _ if p.reg == RSP => ArgKind::Int,
        Class::Shared => ArgKind::Slice { mutbl: false, nullable: p.nullable, elem: None },
        Class::Mut => ArgKind::Slice { mutbl: true, nullable: p.nullable, elem: None },
        Class::NotPointer | Class::Raw => ArgKind::Int,
    }
}

/// Block of definition for every value, then mark values used in another block.
fn hoisted(f: &Function, cfg: &Cfg) -> Vec<bool> {
    let n = f.insts.len();
    let mut def = vec![u32::MAX; n];
    let mut hoist = vec![false; n];
    for &b in &cfg.rpo {
        let blk = &f.blocks[b];
        for &p in blk.params.get(&f.value_pool) {
            def[p.index()] = b.index() as u32;
            hoist[p.index()] = b != f.entry;
        }
        for &id in blk.insts.get(&f.value_pool) {
            def[id.index()] = b.index() as u32;
        }
    }
    for &b in &cfg.rpo {
        let blk = &f.blocks[b];
        let mut use_in = |v: ValueId| {
            if def[v.index()] != b.index() as u32 {
                hoist[v.index()] = true;
            }
        };
        for &id in blk.insts.get(&f.value_pool) {
            for_each_operand(f.insts[id].kind, f, &mut use_in);
        }
        term_uses(f, blk.term, &mut use_in);
    }
    // Entry params are function arguments, already in scope everywhere.
    for &p in f.blocks[f.entry].params.get(&f.value_pool) {
        hoist[p.index()] = false;
    }
    hoist
}

/// Values used by some reachable instruction or terminator.
fn used(f: &Function, cfg: &Cfg) -> Vec<bool> {
    let mut u = vec![false; f.insts.len()];
    for &b in &cfg.rpo {
        let blk = &f.blocks[b];
        for &id in blk.insts.get(&f.value_pool) {
            for_each_operand(f.insts[id].kind, f, |v| u[v.index()] = true);
        }
        term_uses(f, blk.term, |v| u[v.index()] = true);
    }
    u
}

/// Values whose only use is naming the callee of calls emitted by name: the
/// callee's address, its `IntToPtr`, and a GOT slot load.
fn callee_only(f: &Function, cfg: &Cfg, call: &dyn Fn(Site) -> Option<CallInfo>) -> Vec<bool> {
    let mut uses = vec![0u32; f.insts.len()];
    for &b in &cfg.rpo {
        let blk = &f.blocks[b];
        for &id in blk.insts.get(&f.value_pool) {
            for_each_operand(f.insts[id].kind, f, |v| uses[v.index()] += 1);
        }
        term_uses(f, blk.term, |v| uses[v.index()] += 1);
    }
    let mut work = Vec::new();
    let named = |s: Site| call(s).is_some_and(|c| c.path.is_some());
    for &b in &cfg.rpo {
        let blk = &f.blocks[b];
        for &id in blk.insts.get(&f.value_pool) {
            if let InstKind::Call { callee, .. } = f.insts[id].kind {
                if named(Site::Call(id)) {
                    work.push(callee);
                }
            }
        }
        if let Terminator::TailCall { callee, .. } = blk.term {
            if named(Site::Tail(b)) {
                work.push(callee);
            }
        }
    }
    let mut skip = vec![false; f.insts.len()];
    while let Some(v) = work.pop() {
        uses[v.index()] -= 1;
        let k = f.insts[v].kind;
        let pure = matches!(k, InstKind::Const(_) | InstKind::IntToPtr(_) | InstKind::PtrToInt(_) | InstKind::Load { volatile: false, .. });
        if uses[v.index()] == 0 && pure {
            skip[v.index()] = true;
            for_each_operand(k, f, |o| work.push(o));
        }
    }
    skip
}

/// Every value a terminator uses.
pub fn term_operands(f: &Function, t: Terminator, cb: impl FnMut(ValueId)) {
    term_uses(f, t, cb)
}

/// Pure instructions that can move to their use.
fn pure(k: InstKind) -> bool {
    use InstKind::*;
    match k {
        Bin { op, .. } => !matches!(op, BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem),
        Un { .. } | Cmp { .. } | Cast { .. } | Select { .. } | PtrOffset { .. } | IntToPtr(_) | PtrToInt(_)
        | AddrOfLocal(_) | Aggregate { .. } => true,
        _ => false,
    }
}

/// Instructions that read memory or can panic: they can move to their use only
/// past instructions that do neither.
fn movable(k: InstKind) -> bool {
    match k {
        InstKind::Load { volatile, .. } => !volatile,
        InstKind::Bin { op, .. } => matches!(op, BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem),
        _ => false,
    }
}

/// Values written straight into their only use: used once, in the block that
/// defines them, and either pure or (`movable`) with nothing between the
/// definition and the use that writes memory or calls. "The use" is where the
/// expression ends up, which is the use of the user when that is inlined too.
/// In safe mode a load stays out of a store, which may borrow the same slice
/// mutably.
fn inlined(f: &Function, cfg: &Cfg, hoisted: &[bool], skip: &[bool], safe: bool) -> Vec<bool> {
    let n = f.insts.len();
    let mut uses = vec![0u32; n];
    for &b in &cfg.rpo {
        let blk = &f.blocks[b];
        for &id in blk.insts.get(&f.value_pool) {
            for_each_operand(f.insts[id].kind, f, |v| uses[v.index()] += 1);
        }
        term_uses(f, blk.term, |v| uses[v.index()] += 1);
    }
    let mut inline = vec![false; n];
    let mut user = vec![u32::MAX; n];
    let mut pos = vec![0u32; n];
    for &b in &cfg.rpo {
        let blk = &f.blocks[b];
        let insts = blk.insts.get(&f.value_pool);
        let end = insts.len() as u32;
        for (i, &id) in insts.iter().enumerate() {
            for_each_operand(f.insts[id].kind, f, |v| user[v.index()] = i as u32);
        }
        term_uses(f, blk.term, |v| user[v.index()] = end);
        // barriers[i] = instructions before i that write memory or call
        let mut barriers = vec![0u32; insts.len() + 1];
        for (i, &id) in insts.iter().enumerate() {
            let k = f.insts[id].kind;
            let barrier = !skip[id.index()] && !pure(k) && !movable(k) && !matches!(k, InstKind::Const(_) | InstKind::Undef | InstKind::CallOut { .. } | InstKind::AsmOut { .. } | InstKind::BlockParam(_));
            barriers[i + 1] = barriers[i] + barrier as u32;
        }
        for (i, &id) in insts.iter().enumerate().rev() {
            let k = f.insts[id].kind;
            let v = id.index();
            if uses[v] != 1 || hoisted[v] || skip[v] || !(pure(k) || movable(k)) {
                continue;
            }
            let mut p = user[v];
            // `if a { if b { x } else { y } } else { z }` on one line is too much
            if matches!(k, InstKind::Select { .. }) && p < end && matches!(f.insts[insts[p as usize]].kind, InstKind::Select { .. }) {
                continue;
            }
            if p < end && inline[insts[p as usize].index()] {
                p = pos[insts[p as usize].index()];
            }
            pos[v] = p;
            let into_store = p < end && matches!(f.insts[insts[p as usize]].kind, InstKind::Store { .. });
            inline[v] = !movable(k) || barriers[p as usize] == barriers[i + 1] && !(safe && into_store);
        }
    }
    inline
}

/// Values used, and only used, as the base of an address with a variable
/// index (`ptr b + i*8`): the code reaches memory only at some offset from
/// them, which says nothing about what `b` itself points into.
fn index_only(f: &Function) -> Vec<bool> {
    let n = f.insts.len();
    let (mut uses, mut indexed) = (vec![0u32; n], vec![0u32; n]);
    for (_, blk) in f.blocks.iter() {
        for &id in blk.insts.get(&f.value_pool) {
            let k = f.insts[id].kind;
            for_each_operand(k, f, |v| uses[v.index()] += 1);
            if let InstKind::PtrOffset { base, index: Some(i), .. } = k {
                if i != base {
                    indexed[base.index()] += 1;
                }
            }
        }
        term_uses(f, blk.term, |v| uses[v.index()] += 1);
    }
    (0..n).map(|k| uses[k] > 0 && uses[k] == indexed[k]).collect()
}

fn term_uses(f: &Function, t: Terminator, mut cb: impl FnMut(ValueId)) {
    let list = |l: ListRef, cb: &mut dyn FnMut(ValueId)| l.get(&f.value_pool).iter().for_each(|&v| cb(v));
    match t {
        Terminator::Jump { args, .. } => list(args, &mut cb),
        Terminator::Branch { c, args, .. } => {
            cb(c);
            list(args, &mut cb)
        }
        Terminator::Return(Some(v)) | Terminator::Switch { v, .. } => cb(v),
        Terminator::TailCall { callee, args } => {
            cb(callee);
            list(args, &mut cb)
        }
        Terminator::Return(None) | Terminator::Unreachable => {}
    }
}

/// Sorted case values as a pattern, runs as ranges: `0 | 3..=5`.
fn case_pattern(cases: &[u64]) -> String {
    let mut parts = Vec::new();
    let mut i = 0;
    while i < cases.len() {
        let mut j = i;
        while j + 1 < cases.len() && cases[j + 1] == cases[j] + 1 {
            j += 1;
        }
        parts.push(if j == i { cases[i].to_string() } else { format!("{}..={}", cases[i], cases[j]) });
        i = j + 1;
    }
    parts.join(" | ")
}

fn bytes(ty: TyId) -> usize {
    match ty {
        TyId::B1 => 1,
        TyId::B2 => 2,
        TyId::B4 => 4,
        _ => 8,
    }
}

fn signed(ty: TyId) -> &'static str {
    match bytes(ty) {
        1 => "i8",
        2 => "i16",
        4 => "i32",
        _ => "i64",
    }
}

impl Emitter<'_> {
    /// The value as an expression: its variable, its literal, or (if inlined) its
    /// whole expression.
    fn name(&self, v: ValueId) -> String {
        let ty = self.ty(v);
        match (self.entry_param[v.index()], self.f.insts[v].kind) {
            (Some(_), InstKind::BlockParam(r)) if (XMM_PARAM..XMM_PARAM + 32).contains(&r) => {
                let k = r - XMM_PARAM;
                format!("xmm{}{}", k / 2, if k % 2 == 1 { "_hi" } else { "" })
            }
            (Some(_), InstKind::BlockParam(r)) if r >= STACK_ARG_BASE => format!("arg{}", 6 + (r - STACK_ARG_BASE) as usize),
            (Some(_), InstKind::BlockParam(r)) => REG[r as usize & 15].to_string(),
            _ if self.is_lit(v) && ty == TyId::BOOL => (self.konst(v) != Some(0)).to_string(),
            _ if self.is_lit(v) => self.lit_as(self.konst(v).unwrap_or(0), self.vt[v.index()]),
            _ => match &self.exprs[v.index()] {
                Some(e) => e.clone(),
                None => {
                    debug_assert!(!self.inline[v.index()], "v{} used before its block was emitted", v.index());
                    format!("v{}", v.index())
                }
            },
        }
    }

    /// Constants (and undefined values, which are zero) are written as literals
    /// where they are used, and get no variable.
    fn is_lit(&self, v: ValueId) -> bool {
        matches!(self.f.insts[v].kind, InstKind::Const(_) | InstKind::Undef) && !matches!(self.ty(v), TyId::PAIR | TyId::UNIT)
    }

    /// A constant's value, masked to its type.
    fn konst(&self, v: ValueId) -> Option<u64> {
        match self.f.insts[v].kind {
            InstKind::Const(c) => {
                let ty = self.ty(v);
                let mask = if bytes(ty) >= 8 { u64::MAX } else { (1u64 << (bytes(ty) * 8)) - 1 };
                Some(self.f.consts[c.index()] as u64 & mask)
            }
            InstKind::Undef => Some(0),
            _ => None,
        }
    }

    /// The width and signedness of integer type `t`.
    fn int_of(&self, t: TyId) -> Option<(u8, bool)> {
        match self.table.tys[t] {
            Ty::Int { bits, signed } => Some((bits, signed)),
            Ty::Unknown { bytes: b @ (1 | 2 | 4 | 8) } => Some(((b * 8) as u8, false)),
            _ => None,
        }
    }

    /// Bits `c` as a literal of integer type `t`, without a suffix: negative
    /// if `t` is signed and the sign bit is set.
    fn lit_bits(&self, c: u64, t: TyId) -> String {
        match self.int_of(t) {
            Some((bits, true)) => {
                let bits = bits.clamp(8, 64) as u32;
                let x = ((c << (64 - bits)) as i64) >> (64 - bits);
                if x < 0 { format!("-{}", lit(x.unsigned_abs())) } else { lit(x as u64) }
            }
            Some((bits, false)) if bits < 64 => lit(c & ((1u64 << bits) - 1)),
            _ => lit(c),
        }
    }

    /// Bits `c` as a literal of type `t`, with its suffix: `5_u32`, `-1_i64`.
    fn lit_as(&self, c: u64, t: TyId) -> String {
        format!("{}_{}", self.lit_bits(c, t), self.rs(t))
    }

    /// Constant `v` as the same bits in integer type `t`, without a suffix
    /// (`None` if `v` isn't an integer constant): for a place where the other
    /// operand fixes the type.
    fn const_in(&self, v: ValueId, t: TyId) -> Option<String> {
        let c = self.konst(v)?;
        if self.ty(v) == TyId::BOOL || self.int_of(t).is_none() {
            return None;
        }
        // the bits `conv` would give: widened by `v`'s own signedness
        let c = match self.int_of(self.vt[v.index()]) {
            Some((bits, true)) if bits < 64 => (((c << (64 - bits as u32)) as i64) >> (64 - bits as u32)) as u64,
            _ => c,
        };
        Some(self.lit_bits(c, t))
    }

    /// `v` as type `t`, where the other operand fixes the type: a constant needs
    /// no suffix there.
    fn operand_as(&self, v: ValueId, t: TyId) -> String {
        match self.const_in(v, t) {
            Some(c) => c,
            None => expr::rhs(self.val(v, t)),
        }
    }

    /// `v` where its type is already fixed by the other operand (`lhs`).
    fn operand(&self, v: ValueId, lhs: ValueId) -> String {
        self.operand_as(v, self.vt[lhs.index()])
    }

    /// `v` converted to type `t` (just `v` if it already is one).
    fn val(&self, v: ValueId, t: TyId) -> String {
        self.conv(&self.name(v), self.vt[v.index()], t)
    }

    /// `v` as a return value or call argument of type `t`. The ABI only defines
    /// the low byte of a `bool` (`al`), so a wider register is truncated first.
    fn abi_val(&self, v: ValueId, t: TyId) -> String {
        if self.table.tys[t] == Ty::Bool && self.ty(v) != TyId::BOOL && bytes(self.ty(v)) > 1 {
            return self.conv(&format!("{} as u8", expr::cast(self.name(v))), TyId::B1, t);
        }
        self.val(v, t)
    }

    /// `c`, compared with `other`, as the end of the static `other` points
    /// into, when `c` is the address just past it: a loop over one array may
    /// stop at the address of whatever follows it in the binary, which the
    /// output's statics don't keep next to each other. A loop walking down
    /// stops one element before the start the same way.
    fn past_end(&self, c: ValueId, other: ValueId) -> Option<String> {
        let (mut ats, mut seen) = (Vec::new(), Vec::new());
        if !self.constants(c, &mut ats, &mut seen) || ats.is_empty() || ats.iter().any(|&a| a != ats[0]) {
            return None;
        }
        let at = ats[0];
        let (mut bases, mut seen) = (Vec::new(), Vec::new());
        if !self.bases(other, &mut bases, &mut seen) || bases.is_empty() {
            return None;
        }
        let e = (self.global_end)(bases[0], at)?;
        bases[1..].iter().all(|&b| (self.global_end)(b, at).as_ref() == Some(&e)).then_some(e)
    }

    /// The constant addresses `v` is computed from, by offsets and through
    /// block parameters; `false` if it may come from anywhere else.
    fn bases(&self, v: ValueId, out: &mut Vec<u64>, seen: &mut Vec<ValueId>) -> bool {
        if seen.contains(&v) {
            return true;
        }
        if seen.len() >= 32 {
            return false;
        }
        seen.push(v);
        let f = self.f;
        match f.insts[v].kind {
            InstKind::Const(_) => {
                out.extend(self.konst(v));
                true
            }
            InstKind::IntToPtr(x) | InstKind::PtrToInt(x) => self.bases(x, out, seen),
            InstKind::PtrOffset { base, .. } => self.bases(base, out, seen),
            InstKind::Bin { op: BinOp::Add, lhs, rhs } => {
                let addr = |k: Option<u64>| k.filter(|&k| (self.global_of)(k).is_some());
                match (self.konst(lhs), self.konst(rhs)) {
                    (k @ Some(_), _) | (_, k @ Some(_)) if addr(k).is_some() => {
                        out.extend(k);
                        true
                    }
                    (Some(_), None) => self.bases(rhs, out, seen),
                    (None, Some(_)) => self.bases(lhs, out, seen),
                    _ => false,
                }
            }
            InstKind::Bin { op: BinOp::Sub, lhs, rhs } if self.konst(rhs).is_some() => self.bases(lhs, out, seen),
            InstKind::BlockParam(_) => self.joined(v, |x| self.bases(x, out, seen)),
            // A path where the value was never set constrains nothing.
            InstKind::Undef => true,
            _ => false,
        }
    }

    /// The constants `v` may be, through block parameters; `false` if it may
    /// be anything else.
    fn constants(&self, v: ValueId, out: &mut Vec<u64>, seen: &mut Vec<ValueId>) -> bool {
        if seen.contains(&v) {
            return true;
        }
        if seen.len() >= 32 {
            return false;
        }
        seen.push(v);
        match self.f.insts[v].kind {
            InstKind::Undef => true,
            InstKind::Const(_) => {
                out.extend(self.konst(v));
                true
            }
            InstKind::IntToPtr(x) | InstKind::PtrToInt(x) => self.constants(x, out, seen),
            InstKind::BlockParam(_) => self.joined(v, |x| self.constants(x, out, seen)),
            _ => false,
        }
    }

    /// Whether `each` holds for every value block parameter `v` receives. A
    /// parameter of the entry block is the function's own; one of a block no
    /// edge reaches is never set.
    fn joined(&self, v: ValueId, mut each: impl FnMut(ValueId) -> bool) -> bool {
        let f = self.f;
        let Some((b, k)) = f.blocks.iter().find_map(|(b, blk)| {
            blk.params.get(&f.value_pool).iter().position(|&p| p == v).map(|k| (b, k))
        }) else { return false };
        let (mut any, mut ok) = (false, true);
        for (p, blk) in f.blocks.iter() {
            if blk.term.successors(&f.value_pool).any(|s| s == b) {
                crate::abi::incoming(f, p, b, k, |x| {
                    any = true;
                    ok = ok && each(x);
                });
            }
        }
        (any || b != f.entry) && ok
    }

    /// `v as u64`, or just `v` if it already is one.
    fn as_u64(&self, v: ValueId) -> String {
        self.val(v, self.u64_ty)
    }

    fn ty(&self, v: ValueId) -> TyId {
        self.f.insts[v].ty
    }

    /// The Rust spelling of a type (`PAIR` is a tuple, `UNIT` is `()`).
    fn rs(&self, t: TyId) -> String {
        match t {
            TyId::PAIR => "(u64, u64)".into(),
            TyId::UNIT => "()".into(),
            TyId::BOOL => "bool".into(),
            _ => render(self.table, t),
        }
    }

    /// Rust type of value `v`.
    fn rt(&self, v: ValueId) -> String {
        self.rs(self.vt[v.index()])
    }

    fn is_signed(&self, v: ValueId) -> bool {
        matches!(self.table.tys[self.vt[v.index()]], Ty::Int { signed: true, .. })
    }

    /// `v` as the unsigned integer of its width (just `v` if it is one).
    fn as_u(&self, v: ValueId) -> String {
        if self.is_signed(v) { format!("{} as u{}", expr::cast(self.name(v)), bytes(self.ty(v)) * 8) } else { self.name(v) }
    }

    /// Expression `e` of type `from` as type `to`, with the bits the machine
    /// code would have: integers widen by their own signedness, so a signed
    /// value is made unsigned before it is zero-extended.
    fn conv(&self, e: &str, from: TyId, to: TyId) -> String {
        if from == to {
            return e.to_string();
        }
        let t = self.table;
        let to_s = self.rs(to);
        let c = || expr::cast(e.to_string());
        match (t.tys[from], t.tys[to]) {
            (Ty::Bool, _) if self.int_of(to).is_some() => format!("{} as {to_s}", c()),
            (_, Ty::Bool) => format!("{} != 0", expr::lhs(e.to_string(), "!=")),
            (Ty::RawPtr { .. }, Ty::F32 | Ty::F64) => self.conv(&format!("{} as u64", c()), self.u64_ty, to),
            (Ty::RawPtr { .. }, _) => format!("{} as {to_s}", c()),
            (_, Ty::RawPtr { .. }) => match self.int_of(from) {
                Some((bits, true)) if bits < 64 => format!("{} as u{bits} as {to_s}", c()),
                _ => format!("{} as {to_s}", c()),
            },
            (Ty::F32 | Ty::F64, _) => format!("{}.to_bits() as {to_s}", expr::recv(e.to_string())),
            (_, Ty::F32) => format!("f32::from_bits({} as u32)", c()),
            (_, Ty::F64) => format!("f64::from_bits({} as u64)", c()),
            _ => match (self.int_of(from), self.int_of(to)) {
                (Some((fb, true)), Some((tb, _))) if tb > fb => format!("{} as u{fb} as {to_s}", c()),
                (Some(_), Some(_)) => format!("{} as {to_s}", c()),
                // a tuple, `()`: nothing to convert
                _ => e.to_string(),
            },
        }
    }

    /// An argument of type `t` as the register value the body reads. Callers
    /// extend 8- and 16-bit arguments to 32 bits by their signedness (gcc and
    /// clang both do, and clang's code relies on it), so do the same.
    fn arg_to_reg(&self, b: &str, t: TyId, to: TyId) -> String {
        match self.table.tys[t] {
            Ty::Int { bits: 8 | 16, signed } => {
                let w = self.table.get(&Ty::Int { bits: 32, signed }).expect("integer types are interned");
                self.conv(&format!("{b} as {}32", if signed { 'i' } else { 'u' }), w, to)
            }
            Ty::Bool => self.conv(&format!("{b} as u32"), self.table.get(&Ty::Int { bits: 32, signed: false }).unwrap(), to),
            _ => self.conv(b, t, to),
        }
    }

    /// A zero of type `t`.
    fn zero(&self, t: TyId) -> String {
        match self.table.tys[t] {
            Ty::Bool => "false".into(),
            Ty::RawPtr { mutbl: Mutbl::Mut, .. } => "core::ptr::null_mut()".into(),
            Ty::RawPtr { .. } => "core::ptr::null()".into(),
            _ => format!("0_{}", self.rs(t)),
        }
    }

    /// Signature names for the entry parameters that aren't their register.
    fn bind_names(&mut self) {
        let params: Vec<ValueId> = self.f.blocks[self.f.entry].params.get(&self.f.value_pool).to_vec();
        let typed: Vec<(u8, Option<TyId>, Option<String>)> =
            self.types.map(|t| t.args.iter().map(|a| (a.reg, a.ty, a.name.clone())).collect()).unwrap_or_default();
        for (k, &p) in params.iter().enumerate() {
            let reg = match self.f.insts[p].kind {
                InstKind::BlockParam(r) => r,
                _ => continue,
            };
            let n = self.name(p);
            let (ty, dname) = typed.iter().find(|a| a.0 == reg).map(|a| (a.1, a.2.clone())).unwrap_or((None, None));
            self.bind[k] = match self.args[k] {
                ArgKind::Slice { .. } | ArgKind::Struct { .. } => Some(dname.unwrap_or(format!("{n}_ref"))),
                ArgKind::Int => match ty.map(|t| self.table.tys[t]) {
                    Some(Ty::RawPtr { .. }) => Some(dname.unwrap_or(format!("{n}_p"))),
                    _ => dname,
                },
            };
        }
    }

    /// Entry parameters, ordered for the signature: SysV argument registers first,
    /// then any other register the function reads on entry (callee-saved, rsp...).
    fn sig_order(&self) -> Vec<usize> {
        let params = self.f.blocks[self.f.entry].params.get(&self.f.value_pool);
        let reg = |k: usize| match self.f.insts[params[k]].kind {
            InstKind::BlockParam(r) => r,
            _ => u8::MAX,
        };
        let rank = |r: u8| SYSV_ARGS.iter().position(|&a| a == r).unwrap_or(6 + r as usize);
        let mut order: Vec<usize> = (0..params.len()).collect();
        order.sort_by_key(|&k| rank(reg(k)));
        order
    }

    /// Entry parameter indices in signature order, with `None` for an argument the
    /// function takes but never uses (and the name to give it), and the
    /// argument's recovered type.
    fn arg_order(&self) -> Vec<(Option<usize>, String, Option<TyId>)> {
        let params = self.f.blocks[self.f.entry].params.get(&self.f.value_pool);
        let Some(sig) = self.sig else {
            return self.sig_order().into_iter().map(|k| (Some(k), String::new(), None)).collect();
        };
        let find = |reg: u8| params.iter().position(|&p| matches!(self.f.insts[p].kind, InstKind::BlockParam(r) if r == reg));
        let typed = |reg: u8| self.types.and_then(|t| t.args.iter().find(|a| a.reg == reg));
        let unused = |reg: u8, fallback: String| match typed(reg).and_then(|a| a.name.as_ref()) {
            Some(n) => format!("_{n}"),
            None => fallback,
        };
        let mut order = Vec::new();
        for &reg in &SYSV_ARGS[..sig.args as usize] {
            order.push((find(reg), unused(reg, format!("_{}", REG[reg as usize])), typed(reg).and_then(|a| a.ty)));
        }
        for j in 0..sig.stack_args {
            let reg = STACK_ARG_BASE + j;
            order.push((find(reg), unused(reg, format!("_arg{}", 6 + j as usize)), typed(reg).and_then(|a| a.ty)));
        }
        // float arguments arrive as `f64`: the register holds an f32's bits the same way
        for j in 0..sig.fargs {
            let reg = XMM_PARAM + 2 * j;
            order.push((find(reg), format!("_xmm{j}"), Some(TyId::F64)));
        }
        // anything else on entry (there shouldn't be anything after `abi::apply`)
        for k in 0..params.len() {
            if !order.iter().any(|&(x, _, _)| x == Some(k)) {
                order.push((Some(k), String::new(), None));
            }
        }
        order
    }

    fn signature(&self, name: &str, out: &mut String) {
        let params = self.f.blocks[self.f.entry].params.get(&self.f.value_pool);
        let unsafety = if self.stats.raw > 0 { "unsafe " } else { "" };
        let mut sig = Vec::new();
        let mut prologue = String::new();
        for (k, unused, ty) in self.arg_order() {
            let Some(k) = k else {
                sig.push(format!("{unused}: {}", ty.map_or("u64".to_string(), |t| self.rs(t))));
                continue;
            };
            let p = params[k];
            let n = self.name(p);
            let pt = self.rt(p);
            let bind = self.bind[k].clone();
            match self.args[k] {
                ArgKind::Int => match (ty, bind) {
                    (None, None) => sig.push(format!("mut {n}: {pt}")),
                    (Some(t), b) => {
                        let b = b.unwrap_or_else(|| n.clone());
                        sig.push(format!("{b}: {}", self.rs(t)));
                        let _ = writeln!(prologue, "    let mut {n}: {pt} = {};", self.arg_to_reg(&b, t, self.vt[p.index()]));
                    }
                    (None, Some(b)) => {
                        sig.push(format!("{b}: {pt}"));
                        let _ = writeln!(prologue, "    let mut {n}: {pt} = {b};");
                    }
                },
                ArgKind::Slice { mutbl, nullable, elem } => {
                    let b = bind.unwrap_or_else(|| format!("{n}_ref"));
                    let et = elem.map_or("u8".to_string(), |t| self.rs(t));
                    let r = if mutbl { format!("&mut [{et}]") } else { format!("&[{et}]") };
                    let (m, t) = if nullable { (if mutbl { "mut " } else { "" }, format!("Option<{r}>")) } else { ("", r) };
                    sig.push(format!("{m}{b}: {t}"));
                    let addr = if nullable {
                        format!("{b}.as_deref().map_or(0, |s| s.as_ptr() as u64)")
                    } else {
                        format!("{b}.as_ptr() as u64")
                    };
                    let _ = writeln!(prologue, "    let {n}_base: u64 = {addr};");
                    let _ = writeln!(prologue, "    let mut {n}: {pt} = {};", self.conv(&format!("{n}_base"), self.u64_ty, self.vt[p.index()]));
                }
                ArgKind::Struct { mutbl, nullable, ty: st } => {
                    let b = bind.unwrap_or_else(|| format!("{n}_ref"));
                    let sn = self.rs(st);
                    let r = if mutbl { format!("&mut {sn}") } else { format!("&{sn}") };
                    let (m, t) = if nullable { (if mutbl { "mut " } else { "" }, format!("Option<{r}>")) } else { ("", r) };
                    sig.push(format!("{m}{b}: {t}"));
                    let addr = if nullable {
                        format!("{b}.as_deref().map_or(0, |s| s as *const {sn} as u64)")
                    } else {
                        format!("&*{b} as *const {sn} as u64")
                    };
                    let _ = writeln!(prologue, "    let {n}_base: u64 = {addr};");
                    let _ = writeln!(prologue, "    let mut {n}: {pt} = {};", self.conv(&format!("{n}_base"), self.u64_ty, self.vt[p.index()]));
                }
            }
        }
        let ret = match self.sig {
            Some(s) if s.ret2 => " -> (u64, u64)".to_string(),
            Some(s) if s.fret => " -> f64".to_string(),
            Some(s) if !s.ret => String::new(),
            _ => match self.types.and_then(|t| t.ret) {
                Some(t) => format!(" -> {}", self.rs(t)),
                None => " -> u64".to_string(),
            },
        };
        let _ = writeln!(out, "pub {unsafety}fn {name}({}){ret} {{", sig.join(", "));
        out.push_str(&prologue);
    }

    fn body(&mut self, cfg: &Cfg, out: &mut String) {
        let f = self.f;
        // The stack frame (`frame.rs`): u128s, so it is 16-byte aligned like a real one.
        if let Some((_, l)) = f.locals.iter().next() {
            let n = (l.size as usize).div_ceil(16);
            if self.frame_safe() {
                // bytes, so that safe code can index them; aligned like a real frame
                if n * 16 <= 4096 {
                    let _ = writeln!(out, "    #[repr(C, align(16))]\n    struct Frame([u8; {}]);", n * 16);
                    let _ = writeln!(out, "    let mut frame = Frame([0; {}]);", n * 16);
                } else {
                    let _ = writeln!(out, "    struct Frame(Vec<u8>);\n    let mut frame = Frame(vec![0; {}]);", n * 16);
                }
                // (mutable, for the objects of the frame that stay raw)
                let _ = writeln!(out, "    let frame_base: u64 = frame.0.as_mut_ptr() as u64;");
            } else if n * 16 <= 4096 {
                let _ = writeln!(out, "    let mut frame = [0u128; {n}];");
            } else {
                let _ = writeln!(out, "    let mut frame = vec![0u128; {n}];");
            }
        }
        // Allocations that are `Box`es.
        if let Some(a) = self.borrow {
            for (r, root) in a.roots.iter().enumerate() {
                if let (Root::Alloc(id), Some(Some(_))) = (root, self.places.get(r)) {
                    let h = format!("heap{}", id.index());
                    let _ = writeln!(out, "    let mut {h}: Box<[u8]> = Box::default();\n    let mut {h}_base: u64 = 0;");
                }
            }
        }
        // Up-front declarations for block params and cross-block values.
        let mut decls: Vec<(String, String, &str)> = Vec::new();
        for &b in &cfg.rpo {
            let blk = &f.blocks[b];
            for &v in blk.params.get(&f.value_pool).iter().chain(blk.insts.get(&f.value_pool)) {
                if self.hoisted[v.index()] && !self.skip[v.index()] && !self.is_lit(v) {
                    let zero = match self.ty(v) {
                        TyId::BOOL => "false",
                        TyId::PAIR => "(0, 0)",
                        _ if matches!(self.table.tys[self.vt[v.index()]], Ty::RawPtr { .. }) => "core::ptr::null_mut()",
                        _ => "0",
                    };
                    decls.push((self.name(v), self.rt(v), zero));
                }
            }
        }
        let declare = |decls: &[(String, String, &str)], skip: &[bool], out: &mut String| {
            for (k, (n, t, zero)) in decls.iter().enumerate() {
                if !skip.get(k).copied().unwrap_or(false) {
                    let _ = writeln!(out, "    let mut {n}: {t} = {zero};");
                }
            }
        };
        let has_edges = cfg.rpo.iter().any(|&b| f.blocks[b].term.successors(&f.value_pool).next().is_some());
        if !has_edges {
            declare(&decls, &[], out);
            self.block(f.entry, "    ", out);
            return;
        }
        // Structured `if`/`loop` when the CFG allows it. Statements are emitted
        // again below if it doesn't, so count them only once.
        let before = self.stats;
        if self.structure {
            if let Some(mut s) = structure(f, cfg, self) {
                // a variable assigned once, and used only after that in the same
                // scope, is declared there
                let names: Vec<&str> = decls.iter().map(|d| d.0.as_str()).collect();
                let types: Vec<String> = decls.iter().map(|d| d.1.clone()).collect();
                let inline = declare_in_place(&mut s.nodes, &names, &types);
                declare(&decls, &inline, out);
                if s.regions > 0 {
                    self.stats.state_machines += 1;
                    let _ = writeln!(out, "    let mut {DISPATCH_VAR}: u32 = {};", f.entry.index());
                }
                print(&s.nodes, 1, out);
                return;
            }
            self.stats = before;
        }
        declare(&decls, &[], out);
        self.stats.state_machines += 1;
        let _ = writeln!(out, "    let mut bb: u32 = {};", f.entry.index());
        out.push_str("    loop {\n        match bb {\n");
        for &b in &cfg.rpo {
            let _ = writeln!(out, "            {} => {{", b.index());
            self.block(b, "                ", out);
            out.push_str("            }\n");
        }
        out.push_str("            _ => unreachable!(),\n        }\n    }\n");
    }

    /// The block's statements, one per line, without its terminator.
    fn stmt_lines(&mut self, b: BlockId) -> Vec<String> {
        let f = self.f;
        let blk = &f.blocks[b];
        let mut out = Vec::new();
        for &id in blk.insts.get(&f.value_pool) {
            if self.skip[id.index()] || self.is_lit(id) {
                continue;
            }
            let at = f.origin.get(id.index()).copied().unwrap_or(0);
            if self.inline[id.index()] {
                if let Stmt::Value(e) = self.inst(id) {
                    self.exprs[id.index()] = Some(e);
                    continue;
                }
                unreachable!("only plain values are inlined");
            }
            let n = self.name(id);
            match self.inst(id) {
                Stmt::Value(e) if self.hoisted[id.index()] => out.push(format!("{n} = {e}; // {at:#x}")),
                Stmt::Value(e) => out.push(format!("let {n}: {} = {e}; // {at:#x}", self.rt(id))),
                Stmt::Effect(e) => out.push(format!("{e}; // {at:#x}")),
                Stmt::Pair(e) => {
                    // rax:rdx; the rdx half is a `CallOut` reading `vN_pair.1`
                    out.push(format!("let {n}_pair: (u64, u64) = {e}; // {at:#x}"));
                    let first = self.conv(&format!("{n}_pair.0"), self.u64_ty, self.vt[id.index()]);
                    if self.hoisted[id.index()] {
                        out.push(format!("{n} = {first};"));
                    } else {
                        out.push(format!("let {n}: {} = {first};", self.rt(id)));
                    }
                }
            }
        }
        out
    }

    /// The function's return type (`u64` unless types say otherwise).
    fn ret_ty(&self) -> TyId {
        if self.sig.is_some_and(|s| s.fret) {
            return TyId::F64;
        }
        self.types.and_then(|t| t.ret).unwrap_or(self.u64_ty)
    }

    /// A terminator that leaves the function (or can't be expressed yet).
    fn exit_line(&mut self, b: BlockId) -> String {
        let t = self.f.blocks[b].term;
        match t {
            Terminator::Return(Some(v)) if self.ty(v) == TyId::PAIR => format!("return {};", self.name(v)),
            Terminator::Return(Some(v)) => format!("return {};", self.abi_val(v, self.ret_ty())),
            Terminator::Return(None) if self.sig.is_some_and(|s| !s.ret) => "return;".to_string(),
            Terminator::Return(None) if self.types.is_none() => "return 0;".to_string(),
            Terminator::Return(None) => format!("return {};", self.zero(self.ret_ty())),
            Terminator::TailCall { callee, args } => {
                let (call, ret, ret2, rty) = self.call_expr(Site::Tail(b), callee, args);
                let me2 = self.sig.is_some_and(|s| s.ret2);
                match (self.sig.is_none_or(|s| s.ret), ret) {
                    (true, true) if ret2 && !me2 => format!("return {};", self.conv(&format!("{call}.0"), self.u64_ty, self.ret_ty())),
                    (true, true) if ret2 => format!("return {call};"),
                    (true, true) => format!("return {};", self.conv(&call, rty, self.ret_ty())),
                    (true, false) if self.types.is_none() => format!("{call}; return 0;"),
                    (true, false) => format!("{call}; return {};", self.zero(self.ret_ty())),
                    (false, _) => format!("{call}; return;"),
                }
            }
            Terminator::Unreachable => "panic!(\"execution ran past the end of the lifted code\");".to_string(),
            Terminator::Jump { .. } | Terminator::Branch { .. } | Terminator::Switch { .. } => unreachable!("not an exit"),
        }
    }

    /// One `match` arm of the state machine.
    fn block(&mut self, b: BlockId, ind: &str, out: &mut String) {
        let f = self.f;
        for line in self.stmt_lines(b) {
            let _ = writeln!(out, "{ind}{line}");
        }
        match f.blocks[b].term {
            Terminator::Jump { to, args } => {
                self.goto(to, args.get(&f.value_pool), ind, out);
            }
            Terminator::Branch { c, t, f: e, args } => {
                let a = args.get(&f.value_pool);
                let nt = f.blocks[t].params.len as usize;
                let inner = format!("{ind}    ");
                let _ = writeln!(out, "{ind}if {} {{", self.name(c));
                self.goto(t, &a[..nt], &inner, out);
                let _ = writeln!(out, "{ind}}} else {{");
                self.goto(e, &a[nt..], &inner, out);
                let _ = writeln!(out, "{ind}}}");
            }
            Terminator::Switch { v, table, default } => {
                let cases = table.get(&f.value_pool);
                let _ = writeln!(out, "{ind}bb = match {} {{", self.as_u(v));
                for t in f.blocks[b].term.successors(&f.value_pool).skip(1) {
                    let ks: Vec<u64> =
                        (0..cases.len() as u64).filter(|&k| BlockId::from_value(cases[k as usize]) == t).collect();
                    let _ = writeln!(out, "{ind}    {} => {},", case_pattern(&ks), t.index());
                }
                let _ = writeln!(out, "{ind}    _ => {},", default.index());
                let _ = writeln!(out, "{ind}}};");
            }
            _ => {
                let _ = writeln!(out, "{ind}{}", self.exit_line(b));
            }
        }
    }

    /// Pass edge arguments to `to`'s parameters, as a parallel assignment.
    fn assign(&self, to: BlockId, args: &[ValueId]) -> Option<String> {
        let params = self.f.blocks[to].params.get(&self.f.value_pool);
        let pairs: Vec<(String, String)> = params
            .iter()
            .zip(args)
            .map(|(&p, &a)| {
                let (pt, at) = (self.vt[p.index()], self.vt[a.index()]);
                // the lifter can pass a wider register than the parameter holds
                let v = match self.konst(a) {
                    _ if pt == at => self.name(a),
                    Some(c) if self.int_of(pt).is_some() && at != TyId::BOOL => self.lit_as(c, pt),
                    _ => self.val(a, pt),
                };
                (self.name(p), v)
            })
            .filter(|(p, a)| p != a)
            .collect();
        match pairs.len() {
            0 => None,
            1 => Some(format!("{} = {};", pairs[0].0, pairs[0].1)),
            _ => {
                let (l, r): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
                Some(format!("({}) = ({});", l.join(", "), r.join(", ")))
            }
        }
    }

    /// State machine edge: assign `to`'s parameters, then jump.
    fn goto(&self, to: BlockId, args: &[ValueId], ind: &str, out: &mut String) {
        if let Some(a) = self.assign(to, args) {
            let _ = writeln!(out, "{ind}{a}");
        }
        let _ = writeln!(out, "{ind}bb = {};", to.index());
    }

    fn inst(&mut self, id: ValueId) -> Stmt {
        use InstKind::*;
        let f = self.f;
        let ty = self.ty(id);
        let vt = self.vt[id.index()];
        let t = self.rt(id);
        let n = |v: ValueId| self.name(v);
        let e = match f.insts[id].kind {
            Const(c) => match ty {
                TyId::BOOL => (f.consts[c.index()] != 0).to_string(),
                _ => self.lit_as(self.konst(id).unwrap_or(0), vt),
            },
            Undef if ty == TyId::BOOL => "false".to_string(),
            CallOut { call, reg: crate::abi::RDX } if matches!(f.insts[call].kind, Call { .. }) && self.ret2(call) => {
                self.conv(&format!("{}_pair.1", n(call)), self.u64_ty, vt)
            }
            Undef | CallOut { .. } => format!("0_{t}"),
            Aggregate { ty: TyId::PAIR, fields } => {
                let v = fields.get(&f.value_pool);
                format!("({}, {})", self.as_u64(v[0]), self.as_u64(v[1]))
            }
            AddrOfLocal(_) if self.frame_safe() => self.conv("frame_base", self.u64_ty, vt),
            AddrOfLocal(_) => self.conv("frame.as_mut_ptr() as u64", self.u64_ty, vt),
            Call { args, .. } if self.boxed_alloc(id).is_some() => {
                // `malloc` of an allocation that is used, then freed or dropped, like a Box
                let h = self.boxed_alloc(id).unwrap();
                let (a, b) = (self.call)(Site::Call(id)).and_then(|c| c.alloc).unwrap_or((0, None));
                let args = args.get(&f.value_pool);
                let size = |v: ValueId| format!("{} as usize", expr::cast(self.as_u(v)));
                let mut len = format!("({})", size(args[a as usize]));
                if let Some(b) = b {
                    len = format!("{len}.wrapping_mul({})", size(args[b as usize]));
                }
                let e = format!("{{ {h} = vec![0u8; {len}].into_boxed_slice(); {h}_base = {h}.as_ptr() as u64; {h}_base }}");
                self.conv(&e, self.u64_ty, vt)
            }
            Call { args, .. } if self.builtin(id).is_some() => {
                let (b, ps) = self.builtin(id).unwrap();
                let a = args.get(&f.value_pool);
                let off = |k: usize| format!("{}.wrapping_sub({}) as usize", expr::recv(self.as_u64(a[k])), ps[k].base);
                let len = format!("{} as usize", expr::cast(self.as_u(a[2])));
                let dst = ps[0].write.clone().unwrap();
                let e = match b {
                    crate::libc::Builtin::Fill => format!("{dst}[{}..][..{len}].fill({} as u8)", off(0), n(a[1])),
                    crate::libc::Builtin::Copy if ps[0].read == ps[1].read => {
                        format!("let __s = {}; {dst}.copy_within(__s..__s + ({len}), {})", off(1), off(0))
                    }
                    crate::libc::Builtin::Copy => {
                        format!("{dst}[{}..][..{len}].copy_from_slice(&{}[{}..][..{len}])", off(0), ps[1].read, off(1))
                    }
                };
                if !self.used[id.index()] {
                    return Stmt::Effect(format!("{{ {e}; }}"));
                }
                format!("{{ {e}; {} }}", self.val(a[0], vt))
            }
            Call { args, .. } if self.boxed_free(id).is_some() => {
                let _ = args;
                return Stmt::Effect(format!("{} = Box::default()", self.boxed_free(id).unwrap()));
            }
            Call { callee, args } => {
                let (call, ret, ret2, rty) = self.call_expr(Site::Call(id), callee, args);
                if ret2 {
                    return Stmt::Pair(call);
                }
                if !ret || !self.used[id.index()] {
                    return Stmt::Effect(call);
                }
                self.conv(&call, rty, vt)
            }
            Param(i) => format!("todo!(\"param {i}\")"),
            BlockParam(_) => n(id),
            Bin { op, lhs, rhs } => self.bin(id, op, lhs, rhs),
            Un { op, v } => {
                let r = expr::recv(n(v));
                // in the result's own type
                let same = expr::recv(self.val(v, vt));
                match op {
                    UnOp::Neg => format!("{same}.wrapping_neg()"),
                    UnOp::Not => format!("!{}", expr::unary(self.val(v, vt))),
                    UnOp::Bswap => format!("{same}.swap_bytes()"),
                    UnOp::Popcnt => format!("{r}.count_ones() as {t}"),
                    UnOp::Ctz => format!("{r}.trailing_zeros() as {t}"),
                    UnOp::Clz => format!("{r}.leading_zeros() as {t}"),
                    UnOp::Lane(op, w) => {
                        self.conv(&format!("{}({})", crate::simd::un_name(op, w), self.as_u64(v)), self.u64_ty, vt)
                    }
                }
            }
            Cmp { cc, lhs, rhs } => self.cmp(cc, lhs, rhs),
            Cast { kind, v } => self.cast(kind, v, id),
            Select { c, t: a, f: b } => format!("if {} {{ {} }} else {{ {} }}", n(c), self.val(a, vt), self.val(b, vt)),
            PtrOffset { base, index, scale, disp } => {
                let mut s = expr::recv(self.as_u64(base));
                if let Some(i) = index {
                    s = if scale == 1 {
                        format!("{s}.wrapping_add({})", self.as_u64(i))
                    } else {
                        format!("{s}.wrapping_add({}.wrapping_mul({scale}))", expr::recv(self.as_u64(i)))
                    };
                }
                let s = match disp {
                    0 => s,
                    d if d < 0 => format!("{s}.wrapping_sub({})", lit((d as i64).unsigned_abs())),
                    d => format!("{s}.wrapping_add({})", lit(d as u64)),
                };
                self.conv(&s, self.u64_ty, vt)
            }
            IntToPtr(v) => match f.insts[v].kind {
                Const(c) => match self.index_only[id.index()].then(|| (self.global_before)(f.consts[c.index()] as u64)).flatten().or_else(|| (self.global_of)(f.consts[c.index()] as u64)) {
                    Some(g) => self.conv(&g, self.u64_ty, vt),
                    None => self.val(v, vt),
                },
                _ => self.val(v, vt),
            },
            PtrToInt(v) => self.val(v, vt),
            Opaque { asm, args } => return Stmt::Effect(self.asm(id, asm, args)),
            AsmOut { asm, k } => self.conv(&format!("{}_asm[{k}]", n(asm)), self.u64_ty, vt),
            Load { ptr, .. } => self.load(id, ptr),
            Store { ptr, val, .. } => return Stmt::Effect(self.store(ptr, val)),
            MemCopy { dst, src, len } => {
                let s = format!(
                    "unsafe {{ core::ptr::copy({} as *const u8, {} as *mut u8, {} as usize) }}",
                    expr::cast(self.as_u(src)),
                    expr::cast(self.as_u(dst)),
                    expr::cast(self.as_u(len))
                );
                self.stats.raw += 1;
                self.stats.raw_by[bucket(self.src[dst.index()] | self.src[src.index()])] += 1;
                return Stmt::Effect(s);
            }
            MemFill { dst, val, count } => {
                let w = bytes(self.ty(val));
                let (d, v, n) = (expr::cast(self.as_u(dst)), self.as_u(val), expr::cast(self.as_u(count)));
                let s = if w == 1 {
                    format!("unsafe {{ core::ptr::write_bytes({d} as *mut u8, {v}, {n} as usize) }}")
                } else {
                    let bits = w * 8;
                    format!(
                        "for fill_at in 0..{n} as usize {{ unsafe {{ ({d} as *mut u8).add(fill_at * {w}).cast::<u{bits}>().write_unaligned({v}) }} }}"
                    )
                };
                self.stats.raw += 1;
                self.stats.raw_by[bucket(self.src[dst.index()])] += 1;
                return Stmt::Effect(s);
            }
            other => {
                self.stats.todo += 1;
                let s = format!("{other:?}").replace('"', "'").replace('{', "{{").replace('}', "}}");
                if ty == TyId::UNIT {
                    return Stmt::Effect(format!("todo!(\"{s}\")"));
                }
                format!("todo!(\"{s}\")")
            }
        };
        Stmt::Value(e)
    }

    /// An instruction kept as inline assembly: an `asm!` on a copy of each
    /// register it uses, then `let vN_asm = [..]` with what it wrote (`AsmOut`
    /// reads it). An xmm register is a `__m128i` made of its two halves.
    fn asm(&mut self, id: ValueId, a: u32, args: ListRef) -> String {
        let f = self.f;
        let a = &f.asm[a as usize];
        let args = args.get(&f.value_pool);
        let (mut lets, mut operands, mut post, mut outs) = (String::new(), Vec::new(), String::new(), Vec::new());
        for (k, op) in a.ops.iter().enumerate() {
            let spec = if op.named { format!("\"{}\"", op.reg) } else { op.reg.to_string() };
            let x = format!("asm{k}");
            let input = match (op.xmm, op.input) {
                (true, Some(j)) => format!(
                    "core::mem::transmute::<[u64; 2], core::arch::x86_64::__m128i>([{}, {}])",
                    self.as_u64(args[j as usize]),
                    self.as_u64(args[j as usize + 1])
                ),
                (true, None) => "core::mem::transmute::<[u64; 2], core::arch::x86_64::__m128i>([0; 2])".to_string(),
                (false, Some(j)) => self.as_u64(args[j as usize]),
                (false, None) => "0_u64".to_string(),
            };
            if op.output.is_none() {
                operands.push(format!("in({spec}) {input}"));
                continue;
            }
            let _ = write!(lets, "let mut {x} = {input}; ");
            operands.push(format!("inout({spec}) {x}"));
            if op.xmm {
                let _ = write!(post, "let {x} = core::mem::transmute::<core::arch::x86_64::__m128i, [u64; 2]>({x}); ");
                outs.extend([format!("{x}[0]"), format!("{x}[1]")]);
            } else {
                outs.push(x);
            }
        }
        if !a.stack {
            operands.push("options(nostack)".to_string());
        }
        self.stats.raw += 1;
        self.stats.asm += 1;
        let call = format!("core::arch::asm!({:?}, {})", a.text, operands.join(", "));
        if outs.is_empty() {
            return format!("unsafe {{ {lets}{call}; }}");
        }
        format!("let {}_asm: [u64; {}] = unsafe {{ {lets}{call}; {post}[{}] }}", self.name(id), outs.len(), outs.join(", "))
    }

    /// Does the call `call` return rax:rdx?
    fn ret2(&self, call: ValueId) -> bool {
        (self.call)(Site::Call(call)).is_some_and(|c| c.ret2)
    }

    /// The arguments of a call as the callee takes them: converted to its
    /// recovered argument types, or `u64`.
    fn typed_args(&self, args: &[ValueId], info: Option<&CallInfo>) -> Vec<String> {
        let want = |k: usize| info.and_then(|c| c.arg_tys.get(k).copied().flatten()).unwrap_or(self.u64_ty);
        args.iter().enumerate().map(|(k, &v)| self.abi_val(v, want(k))).collect()
    }

    /// A call to the callee at `site`, whether it returns a value (`ret`) and two
    /// (`ret2`, as a `(u64, u64)`), and the Rust type of the value it returns.
    fn call_expr(&mut self, site: Site, callee: ValueId, args: ListRef) -> (String, bool, bool, TyId) {
        let args = args.get(&self.f.value_pool);
        // Straight from the lifter, a call lists all six argument registers and more.
        let args = if self.sig.is_none() { &args[..args.len().min(6)] } else { args };
        let info = (self.call)(site);
        let typed = self.typed_args(args, info.as_ref());
        let a = typed.join(", ");
        let rty = info.as_ref().and_then(|c| c.ret_ty).unwrap_or(self.u64_ty);
        match info {
            Some(CallInfo { path: Some(p), ret, ret2, foreign: false, raw: false, args: pass, .. }) if pass.iter().any(|x| matches!(x, Pass::Borrow { .. })) => {
                let (lets, a) = self.call_args(args, &pass, typed);
                (format!("unsafe {{ {lets}{p}({}) }}", a.join(", ")), ret, ret2, rty)
            }
            Some(CallInfo { path: Some(p), ret, ret2, foreign, raw, args: pass, arg_tys, .. }) => {
                if foreign || raw {
                    self.stats.raw += 1;
                }
                let a = self.raw_lends(args, &pass, &arg_tys, typed).join(", ");
                if ret2 && foreign {
                    // an extern returns `ffi::Pair`, which is FFI-safe; a tuple isn't
                    return (format!("unsafe {{ let pair_ = {p}({a}); (pair_.0, pair_.1) }}"), ret, ret2, rty);
                }
                (format!("unsafe {{ {p}({a}) }}"), ret, ret2, rty)
            }
            _ => {
                self.stats.raw += 1;
                let want = |k: usize| info.as_ref().and_then(|c| c.arg_tys.get(k).copied().flatten());
                let tys: Vec<&str> = (0..args.len()).map(|k| if want(k) == Some(TyId::F64) { "f64" } else { "u64" }).collect();
                let callee = self.as_u64(callee);
                (
                    format!("unsafe {{ core::mem::transmute::<u64, unsafe extern \"C\" fn({}) -> u64>({callee})({a}) }}", tys.join(", ")),
                    true,
                    false,
                    self.u64_ty,
                )
            }
        }
    }

    fn bin(&self, id: ValueId, op: BinOp, lhs: ValueId, rhs: ValueId) -> String {
        if let BinOp::Lane(op, w) = op {
            let call = format!("{}({}, {})", crate::simd::name(op, w), self.as_u64(lhs), self.as_u64(rhs));
            return self.conv(&call, self.u64_ty, self.vt[id.index()]);
        }
        let ty = self.ty(lhs);
        if ty == TyId::BOOL {
            let o = match op {
                BinOp::And => "&",
                BinOp::Or => "|",
                _ => "^",
            };
            return format!("{} {o} {}", expr::lhs(self.name(lhs), o), self.operand(rhs, lhs));
        }
        let vt = self.vt[id.index()];
        let t = self.rt(id);
        let res_signed = self.is_signed(id);
        // the operands in the result's own type
        let a = expr::recv(self.val(lhs, vt));
        // the right operand: its type is fixed by the method, so a literal needs no suffix
        let b = self.const_in(rhs, vt).unwrap_or_else(|| self.val(rhs, vt));
        let infix = |o: &str| format!("{} {o} {}", expr::lhs(self.val(lhs, vt), o), self.operand_as(rhs, vt));
        // computed unsigned, then as the result's type
        let unsigned = |e: String| if res_signed { format!("{} as {t}", expr::cast(e)) } else { e };
        let uinfix = |o: &str| {
            let r = match self.konst(rhs) {
                Some(c) => lit(c),
                None => expr::rhs(self.as_u(rhs)),
            };
            unsigned(format!("{} {o} {r}", expr::lhs(self.as_u(lhs), o)))
        };
        // computed signed, then as the result's type
        let sa = if self.is_signed(lhs) { expr::recv(self.name(lhs)) } else { format!("({})", self.signed(lhs)) };
        let signed = |e: String| if res_signed { e } else { format!("{e} as {t}") };
        // a shift or rotate amount, as the u32 the method takes
        let amount = match self.konst(rhs) {
            Some(c) => lit(c & 0xff),
            None => format!("{} as u32", expr::cast(self.name(rhs))),
        };
        let neg = self
            .konst(rhs)
            .filter(|_| !res_signed)
            .map(|c| c.wrapping_neg() & if bytes(ty) >= 8 { u64::MAX } else { (1 << (bytes(ty) * 8)) - 1 });
        match op {
            // `x - 8` reads better than `x + 0xfffffffffffffff8`
            BinOp::Add if neg.is_some_and(|m| m < 0x10000) => format!("{a}.wrapping_sub({})", lit(neg.unwrap())),
            BinOp::Sub if neg.is_some_and(|m| m < 0x10000) => format!("{a}.wrapping_add({})", lit(neg.unwrap())),
            BinOp::Add => format!("{a}.wrapping_add({b})"),
            BinOp::Sub => format!("{a}.wrapping_sub({b})"),
            BinOp::Mul => format!("{a}.wrapping_mul({b})"),
            BinOp::UMulHi => format!(
                "(({} as u128 * {} as u128) >> 64) as {t}",
                expr::cast(self.as_u(lhs)),
                expr::cast(self.as_u(rhs))
            ),
            BinOp::SMulHi => format!("(({} as i128 * {} as i128) >> 64) as {t}", self.signed(lhs), self.signed(rhs)),
            BinOp::UDiv => uinfix("/"),
            BinOp::URem => uinfix("%"),
            BinOp::SDiv => signed(format!("{sa}.wrapping_div({})", self.signed(rhs))),
            BinOp::SRem => signed(format!("{sa}.wrapping_rem({})", self.signed(rhs))),
            BinOp::And => infix("&"),
            BinOp::Or => infix("|"),
            BinOp::Xor => infix("^"),
            BinOp::Shl => format!("{a}.wrapping_shl({amount})"),
            BinOp::LShr => unsigned(format!("{}.wrapping_shr({amount})", expr::recv(self.as_u(lhs)))),
            BinOp::AShr => signed(format!("{sa}.wrapping_shr({amount})")),
            BinOp::RotL => format!("{a}.rotate_left({amount})"),
            BinOp::RotR => format!("{a}.rotate_right({amount})"),
            BinOp::Lane(..) => unreachable!("handled above"),
        }
    }

    /// How safe-mode code reaches each safe root of the borrow analysis.
    fn places(&self, global_slice: &dyn Fn(u64) -> Option<String>) -> Vec<Option<Place>> {
        let Some(a) = self.borrow else { return Vec::new() };
        let params = self.f.blocks[self.f.entry].params.get(&self.f.value_pool);
        a.roots
            .iter()
            .enumerate()
            .map(|(r, &root)| {
                if !a.safe[r] {
                    return None;
                }
                Some(match root {
                    Root::Param(k) => {
                        let ArgKind::Slice { mutbl, nullable, elem } = self.args[k as usize] else { return None };
                        let n = self.name(params[k as usize]);
                        let b = self.bind[k as usize].clone().unwrap_or_else(|| format!("{n}_ref"));
                        let read = match (nullable, mutbl) {
                            (false, _) => b.clone(),
                            (true, false) => format!("{b}.unwrap()"),
                            (true, true) => format!("{b}.as_deref().unwrap()"),
                        };
                        let write = match nullable {
                            false => b.clone(),
                            true => format!("{b}.as_deref_mut().unwrap()"),
                        };
                        Place { read, write: mutbl.then_some(write), base: format!("{n}_base"), elem }
                    }
                    Root::Frame(_) => Place { read: "frame.0".into(), write: Some("frame.0".into()), base: "frame_base".into(), elem: None },
                    Root::Global(c) => {
                        let g = global_slice(c)?;
                        Place { read: format!("{g}.b"), write: None, base: format!("(core::ptr::addr_of!({g}) as u64)"), elem: None }
                    }
                    Root::Contents(_) => return None,
                    Root::Alloc(id) => {
                        let h = format!("heap{}", id.index());
                        Place { read: h.clone(), write: Some(h.clone()), base: format!("{h}_base"), elem: None }
                    }
                })
            })
            .collect()
    }

    fn cmp(&self, cc: Cond, lhs: ValueId, rhs: ValueId) -> String {
        let ty = self.ty(lhs);
        let unsigned = match cc {
            Cond::Eq => Some("=="),
            Cond::Ne => Some("!="),
            Cond::Ult => Some("<"),
            Cond::Ule => Some("<="),
            Cond::Ugt => Some(">"),
            Cond::Uge => Some(">="),
            _ => None,
        };
        if let Some(op) = unsigned {
            if let Some(e) = self.past_end(lhs, rhs) {
                return format!("{e} {op} {}", expr::rhs(self.as_u64(rhs)));
            }
            if let Some(e) = self.past_end(rhs, lhs) {
                return format!("{} {op} {e}", expr::lhs(self.as_u64(lhs), op));
            }
        }
        let ptr = |v: ValueId| matches!(self.table.tys[self.vt[v.index()]], Ty::RawPtr { .. });
        if ptr(lhs) || ptr(rhs) {
            let null = |p: ValueId, z: ValueId| (ptr(p) && self.konst(z) == Some(0)).then(|| expr::recv(self.name(p)));
            if let (Some(p), Cond::Eq | Cond::Ne) = (null(lhs, rhs).or_else(|| null(rhs, lhs)), cc) {
                return format!("{}{p}.is_null()", if matches!(cc, Cond::Ne) { "!" } else { "" });
            }
            // as the integers the machine code compares
            let (a, b) = (self.as_u64(lhs), self.as_u64(rhs));
            let op = match cc {
                Cond::Eq => "==",
                Cond::Ne => "!=",
                Cond::Ult => "<",
                Cond::Ule => "<=",
                Cond::Ugt => ">",
                Cond::Uge => ">=",
                Cond::Slt => return format!("({a} as i64) < ({b} as i64)"),
                Cond::Sle => return format!("({a} as i64) <= ({b} as i64)"),
                Cond::Sgt => return format!("({a} as i64) > ({b} as i64)"),
                Cond::Sge => return format!("({a} as i64) >= ({b} as i64)"),
            };
            return format!("{} {op} {}", expr::lhs(a, op), expr::rhs(b));
        }
        let (op, sign) = match cc {
            Cond::Eq => ("==", None),
            Cond::Ne => ("!=", None),
            Cond::Ult => ("<", Some(false)),
            Cond::Ule => ("<=", Some(false)),
            Cond::Ugt => (">", Some(false)),
            Cond::Uge => (">=", Some(false)),
            Cond::Slt => ("<", Some(true)),
            Cond::Sle => ("<=", Some(true)),
            Cond::Sgt => (">", Some(true)),
            Cond::Sge => (">=", Some(true)),
        };
        match sign {
            _ if ty == TyId::BOOL => return format!("{} {op} {}", expr::lhs(self.name(lhs), op), self.operand(rhs, lhs)),
            None => return format!("{} {op} {}", expr::lhs(self.name(lhs), op), self.operand(rhs, lhs)),
            Some(false) => {
                let b = match self.konst(rhs) {
                    Some(c) => lit(c),
                    None => expr::rhs(self.as_u(rhs)),
                };
                return format!("{} {op} {b}", expr::lhs(self.as_u(lhs), op));
            }
            Some(true) => {}
        }
        let a = if self.is_signed(lhs) { expr::lhs(self.name(lhs), op) } else { format!("({})", self.signed(lhs)) };
        let b = match self.konst(rhs) {
            // the constant as the signed number it is compared as
            Some(c) => {
                let bits = bytes(ty) as u32 * 8;
                let v = ((c << (64 - bits)) as i64) >> (64 - bits);
                if (-9..10).contains(&v) { v.to_string() } else if v < 0 { format!("-{:#x}", v.unsigned_abs()) } else { format!("{v:#x}") }
            }
            None if self.is_signed(rhs) => expr::rhs(self.name(rhs)),
            None => format!("({})", self.signed(rhs)),
        };
        format!("{a} {op} {b}")
    }

    /// A cast, skipping an inner cast that the outer one makes redundant:
    /// `x as u32 as u64 as u8` is `x as u8`.
    fn cast(&self, kind: CastKind, mut v: ValueId, id: ValueId) -> String {
        let ty = self.ty(id);
        let vt = self.vt[id.index()];
        let t = self.rt(id);
        if let CastKind::Trunc | CastKind::ZExt = kind {
            let trunc = matches!(kind, CastKind::Trunc);
            while self.inline[v.index()] {
                let InstKind::Cast { kind: inner, v: w } = self.f.insts[v].kind else { break };
                let redundant = match inner {
                    CastKind::ZExt => true,
                    CastKind::Trunc => trunc,
                    // only the source's own bits survive
                    CastKind::SExt => trunc && bytes(ty) <= bytes(self.ty(w)),
                    _ => false,
                };
                // `w as T` would sign-extend a signed `w`
                let widens_signed = self.is_signed(w) && bytes(ty) > bytes(self.ty(w));
                if !redundant || widens_signed {
                    break;
                }
                v = w;
            }
        }
        if let Some(c) = self.konst(v) {
            // fold a cast of a constant
            if self.ty(v) != TyId::BOOL && ty != TyId::BOOL && matches!(kind, CastKind::Trunc | CastKind::ZExt | CastKind::SExt) {
                let bits = bytes(self.ty(v)) as u32 * 8;
                let c = if matches!(kind, CastKind::SExt) { (((c << (64 - bits)) as i64) >> (64 - bits)) as u64 } else { c };
                let mask = if bytes(ty) >= 8 { u64::MAX } else { (1u64 << (bytes(ty) * 8)) - 1 };
                return self.lit_as(c & mask, vt);
            }
        }
        if self.vt[v.index()] == vt {
            return self.name(v);
        }
        match kind {
            CastKind::SExt => format!("{} as {t}", self.signed(v)),
            CastKind::ZExt => format!("{} as {t}", expr::cast(self.as_u(v))),
            _ => format!("{} as {t}", expr::cast(self.name(v))),
        }
    }

    /// `v` reinterpreted as the signed integer of its width: `v as i32`, where
    /// casts inside `v` that don't change its low bits are dropped. Just `v`
    /// (ready for an `as`) if it is signed already.
    fn signed(&self, mut v: ValueId) -> String {
        if self.is_signed(v) {
            return expr::cast(self.name(v));
        }
        let width = bytes(self.ty(v));
        let s = signed(self.ty(v));
        while self.inline[v.index()] {
            let InstKind::Cast { kind, v: w } = self.f.insts[v].kind else { break };
            let keeps_low_bits = match kind {
                CastKind::ZExt | CastKind::Trunc => true,
                CastKind::SExt => width <= bytes(self.ty(w)),
                _ => false,
            };
            if !keeps_low_bits || bytes(self.ty(w)) < width || self.ty(w) == TyId::BOOL {
                break;
            }
            v = w;
        }
        format!("{} as {s}", expr::cast(self.name(v)))
    }

    /// The place an access through `ptr` can bounds-check against, if `ptr`
    /// derives from exactly one safe root (that can be written, for a store).
    fn slice_root(&self, ptr: ValueId, write: bool) -> Option<(String, String, Option<TyId>)> {
        let r = self.borrow?.safe_root(ptr)?;
        let p = self.places.get(r as usize)?.as_ref()?;
        match write {
            false => Some((p.read.clone(), p.base.clone(), p.elem)),
            true => Some((p.write.clone()?, p.base.clone(), p.elem)),
        }
    }

    /// A field access for a load or store of `bytes` through `ptr`: the place
    /// (`p.count`, `(*p).count`), whether it is a raw dereference, and the
    /// field's type.
    fn field(&self, ptr: ValueId, bytes: usize, write: bool) -> Option<(String, bool, TyId)> {
        let types = self.types?;
        let (r, d) = base_of(self.f, &self.alias, ptr);
        // an undefined base prints as 0, and `(*(0 as *const S)).f` is rejected
        if matches!(self.f.insts[r].kind, InstKind::Undef | InstKind::Const(_)) {
            return None;
        }
        let s = types.pointee[r.index()].or_else(|| types.pointee[ptr.index()].filter(|_| d == 0))?;
        if !matches!(self.table.tys[s], Ty::Struct(_)) {
            return None;
        }
        let (path, lt) = leaf(self.table, s, d, bytes as u32)?;
        let sn = self.rs(s);
        if let Some(k) = self.entry_param[r.index()] {
            let b = self.bind[k].clone();
            match (self.args[k], b) {
                (ArgKind::Struct { mutbl, nullable, ty }, Some(b)) if ty == s && (mutbl || !write) => {
                    let base = match (nullable, mutbl, write) {
                        (false, _, _) => b,
                        (true, false, _) => format!("{b}.unwrap()"),
                        (true, true, false) => format!("{b}.as_deref().unwrap()"),
                        (true, true, true) => format!("{b}.as_deref_mut().unwrap()"),
                    };
                    return Some((format!("{base}{path}"), false, lt));
                }
                (ArgKind::Int, Some(b)) => {
                    let arg = types.args.iter().find(|a| Some(a.reg) == self.reg_of(r)).and_then(|a| a.ty);
                    if let Some(Ty::RawPtr { pointee, mutbl }) = arg.map(|t| self.table.tys[t]) {
                        if pointee == s && (mutbl == Mutbl::Mut || !write) {
                            return Some((format!("(*{b}){path}"), true, lt));
                        }
                    }
                }
                _ => {}
            }
        }
        // a pointer variable this address is computed from, directly
        if let Ty::RawPtr { pointee, .. } = self.table.tys[self.vt[r.index()]] {
            // (and only if this access's own address is written with `r`: an
            // address the same as an earlier one is written as that one, which
            // may be the only variable in scope, as in a condition the
            // structurer repeats in another branch)
            if pointee == s
                && !self.inline[r.index()]
                && crate::types::decompose(self.f, ptr) == (r, d)
                && mentions(&self.as_u64(ptr), &self.name(r))
            {
                return Some((format!("(*{}){path}", self.name(r)), true, lt));
            }
        }
        let m = if write { "mut" } else { "const" };
        // The struct's address: `r` may be a value from another block that only an
        // alias reaches, so go from `ptr`, which this access has in scope.
        let p = self.as_u64(ptr);
        let base = match d {
            0 => p,
            d if d < 0 => format!("{}.wrapping_add({})", expr::recv(p), lit(d.unsigned_abs())),
            d => format!("{}.wrapping_sub({})", expr::recv(p), lit(d as u64)),
        };
        Some((format!("(*({} as *{m} {sn})){path}", expr::cast(base)), true, lt))
    }

    fn reg_of(&self, v: ValueId) -> Option<u8> {
        match self.f.insts[v].kind {
            InstKind::BlockParam(r) => Some(r),
            _ => None,
        }
    }

    fn load(&mut self, id: ValueId, ptr: ValueId) -> String {
        let ty = self.ty(id);
        let (t, len) = (self.rt(id), bytes(ty));
        let slice = self.slice_root(ptr, false);
        if let Some((place, raw, lt)) = self.field(ptr, len, false).filter(|(_, raw, _)| !raw || slice.is_none()) {
            let e = self.conv(&place, lt, self.vt[id.index()]);
            if raw {
                self.stats.raw += 1;
                self.stats.raw_by[bucket(self.src[ptr.index()])] += 1;
                return format!("unsafe {{ {e} }}");
            }
            self.stats.checked += 1;
            return e;
        }
        let p = self.as_u64(ptr);
        let (pr, pc) = (expr::recv(p.clone()), expr::cast(p));
        if let Some((s, base, elem)) = slice {
            self.stats.checked += 1;
            if let Some(et) = elem {
                // `elems_arg`: a whole element at a multiple of its size
                let i = self.elem_index(ptr, len).unwrap_or_else(|| format!("({pr}.wrapping_sub({base}) / {len}) as usize"));
                return self.conv(&format!("{s}[{i}]"), et, self.vt[id.index()]);
            }
            let bytes = format!("{s}[{pr}.wrapping_sub({base}) as usize..][..{len}].try_into().unwrap()");
            return match self.table.tys[self.vt[id.index()]] {
                // a typed pointer is stored as its address
                Ty::RawPtr { .. } => format!("(u64::from_le_bytes({bytes}) as {t})"),
                _ => format!("{t}::from_le_bytes({bytes})"),
            };
        }
        self.stats.raw += 1;
        self.stats.raw_by[bucket(self.src[ptr.index()])] += 1;
        format!("unsafe {{ ({pc} as *const {t}).read_unaligned() }}")
    }

    fn store(&mut self, ptr: ValueId, val: ValueId) -> String {
        let vt = self.ty(val);
        let (t, len, v) = (self.rt(val), bytes(vt), self.name(val));
        // a pointer is stored as its address
        let (t, v) = match self.table.tys[self.vt[val.index()]] {
            Ty::RawPtr { .. } => ("u64".to_string(), self.as_u64(val)),
            _ => (t, v),
        };
        let slice = self.slice_root(ptr, true);
        if let Some((place, raw, lt)) = self.field(ptr, len, true).filter(|(_, raw, _)| !raw || slice.is_none()) {
            let e = match self.table.tys[lt] {
                // the pointee may not be printed (`render_structs`); let Rust infer it
                Ty::RawPtr { .. } => format!("{} as _", expr::cast(self.as_u64(val))),
                _ => self.val(val, lt),
            };
            if raw {
                self.stats.raw += 1;
                self.stats.raw_by[bucket(self.src[ptr.index()])] += 1;
                return format!("unsafe {{ {place} = {e} }}");
            }
            self.stats.checked += 1;
            return format!("{place} = {e}");
        }
        let p = self.as_u64(ptr);
        let (pr, pc, vr) = (expr::recv(p.clone()), expr::cast(p), expr::recv(v.clone()));
        if let Some((s, base, elem)) = slice {
            self.stats.checked += 1;
            if let Some(et) = elem {
                let i = self.elem_index(ptr, len).unwrap_or_else(|| format!("({pr}.wrapping_sub({base}) / {len}) as usize"));
                return format!("{s}[{i}] = {}", self.val(val, et));
            }
            return format!("{s}[{pr}.wrapping_sub({base}) as usize..][..{len}].copy_from_slice(&{vr}.to_le_bytes())");
        }
        self.stats.raw += 1;
        self.stats.raw_by[bucket(self.src[ptr.index()])] += 1;
        format!("unsafe {{ ({pc} as *mut {t}).write_unaligned({v}) }}")
    }

    /// The index of `ptr` in the `&[T]` argument it points into, with `w`-byte
    /// elements, when the address is plainly that argument plus a constant and
    /// maybe a scaled index (`rdi + 4 * v8 + 8` is `v8 as usize + 2`).
    fn elem_index(&self, ptr: ValueId, w: usize) -> Option<String> {
        use InstKind::*;
        let elems = |v: ValueId| matches!(self.entry_param[v.index()].map(|k| self.args[k]), Some(ArgKind::Slice { elem: Some(_), .. }));
        let (v, d) = crate::types::decompose(self.f, ptr);
        if d < 0 || d % w as i64 != 0 {
            return None;
        }
        let k = d as u64 / w as u64;
        if elems(v) {
            return Some(k.to_string());
        }
        // `i * w`
        let scaled = |x: ValueId| -> Option<ValueId> {
            let c = |y: ValueId| match self.f.insts[y].kind {
                Const(c) => Some(self.f.consts[c.index()] as u64),
                _ => None,
            };
            match self.f.insts[x].kind {
                Bin { op: BinOp::Mul, lhs, rhs } if c(rhs) == Some(w as u64) => Some(lhs),
                Bin { op: BinOp::Mul, lhs, rhs } if c(lhs) == Some(w as u64) => Some(rhs),
                Bin { op: BinOp::Shl, lhs, rhs } if c(rhs).is_some_and(|s| s < 64 && 1u64 << s == w as u64) => Some(lhs),
                _ => None,
            }
        };
        let base = |x: ValueId| {
            let (b, d) = crate::types::decompose(self.f, x);
            (d == 0 && elems(b)).then_some(b)
        };
        let i = match self.f.insts[v].kind {
            PtrOffset { base: b, index: Some(i), scale, disp: 0 } if scale as usize == w && base(b).is_some() => i,
            Bin { op: BinOp::Add, lhs, rhs } if base(lhs).is_some() => scaled(rhs)?,
            Bin { op: BinOp::Add, lhs, rhs } if base(rhs).is_some() => scaled(lhs)?,
            _ => return None,
        };
        let i = format!("{} as usize", expr::cast(self.name(i)));
        Some(if k == 0 { i } else { format!("({i}).wrapping_add({k})") })
    }

    /// The frame is a safe root: an array of bytes that accesses index.
    fn frame_safe(&self) -> bool {
        self.borrow
            .is_some_and(|a| a.roots.iter().enumerate().any(|(r, x)| matches!(x, Root::Frame(_)) && self.places.get(r).is_some_and(|p| p.is_some())))
    }

    /// A call to `memcpy` or `memset` whose pointers all have safe roots: what it
    /// does, and the places of its arguments (`dst`, `src`).
    fn builtin(&self, call: ValueId) -> Option<(crate::libc::Builtin, Vec<&Place>)> {
        let a = self.borrow?;
        let b = (self.call)(Site::Call(call))?.builtin?;
        if a.raw_calls.contains(&Site::Call(call)) {
            return None;
        }
        let InstKind::Call { args, .. } = self.f.insts[call].kind else { return None };
        let args = args.get(&self.f.value_pool);
        let ptrs = if b == crate::libc::Builtin::Copy { 2 } else { 1 };
        if args.len() < 3 {
            return None;
        }
        let mut ps = Vec::new();
        for &p in &args[..ptrs] {
            ps.push(self.places.get(a.safe_root(p)? as usize)?.as_ref()?);
        }
        ps[0].write.as_ref()?;
        Some((b, ps))
    }

    /// The `Box` this `free` call drops, if its argument is one.
    fn boxed_free(&self, call: ValueId) -> Option<String> {
        if !(self.call)(Site::Call(call)).is_some_and(|c| c.free) {
            return None;
        }
        let InstKind::Call { args, .. } = self.f.insts[call].kind else { return None };
        let &p = args.get(&self.f.value_pool).first()?;
        let a = self.borrow?;
        let r = a.safe_root(p)?;
        match a.roots[r as usize] {
            Root::Alloc(_) => self.places.get(r as usize)?.as_ref().map(|p| p.read.clone()),
            _ => None,
        }
    }

    /// The safe root of the allocation this call makes, if it is a `Box`.
    fn boxed_alloc(&self, call: ValueId) -> Option<String> {
        let a = self.borrow?;
        let r = a.roots.iter().position(|&x| x == Root::Alloc(call))?;
        self.places.get(r)?.as_ref().map(|p| p.read.clone())
    }

    /// Safe-mode arguments for a call to a decompiled function: slices for the
    /// arguments it borrows, reborrowed from the root each one points into. Two
    /// borrows of one root, one of them mutable, start at different known offsets
    /// (the borrow analysis downgrades the others); they are split apart with
    /// `split_at_mut`. Returns the statements that do the splitting, and the
    /// argument expressions.
    /// The arguments of a call to a raw twin or a C function: an argument it
    /// borrows or accesses whose root is safe is a pointer made from the root's
    /// slice at the call (`borrow::FactKind::RawLend`), not the address kept
    /// from earlier.
    fn raw_lends(&self, args: &[ValueId], pass: &[Pass], tys: &[Option<TyId>], mut out: Vec<String>) -> Vec<String> {
        let Some(a) = self.borrow else { return out };
        for (k, &v) in args.iter().enumerate() {
            let mutbl = match pass.get(k) {
                Some(&Pass::Borrow { mutbl, .. }) => mutbl,
                Some(&Pass::Access { write }) => write,
                _ => continue,
            };
            let Some(place) = a.safe_root(v).and_then(|r| self.places.get(r as usize)).and_then(|p| p.as_ref()) else { continue };
            let slice = match (&place.write, mutbl) {
                (Some(w), true) => format!("{w}.as_mut_ptr()"),
                (_, false) => format!("{}.as_ptr()", place.read),
                (None, true) => continue,
            };
            let addr = format!("({slice} as u64).wrapping_add({}.wrapping_sub({}))", expr::recv(self.as_u64(v)), place.base);
            out[k] = self.conv(&addr, self.u64_ty, tys.get(k).copied().flatten().unwrap_or(self.u64_ty));
        }
        out
    }

    fn call_args(&mut self, args: &[ValueId], pass: &[Pass], typed: Vec<String>) -> (String, Vec<String>) {
        let mut out = typed;
        let mut lets = String::new();
        let Some(a) = self.borrow else { return (lets, out) };
        // (root, argument) for every borrowed argument that points into a root;
        // the objects of the frame are one array, so they are one group
        let mut groups: Vec<(u8, Vec<usize>)> = Vec::new();
        let same = |r: u8, s: u8| r == s || [r, s].iter().all(|&x| matches!(a.roots.get(x as usize), Some(Root::Frame(_))));
        for (k, &v) in args.iter().enumerate() {
            let Some(&Pass::Borrow { nullable, .. }) = pass.get(k) else { continue };
            match self.borrow.and_then(|a| a.safe_root(v)) {
                Some(r) if self.places.get(r as usize).is_some_and(|p| p.is_some()) => {
                    match groups.iter_mut().find(|g| same(g.0, r)) {
                        Some(g) => g.1.push(k),
                        None => groups.push((r, vec![k])),
                    }
                }
                _ if nullable && self.borrow.is_some_and(|a| a.origin[v.index()].is_none()) => out[k] = "None".into(),
                _ => {
                    self.stats.todo += 1;
                    out[k] = "todo!(\"a slice the borrow analysis couldn't prove\")".into();
                }
            }
        }
        // The other arguments may read the roots lent here (`f(&mut s[..],
        // s[8])`), so they are evaluated first.
        // So may the addresses of the lent ones.
        let simple = |e: &str| e.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        let mut offs: Vec<String> = args.iter().map(|&v| expr::recv(self.as_u64(v))).collect();
        if !groups.is_empty() {
            let lent: Vec<usize> = groups.iter().flat_map(|g| g.1.iter().copied()).collect();
            for k in 0..out.len() {
                if !lent.contains(&k) && !simple(&out[k]) {
                    let _ = write!(lets, "let __a{k} = {}; ", out[k]);
                    out[k] = format!("__a{k}");
                } else if lent.contains(&k) && !simple(&offs[k]) {
                    let _ = write!(lets, "let __o{k}: u64 = {}; ", offs[k]);
                    offs[k] = format!("__o{k}");
                }
            }
        }
        for (r, mut ks) in groups {
            let place = self.places[r as usize].as_ref().unwrap();
            let mutbl = |k: usize| matches!(pass[k], Pass::Borrow { mutbl: true, .. });
            let nullable = |k: usize| matches!(pass[k], Pass::Borrow { nullable: true, .. });
            let off = |k: usize| offs[k].clone();
            let mut exprs: Vec<(usize, String)> = Vec::new();
            if ks.len() == 1 || !ks.iter().any(|&k| mutbl(k)) {
                for &k in &ks {
                    let v = off(k);
                    let e = match (mutbl(k), &place.write) {
                        (true, Some(w)) => format!("&mut {w}[{v}.wrapping_sub({}) as usize..]", place.base),
                        (true, None) => "todo!(\"a mutable borrow of a read-only root\")".to_string(),
                        (false, _) => format!("&{}[{v}.wrapping_sub({}) as usize..]", place.read, place.base),
                    };
                    exprs.push((k, e));
                }
            } else {
                // distinct known offsets in one object, or different objects of
                // the frame, in order
                let at = |k: usize| {
                    let o = a.origin[args[k].index()];
                    let piece = match o.single().and_then(|r| a.roots.get(r as usize)) {
                        Some(&Root::Frame(lo)) => lo as i64,
                        _ => 0,
                    };
                    (piece, match o.off {
                        Off::Known(x) => x,
                        Off::Unknown => i64::MAX,
                    })
                };
                ks.sort_by_key(|&k| at(k));
                let w = place.write.clone().unwrap_or_default();
                let _ = write!(lets, "let __s = &mut {w}[{}.wrapping_sub({}) as usize..]; ", off(ks[0]), place.base);
                for i in 0..ks.len() {
                    let name = format!("__s{r}_{i}");
                    if i + 1 < ks.len() {
                        let _ = write!(lets, "let ({name}, __s) = __s.split_at_mut({}.wrapping_sub({}) as usize); ", off(ks[i + 1]), off(ks[i]));
                    } else {
                        let _ = write!(lets, "let {name} = __s; ");
                    }
                    let k = ks[i];
                    exprs.push((k, if mutbl(k) { format!("&mut *{name}") } else { format!("&*{name}") }));
                }
            }
            for (k, e) in exprs {
                out[k] = if nullable(k) { format!("if {} == 0 {{ None }} else {{ Some({e}) }}", off(k)) } else { e };
            }
        }
        (lets, out)
    }
}

enum Stmt {
    /// Defines the value: `let vN: T = expr;`
    Value(String),
    /// Side effect only.
    Effect(String),
    /// A call returning rax:rdx: `let vN_pair = expr; let vN = vN_pair.0;`
    Pair(String),
}

impl Source for Emitter<'_> {
    fn stmts(&mut self, b: BlockId) -> Vec<Node> {
        self.stmt_lines(b).into_iter().map(Node::Line).collect()
    }

    fn edge(&mut self, to: BlockId, args: &[ValueId]) -> Vec<Node> {
        self.assign(to, args).map(Node::Copy).into_iter().collect()
    }

    fn cond(&self, c: ValueId) -> String {
        self.name(c)
    }

    fn case(&self, v: ValueId, cases: &[u64]) -> String {
        match cases {
            [k] => format!("{} == {k}", self.as_u(v)),
            _ => format!("matches!({}, {})", self.as_u(v), case_pattern(cases)),
        }
    }

    fn exit(&mut self, b: BlockId) -> Node {
        Node::Exit(self.exit_line(b))
    }

    fn quiet(&self, b: BlockId) -> bool {
        let blk = &self.f.blocks[b];
        blk.params.len == 0
            && blk.insts.get(&self.f.value_pool).iter().all(|&v| self.skip[v.index()] || self.is_lit(v) || self.inline[v.index()])
    }
}

/// `e` uses the variable `name` (as a whole identifier).
pub(crate) fn mentions(e: &str, name: &str) -> bool {
    let word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    e.match_indices(name).any(|(i, _)| {
        !e[..i].ends_with(word) && !e[i + name.len()..].starts_with(word)
    })
}
