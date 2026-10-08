//! Safe-mode borrow inference: which objects a function's pointers point into, and
//! which of those objects safe mode can access through a bounds-checked slice.
//!
//! It runs on clean SSA (`opt::clean`) in three steps (docs/ownership.md):
//!
//! 1. **Origins.** A forward dataflow over the CFG assigns every value the set of
//!    *roots* it may point into, plus a byte offset when that offset is a known
//!    constant. A root is an entry parameter, the stack frame (`AddrOfLocal`), a
//!    global (a constant address), or an allocation (the result of a call whose
//!    summary says it allocates, like `malloc`). Pointers stored into the frame or
//!    an allocation are followed through memory: a load from `(root, offset)` gets
//!    the origins stored there (points-to, flow-insensitive per slot).
//! 2. **Facts.** One pass records what happens to derived values: reads, writes,
//!    escapes, returns, null checks, and at calls whatever the callee's summary
//!    (`Callee`) says it does with each argument: ignore it, borrow it as a slice,
//!    free it, or keep it (escape).
//! 3. **Classes**, to a fixpoint that only ever downgrades. A root is *safe* when
//!    every use of it is one safe mode can express: accesses through exactly this
//!    root, reborrows into callees, and pointers to it stored only where they are
//!    tracked. Loans (a callee borrowing two overlapping parts of a root, one of
//!    them mutably) and moves (an allocation used after `free`) are checked with
//!    Polonius-style rules in `loans.rs`; a failure downgrades the loan or the root.
//!
//! An argument counts as a pointer only if something dereferences a value derived
//! from it. Compilers use `lea` and `add` for plain integer arithmetic, so being
//! offset or returned is not evidence on its own.
use crate::abi::Site;
use crate::cfg::Cfg;
use crate::ir::*;
use crate::loans;
use std::collections::HashMap;

/// Byte offset of a derived pointer from its root.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Off {
    Known(i64),
    /// Indexed, advanced in a loop, or merged from different offsets.
    Unknown,
}

/// Which roots a value may point into, and where.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Origin {
    /// Bit `r` set: may be derived from root `r` (`Analysis::roots`; bit
    /// `OTHER` stands for the roots past the 63rd).
    pub roots: u64,
    /// Meaningful only when `roots` is non-zero.
    pub off: Off,
}

/// The root index shared by every root past the 63rd. It is never safe.
pub const OTHER: u8 = 63;

impl Origin {
    pub const NONE: Origin = Origin { roots: 0, off: Off::Known(0) };

    pub fn is_none(self) -> bool {
        self.roots == 0
    }

    fn root(r: u8) -> Origin {
        Origin { roots: 1 << r, off: Off::Known(0) }
    }

    /// The root, if there is exactly one.
    pub fn single(self) -> Option<u8> {
        (self.roots.count_ones() == 1).then(|| self.roots.trailing_zeros() as u8)
    }

    fn join(self, o: Origin) -> Origin {
        if self.is_none() {
            return o;
        }
        if o.is_none() {
            return self;
        }
        let off = if self.roots == o.roots && self.off == o.off { self.off } else { Off::Unknown };
        Origin { roots: self.roots | o.roots, off }
    }

    // `shift` and `unknown` keep NONE canonical (offset Known(0)); otherwise an
    // untracked counter decremented in a loop would never reach a fixpoint.
    fn shift(self, d: i64) -> Origin {
        if self.is_none() {
            return self;
        }
        let off = match self.off {
            Off::Known(x) => x.checked_add(d).map_or(Off::Unknown, Off::Known),
            Off::Unknown => Off::Unknown,
        };
        Origin { off, ..self }
    }

    fn unknown(self) -> Origin {
        if self.is_none() {
            return self;
        }
        Origin { off: Off::Unknown, ..self }
    }

    pub fn each_root(self) -> impl Iterator<Item = u8> {
        (0..64u8).filter(move |&k| self.roots & (1 << k) != 0)
    }
}

/// What a pointer can point into.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Root {
    /// Entry parameter `k`.
    Param(u8),
    /// The function's stack frame (`Function::locals[0]`, from `frame::promote`).
    Frame,
    /// A constant address: a global in the binary's data.
    Global(u64),
    /// The result of the allocating call that defines this value.
    Alloc(ValueId),
}

/// What a callee does with one argument (its summary, from its own analysis, or
/// from a table for C library functions).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Pass {
    /// Uses it as an integer only, and doesn't keep it.
    Ignore,
    /// Takes it as a slice (`&[u8]` / `&mut [u8]`, `Option<..>` if nullable): the
    /// caller reborrows the root the argument points into.
    Borrow { mutbl: bool, nullable: bool },
    /// Frees it (`free`, `operator delete`): a move of an owned allocation.
    Free,
    /// Reads (or writes) through it during the call only, like `memcpy`. Safe
    /// mode emits such a call as slice operations when every pointer argument
    /// has a safe root; otherwise every one of them escapes.
    Access { write: bool },
    /// Anything else: it may keep it, or access it in ways we can't see.
    Escape,
}

/// A callee's summary, by argument position (SysV registers, then stack).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Callee {
    /// Missing arguments `Escape`.
    pub args: Vec<Pass>,
    /// Returns a new allocation of `args[a]` (times `args[b]`) bytes: `malloc(n)`,
    /// `calloc(a, b)`.
    pub alloc: Option<(u8, Option<u8>)>,
    /// The result points into these arguments (bit = position).
    pub ret_from: u64,
}

/// What the analysis needs from the rest of the program.
pub struct Ctx<'a> {
    /// The callee at a call site, or `None` when unknown (everything escapes).
    pub callee: &'a dyn Fn(Site) -> Option<Callee>,
    /// Entry parameters that must stay integers (by entry parameter index),
    /// because some caller can't pass a slice for them.
    pub demoted: &'a dyn Fn(usize) -> bool,
    /// A global at this address can be read through a slice (a read-only static).
    pub global_ok: &'a dyn Fn(u64) -> bool,
}

impl Ctx<'_> {
    /// No calls are known, no argument is demoted, no global is a static.
    pub const ISOLATED: Ctx<'static> = Ctx { callee: &|_| None, demoted: &|_| false, global_ok: &|_| false };
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FactKind {
    Read,
    Write,
    /// The pointer itself leaves what we can track: stored to memory we don't
    /// follow, passed to a callee that keeps it, or fed to an operation that isn't
    /// pointer arithmetic.
    Escape,
    /// Returned to the caller (a reborrow if the root turns out to be a pointer).
    Return,
    /// Compared with zero: the argument is `Option<&T>` / `Option<&mut T>`.
    NullCheck,
    /// Passed to a callee that takes a slice.
    Borrow { mutbl: bool },
    /// Passed to `free`.
    Free,
    /// Stored into the frame or an allocation (root `.0`), where loads find it.
    Stash(u8),
    /// The contents of this root (a frame or an allocation) are copied into root
    /// `.0` by a `memcpy`.
    CopyTo(u8),
    /// The contents of this root leave what we can track (copied elsewhere).
    Spill,
}

/// One observation about root `root`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Fact {
    pub root: u8,
    pub off: Off,
    pub kind: FactKind,
    /// The instruction, or `None` for a terminator.
    pub at: Option<ValueId>,
    /// Position in the CFG (`loans::Point`).
    pub point: u32,
    /// The call, for `Borrow` and `Free`, and the argument position.
    pub site: Option<(Site, u8)>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Class {
    /// Never dereferenced: emit as an integer.
    NotPointer,
    /// Only read through: `&T`.
    Shared,
    /// Written through: `&mut T`.
    Mut,
    /// Escapes where we can't see its lifetime: keep `*const T` / `*mut T`.
    Raw,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamBorrow {
    /// The entry block parameter and the x86 register it arrives in.
    pub value: ValueId,
    pub reg: u8,
    pub class: Class,
    /// A value derived from it is returned. With a pointer class this means the
    /// return borrows from this argument (`fn(&'a T) -> &'a U`).
    pub returned: bool,
    /// Compared against null: emit as `Option<&T>` / `Option<&mut T>`.
    pub nullable: bool,
    /// Constant offsets read or written through it, sorted; `true` if written.
    /// These are the field layout evidence for the pointee type.
    pub fields: Vec<(i64, bool)>,
    /// Accessed at a non-constant offset: evidence for a slice or array.
    pub indexed: bool,
}

#[derive(Clone)]
pub struct Analysis {
    /// Origin of each value, indexed by `ValueId`.
    pub origin: Vec<Origin>,
    pub facts: Vec<Fact>,
    /// One entry per entry-block parameter, in parameter order.
    pub params: Vec<ParamBorrow>,
    /// What each root index stands for (at most 63; see `OTHER`).
    pub roots: Vec<Root>,
    /// Accesses through the root can be bounds-checked, and it can be reborrowed.
    pub safe: Vec<bool>,
    /// Something writes through the root (or borrows it mutably).
    pub written: Vec<bool>,
    /// A pointer into the root leaves what the analysis can follow.
    pub escaped: Vec<bool>,
    /// Why each root that isn't safe isn't (empty for safe roots).
    pub why: Vec<&'static str>,
    /// Calls where a callee takes a slice that this function can't pass, by
    /// argument position: the callee must take that argument as an integer.
    pub unprovable: Vec<(Site, u8)>,
    /// Calls to functions like `memcpy` that must stay raw calls, because some
    /// pointer argument has no safe root.
    pub raw_calls: Vec<Site>,
    /// Mutable borrows downgraded because they conflict with another borrow of
    /// the same object at the same call (`loans.rs`).
    pub loan_errors: usize,
    /// Allocations kept raw because they may be used after they are freed.
    pub move_errors: usize,
}

/// x86 register number of RSP. Facts on an RSP root describe stack slots.
pub const RSP: u8 = 4;

/// Frames up to this size can be safe (an array on the stack, or a `Vec` above
/// 4 KiB). The 64 KiB fallback frame for untrackable stack use stays raw.
pub const MAX_SAFE_FRAME: u32 = crate::frame::FALLBACK_FRAME - 1;

/// The analysis with nothing known about calls (every argument escapes).
pub fn analyze(f: &Function) -> Analysis {
    analyze_with(f, &Ctx::ISOLATED)
}

pub fn analyze_with(f: &Function, ctx: &Ctx) -> Analysis {
    let cfg = Cfg::new(f);
    let callees = callees(f, &cfg, ctx);
    let (roots, starts) = find_roots(f, &cfg, &callees);
    let origin = origins(f, &cfg, &roots, &starts, &callees);
    let points = Points::new(f);
    let (facts, unprovable, bad_calls) = facts(f, &cfg, &origin, &roots, &callees, &points);
    let mut a = Analysis {
        origin,
        facts,
        params: Vec::new(),
        roots,
        safe: Vec::new(),
        written: Vec::new(),
        escaped: Vec::new(),
        why: Vec::new(),
        unprovable,
        raw_calls: bad_calls,
        loan_errors: 0,
        move_errors: 0,
    };
    classify(f, &cfg, ctx, &points, &mut a);
    a
}

fn konst(f: &Function, v: ValueId) -> Option<i64> {
    match f.insts[v].kind {
        InstKind::Const(c) => Some(f.consts[c.index()] as u64 as i64),
        _ => None,
    }
}

/// Edge argument `k` on every edge from `p` into `b`.
pub(crate) fn incoming(f: &Function, p: BlockId, b: BlockId, k: usize, mut cb: impl FnMut(ValueId)) {
    match f.blocks[p].term {
        Terminator::Jump { to, args } if to == b => cb(f.value_pool[args.start as usize + k]),
        Terminator::Branch { t, f: e, args, .. } => {
            if t == b {
                cb(f.value_pool[args.start as usize + k]);
            }
            if e == b {
                cb(f.value_pool[args.start as usize + f.blocks[t].params.len as usize + k]);
            }
        }
        _ => {}
    }
}

/// The summary of every call site's callee, by the call's value (`Call`) or block
/// (`TailCall`).
struct Callees {
    call: HashMap<ValueId, Callee>,
    tail: HashMap<BlockId, Callee>,
}

impl Callees {
    fn get(&self, s: Site) -> Option<&Callee> {
        match s {
            Site::Call(v) => self.call.get(&v),
            Site::Tail(b) => self.tail.get(&b),
        }
    }
}

fn callees(f: &Function, cfg: &Cfg, ctx: &Ctx) -> Callees {
    let mut c = Callees { call: HashMap::new(), tail: HashMap::new() };
    for &b in &cfg.rpo {
        for &id in f.blocks[b].insts.get(&f.value_pool) {
            if matches!(f.insts[id].kind, InstKind::Call { .. }) {
                if let Some(s) = (ctx.callee)(Site::Call(id)) {
                    c.call.insert(id, s);
                }
            }
        }
        if matches!(f.blocks[b].term, Terminator::TailCall { .. }) {
            if let Some(s) = (ctx.callee)(Site::Tail(b)) {
                c.tail.insert(b, s);
            }
        }
    }
    c
}

/// Every root, and the root each value starts (if it starts one).
fn find_roots(f: &Function, cfg: &Cfg, callees: &Callees) -> (Vec<Root>, HashMap<ValueId, u8>) {
    let mut roots = Vec::new();
    let mut starts = HashMap::new();
    let add = |roots: &mut Vec<Root>, r: Root| -> u8 {
        if let Some(i) = roots.iter().position(|&x| x == r) {
            return i as u8;
        }
        if roots.len() >= OTHER as usize {
            return OTHER;
        }
        roots.push(r);
        (roots.len() - 1) as u8
    };
    for k in 0..f.blocks[f.entry].params.len as usize {
        add(&mut roots, Root::Param(k.min(255) as u8));
    }
    for &b in &cfg.rpo {
        for &id in f.blocks[b].insts.get(&f.value_pool) {
            let r = match f.insts[id].kind {
                InstKind::AddrOfLocal(_) => Root::Frame,
                InstKind::IntToPtr(v) => match konst(f, v) {
                    Some(c) => Root::Global(c as u64),
                    None => continue,
                },
                // an allocation whose size arguments are there
                InstKind::Call { args, .. }
                    if callees.call.get(&id).and_then(|c| c.alloc).is_some_and(|(a, b)| {
                        (a.max(b.unwrap_or(0)) as u32) < args.len
                    }) =>
                {
                    Root::Alloc(id)
                }
                _ => continue,
            };
            starts.insert(id, add(&mut roots, r));
        }
    }
    (roots, starts)
}

fn is_container(roots: &[Root], r: u8) -> bool {
    matches!(roots.get(r as usize), Some(Root::Frame | Root::Alloc(_)))
}

/// Points-to: what is stored at `(container root, offset)`; `None` is an unknown
/// offset, which any load from that root may see.
type Mem = HashMap<(u8, Option<i64>), Origin>;

fn mem_key(r: u8, off: Off) -> (u8, Option<i64>) {
    match off {
        Off::Known(x) => (r, Some(x)),
        Off::Unknown => (r, None),
    }
}

fn mem_load(mem: &Mem, r: u8, off: Off) -> Origin {
    match off {
        Off::Known(x) => {
            let a = mem.get(&(r, Some(x))).copied().unwrap_or(Origin::NONE);
            a.join(mem.get(&(r, None)).copied().unwrap_or(Origin::NONE))
        }
        Off::Unknown => mem.iter().filter(|((c, _), _)| *c == r).fold(Origin::NONE, |a, (_, &o)| a.join(o)),
    }
}

fn mem_join(mem: &mut Mem, key: (u8, Option<i64>), o: Origin) -> bool {
    if o.is_none() {
        return false;
    }
    let e = mem.entry(key).or_insert(Origin::NONE);
    let new = e.join(o);
    let changed = new != *e;
    *e = new;
    changed
}

/// What an instruction's result derives from, given its operands' origins.
fn transfer(f: &Function, id: ValueId, o: &[Origin], starts: &HashMap<ValueId, u8>, roots: &[Root], mem: &Mem, callees: &Callees) -> Origin {
    use InstKind::*;
    if let Some(&r) = starts.get(&id) {
        return Origin::root(r);
    }
    let of = |v: ValueId| o[v.index()];
    match f.insts[id].kind {
        PtrOffset { base, index, scale, disp } => {
            let b = of(base).shift(disp as i64);
            match index {
                None => b,
                // With scale 1 either register may be the pointer.
                Some(i) if scale == 1 => sum(roots, of(base), of(i)),
                Some(_) => b.unknown(),
            }
        }
        Bin { op: BinOp::Add, lhs, rhs } => match (of(lhs).is_none(), of(rhs).is_none()) {
            (false, true) => konst(f, rhs).map_or(of(lhs).unknown(), |c| of(lhs).shift(c)),
            (true, false) => konst(f, lhs).map_or(of(rhs).unknown(), |c| of(rhs).shift(c)),
            (true, true) => Origin::NONE,
            (false, false) => sum(roots, of(lhs), of(rhs)),
        },
        Bin { op: BinOp::Sub, lhs, rhs } => match konst(f, rhs) {
            Some(c) if of(rhs).is_none() => of(lhs).shift(c.wrapping_neg()),
            // `p - q` is usually a length; treated as derived from `p` (sound, and
            // monotone), and `facts` doesn't count `q` as escaping.
            _ => of(lhs).unknown(),
        },
        Select { t, f: e, .. } => of(t).join(of(e)),
        Cast { kind: CastKind::Bitcast, v } | IntToPtr(v) | PtrToInt(v) => of(v),
        // A pointer loaded from the frame or an allocation: whatever was stored there.
        Load { ptr, .. } => match of(ptr).single() {
            Some(r) if is_container(roots, r) => mem_load(mem, r, of(ptr).off),
            _ => Origin::NONE,
        },
        Call { args, .. } => match callees.call.get(&id) {
            Some(c) if c.ret_from != 0 => {
                let args = args.get(&f.value_pool);
                (0..args.len().min(64))
                    .filter(|&k| c.ret_from & (1 << k) != 0)
                    .fold(Origin::NONE, |a, k| a.join(of(args[k])))
                    .unknown()
            }
            _ => Origin::NONE,
        },
        _ => Origin::NONE,
    }
}

/// Roots that are certainly pointers: everything but arguments, which may be
/// integers.
fn pointer_roots(roots: &[Root]) -> u64 {
    let mut m = 1 << OTHER;
    for (r, root) in roots.iter().enumerate() {
        if !matches!(root, Root::Param(_)) {
            m |= 1 << r;
        }
    }
    m
}

/// `a + b` where both may be derived. If one side certainly points into an
/// object (the frame, a global, an allocation) and the other only comes from
/// arguments, the argument is an index. Otherwise keep both (over-approximate),
/// so the analysis stays monotone and reaches a fixpoint.
fn sum(roots: &[Root], a: Origin, b: Origin) -> Origin {
    let ptr = pointer_roots(roots);
    match (a.roots & ptr != 0, b.roots & ptr != 0) {
        (true, false) => a.unknown(),
        (false, true) => b.unknown(),
        _ => a.join(b).unknown(),
    }
}

/// An operand of pointer arithmetic that `sum` treated as an index.
fn is_index(roots: &[Root], o: &[Origin], result: ValueId, v: ValueId) -> bool {
    let ptr = pointer_roots(roots);
    o[result.index()].roots & ptr != 0 && o[v.index()].roots & ptr == 0
}

fn origins(f: &Function, cfg: &Cfg, roots: &[Root], starts: &HashMap<ValueId, u8>, callees: &Callees) -> Vec<Origin> {
    let mut o = vec![Origin::NONE; f.insts.len()];
    let mut mem: Mem = HashMap::new();
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &cfg.rpo {
            let blk = &f.blocks[b];
            for (k, &p) in blk.params.get(&f.value_pool).iter().enumerate() {
                let mut new = if b == f.entry { Origin::root(k.min(OTHER as usize) as u8) } else { Origin::NONE };
                for &pred in cfg.preds(b) {
                    incoming(f, pred, b, k, |a| new = new.join(o[a.index()]));
                }
                if new != o[p.index()] {
                    o[p.index()] = new;
                    changed = true;
                }
            }
            for &id in blk.insts.get(&f.value_pool) {
                match f.insts[id].kind {
                    InstKind::Store { ptr, val, .. } => {
                        if let Some(r) = o[ptr.index()].single().filter(|&r| is_container(roots, r)) {
                            changed |= mem_join(&mut mem, mem_key(r, o[ptr.index()].off), o[val.index()]);
                        }
                    }
                    InstKind::MemFill { dst, val, .. } => {
                        if let Some(r) = o[dst.index()].single().filter(|&r| is_container(roots, r)) {
                            changed |= mem_join(&mut mem, (r, None), o[val.index()]);
                        }
                    }
                    InstKind::MemCopy { dst, src, .. } => {
                        let (d, s) = (o[dst.index()].single(), o[src.index()].single());
                        if let (Some(d), Some(s)) = (d, s) {
                            if is_container(roots, d) && is_container(roots, s) {
                                let all = mem_load(&mem, s, Off::Unknown);
                                changed |= mem_join(&mut mem, (d, None), all);
                            }
                        }
                    }
                    _ => {}
                }
                let new = transfer(f, id, &o, starts, roots, &mem, callees);
                if new != o[id.index()] {
                    o[id.index()] = new;
                    changed = true;
                }
            }
        }
    }
    o
}

/// Program points for `loans.rs`: one per instruction, then one for the terminator.
struct Points {
    start: Vec<u32>,
}

impl Points {
    fn new(f: &Function) -> Points {
        let mut start = Vec::with_capacity(f.blocks.len());
        let mut n = 0u32;
        for (_, blk) in f.blocks.iter() {
            start.push(n);
            n += blk.insts.len + 1;
        }
        Points { start }
    }
    fn at(&self, b: BlockId, i: usize) -> u32 {
        self.start[b.index()] + i as u32
    }
    fn term(&self, f: &Function, b: BlockId) -> u32 {
        self.start[b.index()] + f.blocks[b].insts.len
    }
}

type FactsOut = (Vec<Fact>, Vec<(Site, u8)>, Vec<Site>);

fn facts(f: &Function, cfg: &Cfg, o: &[Origin], roots: &[Root], callees: &Callees, points: &Points) -> FactsOut {
    use InstKind::*;
    let mut out = Vec::new();
    let mut unprovable = Vec::new();
    let mut bad_calls = Vec::new();
    let mut opaque = false;
    for &b in &cfg.rpo {
        let blk = &f.blocks[b];
        let mut fx = FactSink { o, out: &mut out, unprovable: &mut unprovable, bad_calls: &mut bad_calls, point: 0, at: None };
        for (i, &id) in blk.insts.get(&f.value_pool).iter().enumerate() {
            fx.point = points.at(b, i);
            fx.at = Some(id);
            match f.insts[id].kind {
                Load { ptr, .. } => fx.access(ptr, FactKind::Read),
                Store { ptr, val, .. } => {
                    fx.access(ptr, FactKind::Write);
                    fx.stash(roots, val, ptr);
                }
                MemFill { dst, val, count } => {
                    fx.access(dst, FactKind::Write);
                    fx.stash(roots, val, dst);
                    fx.add(count, FactKind::Escape);
                }
                MemCopy { dst, src, len } => {
                    fx.access(dst, FactKind::Write);
                    fx.access(src, FactKind::Read);
                    fx.add(len, FactKind::Escape);
                    if let Some(s) = o[src.index()].single().filter(|&s| is_container(roots, s)) {
                        let kind = match o[dst.index()].single().filter(|&d| is_container(roots, d)) {
                            Some(d) => FactKind::CopyTo(d),
                            None => FactKind::Spill,
                        };
                        fx.out.push(Fact { root: s, off: Off::Unknown, kind, at: fx.at, point: fx.point, site: None });
                    }
                }
                Call { callee, args } => {
                    let site = Site::Call(id);
                    fx.call(f, site, callee, args, callees.get(site));
                }
                // Pointer arithmetic the origin pass followed: no fact. If the
                // result lost track of a rooted operand, that operand escapes.
                k @ (PtrOffset { .. } | Bin { op: BinOp::Add | BinOp::Sub, .. } | Select { .. }
                | Cast { kind: CastKind::Bitcast, .. } | IntToPtr(_) | PtrToInt(_)) => {
                    let kept = o[id.index()].roots;
                    if !is_pointer_difference(k, o) {
                        crate::verify::for_each_operand(k, f, |v| {
                            if o[v.index()].roots & !kept != 0 && !is_index(roots, o, id, v) {
                                fx.add(v, FactKind::Escape);
                            }
                        });
                    }
                }
                // Comparisons (null checks, `p < end`) don't let the pointer out.
                Cmp { cc: Cond::Eq | Cond::Ne, lhs, rhs } => {
                    if konst(f, rhs) == Some(0) && o[lhs.index()].off == Off::Known(0) {
                        fx.add(lhs, FactKind::NullCheck);
                    }
                }
                Cmp { .. } => {}
                // registers after a call, and at a return: bookkeeping, not uses
                CallOut { .. } | Exit { .. } => {}
                Opaque { .. } => opaque = true,
                k => crate::verify::for_each_operand(k, f, |v| fx.add(v, FactKind::Escape)),
            }
        }
        fx.point = points.term(f, b);
        fx.at = None;
        match blk.term {
            Terminator::Return(Some(v)) => fx.add(v, FactKind::Return),
            Terminator::TailCall { callee, args } => {
                let site = Site::Tail(b);
                let c = callees.get(site);
                fx.call(f, site, callee, args, c);
                // what the callee returns is our return value
                if let Some(c) = c {
                    for (k, &a) in args.get(&f.value_pool).iter().enumerate().take(64) {
                        if c.ret_from & (1 << k) != 0 {
                            fx.add(a, FactKind::Return);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    // Unliftable code may do anything with any root.
    if opaque {
        for root in 0..roots.len() as u8 {
            out.push(Fact { root, off: Off::Unknown, kind: FactKind::Escape, at: None, point: 0, site: None });
        }
    }
    (out, unprovable, bad_calls)
}

struct FactSink<'a> {
    o: &'a [Origin],
    out: &'a mut Vec<Fact>,
    unprovable: &'a mut Vec<(Site, u8)>,
    bad_calls: &'a mut Vec<Site>,
    point: u32,
    at: Option<ValueId>,
}

impl FactSink<'_> {
    fn add(&mut self, v: ValueId, kind: FactKind) {
        self.add_site(v, kind, None);
    }

    fn add_site(&mut self, v: ValueId, kind: FactKind, site: Option<(Site, u8)>) {
        let og = self.o[v.index()];
        for root in og.each_root() {
            self.out.push(Fact { root, off: og.off, kind, at: self.at, point: self.point, site });
        }
    }

    /// A load or store through `ptr`. Safe mode can only bounds-check it against
    /// one root, so a pointer into several roots makes them all escape.
    fn access(&mut self, ptr: ValueId, kind: FactKind) {
        let og = self.o[ptr.index()];
        self.add(ptr, if og.roots.count_ones() > 1 { FactKind::Escape } else { kind });
    }

    /// `val` is stored through `ptr`: tracked if `ptr` is in the frame or an
    /// allocation, an escape anywhere else.
    fn stash(&mut self, roots: &[Root], val: ValueId, ptr: ValueId) {
        match self.o[ptr.index()].single().filter(|&r| is_container(roots, r)) {
            Some(c) => self.add(val, FactKind::Stash(c)),
            None => self.add(val, FactKind::Escape),
        }
    }

    fn call(&mut self, f: &Function, site: Site, callee: ValueId, args: ListRef, c: Option<&Callee>) {
        self.add(callee, FactKind::Escape);
        for (k, &a) in args.get(&f.value_pool).iter().enumerate() {
            let pass = c.and_then(|c| c.args.get(k)).copied().unwrap_or(Pass::Escape);
            let og = self.o[a.index()];
            let k = k.min(255) as u8;
            match pass {
                Pass::Ignore => {}
                Pass::Escape => self.add(a, FactKind::Escape),
                Pass::Access { write } if og.single().is_some() => {
                    self.add_site(a, if write { FactKind::Write } else { FactKind::Read }, Some((site, k)))
                }
                Pass::Access { .. } => {
                    self.add(a, FactKind::Escape);
                    self.bad_calls.push(site);
                }
                Pass::Free if og.single().is_some() => self.add_site(a, FactKind::Free, Some((site, k))),
                Pass::Free => self.add(a, FactKind::Escape),
                // `None` for a null constant
                Pass::Borrow { nullable: true, .. } if og.is_none() && konst(f, a) == Some(0) => {}
                Pass::Borrow { mutbl, .. } if og.single().is_some() => {
                    self.add_site(a, FactKind::Borrow { mutbl }, Some((site, k)))
                }
                Pass::Borrow { .. } => {
                    self.add(a, FactKind::Escape);
                    self.unprovable.push((site, k));
                }
            }
        }
    }
}

/// `p - q` of two derived pointers is an integer length (`offset_from`), not an escape.
fn is_pointer_difference(k: InstKind, o: &[Origin]) -> bool {
    matches!(k, InstKind::Bin { op: BinOp::Sub, lhs, rhs } if !o[lhs.index()].is_none() && !o[rhs.index()].is_none())
}

/// Per-root summary of the facts.
#[derive(Default, Clone, Copy)]
struct Uses {
    read: bool,
    write: bool,
    escape: bool,
    returned: bool,
    nullable: bool,
    borrowed: bool,
    borrowed_mut: bool,
    freed: bool,
    /// Freed at an offset other than 0.
    bad_free: bool,
    /// Accessed at a negative constant offset (before the start of a slice).
    negative: bool,
    spill: bool,
}

fn classify(f: &Function, cfg: &Cfg, ctx: &Ctx, points: &Points, a: &mut Analysis) {
    let n = a.roots.len();
    let entry = f.blocks[f.entry].params.get(&f.value_pool);
    let reg = |k: usize| match f.insts[entry[k]].kind {
        InstKind::BlockParam(r) => r,
        _ => u8::MAX,
    };
    let frame_ok = f.locals.iter().next().is_some_and(|(_, l)| l.size <= MAX_SAFE_FRAME);
    let mut u = vec![Uses::default(); n];
    for x in a.facts.iter().filter(|x| (x.root as usize) < n) {
        let r = &mut u[x.root as usize];
        match x.kind {
            FactKind::Read => r.read = true,
            FactKind::Write => r.write = true,
            FactKind::Escape => r.escape = true,
            FactKind::Return => r.returned = true,
            FactKind::NullCheck => r.nullable = true,
            FactKind::Borrow { mutbl } => {
                r.borrowed = true;
                r.borrowed_mut |= mutbl;
            }
            FactKind::Free => {
                r.freed = true;
                r.bad_free |= x.off != Off::Known(0);
            }
            FactKind::Spill => r.spill = true,
            FactKind::Stash(_) | FactKind::CopyTo(_) => {}
        }
        if matches!(x.kind, FactKind::Read | FactKind::Write | FactKind::Borrow { .. }) {
            r.negative |= matches!(x.off, Off::Known(o) if o < 0);
        }
    }
    let in_loop: Vec<bool> = a.roots.iter().map(|r| matches!(r, Root::Alloc(id) if in_cycle(f, cfg, *id))).collect();

    // Downgrade to a fixpoint: `safe` only goes from true to false, and the sets
    // of escapes and unprovable borrows only grow.
    let mut safe = vec![true; n];
    let mut move_error = vec![false; n];
    let mut bad_loans: Vec<(Site, u8)> = Vec::new();
    let mut escaped;
    let mut reasons;
    let mut changed_calls = false;
    loop {
        // a container's contents escape if it isn't safe itself, is copied out,
        // or is lent to a callee (which could read the pointers in it)
        let mut contents_escape: Vec<bool> = (0..n).map(|c| !safe[c] || u[c].spill || u[c].borrowed).collect();
        loop {
            let mut more = false;
            for x in &a.facts {
                if let FactKind::CopyTo(d) = x.kind {
                    let (s, d) = (x.root as usize, d as usize);
                    if s < n && !contents_escape[s] && (d >= n || contents_escape[d]) {
                        contents_escape[s] = true;
                        more = true;
                    }
                }
            }
            if !more {
                break;
            }
        }
        let mut escape: Vec<bool> = u.iter().map(|x| x.escape).collect();
        for x in &a.facts {
            let r = x.root as usize;
            if r >= n {
                continue;
            }
            match x.kind {
                FactKind::Stash(c) if c as usize >= n || contents_escape[c as usize] => escape[r] = true,
                FactKind::Borrow { .. } | FactKind::Free if bad_loans.contains(&x.site.unwrap()) => escape[r] = true,
                FactKind::Read | FactKind::Write if x.site.is_some_and(|(s, _)| a.raw_calls.contains(&s)) => escape[r] = true,
                _ => {}
            }
        }
        let why: Vec<&'static str> = (0..n)
            .map(|r| {
                let x = u[r];
                let common = if escape[r] { "escapes" } else { "" };
                let neg = if x.negative { "accessed before its start" } else { "" };
                let first = |l: &[&'static str]| l.iter().copied().find(|w| !w.is_empty()).unwrap_or("");
                match a.roots[r] {
                    Root::Param(k) => {
                        let k = k as usize;
                        first(&[
                            if reg(k) == RSP { "the stack pointer" } else { "" },
                            if (ctx.demoted)(k) { "a caller passes an integer" } else { "" },
                            common,
                            neg,
                            if x.freed { "freed" } else { "" },
                            if !(x.read || x.write || x.borrowed) { "not dereferenced" } else { "" },
                        ])
                    }
                    Root::Frame => first(&[
                        if frame_ok { "" } else { "too large for an array" },
                        common,
                        neg,
                        if x.returned { "returned" } else { "" },
                        if x.freed { "freed" } else { "" },
                    ]),
                    Root::Global(c) => first(&[
                        if (ctx.global_ok)(c) { "" } else { "not a read-only static" },
                        common,
                        if x.write || x.borrowed_mut || x.freed { "written" } else { "" },
                    ]),
                    Root::Alloc(_) => first(&[
                        common,
                        neg,
                        if x.returned { "returned" } else { "" },
                        if in_loop[r] { "allocated in a loop" } else { "" },
                        if x.bad_free { "freed at an offset" } else { "" },
                        if move_error[r] { "used after it may be freed" } else { "" },
                    ]),
                }
            })
            .collect();
        let new: Vec<bool> = why.iter().map(|w| w.is_empty()).collect();


        // Borrows the caller can't provide: from a root that isn't safe, a mutable
        // borrow of a read-only global, or a nullable argument (we can't tell
        // whether it is null at the call).
        let mut bad = Vec::new();
        for x in &a.facts {
            let FactKind::Borrow { .. } = x.kind else { continue };
            let r = x.root as usize;
            let ok = r < n && new[r] && !(matches!(a.roots[r], Root::Param(_)) && u[r].nullable);
            if !ok {
                bad.push(x.site.unwrap());
            }
        }
        // A call like `memcpy` is inlined only if all its pointers have safe roots
        // (and the ones it writes can be written).
        for x in &a.facts {
            let (FactKind::Read | FactKind::Write, Some((site, _))) = (x.kind, x.site) else { continue };
            let r = x.root as usize;
            let writable = x.kind == FactKind::Read || !matches!(a.roots.get(r), Some(Root::Global(_)));
            if (r >= n || !new[r] || !writable) && !a.raw_calls.contains(&site) {
                a.raw_calls.push(site);
                changed_calls = true;
            }
        }
        // Loans and moves (`loans.rs`).
        let (errs, merrs) = check_loans(f, cfg, points, a, &new);
        bad.extend(errs);
        let mut changed = new != safe || std::mem::take(&mut changed_calls);
        for b in bad {
            if !bad_loans.contains(&b) {
                bad_loans.push(b);
                changed = true;
            }
        }
        for r in merrs {
            if !move_error[r] {
                move_error[r] = true;
                changed = true;
            }
        }
        safe = new;
        escaped = escape;
        reasons = why;
        if !changed {
            break;
        }
    }
    a.loan_errors = bad_loans.iter().filter(|s| !a.unprovable.contains(s)).count();
    a.move_errors = move_error.iter().filter(|&&m| m).count();
    a.unprovable.extend(bad_loans);
    a.unprovable.sort_unstable_by_key(|&(s, k)| (site_key(s), k));
    a.unprovable.dedup();
    a.written = u.iter().map(|x| x.write || x.borrowed_mut).collect();
    a.safe = safe;
    a.escaped = escaped;
    a.why = reasons;

    a.params = entry
        .iter()
        .enumerate()
        .map(|(k, &value)| {
            let r = k.min(OTHER as usize);
            let x = u.get(r).copied().unwrap_or_default();
            let mut fields: Vec<(i64, bool)> = Vec::new();
            let mut indexed = false;
            for y in a.facts.iter().filter(|y| y.root as usize == r) {
                if matches!(y.kind, FactKind::Read | FactKind::Write) {
                    match y.off {
                        Off::Known(o) => fields.push((o, y.kind == FactKind::Write)),
                        Off::Unknown => indexed = true,
                    }
                }
            }
            fields.sort_unstable();
            // Merge duplicates, keeping "written" if any access wrote.
            fields.dedup_by(|b, a| {
                if a.0 == b.0 {
                    a.1 |= b.1;
                    true
                } else {
                    false
                }
            });
            let safe = a.safe.get(r).copied().unwrap_or(false);
            let class = if !x.read && !x.write && !x.borrowed {
                Class::NotPointer
            } else if !safe {
                Class::Raw
            } else if a.written[r] {
                Class::Mut
            } else {
                Class::Shared
            };
            ParamBorrow { value, reg: reg(k), class, returned: x.returned, nullable: x.nullable, fields, indexed }
        })
        .collect();
}

fn site_key(s: Site) -> (u8, usize) {
    match s {
        Site::Call(v) => (0, v.index()),
        Site::Tail(b) => (1, b.index()),
    }
}

/// Is the block defining `v` on a cycle?
fn in_cycle(f: &Function, cfg: &Cfg, v: ValueId) -> bool {
    let Some(home) = cfg.rpo.iter().copied().find(|&b| f.blocks[b].insts.get(&f.value_pool).contains(&v)) else {
        return true;
    };
    let mut seen = vec![false; f.blocks.len()];
    let mut work: Vec<BlockId> = f.blocks[home].term.successors(&f.value_pool).collect();
    while let Some(b) = work.pop() {
        if b == home {
            return true;
        }
        if std::mem::replace(&mut seen[b.index()], true) {
            continue;
        }
        work.extend(f.blocks[b].term.successors(&f.value_pool));
    }
    false
}

/// Run the loan and move rules on the borrows and frees of safe roots. Returns
/// the mutable borrows that conflict with another borrow at the same call, and
/// the allocations used after they may have been freed.
fn check_loans(f: &Function, cfg: &Cfg, points: &Points, a: &Analysis, safe: &[bool]) -> (Vec<(Site, u8)>, Vec<usize>) {
    let n = a.roots.len();
    let mut facts = loans::Facts::default();
    // Reborrows at a call: one loan (and origin) each, live during the call.
    let borrows: Vec<&Fact> = a
        .facts
        .iter()
        .filter(|x| matches!(x.kind, FactKind::Borrow { .. }) && (x.root as usize) < n && safe[x.root as usize])
        .collect();
    let has_moves = a.facts.iter().any(|x| x.kind == FactKind::Free && matches!(a.roots.get(x.root as usize), Some(Root::Alloc(_))));
    if borrows.len() < 2 && !has_moves {
        return (Vec::new(), Vec::new());
    }
    for &b in &cfg.rpo {
        let len = f.blocks[b].insts.len as usize;
        for i in 0..len {
            facts.cfg_edge.push((points.at(b, i), points.at(b, i + 1)));
        }
        for s in f.blocks[b].term.successors(&f.value_pool) {
            facts.cfg_edge.push((points.term(f, b), points.at(s, 0)));
        }
    }
    for (l, x) in borrows.iter().enumerate() {
        let l = l as u32;
        facts.loan_issued_at.push((l, l, x.point));
        facts.origin_live_at.push((l, x.point));
    }
    for (i, x) in borrows.iter().enumerate() {
        for (j, y) in borrows.iter().enumerate() {
            let (FactKind::Borrow { mutbl: mx }, FactKind::Borrow { mutbl: my }) = (x.kind, y.kind) else { continue };
            // Two borrows of the same root at one call conflict unless both are
            // shared or they start at different known offsets (`split_at_mut`).
            let disjoint = matches!((x.off, y.off), (Off::Known(p), Off::Known(q)) if p != q);
            if i != j && x.point == y.point && x.root == y.root && (mx || my) && !disjoint {
                facts.invalidates.push((x.point, j as u32));
            }
        }
    }
    // Moves of owned allocations: freed (moved out), then used.
    for x in &a.facts {
        let r = x.root as usize;
        if r >= n || !matches!(a.roots[r], Root::Alloc(_)) {
            continue;
        }
        if x.kind == FactKind::Free {
            facts.moved_at.push((r as u32, x.point));
        }
        facts.accessed_at.push((r as u32, x.point));
    }
    for (r, root) in a.roots.iter().enumerate() {
        if let Root::Alloc(id) = root {
            if let Some(p) = point_of(f, points, *id) {
                facts.assigned_at.push((r as u32, p));
            }
        }
    }
    let out = loans::solve(&facts);
    let mut bad: Vec<(Site, u8)> = Vec::new();
    for (l, _) in out.errors {
        let x = borrows[l as usize];
        if matches!(x.kind, FactKind::Borrow { mutbl: true }) {
            bad.push(x.site.unwrap());
        }
    }
    let moved: Vec<usize> = out.move_errors.iter().map(|&(r, _)| r as usize).collect();
    (bad, moved)
}

fn point_of(f: &Function, points: &Points, v: ValueId) -> Option<u32> {
    for (b, blk) in f.blocks.iter() {
        if let Some(i) = blk.insts.get(&f.value_pool).iter().position(|&x| x == v) {
            return Some(points.at(b, i));
        }
    }
    None
}

/// A stack slot at a constant offset from the entry RSP (negative: the frame;
/// positive: the return address and stack-passed arguments).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StackSlot {
    pub off: i64,
    pub read: bool,
    pub write: bool,
    /// The slot's address is stored, passed or returned, so it must stay a real
    /// local that is borrowed (`&x` / `&mut x`) rather than become an SSA value.
    pub address_taken: bool,
}

impl Analysis {
    /// The root that every access through `v` can be bounds-checked against.
    pub fn safe_root(&self, v: ValueId) -> Option<u8> {
        let r = self.origin[v.index()].single()?;
        self.safe.get(r as usize).copied().unwrap_or(false).then_some(r)
    }

    /// Stack slots, if the function uses RSP. `None` if it doesn't, or if some
    /// access through RSP has an unknown offset (then no slot can be promoted).
    pub fn stack_slots(&self) -> Option<Vec<StackSlot>> {
        let root = self.params.iter().position(|p| p.reg == RSP)? as u8;
        let mut slots: Vec<StackSlot> = Vec::new();
        for x in self.facts.iter().filter(|x| x.root == root) {
            let Off::Known(off) = x.off else { return None };
            let i = match slots.iter().position(|s| s.off == off) {
                Some(i) => i,
                None => {
                    slots.push(StackSlot { off, read: false, write: false, address_taken: false });
                    slots.len() - 1
                }
            };
            match x.kind {
                FactKind::Read => slots[i].read = true,
                FactKind::Write => slots[i].write = true,
                FactKind::NullCheck => {}
                _ => slots[i].address_taken = true,
            }
        }
        slots.sort_unstable_by_key(|s| s.off);
        Some(slots)
    }
}

