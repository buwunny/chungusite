//! Safe-mode borrow inference, first stage: which pointer arguments can be `&T`,
//! which need `&mut T`, and which must stay raw pointers.
//!
//! It runs on clean SSA (`opt::clean`) in two steps:
//!
//! 1. **Origins.** A forward dataflow over the CFG assigns every value the set of
//!    entry parameters it may be derived from, plus a byte offset when that offset
//!    is a known constant. `PtrOffset`, `Add`/`Sub` and block parameters propagate
//!    origins; block parameters join their incoming edge arguments to a fixpoint.
//! 2. **Facts and classes.** One pass records what happens to derived values:
//!    reads, writes, escapes (stored to memory, passed to a call, or used in a
//!    way the analysis can't follow) and returns. Each argument is then classified
//!    from its facts.
//!
//! An argument counts as a pointer only if something dereferences a value derived
//! from it. Compilers use `lea` and `add` for plain integer arithmetic, so being
//! offset or returned is not evidence on its own.
//!
//! Moves (ownership transfer) can't be decided inside one function: they need call
//! summaries such as "this callee frees its argument". The lifter does not lift
//! `CALL` yet. See docs/ownership.md for that step, and for loans and lifetimes.
use crate::cfg::Cfg;
use crate::ir::*;

/// Byte offset of a derived pointer from its root.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Off {
    Known(i64),
    /// Indexed, advanced in a loop, or merged from different offsets.
    Unknown,
}

/// Which entry parameters a value may point into, and where.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Origin {
    /// Bit `k` set: may be derived from entry parameter `k`.
    pub roots: u32,
    /// Meaningful only when `roots` is non-zero.
    pub off: Off,
}

impl Origin {
    pub const NONE: Origin = Origin { roots: 0, off: Off::Known(0) };

    pub fn is_none(self) -> bool {
        self.roots == 0
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

    fn each_root(self) -> impl Iterator<Item = u8> {
        (0..32u8).filter(move |&k| self.roots & (1 << k) != 0)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FactKind {
    Read,
    Write,
    /// The pointer itself leaves what we can track: stored to memory, passed to a
    /// callee, or fed to an operation that isn't pointer arithmetic.
    Escape,
    /// Returned to the caller (a reborrow if the root turns out to be a pointer).
    Return,
    /// Compared with zero: the argument is `Option<&T>` / `Option<&mut T>`.
    NullCheck,
}

/// One observation about entry parameter `root`, at instruction `at`
/// (`None` for terminators).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Fact {
    pub root: u8,
    pub off: Off,
    pub kind: FactKind,
    pub at: Option<ValueId>,
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

pub struct Analysis {
    /// Origin of each value, indexed by `ValueId`.
    pub origin: Vec<Origin>,
    pub facts: Vec<Fact>,
    /// One entry per entry-block parameter, in parameter order.
    pub params: Vec<ParamBorrow>,
}

/// x86 register number of RSP. Facts on an RSP root describe stack slots.
pub const RSP: u8 = 4;

pub fn analyze(f: &Function) -> Analysis {
    let cfg = Cfg::new(f);
    let origin = origins(f, &cfg);
    let facts = facts(f, &cfg, &origin);
    let params = classify(f, &facts);
    Analysis { origin, facts, params }
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

/// What an instruction's result derives from, given its operands' origins.
fn transfer(f: &Function, k: InstKind, o: &[Origin]) -> Origin {
    use InstKind::*;
    let of = |v: ValueId| o[v.index()];
    match k {
        PtrOffset { base, index, scale, disp } => {
            let b = of(base).shift(disp as i64);
            match index {
                None => b,
                // With scale 1 either register may be the pointer.
                Some(i) if scale == 1 => b.join(of(i)).unknown(),
                Some(_) => b.unknown(),
            }
        }
        Bin { op: BinOp::Add, lhs, rhs } => match (of(lhs).is_none(), of(rhs).is_none()) {
            (false, true) => konst(f, rhs).map_or(of(lhs).unknown(), |c| of(lhs).shift(c)),
            (true, false) => konst(f, lhs).map_or(of(rhs).unknown(), |c| of(rhs).shift(c)),
            (true, true) => Origin::NONE,
            // Both derived: keep both (over-approximate) so the analysis stays
            // monotone and reaches a fixpoint.
            (false, false) => of(lhs).join(of(rhs)).unknown(),
        },
        Bin { op: BinOp::Sub, lhs, rhs } => match konst(f, rhs) {
            Some(c) if of(rhs).is_none() => of(lhs).shift(c.wrapping_neg()),
            // `p - q` is usually a length; treated as derived from `p` (sound, and
            // monotone), and `facts` doesn't count `q` as escaping.
            _ => of(lhs).unknown(),
        },
        Select { t, f: e, .. } => of(t).join(of(e)),
        Cast { kind: CastKind::Bitcast, v } => of(v),
        // Loads produce values we don't track through memory (yet); everything
        // else is not pointer arithmetic.
        _ => Origin::NONE,
    }
}

fn origins(f: &Function, cfg: &Cfg) -> Vec<Origin> {
    let mut o = vec![Origin::NONE; f.insts.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &cfg.rpo {
            let blk = &f.blocks[b];
            for (k, &p) in blk.params.get(&f.value_pool).iter().enumerate() {
                let mut new = if b == f.entry {
                    Origin { roots: 1 << k, off: Off::Known(0) }
                } else {
                    Origin::NONE
                };
                for &pred in cfg.preds(b) {
                    incoming(f, pred, b, k, |a| new = new.join(o[a.index()]));
                }
                if new != o[p.index()] {
                    o[p.index()] = new;
                    changed = true;
                }
            }
            for &id in blk.insts.get(&f.value_pool) {
                let new = transfer(f, f.insts[id].kind, &o);
                if new != o[id.index()] {
                    o[id.index()] = new;
                    changed = true;
                }
            }
        }
    }
    o
}

fn facts(f: &Function, cfg: &Cfg, o: &[Origin]) -> Vec<Fact> {
    use InstKind::*;
    let mut out = Vec::new();
    let mut add = |v: ValueId, kind: FactKind, at: Option<ValueId>| {
        let og = o[v.index()];
        for root in og.each_root() {
            out.push(Fact { root, off: og.off, kind, at });
        }
    };
    let mut opaque = false;
    for &b in &cfg.rpo {
        let blk = &f.blocks[b];
        for &id in blk.insts.get(&f.value_pool) {
            let at = Some(id);
            match f.insts[id].kind {
                Load { ptr, .. } => add(ptr, FactKind::Read, at),
                Store { ptr, val, .. } => {
                    add(ptr, FactKind::Write, at);
                    add(val, FactKind::Escape, at);
                }
                MemCopy { dst, src, len } => {
                    add(dst, FactKind::Write, at);
                    add(src, FactKind::Read, at);
                    add(len, FactKind::Escape, at);
                }
                Call { callee, args } => {
                    add(callee, FactKind::Escape, at);
                    for &a in args.get(&f.value_pool) {
                        add(a, FactKind::Escape, at);
                    }
                }
                // Pointer arithmetic the origin pass followed: no fact. If the
                // result lost track of a rooted operand, that operand escapes.
                k @ (PtrOffset { .. } | Bin { op: BinOp::Add | BinOp::Sub, .. } | Select { .. }
                | Cast { kind: CastKind::Bitcast, .. }) => {
                    let kept = o[id.index()].roots;
                    if !is_pointer_difference(k, o) {
                        crate::verify::for_each_operand(k, f, |v| {
                            if o[v.index()].roots & !kept != 0 {
                                add(v, FactKind::Escape, at);
                            }
                        });
                    }
                }
                // Comparisons (null checks, `p < end`) don't let the pointer out.
                Cmp { cc: Cond::Eq | Cond::Ne, lhs, rhs } => {
                    if konst(f, rhs) == Some(0) && o[lhs.index()].off == Off::Known(0) {
                        add(lhs, FactKind::NullCheck, at);
                    }
                }
                Cmp { .. } => {}
                // registers after a call, and at a return: bookkeeping, not uses
                CallOut { .. } | Exit { .. } => {}
                Opaque { .. } => opaque = true,
                k => crate::verify::for_each_operand(k, f, |v| add(v, FactKind::Escape, at)),
            }
        }
        match blk.term {
            Terminator::Return(Some(v)) => add(v, FactKind::Return, None),
            Terminator::TailCall { callee, args } => {
                add(callee, FactKind::Escape, None);
                for &a in args.get(&f.value_pool) {
                    add(a, FactKind::Escape, None);
                }
            }
            _ => {}
        }
    }
    // Unliftable code may do anything with any argument.
    if opaque {
        let n = f.blocks[f.entry].params.len;
        for root in 0..n as u8 {
            out.push(Fact { root, off: Off::Unknown, kind: FactKind::Escape, at: None });
        }
    }
    out
}

/// `p - q` of two derived pointers is an integer length (`offset_from`), not an escape.
fn is_pointer_difference(k: InstKind, o: &[Origin]) -> bool {
    matches!(k, InstKind::Bin { op: BinOp::Sub, lhs, rhs } if !o[lhs.index()].is_none() && !o[rhs.index()].is_none())
}

fn classify(f: &Function, facts: &[Fact]) -> Vec<ParamBorrow> {
    let entry = &f.blocks[f.entry];
    entry
        .params
        .get(&f.value_pool)
        .iter()
        .enumerate()
        .map(|(k, &value)| {
            let reg = match f.insts[value].kind {
                InstKind::BlockParam(r) => r,
                _ => u8::MAX,
            };
            let mine = facts.iter().filter(|x| x.root as usize == k);
            let (mut read, mut write, mut escape, mut returned, mut nullable, mut indexed) =
                (false, false, false, false, false, false);
            let mut fields: Vec<(i64, bool)> = Vec::new();
            for x in mine {
                match x.kind {
                    FactKind::Read => read = true,
                    FactKind::Write => write = true,
                    FactKind::Escape => escape = true,
                    FactKind::Return => returned = true,
                    FactKind::NullCheck => nullable = true,
                }
                if matches!(x.kind, FactKind::Read | FactKind::Write) {
                    match x.off {
                        Off::Known(o) => fields.push((o, x.kind == FactKind::Write)),
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
            let class = if !read && !write {
                Class::NotPointer
            } else if escape {
                Class::Raw
            } else if write {
                Class::Mut
            } else {
                Class::Shared
            };
            ParamBorrow { value, reg, class, returned, nullable, fields, indexed }
        })
        .collect()
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
                FactKind::Escape | FactKind::Return => slots[i].address_taken = true,
                FactKind::NullCheck => {}
            }
        }
        slots.sort_unstable_by_key(|s| s.off);
        Some(slots)
    }
}
