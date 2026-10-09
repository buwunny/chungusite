//! Stack frames (roadmap item 3).
//!
//! After `abi::apply`, rsp is the only non-argument register left on entry. This
//! pass removes it from the signature:
//!
//! 1. **Offsets.** A forward dataflow gives every value its offset from the entry
//!    rsp, if it is derived from it: `Known(x)`, or `Unknown` (indexed, advanced in
//!    a loop, merged from different offsets).
//! 2. **Address-taken objects.** Where a stack address leaves what the dataflow
//!    can follow (stored to memory, passed to a call, returned, indexed, merged),
//!    the object at that offset is address-taken. C pointer arithmetic only moves
//!    up within an object, so everything from that offset up to the return address
//!    stays in memory. (Up to infinity for an address in the caller's argument
//!    area, as `va_start` takes.)
//! 3. **Promotion** (mem2reg). Every other slot that is accessed at one offset with
//!    one store width (narrower loads become truncations) becomes SSA values with
//!    block parameters, built like the lifter builds registers. A slot read before
//!    it is written is `Undef` in the frame, or a new parameter for a stack argument
//!    (entry offsets 8, 16, ...).
//! 4. **The frame.** If rsp is still used, it becomes the address of a local array
//!    (`Function::locals[0]`), laid out like the real frame: 16-byte aligned, with
//!    the entry rsp 8 bytes off alignment. Stack arguments that stay in memory are
//!    copied into it from new parameters.
//!
//! If some stack address goes through an operation that isn't pointer arithmetic
//! (`and rsp, -16`), nothing is promoted and the frame is a fixed 64 KiB.
use crate::abi::{bytes, incoming, new_inst, prepend, SLOT_PARAM, STACK_ARG_BASE};
use crate::borrow::RSP;
use crate::cfg::Cfg;
use crate::ir::*;
use crate::opt::{clean, list_remove, rewrite_uses};
use crate::verify::for_each_operand;

/// Frame used when the stack offsets can't all be followed.
pub const FALLBACK_FRAME: u32 = 1 << 16;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum S {
    /// Not computed yet (unreachable, or later in a loop).
    Bot,
    /// Not derived from rsp.
    Not,
    Known(i64),
    Unknown,
}

fn join(a: S, b: S) -> S {
    match (a, b) {
        (S::Bot, x) | (x, S::Bot) => x,
        (S::Not, S::Not) => S::Not,
        (S::Known(x), S::Known(y)) if x == y => S::Known(x),
        _ => S::Unknown,
    }
}

fn konst(f: &Function, v: ValueId) -> Option<i64> {
    match f.insts[v].kind {
        InstKind::Const(c) => Some(f.consts[c.index()] as u64 as i64),
        _ => None,
    }
}

struct Offsets {
    s: Vec<S>,
    /// Offsets of address-taken objects.
    taken: Vec<i64>,
    /// A stack address went somewhere the analysis can't follow.
    lost: bool,
}

impl Offsets {
    fn of(&self, v: ValueId) -> S {
        self.s[v.index()]
    }
    fn stack(&self, v: ValueId) -> bool {
        matches!(self.of(v), S::Known(_) | S::Unknown)
    }
    fn take(&mut self, v: ValueId) {
        if let S::Known(x) = self.of(v) {
            self.taken.push(x);
        }
    }
}

/// Offset of an instruction's result. With `record`, also note address-taken
/// objects and lost addresses.
fn transfer(f: &Function, k: InstKind, o: &mut Offsets, record: bool) -> S {
    use InstKind::*;
    let of = |o: &Offsets, v: ValueId| o.of(v);
    let s = match k {
        PtrOffset { base, index, scale, disp } => match (of(o, base), index.map(|i| of(o, i))) {
            (S::Bot, _) | (_, Some(S::Bot)) => S::Bot,
            (S::Known(x), None) => S::Known(x + disp as i64),
            (b, None) => b,
            (S::Not, Some(S::Not)) => S::Not,
            (S::Known(x), Some(S::Not)) => {
                if record {
                    o.taken.push(x + disp as i64);
                }
                S::Unknown
            }
            (S::Not, Some(S::Known(x))) if scale == 1 => {
                if record {
                    o.taken.push(x);
                }
                S::Unknown
            }
            (S::Unknown, Some(S::Not)) | (S::Not, Some(S::Unknown)) if scale == 1 || of(o, base) == S::Unknown => S::Unknown,
            _ => {
                if record {
                    o.lost = true;
                }
                S::Not
            }
        },
        Bin { op: op @ (BinOp::Add | BinOp::Sub), lhs, rhs } => {
            let (a, b) = (of(o, lhs), of(o, rhs));
            match (a, b) {
                (S::Bot, _) | (_, S::Bot) => S::Bot,
                (S::Not, S::Not) => S::Not,
                (S::Known(x), S::Not) => match konst(f, rhs) {
                    Some(c) => S::Known(if matches!(op, BinOp::Add) { x.wrapping_add(c) } else { x.wrapping_sub(c) }),
                    None => {
                        if record {
                            o.taken.push(x);
                        }
                        S::Unknown
                    }
                },
                (S::Not, S::Known(x)) if matches!(op, BinOp::Add) => match konst(f, lhs) {
                    Some(c) => S::Known(x.wrapping_add(c)),
                    None => {
                        if record {
                            o.taken.push(x);
                        }
                        S::Unknown
                    }
                },
                (S::Unknown, S::Not) => S::Unknown,
                (S::Not, S::Unknown) if matches!(op, BinOp::Add) => S::Unknown,
                // the distance between two stack addresses is an integer
                (S::Known(_) | S::Unknown, S::Known(_) | S::Unknown) if matches!(op, BinOp::Sub) => S::Not,
                _ => {
                    if record {
                        o.lost = true;
                    }
                    S::Not
                }
            }
        }
        Select { t, f: e, .. } => {
            let s = join(of(o, t), of(o, e));
            if record && s == S::Unknown {
                o.take(t);
                o.take(e);
            }
            s
        }
        Cast { kind: CastKind::Bitcast, v } | IntToPtr(v) | PtrToInt(v) => of(o, v),
        Load { .. } => S::Not,
        Store { val, .. } => {
            if record && o.stack(val) {
                o.take(val);
            }
            S::Not
        }
        MemCopy { dst, src, len: count } | MemFill { dst, val: src, count } => {
            if record {
                for v in [dst, src, count] {
                    o.take(v);
                }
            }
            S::Not
        }
        Call { callee, args } => {
            if record {
                o.take(callee);
                for &a in args.get(&f.value_pool) {
                    o.take(a);
                }
            }
            S::Not
        }
        Cmp { .. } | Const(_) | Undef | CallOut { .. } => S::Not,
        Opaque { .. } => {
            if record {
                o.lost = true;
            }
            S::Not
        }
        k => {
            if record {
                let mut any = false;
                for_each_operand(k, f, |v| any |= o.stack(v));
                o.lost |= any;
            }
            S::Not
        }
    };
    s
}

fn offsets(f: &Function, cfg: &Cfg, sp: ValueId) -> Offsets {
    let mut o = Offsets { s: vec![S::Bot; f.insts.len()], taken: Vec::new(), lost: false };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &cfg.rpo {
            let blk = &f.blocks[b];
            for (k, &p) in blk.params.get(&f.value_pool).iter().enumerate() {
                let new = if b == f.entry {
                    if p == sp { S::Known(0) } else { S::Not }
                } else {
                    let mut s = S::Bot;
                    for &pred in cfg.preds(b) {
                        incoming(f, pred, b, k, |a| s = join(s, o.s[a.index()]));
                    }
                    s
                };
                if new != o.s[p.index()] {
                    o.s[p.index()] = new;
                    changed = true;
                }
            }
            for &id in blk.insts.get(&f.value_pool) {
                let new = transfer(f, f.insts[id].kind, &mut o, false);
                if new != o.s[id.index()] {
                    o.s[id.index()] = new;
                    changed = true;
                }
            }
        }
    }
    // Recording pass, now that offsets are final.
    for &b in &cfg.rpo {
        let blk = &f.blocks[b];
        if b != f.entry {
            for (k, &p) in blk.params.get(&f.value_pool).iter().enumerate() {
                if o.s[p.index()] == S::Unknown {
                    for &pred in cfg.preds(b) {
                        incoming(f, pred, b, k, |a| {
                            if let S::Known(x) = o.s[a.index()] {
                                o.taken.push(x);
                            }
                        });
                    }
                }
            }
        }
        for &id in blk.insts.get(&f.value_pool) {
            transfer(f, f.insts[id].kind, &mut o, true);
        }
        match blk.term {
            Terminator::Return(Some(v)) | Terminator::Switch { v, .. } => o.take(v),
            Terminator::TailCall { callee, args } => {
                o.take(callee);
                for &a in args.get(&f.value_pool) {
                    o.take(a);
                }
            }
            _ => {}
        }
    }
    o
}

#[derive(Copy, Clone, Debug)]
struct Access {
    id: ValueId,
    off: i64,
    size: i64,
    store: bool,
}

#[derive(Copy, Clone, Debug)]
struct Slot {
    off: i64,
    size: i64,
}

/// Promote stack slots and give the function a frame; see the module docs.
/// `stack_args` is how many stack arguments the signature takes: a stack
/// argument whose address escapes copies in only those, as callers pass no more.
pub fn promote(f: &mut Function, stack_args: u8) {
    let entry = f.entry;
    let Some(k_sp) = f.blocks[entry]
        .params
        .get(&f.value_pool)
        .iter()
        .position(|&p| matches!(f.insts[p].kind, InstKind::BlockParam(RSP)))
    else {
        return;
    };
    let sp = f.value_pool[f.blocks[entry].params.start as usize + k_sp];
    let cfg = Cfg::new(f);
    let o = offsets(f, &cfg, sp);

    // Accesses at known offsets.
    let mut accesses = Vec::new();
    for &b in &cfg.rpo {
        for &id in f.blocks[b].insts.get(&f.value_pool) {
            let (ptr, size, store) = match f.insts[id].kind {
                InstKind::Load { ptr, volatile: false, .. } => (ptr, bytes(f.insts[id].ty), false),
                InstKind::Store { ptr, val, .. } => (ptr, bytes(f.insts[val].ty), true),
                _ => continue,
            };
            if let S::Known(off) = o.of(ptr) {
                accesses.push(Access { id, off, size: size as i64, store });
            }
        }
    }

    // Which slots can become values.
    let neg_taken = o.taken.iter().copied().filter(|&x| x < 0).min();
    let pos_taken = o.taken.iter().copied().filter(|&x| x >= 0).min();
    let tainted = |lo: i64, hi: i64| neg_taken.is_some_and(|t| hi > t && lo < 0) || pos_taken.is_some_and(|t| hi > t);
    let mut slots: Vec<Slot> = Vec::new();
    let mut ok_slot: Vec<bool> = Vec::new();
    if !o.lost {
        let mut offs: Vec<i64> = accesses.iter().map(|a| a.off).collect();
        offs.sort_unstable();
        offs.dedup();
        for &off in &offs {
            let at = accesses.iter().filter(|a| a.off == off);
            let size = at.clone().map(|a| a.size).max().unwrap();
            let stores_ok = at.clone().all(|a| !a.store || a.size == size);
            // (stack arguments are the first 64 words above the return
            // address, as `hi` below; a slot further up is something else)
            let in_args = (8..8 + 8 * 64).contains(&off);
            let ok = stores_ok
                && !tainted(off, off + size)
                && (off + size <= 0 || (in_args && off % 8 == 0 && size <= 8));
            slots.push(Slot { off, size });
            ok_slot.push(ok);
        }
        // Overlapping slots at different offsets stay in memory.
        for i in 0..slots.len() {
            for j in i + 1..slots.len() {
                let (a, b) = (slots[i], slots[j]);
                if a.off < b.off + b.size && b.off < a.off + a.size {
                    ok_slot[i] = false;
                    ok_slot[j] = false;
                }
            }
        }
        // A stack-argument word with any access left in memory stays whole.
        for i in 0..slots.len() {
            if slots[i].off >= 8 && !ok_slot[i] {
                let word = (slots[i].off - 8) / 8;
                for j in 0..slots.len() {
                    if slots[j].off >= 8 && (slots[j].off - 8) / 8 == word {
                        ok_slot[j] = false;
                    }
                }
            }
        }
    } else {
        // nothing is promoted; still note the slots for the frame size
        for a in &accesses {
            slots.push(Slot { off: a.off, size: a.size });
            ok_slot.push(false);
        }
    }

    let promoted: Vec<usize> = (0..slots.len()).filter(|&i| ok_slot[i]).collect();
    if !promoted.is_empty() {
        mem2reg(f, &cfg, &o, &accesses, &slots, &promoted);
        clean(f);
    }

    // Is rsp still used?
    let used = uses_value(f, sp);
    let k_sp = f.blocks[entry].params.get(&f.value_pool).iter().position(|&p| p == sp).expect("rsp param");
    let mut params = f.blocks[entry].params;
    let pos = params.start as usize + k_sp;
        list_remove(&mut f.value_pool, &mut params, pos);
    f.blocks[entry].params = params;
    if !used {
        return;
    }

    // The frame: everything still in memory, plus address-taken objects.
    // (a lost frame still holds the stack arguments, which the code reads
    // through rbp or rsp like anything else)
    let (lo, hi) = if o.lost {
        (-(FALLBACK_FRAME as i64) + 64, 8 + 8 * stack_args as i64)
    } else {
        let mut lo = 0i64;
        let mut hi = 8i64;
        for (i, s) in slots.iter().enumerate() {
            if !ok_slot[i] {
                lo = lo.min(s.off);
                hi = hi.max(s.off + s.size);
            }
        }
        if let Some(t) = neg_taken {
            lo = lo.min(t);
        }
        if let Some(t) = pos_taken {
            hi = hi.max(t + 8);
        }
        (lo, hi.min(8 + 8 * 64))
    };
    let t = (-lo).max(8);
    let top = (t - 8 + 15) / 16 * 16 + 8;
    let total = ((top + hi) + 15) / 16 * 16;
    let local = f.locals.push(Local { ty: TyId::B1, frame_offset: -(top as i32), size: total as u32, name: None });
    let at = f.blocks[entry].insts.get(&f.value_pool).first().and_then(|v| f.origin.get(v.index())).copied().unwrap_or(0);
    let base = new_inst(f, InstKind::AddrOfLocal(local), TyId::PTR, at);
    let new_sp = new_inst(f, InstKind::PtrOffset { base, index: None, scale: 1, disp: top as i32 }, TyId::PTR, at);
    let mut head = vec![base, new_sp];
    // Stack arguments still read from memory are copied in from parameters.
    let mut new_params = Vec::new();
    if hi > 8 {
        for word in 0..((hi - 8 + 7) / 8).min(stack_args as i64) {
            let reg = STACK_ARG_BASE + word as u8;
            let exists = f.blocks[entry].params.get(&f.value_pool).iter().copied().find(|&p| matches!(f.insts[p].kind, InstKind::BlockParam(r) if r == reg));
            let in_memory = o.lost
                || slots.iter().enumerate().any(|(i, s)| !ok_slot[i] && s.off >= 8 && (s.off - 8) / 8 == word)
                || pos_taken.is_some_and(|t| 8 + 8 * word + 8 > t);
            if !in_memory || (exists.is_some() && !o.lost) {
                continue;
            }
            let p = exists.unwrap_or_else(|| {
                let p = new_inst(f, InstKind::BlockParam(reg), TyId::B8, at);
                new_params.push(p);
                p
            });
            let ptr = new_inst(f, InstKind::PtrOffset { base: new_sp, index: None, scale: 1, disp: 8 + 8 * word as i32 }, TyId::PTR, at);
            let st = new_inst(f, InstKind::Store { ptr, val: p, align: 1 }, TyId::UNIT, at);
            head.extend([ptr, st]);
        }
    }
    add_entry_params(f, &new_params);
    let mut repl = vec![None; f.insts.len()];
    repl[sp.index()] = Some(new_sp);
    rewrite_uses(f, &repl);
    prepend(f, entry, &head);
}

fn add_entry_params(f: &mut Function, new: &[ValueId]) {
    if new.is_empty() {
        return;
    }
    let e = f.entry;
    let old = f.blocks[e].params;
    let start = f.value_pool.len();
    f.value_pool.extend_from_within(old.start as usize..(old.start + old.len) as usize);
    f.value_pool.extend_from_slice(new);
    f.blocks[e].params = ListRef { start: start as u32, len: (f.value_pool.len() - start) as u32 };
}

/// Does anything placed use `v`?
fn uses_value(f: &Function, v: ValueId) -> bool {
    for (_, blk) in f.blocks.iter() {
        for &id in blk.insts.get(&f.value_pool) {
            let mut used = false;
            for_each_operand(f.insts[id].kind, f, |o| used |= o == v);
            if used {
                return true;
            }
        }
        let mut used = false;
        let list = |l: ListRef| l.get(&f.value_pool).contains(&v);
        match blk.term {
            Terminator::Jump { args, .. } => used |= list(args),
            Terminator::Branch { c, args, .. } => used |= c == v || list(args),
            Terminator::Return(Some(r)) | Terminator::Switch { v: r, .. } => used |= r == v,
            Terminator::TailCall { callee, args } => used |= callee == v || list(args),
            _ => {}
        }
        if used {
            return true;
        }
    }
    false
}

/// SSA construction for the promoted slots, with block parameters.
fn mem2reg(f: &mut Function, cfg: &Cfg, o: &Offsets, accesses: &[Access], slots: &[Slot], promoted: &[usize]) {
    let n = f.blocks.len();
    let ns = promoted.len();
    // slot of each promoted access
    let mut slot_of: Vec<Option<usize>> = vec![None; f.insts.len()];
    for a in accesses {
        if let Some(i) = promoted.iter().position(|&p| slots[p].off == a.off) {
            slot_of[a.id.index()] = Some(i);
        }
    }
    let _ = o;
    let entry = f.entry;
    let at = f.blocks[entry].insts.get(&f.value_pool).first().and_then(|v| f.origin.get(v.index())).copied().unwrap_or(0);

    let mut live_in: Vec<Vec<Option<ValueId>>> = vec![vec![None; ns]; n];
    let mut out: Vec<Vec<Option<ValueId>>> = vec![vec![None; ns]; n];
    let mut repl: Vec<Option<ValueId>> = Vec::new();
    let mut removed = vec![false; f.insts.len()];
    let mut entry_head: Vec<ValueId> = Vec::new();
    let mut entry_params: Vec<ValueId> = Vec::new();

    // A stack argument (a whole word) as the value of slot `s`, which may be narrower.
    let narrow = |f: &mut Function, p: ValueId, s: usize, entry_head: &mut Vec<ValueId>| -> ValueId {
        let size = slots[promoted[s]].size as usize;
        if size >= 8 {
            return p;
        }
        let t = new_inst(f, InstKind::Cast { kind: CastKind::Trunc, v: p }, TyId::unknown(size), at);
        entry_head.push(t);
        t
    };
    // A slot's value on entry to block `b`.
    let live_in_value = |f: &mut Function, b: BlockId, s: usize, entry_head: &mut Vec<ValueId>, entry_params: &mut Vec<ValueId>| -> ValueId {
        let slot = slots[promoted[s]];
        if b == entry || cfg.preds(b).is_empty() {
            if b == entry && slot.off >= 8 {
                let p = new_inst(f, InstKind::BlockParam(STACK_ARG_BASE + ((slot.off - 8) / 8) as u8), TyId::B8, at);
                entry_params.push(p);
                narrow(f, p, s, entry_head)
            } else {
                let u = new_inst(f, InstKind::Undef, TyId::unknown(slot.size as usize), at);
                entry_head.push(u);
                u
            }
        } else {
            new_inst(f, InstKind::BlockParam(SLOT_PARAM), TyId::unknown(slot.size as usize), at)
        }
    };

    for bi in 0..n {
        let b = BlockId::new(bi);
        let list: Vec<ValueId> = f.blocks[b].insts.get(&f.value_pool).to_vec();
        let mut cur: Vec<Option<ValueId>> = vec![None; ns];
        for id in list {
            let Some(s) = slot_of.get(id.index()).copied().flatten() else { continue };
            match f.insts[id].kind {
                InstKind::Store { val, .. } => {
                    cur[s] = Some(val);
                    removed[id.index()] = true;
                }
                InstKind::Load { .. } => {
                    let v = match cur[s] {
                        Some(v) => v,
                        None => {
                            // only an entry-reaching stack argument can be shared:
                            // one parameter per word
                            let v = if b == entry && slots[promoted[s]].off >= 8 {
                                let reg = STACK_ARG_BASE + ((slots[promoted[s]].off - 8) / 8) as u8;
                                match entry_params.iter().copied().find(|&p| matches!(f.insts[p].kind, InstKind::BlockParam(r) if r == reg)) {
                                    Some(p) => narrow(f, p, s, &mut entry_head),
                                    None => live_in_value(f, b, s, &mut entry_head, &mut entry_params),
                                }
                            } else {
                                live_in_value(f, b, s, &mut entry_head, &mut entry_params)
                            };
                            live_in[bi][s] = Some(v);
                            cur[s] = Some(v);
                            v
                        }
                    };
                    let (lt, vt) = (bytes(f.insts[id].ty), bytes(f.insts[v].ty));
                    if lt == vt {
                        repl.resize(f.insts.len(), None);
                        repl[id.index()] = Some(v);
                        removed[id.index()] = true;
                    } else {
                        // narrower load of a wider value: its low bytes
                        f.insts[id].kind = InstKind::Cast { kind: CastKind::Trunc, v };
                    }
                }
                _ => {}
            }
        }
        out[bi] = cur;
    }

    // Live-ins must be defined at the end of every predecessor. A worklist of
    // (block, slot) live-ins whose predecessors haven't been checked yet.
    let mut work: Vec<(usize, usize)> =
        (0..n).flat_map(|bi| (0..ns).map(move |s| (bi, s))).filter(|&(bi, s)| live_in[bi][s].is_some()).collect();
    work.reverse();
    while let Some((bi, s)) = work.pop() {
        let b = BlockId::new(bi);
        if b == entry {
            continue;
        }
        for &p in cfg.preds(b) {
            let pi = p.index();
            if out[pi][s].is_none() {
                let v = if p == entry {
                    let reg_off = slots[promoted[s]].off;
                    let existing = if reg_off >= 8 {
                        let reg = STACK_ARG_BASE + ((reg_off - 8) / 8) as u8;
                        entry_params.iter().copied().find(|&q| matches!(f.insts[q].kind, InstKind::BlockParam(r) if r == reg))
                    } else {
                        None
                    };
                    match existing {
                        Some(q) => narrow(f, q, s, &mut entry_head),
                        None => live_in_value(f, p, s, &mut entry_head, &mut entry_params),
                    }
                } else {
                    live_in_value(f, p, s, &mut entry_head, &mut entry_params)
                };
                live_in[pi][s] = Some(v);
                out[pi][s] = Some(v);
                work.push((pi, s));
            }
        }
    }

    // Block parameters (non-entry blocks with predecessors).
    let has_param = |bi: usize, s: usize, live_in: &Vec<Vec<Option<ValueId>>>, f: &Function| -> bool {
        live_in[bi][s].is_some_and(|v| matches!(f.insts[v].kind, InstKind::BlockParam(SLOT_PARAM)))
    };
    // A `Switch` passes no arguments, so a block it enters takes the slot's value
    // at the switch directly (the switch is its only predecessor).
    for bi in 0..n {
        let b = BlockId::new(bi);
        let switch = cfg.preds(b).first().filter(|p| matches!(f.blocks[**p].term, Terminator::Switch { .. }));
        let Some(&p) = switch else { continue };
        let mut casts = Vec::new();
        for s in 0..ns {
            if has_param(bi, s, &live_in, f) {
                let (param, val) = (live_in[bi][s].unwrap(), out[p.index()][s].expect("filled above"));
                let (pt, vt) = (bytes(f.insts[param].ty), bytes(f.insts[val].ty));
                if pt == vt {
                    repl.resize(f.insts.len(), None);
                    repl[param.index()] = Some(val);
                } else {
                    // the parameter becomes the conversion, at the top of the block
                    let kind = if pt < vt { CastKind::Trunc } else { CastKind::ZExt };
                    f.insts[param].kind = InstKind::Cast { kind, v: val };
                    casts.push(param);
                }
                live_in[bi][s] = None;
            }
        }
        prepend(f, b, &casts);
    }
    for bi in 0..n {
        let b = BlockId::new(bi);
        let new: Vec<ValueId> = (0..ns).filter(|&s| has_param(bi, s, &live_in, f)).map(|s| live_in[bi][s].unwrap()).collect();
        if new.is_empty() {
            continue;
        }
        let old = f.blocks[b].params;
        let start = f.value_pool.len();
        f.value_pool.extend_from_within(old.start as usize..(old.start + old.len) as usize);
        f.value_pool.extend_from_slice(&new);
        f.blocks[b].params = ListRef { start: start as u32, len: (f.value_pool.len() - start) as u32 };
    }
    // Edge arguments, rebuilt per terminator: old arguments for each successor,
    // then the new ones. `params.len` already counts the new parameters.
    for bi in 0..n {
        let b = BlockId::new(bi);
        let term = f.blocks[b].term;
        let (old_args, succs) = match term {
            Terminator::Jump { to, args } => (args, vec![to]),
            Terminator::Branch { t, f: e, args, .. } => (args, vec![t, e]),
            _ => continue,
        };
        if !succs.iter().any(|s| (0..ns).any(|k| has_param(s.index(), k, &live_in, f))) {
            continue;
        }
        let old: Vec<ValueId> = old_args.get(&f.value_pool).to_vec();
        let mut new = Vec::new();
        let mut pos = 0;
        for &s in &succs {
            let added = (0..ns).filter(|&k| has_param(s.index(), k, &live_in, f)).count();
            let n_old = f.blocks[s].params.len as usize - added;
            new.extend_from_slice(&old[pos..pos + n_old]);
            pos += n_old;
            for k in 0..ns {
                if has_param(s.index(), k, &live_in, f) {
                    new.push(out[bi][k].expect("filled above"));
                }
            }
        }
        let start = f.value_pool.len() as u32;
        f.value_pool.extend_from_slice(&new);
        let list = ListRef { start, len: new.len() as u32 };
        match &mut f.blocks[b].term {
            Terminator::Jump { args, .. } | Terminator::Branch { args, .. } => *args = list,
            _ => {}
        }
    }

    // Drop the promoted loads and stores, rewire uses.
    for bi in 0..n {
        let b = BlockId::new(bi);
        let l = f.blocks[b].insts;
        let mut w = l.start as usize;
        for r in l.start as usize..(l.start + l.len) as usize {
            let id = f.value_pool[r];
            if !removed.get(id.index()).copied().unwrap_or(false) {
                f.value_pool[w] = id;
                w += 1;
            }
        }
        f.blocks[b].insts.len = (w - l.start as usize) as u32;
    }
    add_entry_params(f, &entry_params);
    prepend(f, entry, &entry_head);
    repl.resize(f.insts.len(), None);
    rewrite_uses(f, &repl);
}
