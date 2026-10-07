//! Structural checks on a lifted `Function`. Cheap enough to run after every pass in
//! debug builds; tests run it on everything the lifter produces.
//!
//! Catches dangling ids, edges whose argument count doesn't match the target's
//! parameters (a value dropped on the floor), values used but never placed in a
//! block, and instructions that use a store as if it produced a value. Dominance
//! (every use reached by its definition) is not checked yet.
use crate::ir::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    ListOutOfRange { block: BlockId },
    DanglingValue { user: Option<ValueId>, block: BlockId, value: ValueId },
    UsesNonValue { user: Option<ValueId>, block: BlockId, value: ValueId },
    DanglingBlock { block: BlockId, target: BlockId },
    DanglingConst { value: ValueId },
    EdgeArity { block: BlockId, expected: usize, got: usize },
    NotPlaced { value: ValueId },
    PlacedTwice { value: ValueId },
    OriginMismatch { insts: usize, origin: usize },
}

pub fn verify(f: &Function) -> Result<(), VerifyError> {
    let n = f.insts.len();
    if f.origin.len() != n {
        return Err(VerifyError::OriginMismatch { insts: n, origin: f.origin.len() });
    }
    let in_pool = |l: ListRef| (l.start as usize).checked_add(l.len as usize).is_some_and(|e| e <= f.value_pool.len());
    let mut placed = vec![false; n];

    for (b, blk) in f.blocks.iter() {
        if !in_pool(blk.insts) || !in_pool(blk.params) {
            return Err(VerifyError::ListOutOfRange { block: b });
        }
        for &v in blk.params.get(&f.value_pool).iter().chain(blk.insts.get(&f.value_pool)) {
            if v.index() >= n {
                return Err(VerifyError::DanglingValue { user: None, block: b, value: v });
            }
            if std::mem::replace(&mut placed[v.index()], true) {
                return Err(VerifyError::PlacedTwice { value: v });
            }
        }

        let use_ok = |user: Option<ValueId>, v: ValueId| -> Result<(), VerifyError> {
            if v.index() >= n {
                return Err(VerifyError::DanglingValue { user, block: b, value: v });
            }
            if f.insts[v].ty == TyId::UNIT {
                return Err(VerifyError::UsesNonValue { user, block: b, value: v });
            }
            Ok(())
        };

        for &id in blk.insts.get(&f.value_pool) {
            let k = f.insts[id].kind;
            if let InstKind::Const(c) = k {
                if c.index() >= f.consts.len() {
                    return Err(VerifyError::DanglingConst { value: id });
                }
            }
            let mut err = Ok(());
            for_each_operand(k, f, |v| {
                if err.is_ok() {
                    err = use_ok(Some(id), v);
                }
            });
            err?;
        }

        let (targets, args, cond) = match blk.term {
            Terminator::Jump { to, args } => ([Some(to), None], args, None),
            Terminator::Branch { c, t, f: e, args } => ([Some(t), Some(e)], args, Some(c)),
            Terminator::Switch { v, .. } => ([None, None], ListRef::EMPTY, Some(v)),
            Terminator::Return(r) => ([None, None], ListRef::EMPTY, r),
            Terminator::TailCall { callee, args } => ([None, None], args, Some(callee)),
            Terminator::Unreachable => ([None, None], ListRef::EMPTY, None),
        };
        if let Some(c) = cond {
            use_ok(None, c)?;
        }
        if !in_pool(args) {
            return Err(VerifyError::ListOutOfRange { block: b });
        }
        let mut expected = 0;
        for t in targets.into_iter().flatten() {
            if t.index() >= f.blocks.len() {
                return Err(VerifyError::DanglingBlock { block: b, target: t });
            }
            expected += f.blocks[t].params.len as usize;
        }
        if !matches!(blk.term, Terminator::TailCall { .. }) && args.len as usize != expected {
            return Err(VerifyError::EdgeArity { block: b, expected, got: args.len as usize });
        }
        for &a in args.get(&f.value_pool) {
            use_ok(None, a)?;
        }
    }

    // Every value something uses belongs to some block. (Passes may drop unused
    // instructions from block lists; they stay in the arena so ids remain stable.)
    let mut unplaced = None;
    for (_, blk) in f.blocks.iter() {
        for &id in blk.insts.get(&f.value_pool) {
            for_each_operand(f.insts[id].kind, f, |v| {
                if !placed[v.index()] { unplaced.get_or_insert(v); }
            });
        }
        let term_uses = match blk.term {
            Terminator::Jump { args, .. } | Terminator::TailCall { args, .. } => args.get(&f.value_pool),
            Terminator::Branch { args, .. } => args.get(&f.value_pool),
            _ => &[],
        };
        let extra = match blk.term {
            Terminator::Branch { c, .. } => Some(c),
            Terminator::Return(r) => r,
            Terminator::TailCall { callee, .. } => Some(callee),
            Terminator::Switch { v, .. } => Some(v),
            _ => None,
        };
        for &v in term_uses.iter().chain(extra.as_ref()) {
            if !placed[v.index()] { unplaced.get_or_insert(v); }
        }
    }
    match unplaced {
        Some(value) => Err(VerifyError::NotPlaced { value }),
        None => Ok(()),
    }
}

/// Calls `cb` with every value an instruction reads.
pub fn for_each_operand(k: InstKind, f: &Function, mut cb: impl FnMut(ValueId)) {
    use InstKind::*;
    match k {
        Const(_) | Undef | Param(_) | BlockParam(_) | FuncRef(_) | ImportRef(_) | AddrOfLocal(_)
        | AddrOfGlobal(_) | Opaque { .. } => {}
        Bin { lhs, rhs, .. } | Cmp { lhs, rhs, .. } => { cb(lhs); cb(rhs) }
        Un { v, .. } | Cast { v, .. } | IntToPtr(v) | PtrToInt(v) | CallOut { call: v, .. } => cb(v),
        Exit { regs } => regs.get(&f.value_pool).iter().copied().for_each(&mut cb),
        Select { c, t, f: e } => { cb(c); cb(t); cb(e) }
        Call { callee, args } => { cb(callee); args.get(&f.value_pool).iter().copied().for_each(&mut cb) }
        PtrOffset { base, index, .. } => { cb(base); if let Some(i) = index { cb(i) } }
        Load { ptr, .. } => cb(ptr),
        Store { ptr, val, .. } => { cb(ptr); cb(val) }
        MemCopy { dst, src, len } => { cb(dst); cb(src); cb(len) }
        Aggregate { fields, .. } => fields.get(&f.value_pool).iter().copied().for_each(&mut cb),
        Assign { val, .. } => cb(val),
        // Place operands are checked when places are introduced (safe mode).
        Copy(_) | Move(_) | Borrow { .. } => {}
    }
}
