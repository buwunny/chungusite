//! SSA cleanup: remove trivial block parameters and dead code.
//!
//! The lifter creates a block parameter for every register live into a block, so
//! loops carry parameters whose value is the same on every edge (the register was
//! never written in the loop). Those are Braun et al.'s "trivial phis": a parameter
//! whose incoming values are all one value `v` (or the parameter itself) is replaced
//! by `v`. Dead-code elimination then drops pure instructions and parameters
//! nothing uses. Instructions stay in the arena (ids are stable); they are only
//! removed from their block's list.
use crate::ir::*;
use crate::verify::for_each_operand;

#[derive(Default, Debug, PartialEq, Eq)]
pub struct CleanStats {
    pub trivial_params: usize,
    pub dead_params: usize,
    pub dead_insts: usize,
}

pub fn clean(f: &mut Function) -> CleanStats {
    let mut stats = CleanStats::default();
    let mut repl: Vec<Option<ValueId>> = vec![None; f.insts.len()];
    loop {
        let t = remove_trivial_params(f, &mut repl);
        let (dp, di) = remove_dead(f, &repl);
        stats.trivial_params += t;
        stats.dead_params += dp;
        stats.dead_insts += di;
        if t + dp + di == 0 {
            break;
        }
    }
    rewrite_uses(f, &repl);
    stats
}

fn resolve(repl: &[Option<ValueId>], mut v: ValueId) -> ValueId {
    while let Some(r) = repl[v.index()] {
        v = r;
    }
    v
}

/// (start, len) of each edge argument slice into `target` from block `from`.
fn edges_into(f: &Function, from: BlockId, target: BlockId, out: &mut Vec<usize>) {
    match f.blocks[from].term {
        Terminator::Jump { to, args } if to == target => out.push(args.start as usize),
        Terminator::Branch { t, f: e, args, .. } => {
            if t == target {
                out.push(args.start as usize);
            }
            if e == target {
                out.push(args.start as usize + f.blocks[t].params.len as usize);
            }
        }
        _ => {}
    }
}

/// Remove element `at` from the list `l` stored in `pool`, shifting the rest left.
fn list_remove(pool: &mut [ValueId], l: &mut ListRef, at: usize) {
    let (s, e) = (l.start as usize, (l.start + l.len) as usize);
    pool.copy_within(at + 1..e, at);
    debug_assert!(at >= s && at < e);
    l.len -= 1;
}

/// Remove parameter `k` of block `b` and the matching argument on every edge into it.
fn remove_param(f: &mut Function, b: BlockId, k: usize, scratch: &mut Vec<usize>) {
    for pi in 0..f.blocks.len() {
        let p = BlockId::new(pi);
        scratch.clear();
        edges_into(f, p, b, scratch);
        scratch.sort_unstable_by(|a, b| b.cmp(a)); // back to front keeps offsets valid
        for &start in scratch.iter() {
            let pool = &mut f.value_pool;
            match &mut f.blocks[p].term {
                Terminator::Jump { args, .. } | Terminator::Branch { args, .. } => list_remove(pool, args, start + k),
                _ => unreachable!(),
            }
        }
    }
    let mut params = f.blocks[b].params;
    let at = params.start as usize + k;
    list_remove(&mut f.value_pool, &mut params, at);
    f.blocks[b].params = params;
}

fn remove_trivial_params(f: &mut Function, repl: &mut [Option<ValueId>]) -> usize {
    let mut removed = 0;
    let mut starts = Vec::new();
    let mut scratch = Vec::new();
    for bi in 0..f.blocks.len() {
        let b = BlockId::new(bi);
        if b == f.entry {
            continue; // the caller is an extra, unknown predecessor
        }
        let mut k = 0;
        while k < f.blocks[b].params.len as usize {
            let p = f.value_pool[f.blocks[b].params.start as usize + k];
            let mut only: Option<ValueId> = None;
            let mut trivial = true;
            let mut any_edge = false;
            for pi in 0..f.blocks.len() {
                starts.clear();
                edges_into(f, BlockId::new(pi), b, &mut starts);
                for &s in &starts {
                    any_edge = true;
                    let v = resolve(repl, f.value_pool[s + k]);
                    if v == p || only == Some(v) {
                        continue;
                    }
                    if only.is_some() {
                        trivial = false;
                    }
                    only = Some(v);
                }
            }
            match only {
                Some(v) if trivial && any_edge => {
                    repl[p.index()] = Some(v);
                    remove_param(f, b, k, &mut scratch);
                    removed += 1;
                }
                _ => k += 1,
            }
        }
    }
    removed
}

fn is_root(k: InstKind) -> bool {
    matches!(
        k,
        InstKind::Store { .. } | InstKind::Call { .. } | InstKind::MemCopy { .. } | InstKind::Opaque { .. }
            | InstKind::Assign { .. } | InstKind::Load { volatile: true, .. }
    )
}

fn remove_dead(f: &mut Function, repl: &[Option<ValueId>]) -> (usize, usize) {
    let n = f.insts.len();
    // Where each block parameter lives, so a live parameter can mark its incoming args.
    let mut param_of: Vec<Option<(BlockId, usize)>> = vec![None; n];
    for (b, blk) in f.blocks.iter() {
        for (k, &p) in blk.params.get(&f.value_pool).iter().enumerate() {
            param_of[p.index()] = Some((b, k));
        }
    }
    let mut live = vec![false; n];
    let mut work: Vec<ValueId> = Vec::new();
    let mark = |v: ValueId, live: &mut Vec<bool>, work: &mut Vec<ValueId>| {
        let v = resolve(repl, v);
        if !std::mem::replace(&mut live[v.index()], true) {
            work.push(v);
        }
    };
    // Entry params are kept (they are the signature), so the values that loop back
    // into them stay live too.
    for &p in f.blocks[f.entry].params.get(&f.value_pool) {
        mark(p, &mut live, &mut work);
    }
    for (_, blk) in f.blocks.iter() {
        for &id in blk.insts.get(&f.value_pool) {
            if is_root(f.insts[id].kind) {
                mark(id, &mut live, &mut work);
            }
        }
        match blk.term {
            Terminator::Branch { c, .. } => mark(c, &mut live, &mut work),
            Terminator::Return(Some(v)) | Terminator::Switch { v, .. } => mark(v, &mut live, &mut work),
            Terminator::TailCall { callee, args } => {
                mark(callee, &mut live, &mut work);
                for &a in args.get(&f.value_pool) {
                    mark(a, &mut live, &mut work);
                }
            }
            _ => {}
        }
    }
    let mut starts = Vec::new();
    while let Some(v) = work.pop() {
        let mut ops = Vec::new();
        for_each_operand(f.insts[v].kind, f, |o| ops.push(o));
        if let Some((b, k)) = param_of[v.index()] {
            for pi in 0..f.blocks.len() {
                starts.clear();
                edges_into(f, BlockId::new(pi), b, &mut starts);
                ops.extend(starts.iter().map(|&s| f.value_pool[s + k]));
            }
        }
        for o in ops {
            mark(o, &mut live, &mut work);
        }
    }

    // Drop dead instructions from block lists (in place).
    let mut dead_insts = 0;
    for bi in 0..f.blocks.len() {
        let b = BlockId::new(bi);
        let l = f.blocks[b].insts;
        let mut w = l.start as usize;
        for r in l.start as usize..(l.start + l.len) as usize {
            let id = f.value_pool[r];
            if live[id.index()] || is_root(f.insts[id].kind) {
                f.value_pool[w] = id;
                w += 1;
            } else {
                dead_insts += 1;
            }
        }
        f.blocks[b].insts.len = (w - l.start as usize) as u32;
    }

    // Drop dead parameters, back to front so indices stay valid.
    let mut dead_params = 0;
    let mut scratch = Vec::new();
    for bi in 0..f.blocks.len() {
        let b = BlockId::new(bi);
        if b == f.entry {
            continue; // entry params are the function's arguments; keep the signature
        }
        for k in (0..f.blocks[b].params.len as usize).rev() {
            let p = f.value_pool[f.blocks[b].params.start as usize + k];
            if !live[p.index()] {
                remove_param(f, b, k, &mut scratch);
                dead_params += 1;
            }
        }
    }
    (dead_params, dead_insts)
}

/// Apply the replacement map to every use: instruction operands, conditions,
/// return values and edge arguments.
fn rewrite_uses(f: &mut Function, repl: &[Option<ValueId>]) {
    let r = |v: &mut ValueId| *v = resolve(repl, *v);
    for bi in 0..f.blocks.len() {
        let b = BlockId::new(bi);
        let l = f.blocks[b].insts;
        for i in l.start as usize..(l.start + l.len) as usize {
            let id = f.value_pool[i];
            let mut k = f.insts[id].kind;
            map_operands(&mut k, &mut f.value_pool, r);
            f.insts[id].kind = k;
        }
        let mut t = f.blocks[b].term;
        match &mut t {
            Terminator::Jump { args, .. } => map_list(&mut f.value_pool, *args, r),
            Terminator::Branch { c, args, .. } => { r(c); map_list(&mut f.value_pool, *args, r) }
            Terminator::Return(Some(v)) | Terminator::Switch { v, .. } => r(v),
            Terminator::TailCall { callee, args } => { r(callee); map_list(&mut f.value_pool, *args, r) }
            _ => {}
        }
        f.blocks[b].term = t;
    }
}

fn map_list(pool: &mut [ValueId], l: ListRef, r: impl Fn(&mut ValueId)) {
    for v in &mut pool[l.start as usize..(l.start + l.len) as usize] {
        r(v);
    }
}

/// Mutable counterpart of `verify::for_each_operand`.
pub fn map_operands(k: &mut InstKind, pool: &mut [ValueId], r: impl Fn(&mut ValueId)) {
    use InstKind::*;
    match k {
        Const(_) | Param(_) | BlockParam(_) | FuncRef(_) | ImportRef(_) | AddrOfLocal(_)
        | AddrOfGlobal(_) | Opaque { .. } | Copy(_) | Move(_) | Borrow { .. } => {}
        Bin { lhs, rhs, .. } | Cmp { lhs, rhs, .. } => { r(lhs); r(rhs) }
        Un { v, .. } | Cast { v, .. } | IntToPtr(v) | PtrToInt(v) | CallHi(v) => r(v),
        Select { c, t, f } => { r(c); r(t); r(f) }
        Call { callee, args } => { r(callee); map_list(pool, *args, r) }
        PtrOffset { base, index, .. } => { r(base); if let Some(i) = index { r(i) } }
        Load { ptr, .. } => r(ptr),
        Store { ptr, val, .. } => { r(ptr); r(val) }
        MemCopy { dst, src, len } => { r(dst); r(src); r(len) }
        Aggregate { fields, .. } => map_list(pool, *fields, r),
        Assign { val, .. } => r(val),
    }
}
