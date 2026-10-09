//! SSA cleanup: remove trivial block parameters and dead code.
//!
//! The lifter creates a block parameter for every register live into a block, so
//! loops carry parameters whose value is the same on every edge (the register was
//! never written in the loop). Those are Braun et al.'s "trivial phis": a parameter
//! whose incoming values are all one value `v` (or the parameter itself) is replaced
//! by `v`. Dead-code elimination then drops pure instructions and parameters
//! nothing uses. Instructions stay in the arena (ids are stable); they are only
//! removed from their block's list.
use crate::abi::Site;
use std::collections::HashMap;
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
    // Removing parameters never changes the edges, so the predecessors stay valid.
    let preds = preds(f);
    loop {
        let t = remove_trivial_params(f, &mut repl, &preds);
        let (dp, di) = remove_dead(f, &repl, &preds);
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

pub(crate) fn resolve(repl: &[Option<ValueId>], mut v: ValueId) -> ValueId {
    while let Some(r) = repl[v.index()] {
        v = r;
    }
    v
}

/// (start, len) of each edge argument slice into `target` from block `from`.
pub(crate) fn edges_into(f: &Function, from: BlockId, target: BlockId, out: &mut Vec<usize>) {
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
pub(crate) fn list_remove(pool: &mut [ValueId], l: &mut ListRef, at: usize) {
    let (s, e) = (l.start as usize, (l.start + l.len) as usize);
    pool.copy_within(at + 1..e, at);
    debug_assert!(at >= s && at < e);
    l.len -= 1;
}

/// Each block's predecessors, each listed once: `preds[start[b]..start[b + 1]]`.
struct Preds {
    start: Vec<u32>,
    preds: Vec<BlockId>,
}

impl std::ops::Index<usize> for Preds {
    type Output = [BlockId];
    fn index(&self, b: usize) -> &[BlockId] {
        &self.preds[self.start[b] as usize..self.start[b + 1] as usize]
    }
}

fn preds(f: &Function) -> Preds {
    let mut edges: Vec<(u32, BlockId)> = Vec::new();
    for (b, blk) in f.blocks.iter() {
        for s in blk.term.successors(&f.value_pool) {
            edges.push((s.index() as u32, b));
        }
    }
    edges.sort_unstable_by_key(|&(s, b)| (s, b.index()));
    edges.dedup_by_key(|e| (e.0, e.1.index()));
    let mut start = vec![0u32; f.blocks.len() + 1];
    for &(s, _) in &edges {
        start[s as usize + 1] += 1;
    }
    for i in 0..f.blocks.len() {
        start[i + 1] += start[i];
    }
    Preds { start, preds: edges.into_iter().map(|e| e.1).collect() }
}

/// Remove parameter `k` of block `b` and the matching argument on every edge into
/// it; `preds` are `b`'s predecessors.
fn remove_param(f: &mut Function, b: BlockId, k: usize, preds: &[BlockId], scratch: &mut Vec<usize>) {
    for &p in preds {
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

fn remove_trivial_params(f: &mut Function, repl: &mut [Option<ValueId>], preds: &Preds) -> usize {
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
            for &pb in &preds[bi] {
                starts.clear();
                edges_into(f, pb, b, &mut starts);
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
                    remove_param(f, b, k, &preds[bi], &mut scratch);
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
        InstKind::Store { .. } | InstKind::Call { .. } | InstKind::MemCopy { .. } | InstKind::MemFill { .. } | InstKind::Opaque { .. } | InstKind::Exit { .. }
            | InstKind::Assign { .. } | InstKind::Load { volatile: true, .. }
    )
}

fn remove_dead(f: &mut Function, repl: &[Option<ValueId>], preds: &Preds) -> (usize, usize) {
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
            for &pb in &preds[b.index()] {
                starts.clear();
                edges_into(f, pb, b, &mut starts);
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
                remove_param(f, b, k, &preds[bi], &mut scratch);
                dead_params += 1;
            }
        }
    }
    (dead_params, dead_insts)
}

/// Apply the replacement map to every use: instruction operands, conditions,
/// return values and edge arguments.
pub(crate) fn rewrite_uses(f: &mut Function, repl: &[Option<ValueId>]) {
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
        Const(_) | Undef | Param(_) | BlockParam(_) | FuncRef(_) | ImportRef(_) | AddrOfLocal(_)
        | AddrOfGlobal(_) | Opaque { .. } | Copy(_) | Move(_) | Borrow { .. } => {}
        Bin { lhs, rhs, .. } | Cmp { lhs, rhs, .. } => { r(lhs); r(rhs) }
        Un { v, .. } | Cast { v, .. } | IntToPtr(v) | PtrToInt(v) | CallOut { call: v, .. } => r(v),
        Exit { regs } => map_list(pool, *regs, r),
        Select { c, t, f } => { r(c); r(t); r(f) }
        Call { callee, args } => { r(callee); map_list(pool, *args, r) }
        PtrOffset { base, index, .. } => { r(base); if let Some(i) = index { r(i) } }
        Load { ptr, .. } => r(ptr),
        Store { ptr, val, .. } => { r(ptr); r(val) }
        MemCopy { dst, src, len } => { r(dst); r(src); r(len) }
        MemFill { dst, val, count } => { r(dst); r(val); r(count) }
        Aggregate { fields, .. } => map_list(pool, *fields, r),
        Assign { val, .. } => r(val),
    }
}

/// Most instructions a return block may have and still be copied into each of
/// its predecessors (`split_returns`), and most it may add to the function.
const SPLIT_RETURN_MAX: usize = 16;
const SPLIT_RETURN_BUDGET: usize = 64;

/// Give each predecessor of a small return block its own copy of it, the way
/// the source most likely had one `return` per path before the compiler merged
/// them into one epilogue. The copy has one predecessor, so `clean` replaces its
/// parameters by the edge's arguments: `if c { return a; } return b;` instead of
/// a join that assigns a variable on both paths (and, when the paths have code
/// of their own, a labeled block to reach it). Returns how many copies were
/// made, and each call site that was copied with its copy; the caller runs
/// `clean` after.
pub fn split_returns(f: &mut Function) -> (usize, Vec<(Site, Site)>) {
    let mut sites = Vec::new();
    let mut budget = SPLIT_RETURN_BUDGET;
    let mut made = 0;
    // A copy can make its predecessor the start of a straight line to a return
    // (`tail`), which the next round copies in turn.
    for _ in 0..4 {
        let n = split_round(f, &mut budget, &mut sites);
        made += n;
        if n == 0 {
            break;
        }
    }
    (made, sites)
}

fn split_round(f: &mut Function, budget: &mut usize, sites: &mut Vec<(Site, Site)>) -> usize {
    use InstKind::*;
    let preds = preds(f);
    let mut made = 0;
    for bi in 0..f.blocks.len() {
        let b = BlockId::new(bi);
        let n = preds[bi].len();
        if b == f.entry || n < 2 {
            continue;
        }
        let Some(chain) = tail(f, &preds, b) else { continue };
        let size: usize = chain.iter().map(|&c| f.blocks[c].insts.len as usize).sum();
        let cost = size.max(1) * (n - 1);
        let small = size <= SPLIT_RETURN_MAX
            && cost <= *budget
            && chain.iter().flat_map(|&c| f.blocks[c].insts.get(&f.value_pool)).all(|&v| {
                !matches!(f.insts[v].kind, Store { .. } | MemCopy { .. } | MemFill { .. } | Opaque { .. } | Exit { .. } | Assign { .. } | Borrow { .. } | Move(_))
            });
        if !small {
            continue;
        }
        *budget -= cost;
        for &p in &preds[bi][1..] {
            let copy = copy_block(f, b, sites, &mut HashMap::new());
            redirect(f, p, b, copy);
            made += 1;
        }
    }
    made
}

/// The blocks from `b` to the end of the function, if that is a straight line:
/// each one after `b` reached only from the one before, the last one leaving
/// the function. Blocks copied since `preds` was made aren't followed.
fn tail(f: &Function, preds: &Preds, b: BlockId) -> Option<Vec<BlockId>> {
    let mut chain = vec![b];
    loop {
        match f.blocks[*chain.last().unwrap()].term {
            Terminator::Return(_) | Terminator::Unreachable | Terminator::TailCall { .. } => return Some(chain),
            Terminator::Jump { to, .. } if chain.len() < 8 && to != f.entry && !chain.contains(&to) && to.index() + 1 < preds.start.len() && preds[to.index()].len() == 1 => chain.push(to),
            _ => return None,
        }
    }
}

/// A copy of block `b` and the blocks it jumps on to (`tail`), with fresh values
/// (its own parameters), unreachable until an edge is pointed at it. Its calls
/// are added to `sites`.
fn copy_block(f: &mut Function, b: BlockId, sites: &mut Vec<(Site, Site)>, map: &mut HashMap<ValueId, ValueId>) -> BlockId {
    let blk = &f.blocks[b];
    let (params, insts, term) = (blk.params.get(&f.value_pool).to_vec(), blk.insts.get(&f.value_pool).to_vec(), blk.term);
    // a list operand gets a list of its own, so mapping it doesn't touch the original's
    let own_list = |f: &mut Function, l: &mut ListRef| {
        let start = f.value_pool.len() as u32;
        f.value_pool.extend_from_within(l.start as usize..(l.start + l.len) as usize);
        l.start = start;
    };
    let fresh = |f: &mut Function, v: ValueId, map: &mut HashMap<ValueId, ValueId>| {
        let mut inst = f.insts[v];
        if let InstKind::Aggregate { fields: l, .. } | InstKind::Call { args: l, .. } = &mut inst.kind {
            own_list(f, l);
        }
        map_operands(&mut inst.kind, &mut f.value_pool, |x| {
            if let Some(&y) = map.get(x) {
                *x = y;
            }
        });
        let id = f.insts.push(inst);
        let at = f.origin.get(v.index()).copied().unwrap_or(0);
        f.origin.push(at);
        map.insert(v, id);
        id
    };
    let new_params: Vec<ValueId> = params.iter().map(|&v| fresh(f, v, map)).collect();
    let mut new_insts = Vec::with_capacity(insts.len());
    for &v in &insts {
        let id = fresh(f, v, map);
        if matches!(f.insts[v].kind, InstKind::Call { .. }) {
            sites.push((Site::Call(v), Site::Call(id)));
        }
        new_insts.push(id);
    }
    let list = |f: &mut Function, vs: &[ValueId]| {
        let start = f.value_pool.len() as u32;
        f.value_pool.extend_from_slice(vs);
        ListRef { start, len: vs.len() as u32 }
    };
    let params = list(f, &new_params);
    let insts = list(f, &new_insts);
    let term = match term {
        Terminator::Return(Some(v)) => Terminator::Return(Some(map.get(&v).copied().unwrap_or(v))),
        Terminator::Jump { to, mut args } => {
            own_list(f, &mut args);
            for x in &mut f.value_pool[args.start as usize..(args.start + args.len) as usize] {
                *x = map.get(x).copied().unwrap_or(*x);
            }
            Terminator::Jump { to: copy_block(f, to, sites, map), args }
        }
        Terminator::TailCall { callee, mut args } => {
            own_list(f, &mut args);
            for x in &mut f.value_pool[args.start as usize..(args.start + args.len) as usize] {
                *x = map.get(x).copied().unwrap_or(*x);
            }
            Terminator::TailCall { callee: map.get(&callee).copied().unwrap_or(callee), args }
        }
        t => t,
    };
    let copy = f.blocks.push(Block { insts, params, term });
    if matches!(term, Terminator::TailCall { .. }) {
        sites.push((Site::Tail(b), Site::Tail(copy)));
    }
    copy
}
/// Point every edge from `p` into `from` at `to` instead (a block with the same
/// parameters).
fn redirect(f: &mut Function, p: BlockId, from: BlockId, to: BlockId) {
    match &mut f.blocks[p].term {
        Terminator::Jump { to: t, .. } => {
            if *t == from {
                *t = to;
            }
        }
        Terminator::Branch { t, f: e, .. } => {
            for x in [t, e] {
                if *x == from {
                    *x = to;
                }
            }
        }
        Terminator::Switch { table, default, .. } => {
            if *default == from {
                *default = to;
            }
            let table = *table;
            for x in &mut f.value_pool[table.start as usize..(table.start + table.len) as usize] {
                if BlockId::from_value(*x) == from {
                    *x = to.as_value();
                }
            }
        }
        _ => {}
    }
}

/// Turn each branch whose condition is a constant into a jump: `xor ecx, ecx;
/// test ecx, ecx; jnz junk` never goes to `junk`, which obfuscators fill with
/// bytes meant to derail a disassembler (an opaque predicate). The condition is
/// evaluated through constants, casts, arithmetic and comparisons; anything that
/// depends on an input stays a branch. Returns the number of branches folded.
/// The block no longer branched to keeps its code but loses the edge.
pub fn fold_branches(f: &mut Function) -> usize {
    let mut n = 0;
    for b in 0..f.blocks.len() {
        let b = BlockId::new(b);
        let Terminator::Branch { c, t, f: e, args } = f.blocks[b].term else { continue };
        let Some(v) = eval(f, c, 16) else { continue };
        let tn = f.blocks[t].params.len;
        let (to, start, len) = if v != 0 { (t, args.start, tn) } else { (e, args.start + tn, f.blocks[e].params.len) };
        f.blocks[b].term = Terminator::Jump { to, args: ListRef { start, len } };
        n += 1;
    }
    n
}

/// The value of `v` if it is a constant expression, truncated to its width.
fn eval(f: &Function, v: ValueId, depth: u32) -> Option<u64> {
    let width = |v: ValueId| match f.insts[v].ty {
        TyId::B1 | TyId::BOOL => 8,
        TyId::B2 => 16,
        TyId::B4 => 32,
        TyId::B8 | TyId::PTR => 64,
        _ => 0,
    };
    let mask = |bits: u32, x: u64| if bits >= 64 { x } else { x & ((1u64 << bits) - 1) };
    let sext = |bits: u32, x: u64| if bits >= 64 { x } else { ((x << (64 - bits)) as i64 >> (64 - bits)) as u64 };
    let w = width(v);
    if w == 0 || depth == 0 {
        return None;
    }
    let go = |x: ValueId| eval(f, x, depth - 1);
    let r = match f.insts[v].kind {
        InstKind::Const(k) => f.consts[k.index()] as u64,
        InstKind::Cast { kind: CastKind::Trunc | CastKind::ZExt, v: x } => go(x)?,
        InstKind::Cast { kind: CastKind::SExt, v: x } => sext(width(x), go(x)?),
        InstKind::Un { op: UnOp::Not, v: x } => !go(x)?,
        InstKind::Un { op: UnOp::Neg, v: x } => go(x)?.wrapping_neg(),
        InstKind::Bin { op, lhs, rhs } => {
            let (a, b) = (go(lhs)?, go(rhs)?);
            match op {
                BinOp::Add => a.wrapping_add(b),
                BinOp::Sub => a.wrapping_sub(b),
                BinOp::Mul => a.wrapping_mul(b),
                BinOp::And => a & b,
                BinOp::Or => a | b,
                BinOp::Xor => a ^ b,
                BinOp::Shl => a.checked_shl(b as u32).unwrap_or(0),
                BinOp::LShr => mask(w, a).checked_shr(b as u32).unwrap_or(0),
                _ => return None,
            }
        }
        InstKind::Cmp { cc, lhs, rhs } => {
            let bits = width(lhs);
            if bits == 0 {
                return None;
            }
            let (a, b) = (mask(bits, go(lhs)?), mask(bits, go(rhs)?));
            let (sa, sb) = (sext(bits, a) as i64, sext(bits, b) as i64);
            (match cc {
                Cond::Eq => a == b,
                Cond::Ne => a != b,
                Cond::Ult => a < b,
                Cond::Ule => a <= b,
                Cond::Ugt => a > b,
                Cond::Uge => a >= b,
                Cond::Slt => sa < sb,
                Cond::Sle => sa <= sb,
                Cond::Sgt => sa > sb,
                Cond::Sge => sa >= sb,
            }) as u64
        }
        _ => return None,
    };
    Some(mask(w, r))
}
