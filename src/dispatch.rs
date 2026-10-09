//! One entry per irreducible loop.
//!
//! A cycle that can be entered at more than one block has no header, so the
//! structurer emits it as a `loop { match bb { .. } }` state machine where
//! every edge between its blocks, not only the edges into it, sets `bb` and
//! goes round the dispatch: an indirect jump, with every variable live across
//! it. gcc makes such loops out of ordinary C ones (a loop rotated so that it
//! is entered in the middle, or a `switch` in a `for (;;)` whose first case is
//! also jumped to directly), and they are where the decompiled code of zlib's
//! `deflate_slow` and `inflate` spent most of its time.
//!
//! So each such region gets a header of its own: a new block `D` that takes a
//! selector and the parameters of every entry (one slot per register, shared
//! by the entries), and switches on the selector to the entry. An edge into
//! entry `k` goes to `D` instead, with `k` and its arguments in entry `k`'s
//! slots (the other slots get `Undef`), through a small block `T_k` that adds
//! them, so every predecessor keeps its edge as it was. Each entry has one
//! `T_k` for the edges from outside the region and one for those inside it,
//! or the `T_k`s would be entries of the new loop themselves. The entries then
//! take their values from `D`'s parameters, which dominate them. The region is
//! now a natural loop headed by `D`: only edges into its entries go through
//! the selector test (`D`'s `Switch`), and everything else in it is
//! structured as usual.
//!
//! Only edges into entries pay for the selector, so a region whose blocks are
//! mostly entries gains little, but loses nothing either. A region is left
//! alone if it contains the function's entry, or if some value would no longer
//! be defined on every path to its use (a value computed in the region and used
//! at one entry after leaving the region and coming back in).
use crate::cfg::Cfg;
use crate::ir::*;
use crate::opt::{redirect, rewrite_uses};
use crate::structure::{irreducible_regions, reducible};
use crate::verify::for_each_operand;
use std::collections::HashMap;

/// Most regions changed in one function: each change recomputes the
/// dominator tree, and a region's new header can reveal another irreducible
/// region inside it, which a later step handles.
const MAX_REGIONS: usize = 64;

/// Give every irreducible region a single entry, where that is safe. Returns
/// how many regions were changed; the caller runs `opt::clean` after.
pub fn single_entry(f: &mut Function) -> usize {
    let mut done = 0;
    let mut refused: Vec<Vec<BlockId>> = Vec::new();
    while done < MAX_REGIONS {
        let cfg = Cfg::new(f);
        if reducible(f, &cfg) {
            break;
        }
        let next = irreducible_regions(f, &cfg).into_iter().find(|r| !refused.contains(r));
        let Some(r) = next else { break };
        if one_region(f, &cfg, &r) {
            done += 1;
        } else {
            refused.push(r);
        }
    }
    done
}

fn one_region(f: &mut Function, cfg: &Cfg, region: &[BlockId]) -> bool {
    if region.contains(&f.entry) {
        return false;
    }
    let inside = |b: BlockId| region.contains(&b);
    let entries: Vec<BlockId> =
        region.iter().copied().filter(|&b| cfg.preds(b).iter().any(|&p| cfg.reachable(p) && !inside(p))).collect();
    if entries.len() < 2 || !still_dominated(f, cfg, region, &entries) {
        return false;
    }

    let k = entries.len();
    let at = |f: &Function, b: BlockId| {
        f.blocks[b].insts.get(&f.value_pool).first().map_or(0, |v| f.origin[v.index()])
    };
    let new_inst = |f: &mut Function, kind: InstKind, ty: TyId, origin: u64| {
        let id = f.insts.push(Inst { kind, ty });
        f.origin.push(origin);
        id
    };
    let list = |f: &mut Function, vs: &[ValueId]| {
        let start = f.value_pool.len() as u32;
        f.value_pool.extend_from_slice(vs);
        ListRef { start, len: vs.len() as u32 }
    };
    let origin = at(f, entries[0]);

    // D: the selector, then a slot for each parameter of the entries. Entries
    // share a slot for the same register (and type): one entry's values are
    // in the slots at a time, and most of them are the same register's value
    // on every path.
    let sel = new_inst(f, InstKind::BlockParam(u8::MAX), TyId::B4, origin);
    let mut slots: Vec<(InstKind, TyId, usize, ValueId)> = Vec::new(); // kind, type, occurrence, param
    let mut slot_of: Vec<Vec<usize>> = Vec::with_capacity(k); // entry -> its parameters' slots
    for &e in &entries {
        let ps = f.blocks[e].params.get(&f.value_pool).to_vec();
        let mut mine = Vec::with_capacity(ps.len());
        for p in ps {
            let Inst { kind, ty } = f.insts[p];
            let same = |s: &(InstKind, TyId, usize, ValueId)| {
                s.1 == ty && matches!((s.0, kind), (InstKind::BlockParam(a), InstKind::BlockParam(b)) if a == b)
            };
            let nth = mine.iter().filter(|&&j: &&usize| same(&slots[j])).count();
            let j = match slots.iter().position(|s| same(s) && s.2 == nth) {
                Some(j) => j,
                None => {
                    let o = f.origin[p.index()];
                    slots.push((kind, ty, nth, new_inst(f, kind, ty, o)));
                    slots.len() - 1
                }
            };
            mine.push(j);
        }
        slot_of.push(mine);
    }
    let d_params: Vec<ValueId> = std::iter::once(sel).chain(slots.iter().map(|s| s.3)).collect();
    let table: Vec<ValueId> = entries[..k - 1].iter().map(|b| b.as_value()).collect();
    let d = {
        let params = list(f, &d_params);
        let table = list(f, &table);
        let insts = list(f, &[]);
        f.blocks.push(Block { insts, params, term: Terminator::Switch { v: sel, table, default: entries[k - 1] } })
    };

    let mut repl: Vec<Option<ValueId>> = vec![None; f.insts.len()];
    for (i, &e) in entries.iter().enumerate() {
        // T_i: entry i's parameters in, the selector and every slot out. One
        // for the edges from outside the region and one for those inside it,
        // so that `D` is the loop's only entry.
        let ps = f.blocks[e].params.get(&f.value_pool).to_vec();
        let preds: Vec<BlockId> = cfg.preds(e).to_vec();
        for from_inside in [false, true] {
            if !preds.iter().any(|&p| inside(p) == from_inside) {
                continue;
            }
            let mut t_params = Vec::with_capacity(ps.len());
            for &p in &ps {
                let Inst { kind, ty } = f.insts[p];
                let o = f.origin[p.index()];
                t_params.push(new_inst(f, kind, ty, o));
            }
            let c = ConstId::new(f.consts.len());
            f.consts.push(i as u128);
            let o = at(f, e);
            let mut t_insts = vec![new_inst(f, InstKind::Const(c), TyId::B4, o)];
            let mut args = vec![t_insts[0]];
            for (j, s) in slots.iter().enumerate() {
                match slot_of[i].iter().position(|&x| x == j) {
                    Some(n) => args.push(t_params[n]),
                    None => {
                        let u = new_inst(f, InstKind::Undef, s.1, o);
                        t_insts.push(u);
                        args.push(u);
                    }
                }
            }
            let t = {
                let params = list(f, &t_params);
                let insts = list(f, &t_insts);
                let args = list(f, &args);
                f.blocks.push(Block { insts, params, term: Terminator::Jump { to: d, args } })
            };
            for &p in preds.iter().filter(|&&p| inside(p) == from_inside) {
                redirect(f, p, e, t);
            }
        }
        // the entry's own parameters are now `D`'s
        for (&p, &j) in ps.iter().zip(&slot_of[i]) {
            repl[p.index()] = Some(slots[j].3);
        }
        f.blocks[e].params = ListRef { start: 0, len: 0 };
    }
    repl.resize(f.insts.len(), None);
    rewrite_uses(f, &repl);
    true
}

/// After the change, would every use still be dominated by its definition?
/// Checked on the graph the change would make: `D` (node `n`) reached from
/// every predecessor of an entry, the entries reached only from `D`, and the
/// entries' parameters defined at `D`. (`T_k` sits on an edge and changes
/// nothing about dominance.)
fn still_dominated(f: &Function, cfg: &Cfg, region: &[BlockId], entries: &[BlockId]) -> bool {
    let n = f.blocks.len();
    let d = n;
    let is_entry = |b: BlockId| entries.contains(&b);
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    for &b in &cfg.rpo {
        for s in f.blocks[b].term.successors(&f.value_pool) {
            succ[b.index()].push(if is_entry(s) { d } else { s.index() });
        }
    }
    succ[d] = entries.iter().map(|e| e.index()).collect();
    let idom = dominators(f.entry.index(), &succ);
    let dominates = |a: usize, mut b: usize| loop {
        if a == b {
            return true;
        }
        match idom[b] {
            Some(x) if x != b => b = x,
            _ => return false,
        }
    };

    // Where each value is defined.
    let mut def: HashMap<ValueId, usize> = HashMap::new();
    for &b in &cfg.rpo {
        let blk = &f.blocks[b];
        let pb = if is_entry(b) { d } else { b.index() };
        for &p in blk.params.get(&f.value_pool) {
            def.insert(p, pb);
        }
        for &v in blk.insts.get(&f.value_pool) {
            def.insert(v, b.index());
        }
    }
    // Only uses in blocks whose dominators can change: those the region reaches.
    let mut seen = vec![false; n];
    let mut work: Vec<BlockId> = region.to_vec();
    for &b in &work {
        seen[b.index()] = true;
    }
    while let Some(b) = work.pop() {
        for s in f.blocks[b].term.successors(&f.value_pool) {
            if !std::mem::replace(&mut seen[s.index()], true) {
                work.push(s);
            }
        }
    }
    let mut ok = true;
    for &b in &cfg.rpo {
        if !seen[b.index()] {
            continue;
        }
        let blk = &f.blocks[b];
        let mut check = |v: ValueId| {
            if let Some(&x) = def.get(&v) {
                ok &= dominates(x, b.index());
            }
        };
        for &v in blk.insts.get(&f.value_pool) {
            for_each_operand(f.insts[v].kind, f, &mut check);
        }
        crate::emit::term_operands(f, blk.term, &mut check);
    }
    ok
}

/// Immediate dominators of a graph given by successor lists (Cooper, Harvey &
/// Kennedy); `None` for nodes `entry` doesn't reach, the entry its own.
fn dominators(entry: usize, succ: &[Vec<usize>]) -> Vec<Option<usize>> {
    let n = succ.len();
    // reverse postorder
    let mut order = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    let mut stack = vec![(entry, 0usize)];
    seen[entry] = true;
    while let Some((b, i)) = stack.last_mut() {
        if let Some(&s) = succ[*b].get(*i) {
            *i += 1;
            if !std::mem::replace(&mut seen[s], true) {
                stack.push((s, 0));
            }
        } else {
            order.push(*b);
            stack.pop();
        }
    }
    order.reverse();
    let mut index = vec![usize::MAX; n];
    for (i, &b) in order.iter().enumerate() {
        index[b] = i;
    }
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, ss) in succ.iter().enumerate() {
        if index[b] != usize::MAX {
            for &s in ss {
                preds[s].push(b);
            }
        }
    }
    let mut idom: Vec<Option<usize>> = vec![None; n];
    idom[entry] = Some(entry);
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &order[1..] {
            let mut new: Option<usize> = None;
            for &p in &preds[b] {
                if idom[p].is_none() {
                    continue;
                }
                new = Some(match new {
                    None => p,
                    Some(mut x) => {
                        let mut y = p;
                        while x != y {
                            while index[x] > index[y] {
                                x = idom[x].unwrap();
                            }
                            while index[y] > index[x] {
                                y = idom[y].unwrap();
                            }
                        }
                        x
                    }
                });
            }
            if new.is_some() && idom[b] != new {
                idom[b] = new;
                changed = true;
            }
        }
    }
    idom
}
