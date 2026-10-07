//! Control-flow graph views over a `Function`: predecessors, reverse postorder and
//! dominators. Built on demand from the terminators, in flat arrays.
use crate::ir::*;

pub struct Cfg {
    /// Predecessors of block `b` are `preds[pred_start[b]..pred_start[b + 1]]`.
    /// A block that branches to the same target on both edges is listed twice.
    pred_start: Vec<u32>,
    preds: Vec<BlockId>,
    /// Reachable blocks in reverse postorder; the entry is first.
    pub rpo: Vec<BlockId>,
    /// Position of each block in `rpo`, or `u32::MAX` if unreachable.
    pub rpo_index: Vec<u32>,
    /// Immediate dominator; `None` for the entry and for unreachable blocks.
    pub idom: Vec<Option<BlockId>>,
}

impl Cfg {
    pub fn new(f: &Function) -> Cfg {
        Cfg::from_succs(f.blocks.len(), f.entry, |b| f.blocks[b].term.successors())
    }

    /// The CFG of any graph of `n` blocks with at most two successors each.
    /// The structurer uses it on the graph it rewrites.
    pub fn from_succs(n: usize, entry: BlockId, succ: impl Fn(BlockId) -> [Option<BlockId>; 2]) -> Cfg {
        // Predecessors, counting sort style.
        let mut count = vec![0u32; n + 1];
        for b in (0..n).map(BlockId::new) {
            for s in succ(b).into_iter().flatten() {
                count[s.index() + 1] += 1;
            }
        }
        for i in 0..n {
            count[i + 1] += count[i];
        }
        let pred_start = count.clone();
        let mut fill = count;
        let mut preds = vec![entry; pred_start[n] as usize];
        for b in (0..n).map(BlockId::new) {
            for s in succ(b).into_iter().flatten() {
                preds[fill[s.index()] as usize] = b;
                fill[s.index()] += 1;
            }
        }

        // Reverse postorder with an explicit stack.
        let mut post = Vec::with_capacity(n);
        let mut seen = vec![false; n];
        let mut stack: Vec<(BlockId, u8)> = vec![(entry, 0)];
        seen[entry.index()] = true;
        while let Some(&mut (b, ref mut next)) = stack.last_mut() {
            let succ = succ(b);
            if (*next as usize) < succ.len() {
                let s = succ[*next as usize];
                *next += 1;
                if let Some(s) = s {
                    if !std::mem::replace(&mut seen[s.index()], true) {
                        stack.push((s, 0));
                    }
                }
            } else {
                post.push(b);
                stack.pop();
            }
        }
        post.reverse();
        let rpo = post;
        let mut rpo_index = vec![u32::MAX; n];
        for (i, b) in rpo.iter().enumerate() {
            rpo_index[b.index()] = i as u32;
        }

        let mut cfg = Cfg { pred_start, preds, rpo, rpo_index, idom: vec![None; n] };
        cfg.compute_dominators(entry);
        cfg
    }

    pub fn preds(&self, b: BlockId) -> &[BlockId] {
        &self.preds[self.pred_start[b.index()] as usize..self.pred_start[b.index() + 1] as usize]
    }

    pub fn reachable(&self, b: BlockId) -> bool {
        self.rpo_index[b.index()] != u32::MAX
    }

    /// Does `a` dominate `b`? (Every block dominates itself.)
    pub fn dominates(&self, a: BlockId, mut b: BlockId) -> bool {
        loop {
            if a == b {
                return true;
            }
            match self.idom[b.index()] {
                Some(d) => b = d,
                None => return false,
            }
        }
    }

    /// Cooper, Harvey & Kennedy, "A Simple, Fast Dominance Algorithm".
    fn compute_dominators(&mut self, entry: BlockId) {
        let mut idom: Vec<Option<u32>> = vec![None; self.rpo.len()]; // in rpo indices
        idom[0] = Some(0);
        let mut changed = true;
        while changed {
            changed = false;
            for i in 1..self.rpo.len() {
                let b = self.rpo[i];
                let mut new: Option<u32> = None;
                for &p in self.preds(b) {
                    let pi = self.rpo_index[p.index()];
                    if pi == u32::MAX || idom[pi as usize].is_none() {
                        continue;
                    }
                    new = Some(match new {
                        None => pi,
                        Some(mut x) => {
                            let mut y = pi;
                            while x != y {
                                while x > y { x = idom[x as usize].unwrap(); }
                                while y > x { y = idom[y as usize].unwrap(); }
                            }
                            x
                        }
                    });
                }
                if new.is_some() && idom[i] != new {
                    idom[i] = new;
                    changed = true;
                }
            }
        }
        for (i, d) in idom.iter().enumerate() {
            let b = self.rpo[i];
            if b != entry {
                self.idom[b.index()] = d.map(|d| self.rpo[d as usize]);
            }
        }
    }
}
