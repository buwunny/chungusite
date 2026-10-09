//! Control-flow structuring: turn a CFG into nested `if`/`else`, `while`, `loop`,
//! labeled blocks, `break` and `continue`, instead of a `loop { match bb { .. } }`
//! state machine.
//!
//! The construction is Ramsey's "Beyond Relooper" (ICFP 2022): walk the dominator
//! tree; a block's dominator-tree children that are join points (two or more
//! forward in-edges) are emitted after it, each wrapped in a labeled block that
//! branches `break` out of; a loop header wraps its subtree in a `loop` that back
//! edges `continue`; every other edge goes to a block with a single predecessor,
//! which is emitted inline at the branch. That is correct for any reducible CFG.
//! Rust's labeled blocks (`'b: { .. break 'b; }`) make it a direct translation.
//!
//! It runs on a view of the CFG with two changes:
//!
//! * **Short-circuit conditions.** A block that does nothing but branch (all its
//!   values are inlined into the condition), whose only predecessor branches to it
//!   and to one of its own targets, is folded into that predecessor's condition:
//!   `if a { T } else if b { T } else { F }` becomes `if a || b { T } else { F }`.
//!   Without this, `T` has two predecessors and needs a labeled block.
//! * **Irreducible regions.** A cycle with more than one entry has no nesting.
//!   Each such strongly connected component (found by Steensgaard's loop-nesting
//!   decomposition, so as small as it can be) becomes one node of the view, and is
//!   emitted as a `loop { match bb { .. } }` over just its own blocks: edges into
//!   it set `bb` to the block they enter. The rest of the function is structured
//!   around it.
//!
//! The raw result is correct but noisy, so `tidy` then removes what isn't needed:
//! a `break`/`continue` that control would reach anyway by falling off the end,
//! labeled blocks nothing breaks out of, and `if c { A } else { B }` where `A`
//! always leaves (`if c { A } B`); turns a loop that starts with
//! `if c { break; }` into `while !c`; and moves an arm that only assigns a join's
//! variables in front of the `if` (`x = b; if c { A; x = a; }`), so the `if` has
//! no `else` where the source most likely had none. Labels are printed only where a plain
//! `break`/`continue` would not mean the same thing.
use crate::cfg::Cfg;
use crate::emit::mentions;
use crate::expr::{self, not};
use crate::ir::*;
use std::fmt::Write;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Label {
    /// The `loop` (or `while`) headed by this block.
    Loop(u32),
    /// The labeled block that this block's code follows.
    Block(u32),
}

#[derive(Debug, Clone)]
pub enum Node {
    /// A statement; control continues after it.
    Line(String),
    /// Assignment of a block's parameters on an edge into it (`v3 = v1;`,
    /// `(v3, v4) = (v1, 0_u64);`); control continues after it.
    Copy(String),
    /// A statement that never continues (`return`, `panic!`, `todo!`).
    Exit(String),
    If { c: String, then: Vec<Node>, els: Vec<Node> },
    Loop { head: u32, body: Vec<Node> },
    /// `while c { body }`: made by `tidy` from a `loop` that starts with
    /// `if !c { break; }`.
    While { head: u32, c: String, body: Vec<Node> },
    Block { label: u32, body: Vec<Node> },
    /// `match bb { k => { .. } }` over the blocks of an irreducible region; always
    /// the whole body of its `loop`, so control falling off an arm goes round
    /// again.
    Dispatch { arms: Vec<(u32, Vec<Node>)> },
    /// Leave the labeled block or loop.
    Break(Label),
    /// Next iteration of the loop headed by this block.
    Continue(u32),
}

/// What the structurer needs from the emitter, per block.
pub trait Source {
    /// The block's statements, without its terminator.
    fn stmts(&mut self, b: BlockId) -> Vec<Node>;
    /// Statements that pass `args` to `to`'s parameters.
    fn edge(&mut self, to: BlockId, args: &[ValueId]) -> Vec<Node>;
    /// The condition of a `Branch`, as an expression.
    fn cond(&self, c: ValueId) -> String;
    /// "`v` is one of `cases`", for a `Switch`, as an expression.
    fn case(&self, v: ValueId, cases: &[u64]) -> String;
    /// A terminator without successors (return, tail call, ...), as a statement.
    fn exit(&mut self, b: BlockId) -> Node;
    /// The block has no parameters and no statements: all it computes is inlined
    /// into its terminator.
    fn quiet(&self, b: BlockId) -> bool;
}

/// The variable the dispatch of an irreducible region matches on.
pub const DISPATCH_VAR: &str = "bb";

/// Deepest dominator tree the structurer will recurse into. Past this the caller
/// keeps the state machine, which uses no recursion.
const MAX_DEPTH: usize = 512;

/// Is every cycle entered only through its header? Then every retreating edge
/// (to a block no later in reverse postorder) goes to a block that dominates its
/// source, and is a back edge of a natural loop.
pub fn reducible(f: &Function, cfg: &Cfg) -> bool {
    cfg.rpo.iter().all(|&b| f.blocks[b].term.successors(&f.value_pool).all(|s| back_ok(cfg, b, s)))
}

fn back_ok(cfg: &Cfg, b: BlockId, s: BlockId) -> bool {
    cfg.rpo_index[s.index()] > cfg.rpo_index[b.index()] || cfg.dominates(s, b)
}

/// A structured function body.
pub struct Structured {
    pub nodes: Vec<Node>,
    /// Irreducible regions kept as a `loop { match bb }`.
    pub regions: usize,
}

/// A branch condition built from the conditions of several blocks.
#[derive(Clone, Debug)]
enum Test {
    Leaf(ValueId),
    Not(Box<Test>),
    And(Box<Test>, Box<Test>),
    Or(Box<Test>, Box<Test>),
}

/// A node's terminator in the view the structurer works on.
#[derive(Clone, Debug)]
enum VTerm {
    /// Its block's own terminator, which leaves the function.
    Exit,
    Jump { to: BlockId, args: Vec<ValueId> },
    /// `quiet`: blocks folded into the condition, whose (empty) statements are
    /// emitted first so their inlined values are ready.
    Branch { c: Test, t: BlockId, e: BlockId, ta: Vec<ValueId>, ea: Vec<ValueId>, quiet: Vec<BlockId> },
    /// A jump table: each arm's cases go to its block, any other value to `default`.
    /// Its edges carry no arguments.
    Switch { v: ValueId, default: BlockId, arms: Vec<(BlockId, Vec<u64>)> },
    /// An irreducible region, in reverse postorder.
    Region(Vec<BlockId>),
    /// Folded into another node, or part of a region.
    Gone,
}

/// Structure `f`, or `None` if its CFG nests too deeply.
pub fn structure(f: &Function, cfg: &Cfg, src: &mut dyn Source) -> Option<Structured> {
    let n = f.blocks.len();
    let regions = if reducible(f, cfg) { Vec::new() } else { irreducible_regions(f, cfg) };
    let mut node_of: Vec<BlockId> = (0..n).map(BlockId::new).collect();
    let mut vterm: Vec<VTerm> = (0..n).map(|i| own_term(f, BlockId::new(i))).collect();
    // Skip empty blocks that only jump on: their predecessors go straight on.
    let fwd: Vec<BlockId> = (0..n).map(|i| forward(f, BlockId::new(i), &*src)).collect();
    for t in &mut vterm {
        match t {
            VTerm::Jump { to, .. } => *to = fwd[to.index()],
            VTerm::Branch { t, e, .. } => (*t, *e) = (fwd[t.index()], fwd[e.index()]),
            VTerm::Switch { default, arms, .. } => {
                *default = fwd[default.index()];
                for (t, _) in arms {
                    *t = fwd[t.index()];
                }
            }
            _ => {}
        }
    }
    for (k, r) in regions.iter().enumerate() {
        for &b in r {
            node_of[b.index()] = BlockId::new(n + k);
            vterm[b.index()] = VTerm::Gone;
        }
        vterm.push(VTerm::Region(r.clone()));
    }
    short_circuit(f, cfg, &node_of, &mut vterm, &*src);

    // The view's edges, one per real edge (so a join stays a join).
    let mut succ: Vec<Vec<BlockId>> = vec![Vec::new(); vterm.len()];
    for (i, t) in vterm.iter().enumerate() {
        succ[i] = match t {
            VTerm::Exit | VTerm::Gone => Vec::new(),
            VTerm::Jump { to, .. } => vec![node_of[to.index()]],
            VTerm::Branch { t, e, .. } => vec![node_of[t.index()], node_of[e.index()]],
            VTerm::Switch { default, arms, .. } => {
                std::iter::once(*default).chain(arms.iter().map(|a| a.0)).map(|t| node_of[t.index()]).collect()
            }
            VTerm::Region(r) => {
                let me = BlockId::new(i);
                let out = r.iter().flat_map(|&b| f.blocks[b].term.successors(&f.value_pool));
                out.map(|s| node_of[s.index()]).filter(|&s| s != me).collect()
            }
        };
    }
    let entry = node_of[f.entry.index()];
    let view = Cfg::from_succs(&succ, entry);
    // Collapsing the irreducible regions leaves a reducible graph; if it somehow
    // didn't, the caller keeps the whole-function state machine.
    if !view.rpo.iter().all(|&b| succ[b.index()].iter().all(|&s| back_ok(&view, b, s))) {
        return None;
    }

    let m = vterm.len();
    let mut forward_in = vec![0u32; m];
    let mut loop_head = vec![false; m];
    let mut children: Vec<Vec<BlockId>> = vec![Vec::new(); m];
    let mut depth = vec![0usize; m];
    for &b in &view.rpo {
        if let Some(d) = view.idom[b.index()] {
            children[d.index()].push(b);
            depth[b.index()] = depth[d.index()] + 1;
            if depth[b.index()] > MAX_DEPTH {
                return None;
            }
        }
        for &s in &succ[b.index()] {
            if view.rpo_index[s.index()] > view.rpo_index[b.index()] {
                forward_in[s.index()] += 1;
            } else {
                loop_head[s.index()] = true;
            }
        }
    }
    // Children are pushed in rpo order; nesting wants the latest join outermost.
    for c in &mut children {
        c.reverse();
    }
    let follows = vec![false; m];
    let mut s = Structurer { f, cfg: &view, src, node_of, vterm, forward_in, loop_head, children, follows, n };
    let body = s.tree(entry);
    Some(Structured { nodes: tidy(body), regions: regions.len() })
}

/// Where control really goes when it enters `b`: past any blocks that do nothing
/// but jump on (no parameters, no statements, no arguments).
fn forward(f: &Function, mut b: BlockId, src: &dyn Source) -> BlockId {
    for _ in 0..64 {
        match f.blocks[b].term {
            Terminator::Jump { to, args } if args.len == 0 && to != b && b != f.entry && src.quiet(b) => b = to,
            _ => break,
        }
    }
    b
}

fn own_term(f: &Function, b: BlockId) -> VTerm {
    match f.blocks[b].term {
        Terminator::Jump { to, args } => VTerm::Jump { to, args: args.get(&f.value_pool).to_vec() },
        Terminator::Branch { c, t, f: e, args } => {
            let a = args.get(&f.value_pool);
            let nt = f.blocks[t].params.len as usize;
            VTerm::Branch { c: Test::Leaf(c), t, e, ta: a[..nt].to_vec(), ea: a[nt..].to_vec(), quiet: Vec::new() }
        }
        Terminator::Switch { v, table, default } => {
            let cases = table.get(&f.value_pool);
            let arms = f.blocks[b]
                .term
                .successors(&f.value_pool)
                .skip(1)
                .map(|t| {
                    let ks = (0..cases.len() as u64).filter(|&k| BlockId::from_value(cases[k as usize]) == t).collect();
                    (t, ks)
                })
                .collect();
            VTerm::Switch { v, default, arms }
        }
        _ => VTerm::Exit,
    }
}

/// Fold quiet blocks into the condition of their only predecessor (see the module
/// comment). `a || b || c` takes one round per operand.
fn short_circuit(f: &Function, cfg: &Cfg, node_of: &[BlockId], vterm: &mut [VTerm], src: &dyn Source) {
    let n = f.blocks.len();
    // The node a block was folded into, for "its only predecessor is X".
    let mut owner: Vec<BlockId> = (0..n).map(BlockId::new).collect();
    for &x in &cfg.rpo {
        while let VTerm::Branch { c, t, e, ta, ea, .. } = &vterm[x.index()] {
            let (c, t, e) = (c.clone(), *t, *e);
            let foldable = |y: BlockId, other: BlockId| {
                y != other
                    && y != f.entry
                    && node_of[y.index()] == y
                    && matches!(cfg.preds(y), [p] if owner[p.index()] == x)
                    && src.quiet(y)
            };
            let merged = if foldable(e, t) {
                // c ? S : (c2 ? t2 : e2)
                let VTerm::Branch { c: c2, t: t2, e: e2, ta: ta2, ea: ea2, .. } = &vterm[e.index()] else { break };
                if *t2 == t && ta2 == ta && *e2 != t {
                    Some((e, Test::Or(c.into(), c2.clone().into()), t, *e2, ta.clone(), ea2.clone()))
                } else if *e2 == t && ea2 == ta && *t2 != t {
                    Some((e, Test::Or(c.into(), Test::Not(c2.clone().into()).into()), t, *t2, ta.clone(), ta2.clone()))
                } else {
                    None
                }
            } else if foldable(t, e) {
                // c ? (c2 ? t2 : e2) : S
                let VTerm::Branch { c: c2, t: t2, e: e2, ta: ta2, ea: ea2, .. } = &vterm[t.index()] else { break };
                if *e2 == e && ea2 == ea && *t2 != e {
                    Some((t, Test::And(c.into(), c2.clone().into()), *t2, e, ta2.clone(), ea.clone()))
                } else if *t2 == e && ta2 == ea && *e2 != e {
                    Some((t, Test::And(c.into(), Test::Not(c2.clone().into()).into()), *e2, e, ea2.clone(), ea.clone()))
                } else {
                    None
                }
            } else {
                None
            };
            let Some((y, c, t, e, ta, ea)) = merged else { break };
            let VTerm::Branch { quiet, .. } = std::mem::replace(&mut vterm[x.index()], VTerm::Gone) else { unreachable!() };
            let mut quiet = quiet;
            quiet.push(y);
            vterm[x.index()] = VTerm::Branch { c, t, e, ta, ea, quiet };
            vterm[y.index()] = VTerm::Gone;
            owner[y.index()] = x;
        }
    }
}

/// The irreducible strongly connected components, smallest first by nesting:
/// Steensgaard's decomposition. Find the SCCs; one with a single entry is a natural
/// loop, so drop the edges back to its header and look inside it again; one with
/// several entries is a region.
fn irreducible_regions(f: &Function, cfg: &Cfg) -> Vec<Vec<BlockId>> {
    let n = f.blocks.len();
    let mut cut = vec![false; n]; // headers whose in-edges inside their loop are gone
    let mut in_set = vec![false; n];
    let mut out = Vec::new();
    let mut work: Vec<Vec<BlockId>> = vec![cfg.rpo.clone()];
    while let Some(set) = work.pop() {
        for &b in &set {
            in_set[b.index()] = true;
        }
        let sccs = sccs(&set, |b| {
            f.blocks[b].term.successors(&f.value_pool).filter(|s| in_set[s.index()] && !cut[s.index()]).collect()
        });
        for &b in &set {
            in_set[b.index()] = false;
        }
        for mut c in sccs.into_iter().filter(|c| c.len() > 1) {
            for &b in &c {
                in_set[b.index()] = true;
            }
            let entries: Vec<BlockId> = c
                .iter()
                .copied()
                .filter(|&b| b == f.entry || cfg.preds(b).iter().any(|p| cfg.reachable(*p) && !in_set[p.index()]))
                .collect();
            for &b in &c {
                in_set[b.index()] = false;
            }
            if let [h] = entries[..] {
                cut[h.index()] = true;
                work.push(c);
            } else {
                c.sort_by_key(|b| cfg.rpo_index[b.index()]);
                out.push(c);
            }
        }
    }
    out
}

/// Tarjan's strongly connected components of the subgraph on `set`.
fn sccs(set: &[BlockId], succ: impl Fn(BlockId) -> Vec<BlockId>) -> Vec<Vec<BlockId>> {
    use std::collections::HashMap;
    let pos: HashMap<BlockId, usize> = set.iter().enumerate().map(|(i, &b)| (b, i)).collect();
    let m = set.len();
    let edges: Vec<Vec<usize>> = set.iter().map(|&b| succ(b).iter().map(|s| pos[s]).collect()).collect();
    let (mut index, mut low, mut on) = (vec![usize::MAX; m], vec![0; m], vec![false; m]);
    let (mut stack, mut out, mut next) = (Vec::new(), Vec::new(), 0);
    for root in 0..m {
        if index[root] != usize::MAX {
            continue;
        }
        let mut call: Vec<(usize, usize)> = vec![(root, 0)];
        index[root] = next;
        low[root] = next;
        next += 1;
        stack.push(root);
        on[root] = true;
        while let Some(&mut (v, ref mut k)) = call.last_mut() {
            if let Some(&w) = edges[v].get(*k) {
                *k += 1;
                if index[w] == usize::MAX {
                    index[w] = next;
                    low[w] = next;
                    next += 1;
                    stack.push(w);
                    on[w] = true;
                    call.push((w, 0));
                } else if on[w] {
                    low[v] = low[v].min(index[w]);
                }
                continue;
            }
            call.pop();
            if let Some(&(u, _)) = call.last() {
                low[u] = low[u].min(low[v]);
            }
            if low[v] == index[v] {
                let mut c = Vec::new();
                loop {
                    let w = stack.pop().unwrap();
                    on[w] = false;
                    c.push(set[w]);
                    if w == v {
                        break;
                    }
                }
                out.push(c);
            }
        }
    }
    out
}

struct Structurer<'a, 'b> {
    f: &'a Function,
    /// The view: blocks, folded blocks (unreachable) and region nodes (`n..`).
    cfg: &'a Cfg,
    src: &'b mut dyn Source,
    /// The view node each block belongs to: itself, or its region.
    node_of: Vec<BlockId>,
    vterm: Vec<VTerm>,
    forward_in: Vec<u32>,
    loop_head: Vec<bool>,
    /// Dominator-tree children, latest in reverse postorder first.
    children: Vec<Vec<BlockId>>,
    /// Loop exits emitted after their loop: edges to them are `break`s.
    follows: Vec<bool>,
    /// Number of blocks; region `k` is node `n + k`.
    n: usize,
}

impl Structurer<'_, '_> {
    fn is_join(&self, b: BlockId) -> bool {
        self.forward_in[b.index()] >= 2
    }

    /// Nodes of the natural loop headed by `h`: those that reach a back edge to
    /// `h` without passing through `h`.
    fn loop_body(&self, h: BlockId) -> Vec<bool> {
        let mut inside = vec![false; self.vterm.len()];
        inside[h.index()] = true;
        let hi = self.cfg.rpo_index[h.index()];
        let reachable = |p: &BlockId| self.cfg.reachable(*p);
        let mut stack: Vec<BlockId> = self.cfg.preds(h).iter().copied().filter(reachable).collect();
        stack.retain(|p| self.cfg.rpo_index[p.index()] >= hi); // back edges
        while let Some(b) = stack.pop() {
            if !std::mem::replace(&mut inside[b.index()], true) {
                stack.extend(self.cfg.preds(b).iter().copied().filter(reachable));
            }
        }
        inside
    }

    /// `b` and everything it dominates.
    fn tree(&mut self, b: BlockId) -> Vec<Node> {
        let kids = self.children[b.index()].clone();
        if !self.loop_head[b.index()] {
            let joins: Vec<BlockId> = kids.into_iter().filter(|&c| self.is_join(c)).collect();
            return self.within(b, &joins);
        }
        // A loop. Children outside it are where the loop exits to: emit them after
        // the loop, so leaving it is a `break` rather than code nested inside it.
        let inside = self.loop_body(b);
        let (kids_in, exits): (Vec<BlockId>, Vec<BlockId>) = kids.into_iter().partition(|c| inside[c.index()]);
        let joins: Vec<BlockId> = kids_in.into_iter().filter(|&c| self.is_join(c)).collect();
        for &y in &exits {
            self.follows[y.index()] = true;
        }
        let body = self.within(b, &joins);
        let mut out = vec![Node::Loop { head: b.index() as u32, body }];
        // `exits` is latest first, and the latest one is the outermost block.
        for &y in exits.iter().rev() {
            out = vec![Node::Block { label: y.index() as u32, body: out }];
            out.extend(self.tree(y));
        }
        out
    }

    /// `b`'s code, followed by its join children `joins` (latest first).
    fn within(&mut self, b: BlockId, joins: &[BlockId]) -> Vec<Node> {
        if let Some((&y, rest)) = joins.split_first() {
            let inner = self.within(b, rest);
            let mut out = vec![Node::Block { label: y.index() as u32, body: inner }];
            out.extend(self.tree(y));
            return out;
        }
        match self.vterm[b.index()].clone() {
            VTerm::Region(blocks) => self.region(b, &blocks),
            VTerm::Jump { to, args } => {
                let mut out = self.src.stmts(b);
                out.extend(self.branch(b, to, &args));
                out
            }
            VTerm::Switch { v, default, arms } => {
                let mut out = self.src.stmts(b);
                out.extend(self.switch(v, default, &arms, |this, t| this.branch(b, t, &[])));
                out
            }
            VTerm::Branch { c, t, e, ta, ea, quiet } => {
                let mut out = self.src.stmts(b);
                for y in quiet {
                    out.extend(self.src.stmts(y));
                }
                let then = self.branch(b, t, &ta);
                let els = self.branch(b, e, &ea);
                out.push(Node::If { c: self.test(&c), then, els });
                out
            }
            VTerm::Exit => {
                let mut out = self.src.stmts(b);
                out.push(self.src.exit(b));
                out
            }
            VTerm::Gone => unreachable!("folded block reached"),
        }
    }

    fn test(&self, t: &Test) -> String {
        match t {
            Test::Leaf(c) => self.src.cond(*c),
            Test::Not(x) => expr::not(&self.test(x)),
            Test::And(a, b) => format!("{} && {}", expr::logic(self.test(a), true), expr::logic(self.test(b), true)),
            Test::Or(a, b) => format!("{} || {}", expr::logic(self.test(a), false), expr::logic(self.test(b), false)),
        }
    }

    /// Control passes from node `from` to block `to`.
    fn branch(&mut self, from: BlockId, to: BlockId, args: &[ValueId]) -> Vec<Node> {
        let mut out = self.src.edge(to, args);
        let node = self.node_of[to.index()];
        if node.index() >= self.n {
            out.push(Node::Line(format!("{DISPATCH_VAR} = {};", to.index())));
        }
        if self.cfg.rpo_index[node.index()] <= self.cfg.rpo_index[from.index()] {
            out.push(Node::Continue(node.index() as u32));
        } else if self.is_join(node) || self.follows[node.index()] {
            out.push(Node::Break(Label::Block(node.index() as u32)));
        } else {
            out.extend(self.tree(node));
        }
        out
    }

    /// An irreducible region `r`: a loop around a `match` with an arm per block.
    /// Edges between its blocks set the dispatch variable and go round again.
    fn region(&mut self, r: BlockId, blocks: &[BlockId]) -> Vec<Node> {
        // A label of its own: `r` may also head a loop that encloses this one.
        let me = (self.vterm.len() + r.index()) as u32;
        let mut arms = Vec::new();
        for &b in blocks {
            let mut body = self.src.stmts(b);
            match own_term(self.f, b) {
                VTerm::Jump { to, args } => body.extend(self.go(r, me, to, &args)),
                VTerm::Branch { c, t, e, ta, ea, .. } => {
                    let then = self.go(r, me, t, &ta);
                    let els = self.go(r, me, e, &ea);
                    body.push(Node::If { c: self.test(&c), then, els });
                }
                VTerm::Switch { v, default, arms } => {
                    body.extend(self.switch(v, default, &arms, |this, t| this.go(r, me, t, &[])))
                }
                _ => body.push(self.src.exit(b)),
            }
            arms.push((b.index() as u32, body));
        }
        vec![Node::Loop { head: me, body: vec![Node::Dispatch { arms }] }]
    }

    /// `if v == 0 { .. } else if matches!(v, 1 | 3) { .. } else { default }`
    fn switch(
        &mut self,
        v: ValueId,
        default: BlockId,
        arms: &[(BlockId, Vec<u64>)],
        mut to: impl FnMut(&mut Self, BlockId) -> Vec<Node>,
    ) -> Vec<Node> {
        let mut conds = Vec::new();
        for (t, ks) in arms {
            conds.push((self.src.case(v, ks), to(self, *t)));
        }
        let mut chain = to(self, default);
        for (c, then) in conds.into_iter().rev() {
            chain = vec![Node::If { c, then, els: chain }];
        }
        chain
    }

    fn go(&mut self, r: BlockId, me: u32, to: BlockId, args: &[ValueId]) -> Vec<Node> {
        if self.node_of[to.index()] != r {
            return self.branch(r, to, args);
        }
        let mut out = self.src.edge(to, args);
        out.push(Node::Line(format!("{DISPATCH_VAR} = {};", to.index())));
        out.push(Node::Continue(me));
        out
    }
}


// ---------------------------------------------------------------------------
// Tidying

/// Does any `break`/`continue` in `nodes` target `l`?
fn uses(nodes: &[Node], l: Label) -> bool {
    nodes.iter().any(|n| match n {
        Node::Break(x) => *x == l,
        Node::Continue(h) => l == Label::Loop(*h),
        Node::If { then, els, .. } => uses(then, l) || uses(els, l),
        Node::Loop { body, .. } | Node::While { body, .. } | Node::Block { body, .. } => uses(body, l),
        Node::Dispatch { arms } => arms.iter().any(|(_, a)| uses(a, l)),
        Node::Line(_) | Node::Copy(_) | Node::Exit(_) => false,
    })
}

/// Control never reaches the end of `nodes`.
fn diverges(nodes: &[Node]) -> bool {
    match nodes.last() {
        Some(Node::Exit(_) | Node::Break(_) | Node::Continue(_)) => true,
        Some(Node::If { then, els, .. }) => diverges(then) && diverges(els),
        Some(Node::Block { label, body }) => diverges(body) && !uses(body, Label::Block(*label)),
        Some(Node::Loop { head, body }) => !uses_break(body, Label::Loop(*head)),
        Some(Node::Dispatch { arms }) => arms.iter().all(|(_, a)| diverges(a)),
        Some(Node::Line(_) | Node::Copy(_) | Node::While { .. }) | None => false,
    }
}

/// Like `uses`, counting only `break`s (a `continue` stays inside the loop).
fn uses_break(nodes: &[Node], l: Label) -> bool {
    nodes.iter().any(|n| match n {
        Node::Break(x) => *x == l,
        Node::If { then, els, .. } => uses_break(then, l) || uses_break(els, l),
        Node::Loop { body, .. } | Node::While { body, .. } | Node::Block { body, .. } => uses_break(body, l),
        Node::Dispatch { arms } => arms.iter().any(|(_, a)| uses_break(a, l)),
        _ => false,
    })
}

/// Drop jumps in tail position that just go where falling off the end of `nodes`
/// goes anyway: `break 'b` at the end of block `'b`, `continue 'l` at the end of
/// loop `'l`. Returns whether anything changed; changed lists are re-tidied.
fn drop_tail(nodes: &mut Vec<Node>, jump: &Node) -> bool {
    let same = |n: &Node| match (n, jump) {
        (Node::Break(a), Node::Break(b)) => a == b,
        (Node::Continue(a), Node::Continue(b)) => a == b,
        _ => false,
    };
    let changed = match nodes.last_mut() {
        Some(n) if same(n) => {
            nodes.pop();
            true
        }
        Some(Node::If { then, els, .. }) => drop_tail(then, jump) | drop_tail(els, jump),
        // Falling off an inner block's end also falls off ours.
        Some(Node::Block { body, .. }) => drop_tail(body, jump),
        // ... and so does falling off a dispatch arm, which goes round its loop.
        Some(Node::Dispatch { arms }) => arms.iter_mut().fold(false, |c, (_, a)| drop_tail(a, jump) | c),
        _ => false,
    };
    if changed {
        *nodes = level(std::mem::take(nodes));
    }
    changed
}

/// Can evaluating `c` panic or fault? Then it stays even where its value doesn't
/// matter.
fn may_fault(c: &str) -> bool {
    ["unsafe", "[", " / ", " % ", "wrapping_div", "wrapping_rem", "todo!"].iter().any(|k| c.contains(k))
}

fn size(nodes: &[Node]) -> usize {
    nodes
        .iter()
        .map(|n| match n {
            Node::If { then, els, .. } => 1 + size(then) + size(els),
            Node::Loop { body, .. } | Node::While { body, .. } | Node::Block { body, .. } => 1 + size(body),
            Node::Dispatch { arms } => arms.iter().map(|(_, a)| 1 + size(a)).sum(),
            _ => 1,
        })
        .sum()
}

/// Simplify bottom-up: children first, then this level.
pub fn tidy(nodes: Vec<Node>) -> Vec<Node> {
    let nodes = nodes
        .into_iter()
        .map(|n| match n {
            Node::If { c, then, els } => Node::If { c, then: tidy(then), els: tidy(els) },
            // Tail jumps go first, while both arms of an `if` still end in one:
            // tidying the body first could turn `if c { A; break 'b } else { B; break 'b }`
            // into `if c { A; break 'b } B; break 'b`, which keeps the label.
            Node::Loop { head, mut body } => {
                drop_tail(&mut body, &Node::Continue(head));
                body = tidy(body);
                // `loop { if c { break; } .. }` is `while !c { .. }`
                match body.first() {
                    Some(Node::If { c, then, els }) if els.is_empty() && matches!(then[..], [Node::Break(Label::Loop(h))] if h == head) => {
                        let c = not(c);
                        body.remove(0);
                        Node::While { head, c, body }
                    }
                    _ => Node::Loop { head, body },
                }
            }
            Node::Dispatch { arms } => Node::Dispatch { arms: arms.into_iter().map(|(k, a)| (k, tidy(a))).collect() },
            Node::Block { label, mut body } => {
                drop_tail(&mut body, &Node::Break(Label::Block(label)));
                // (before the loop's own tidying too, which looks for `break`s of it)
                if let Some(Node::Loop { head, body: lb }) = body.last_mut() {
                    retarget(lb, Label::Block(label), Label::Loop(*head));
                }
                body = tidy(body);
                // `'b: { ..; 'l: loop { .. break 'b .. } }`: leaving the loop
                // lands at the end of the block too, so `break 'b` there is `break 'l`.
                if let Some(Node::Loop { head, body: lb } | Node::While { head, body: lb, .. }) = body.last_mut() {
                    retarget(lb, Label::Block(label), Label::Loop(*head));
                }
                if uses(&body, Label::Block(label)) {
                    let mut budget = UNBREAK_COPIES;
                    if let Some(b) = unbreak(body.clone(), Label::Block(label), &mut budget) {
                        body = b;
                    }
                }
                Node::Block { label, body }
            }
            n => n,
        })
        .collect();
    level(nodes)
}

/// Most nodes `unbreak` may copy to do away with one labeled block.
const UNBREAK_COPIES: usize = 8;

/// `'l: { nodes }` as plain nesting, or `None` if it needs the label. Each
/// `break 'l` must be the last thing on its path through nested `if`s; it becomes
/// falling off the end, and what follows such an `if` moves into the arms that
/// fall off it: `'l: { if c { A; break 'l; } B }` is `if c { A } else { B }`.
/// What follows is copied when both arms fall off, at most `budget` nodes in all.
fn unbreak(mut nodes: Vec<Node>, l: Label, budget: &mut usize) -> Option<Vec<Node>> {
    // From the last node that leaves to `'l` back: each takes in what follows it.
    while let Some(i) = nodes.iter().rposition(|n| uses(std::slice::from_ref(n), l)) {
        let rest = nodes.split_off(i + 1);
        match nodes.pop() {
            // (anything after it is dead)
            Some(Node::Break(x)) if x == l => {}
            Some(Node::If { c, mut then, mut els }) => {
                let (tf, ef) = (!diverges(&then), !diverges(&els));
                if tf && ef {
                    *budget = budget.checked_sub(size(&rest))?;
                    then.extend(rest.iter().cloned());
                    els.extend(rest);
                } else if tf {
                    then.extend(rest);
                } else if ef {
                    els.extend(rest);
                }
                let then = level(unbreak(then, l, budget)?);
                let els = level(unbreak(els, l, budget)?);
                nodes.push(Node::If { c, then, els });
            }
            // Leaving the loop goes on to `rest`: copy it in front of each of the
            // loop's own `break`s, and the loop ends the block, so `break 'l` is
            // a `break` of the loop.
            Some(Node::Loop { head, mut body }) => {
                let me = Label::Loop(head);
                if !rest.is_empty() {
                    let n = breaks(&body, me);
                    *budget = budget.checked_sub(size(&rest) * n.saturating_sub(1))?;
                    before_breaks(&mut body, me, &rest);
                }
                retarget(&mut body, l, me);
                nodes.push(Node::Loop { head, body });
            }
            Some(Node::While { head, c, mut body }) if rest.is_empty() => {
                retarget(&mut body, l, Label::Loop(head));
                nodes.push(Node::While { head, c, body });
            }
            _ => return None,
        }
    }
    Some(nodes)
}

/// How many `break`s in `nodes` leave `l`.
fn breaks(nodes: &[Node], l: Label) -> usize {
    nodes
        .iter()
        .map(|n| match n {
            Node::Break(x) => (*x == l) as usize,
            Node::If { then, els, .. } => breaks(then, l) + breaks(els, l),
            Node::Loop { body, .. } | Node::While { body, .. } | Node::Block { body, .. } => breaks(body, l),
            Node::Dispatch { arms } => arms.iter().map(|(_, a)| breaks(a, l)).sum(),
            _ => 0,
        })
        .sum()
}

/// Put `rest` in front of each `break` of `l` in `nodes`, or in its place if
/// `rest` never falls off its end.
fn before_breaks(nodes: &mut Vec<Node>, l: Label, rest: &[Node]) {
    let keep = !diverges(rest);
    let mut out = Vec::with_capacity(nodes.len());
    for mut n in std::mem::take(nodes) {
        match &mut n {
            Node::Break(x) if *x == l => {
                out.extend(rest.iter().cloned());
                if !keep {
                    continue;
                }
            }
            Node::If { then, els, .. } => {
                before_breaks(then, l, rest);
                before_breaks(els, l, rest);
            }
            Node::Loop { body, .. } | Node::While { body, .. } | Node::Block { body, .. } => before_breaks(body, l, rest),
            Node::Dispatch { arms } => arms.iter_mut().for_each(|(_, a)| before_breaks(a, l, rest)),
            _ => {}
        }
        out.push(n);
    }
    *nodes = out;
}

fn retarget(nodes: &mut [Node], from: Label, to: Label) {
    for n in nodes {
        match n {
            Node::Break(l) if *l == from => *l = to,
            Node::If { then, els, .. } => {
                retarget(then, from, to);
                retarget(els, from, to);
            }
            Node::Loop { body, .. } | Node::While { body, .. } | Node::Block { body, .. } => retarget(body, from, to),
            Node::Dispatch { arms } => arms.iter_mut().for_each(|(_, a)| retarget(a, from, to)),
            _ => {}
        }
    }
}

/// One level's rewrites, assuming the children are already tidy.
fn level(nodes: Vec<Node>) -> Vec<Node> {
    let mut out = Vec::with_capacity(nodes.len());
    // A stack so spliced-in nodes get the same treatment.
    let mut todo: Vec<Node> = nodes.into_iter().rev().collect();
    while let Some(n) = todo.pop() {
        match n {
            Node::Block { label, body } if !uses(&body, Label::Block(label)) => {
                todo.extend(body.into_iter().rev());
            }
            Node::If { c, then, els } => {
                let (td, ed) = (diverges(&then), diverges(&els));
                match (then.is_empty(), els.is_empty()) {
                    // the condition may read memory (an inlined load) or call, so keep it
                    (true, true) if may_fault(&c) || calls(&c) => out.push(Node::Line(format!("let _ = {c};"))),
                    (true, true) => {}
                    // `if a { if b { .. } }` is `if a && b { .. }`
                    (false, true) => {
                        let mut then = then;
                        match then.pop() {
                            Some(Node::If { c: c2, then: t2, els: e2 }) if then.is_empty() && e2.is_empty() => {
                                let c = format!("{} && {}", expr::logic(c, true), expr::logic(c2, true));
                                out.push(Node::If { c, then: t2, els });
                            }
                            last => {
                                then.extend(last);
                                out.push(Node::If { c, then, els });
                            }
                        }
                    }
                    (true, false) => todo.push(Node::If { c: not(&c), then: els, els: then }),
                    // Early exit: keep the branch that leaves (the shorter one if
                    // both do) under the `if`, and the other after it.
                    _ if td && (!ed || size(&then) <= size(&els)) => {
                        out.push(Node::If { c, then, els: Vec::new() });
                        todo.extend(els.into_iter().rev());
                    }
                    _ if ed => {
                        out.push(Node::If { c: not(&c), then: els, els: Vec::new() });
                        todo.extend(then.into_iter().rev());
                    }
                    // `if c { A; x = a; } else { x = b; }` is `x = b; if c { A; x = a; }`
                    _ if hoistable(&c, &els, &then, &todo) => {
                        out.extend(els);
                        out.push(Node::If { c, then, els: Vec::new() });
                    }
                    _ if hoistable(&c, &then, &els, &todo) => {
                        out.extend(then);
                        out.push(Node::If { c: not(&c), then: els, els: Vec::new() });
                    }
                    _ => out.push(Node::If { c, then, els }),
                }
            }
            n => out.push(n),
        }
    }
    out
}

/// Can the arm `copies` of `if c { other } else { copies }` run before the `if`
/// instead? It must be nothing but edge copies of values that are cheap and
/// harmless to compute on the other path too (no call, no memory access, nothing
/// that can fault), and nothing that runs after them on that path may read the
/// variables they set: not the condition, not `other` (which may only overwrite
/// them, as its own edge copies into the same join do), not the code after the
/// `if` (`rest`). The other arm then needs no `else`, as in the source.
fn hoistable(c: &str, copies: &[Node], other: &[Node], rest: &[Node]) -> bool {
    let mut vars = Vec::new();
    for n in copies {
        let Node::Copy(line) = n else { return false };
        let Some((lhs, rhs)) = line.split_once(" = ") else { return false };
        if may_fault(rhs) || calls(rhs) {
            return false;
        }
        vars.extend(lhs.trim_matches(|ch| ch == '(' || ch == ')').split(", ").map(str::to_owned));
    }
    // `other` must leave by falling off its end, through its own copies to the
    // same variables: then both paths reach the join with what they did before.
    // (A copy that would assign a variable itself is left out, so a variable
    // `other` doesn't copy may still be read at the join with its old value.)
    let Some(Node::Copy(last)) = other.last() else { return false };
    let Some((lhs, _)) = last.split_once(" = ") else { return false };
    let set: Vec<&str> = lhs.trim_matches(|ch| ch == '(' || ch == ')').split(", ").collect();
    !copies.is_empty()
        && !jumps(other)
        && vars.iter().all(|v| set.contains(&v.as_str()) && !mentions(c, v) && !reads(other, v) && !reads(rest, v))
}

/// Does anything in `nodes` leave by `break` or `continue`?
fn jumps(nodes: &[Node]) -> bool {
    nodes.iter().any(|n| match n {
        Node::Break(_) | Node::Continue(_) => true,
        Node::If { then, els, .. } => jumps(then) || jumps(els),
        Node::Loop { body, .. } | Node::While { body, .. } | Node::Block { body, .. } => jumps(body),
        Node::Dispatch { .. } => true,
        Node::Line(_) | Node::Copy(_) | Node::Exit(_) => false,
    })
}

/// Does `e` call a function (anything but a method like `.wrapping_add(..)` or a
/// macro like `addr_of!(..)`)?
fn calls(e: &str) -> bool {
    let b = e.as_bytes();
    let word = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    (0..b.len()).any(|i| {
        if b[i] != b'(' || i == 0 || !word(b[i - 1]) {
            return false;
        }
        let mut j = i;
        while j > 0 && word(b[j - 1]) {
            j -= 1;
        }
        j == 0 || b[j - 1] != b'.'
    })
}

/// Does anything in `nodes` read the variable `v`? Being assigned by an edge copy
/// isn't a read.
fn reads(nodes: &[Node], v: &str) -> bool {
    nodes.iter().any(|n| match n {
        Node::Line(s) | Node::Exit(s) => mentions(s, v),
        // the last copy of `other` may assign `v` (that's the point), but not read it
        Node::Copy(s) => s.split_once(" = ").is_none_or(|(_, rhs)| mentions(rhs, v)),
        Node::If { c, then, els } => mentions(c, v) || reads(then, v) || reads(els, v),
        Node::While { c, body, .. } => mentions(c, v) || reads(body, v),
        Node::Loop { body, .. } | Node::Block { body, .. } => reads(body, v),
        Node::Dispatch { arms } => arms.iter().any(|(_, a)| reads(a, v)),
        Node::Break(_) | Node::Continue(_) => false,
    })
}

// ---------------------------------------------------------------------------
// Printing

/// Print `nodes` at indentation `depth` (4 spaces each).
pub fn print(nodes: &[Node], depth: usize, out: &mut String) {
    let mut named = std::collections::HashSet::new();
    let mut stack = Vec::new();
    needs_label(nodes, &mut stack, &mut named);
    Printer { named, stack: Vec::new() }.list(nodes, depth, out);
}

fn label_name(l: Label) -> String {
    match l {
        Label::Loop(h) => format!("'l{h}"),
        Label::Block(b) => format!("'b{b}"),
    }
}

/// A plain `break`/`continue` targets the innermost loop, and is not allowed
/// directly inside a labeled block. So it can drop its label only when its target
/// is the innermost enclosing construct and a loop.
fn plain(stack: &[Label], target: Label) -> bool {
    matches!(target, Label::Loop(_)) && stack.last() == Some(&target)
}

fn needs_label(nodes: &[Node], stack: &mut Vec<Label>, named: &mut std::collections::HashSet<Label>) {
    for n in nodes {
        match n {
            Node::Break(l) if !plain(stack, *l) => {
                named.insert(*l);
            }
            Node::Continue(h) if !plain(stack, Label::Loop(*h)) => {
                named.insert(Label::Loop(*h));
            }
            Node::If { then, els, .. } => {
                needs_label(then, stack, named);
                needs_label(els, stack, named);
            }
            Node::Loop { head, body } | Node::While { head, body, .. } => {
                stack.push(Label::Loop(*head));
                needs_label(body, stack, named);
                stack.pop();
            }
            Node::Dispatch { arms } => arms.iter().for_each(|(_, a)| needs_label(a, stack, named)),
            Node::Block { label, body } => {
                stack.push(Label::Block(*label));
                needs_label(body, stack, named);
                stack.pop();
            }
            _ => {}
        }
    }
}

struct Printer {
    named: std::collections::HashSet<Label>,
    stack: Vec<Label>,
}

impl Printer {
    fn jump(&self, kw: &str, target: Label) -> String {
        if plain(&self.stack, target) {
            format!("{kw};")
        } else {
            format!("{kw} {};", label_name(target))
        }
    }

    fn list(&mut self, nodes: &[Node], depth: usize, out: &mut String) {
        let ind = "    ".repeat(depth);
        for n in nodes {
            match n {
                Node::Line(s) | Node::Copy(s) | Node::Exit(s) => {
                    let _ = writeln!(out, "{ind}{s}");
                }
                Node::Break(l) => {
                    let _ = writeln!(out, "{ind}{}", self.jump("break", *l));
                }
                Node::Continue(h) => {
                    let _ = writeln!(out, "{ind}{}", self.jump("continue", Label::Loop(*h)));
                }
                Node::If { c, then, els } => {
                    let _ = writeln!(out, "{ind}if {c} {{");
                    self.list(then, depth + 1, out);
                    if els.is_empty() {
                        let _ = writeln!(out, "{ind}}}");
                    } else if let [Node::If { .. }] = els.as_slice() {
                        // `else if`: print the inner `if` on this line.
                        let mut inner = String::new();
                        self.list(els, depth, &mut inner);
                        let _ = write!(out, "{ind}}} else {}", inner.trim_start());
                    } else {
                        let _ = writeln!(out, "{ind}}} else {{");
                        self.list(els, depth + 1, out);
                        let _ = writeln!(out, "{ind}}}");
                    }
                }
                Node::Loop { head, body } | Node::While { head, body, .. } => {
                    let l = Label::Loop(*head);
                    let name = if self.named.contains(&l) { format!("{}: ", label_name(l)) } else { String::new() };
                    match n {
                        Node::While { c, .. } => writeln!(out, "{ind}{name}while {c} {{"),
                        _ => writeln!(out, "{ind}{name}loop {{"),
                    }
                    .unwrap();
                    self.stack.push(l);
                    self.list(body, depth + 1, out);
                    self.stack.pop();
                    let _ = writeln!(out, "{ind}}}");
                }
                Node::Dispatch { arms } => {
                    let _ = writeln!(out, "{ind}match {DISPATCH_VAR} {{");
                    for (k, a) in arms {
                        let _ = writeln!(out, "{ind}    {k} => {{");
                        self.list(a, depth + 2, out);
                        let _ = writeln!(out, "{ind}    }}");
                    }
                    let _ = writeln!(out, "{ind}    _ => unreachable!(),");
                    let _ = writeln!(out, "{ind}}}");
                }
                Node::Block { label, body } => {
                    let l = Label::Block(*label);
                    let _ = writeln!(out, "{ind}{}: {{", label_name(l));
                    self.stack.push(l);
                    self.list(body, depth + 1, out);
                    self.stack.pop();
                    let _ = writeln!(out, "{ind}}}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Declarations

/// Which of `vars` (names of variables declared up front, `let mut vN = 0;`)
/// can be declared where they are assigned instead: assigned by exactly one
/// statement (`vN = e;`), and used only after it in the same list of
/// statements, or nested inside what follows it there. That statement becomes
/// `let vN: T = e;` (with `types[k]` for `vars[k]`), and the caller drops the
/// up-front declarations of the ones this returns `true` for.
pub fn declare_in_place(nodes: &mut [Node], vars: &[&str], types: &[String]) -> Vec<bool> {
    let index: std::collections::HashMap<&str, usize> = vars.iter().enumerate().map(|(k, &v)| (v, k)).collect();
    let mut seen = vec![Seen::default(); vars.len()];
    let mut path = Vec::new();
    scan(nodes, &index, &mut path, &mut seen);
    let ok: Vec<bool> = seen
        .iter()
        .map(|s| match &s.assign {
            Some(a) if s.assigns == 1 => {
                let (scope, at) = a.split_at(a.len() - 1);
                s.uses.iter().all(|u| u.len() > scope.len() && u.starts_with(scope) && u[scope.len()] > at[0])
            }
            _ => false,
        })
        .collect();
    for (k, s) in seen.iter().enumerate() {
        if ok[k] {
            let line = at_path(nodes, s.assign.as_ref().unwrap());
            if let Node::Line(l) = line {
                *l = format!("let {}: {} = {}", vars[k], types[k], &l[vars[k].len() + 3..]);
            }
        }
    }
    ok
}

#[derive(Clone, Default)]
struct Seen {
    /// Statements that assign it (`vN = e;` lines count, anything else makes it
    /// ineligible by counting twice).
    assigns: usize,
    /// Where the `vN = e;` line is: indices into nested statement lists, each
    /// list's index followed by which of the node's lists (or its condition) it is.
    assign: Option<Vec<u32>>,
    uses: Vec<Vec<u32>>,
}

/// Which list of a node a path goes into, after the node's index.
const COND: u32 = u32::MAX;

fn scan(nodes: &[Node], index: &std::collections::HashMap<&str, usize>, path: &mut Vec<u32>, seen: &mut [Seen]) {
    for (i, n) in nodes.iter().enumerate() {
        path.push(i as u32);
        let text = |s: &str, line: bool, sub: Option<u32>, path: &mut Vec<u32>, seen: &mut [Seen]| {
            for (at, name) in idents(s) {
                let Some(&k) = index.get(name) else { continue };
                let rest = &s[at + name.len()..];
                let assigning = rest.starts_with(" = ") || (!line && at < s.find(" = ").unwrap_or(0));
                if let Some(x) = sub {
                    path.push(x);
                }
                if assigning && line && at == 0 {
                    seen[k].assigns += 1;
                    seen[k].assign = Some(path.clone());
                } else if assigning {
                    seen[k].assigns += 2;
                } else {
                    seen[k].uses.push(path.clone());
                }
                if sub.is_some() {
                    path.pop();
                }
            }
        };
        match n {
            Node::Line(s) => text(s, true, None, path, seen),
            // edge copies assign on the left
            Node::Copy(s) => text(s, false, None, path, seen),
            Node::Exit(s) => text(s, false, None, path, seen),
            Node::Break(_) | Node::Continue(_) => {}
            Node::If { c, then, els } => {
                text(c, true, Some(COND), path, seen);
                for (x, l) in [then, els].into_iter().enumerate() {
                    path.push(x as u32);
                    scan(l, index, path, seen);
                    path.pop();
                }
            }
            Node::While { c, body, .. } => {
                text(c, true, Some(COND), path, seen);
                path.push(0);
                scan(body, index, path, seen);
                path.pop();
            }
            Node::Loop { body, .. } | Node::Block { body, .. } => {
                path.push(0);
                scan(body, index, path, seen);
                path.pop();
            }
            Node::Dispatch { arms } => {
                for (x, (_, l)) in arms.iter().enumerate() {
                    path.push(x as u32);
                    scan(l, index, path, seen);
                    path.pop();
                }
            }
        }
        path.pop();
    }
}

/// The node at a path from `scan`.
fn at_path<'a>(nodes: &'a mut [Node], path: &[u32]) -> &'a mut Node {
    let n = &mut nodes[path[0] as usize];
    if path.len() == 1 {
        return n;
    }
    let list: &mut [Node] = match n {
        Node::If { then, els, .. } => if path[1] == 0 { then } else { els },
        Node::While { body, .. } | Node::Loop { body, .. } | Node::Block { body, .. } => body,
        Node::Dispatch { arms } => &mut arms[path[1] as usize].1,
        _ => unreachable!("a path into a statement"),
    };
    at_path(list, &path[2..])
}

/// The identifiers in `s` (outside string literals) with their offsets.
fn idents(s: &str) -> impl Iterator<Item = (usize, &str)> {
    let b = s.as_bytes();
    let word = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut i = 0;
    let mut quoted = false;
    std::iter::from_fn(move || {
        while i < b.len() {
            let c = b[i];
            if quoted {
                if c == b'\\' {
                    i += 1;
                } else if c == b'"' {
                    quoted = false;
                }
                i += 1;
                continue;
            }
            if c == b'"' {
                quoted = true;
                i += 1;
                continue;
            }
            if word(c) && (i == 0 || !word(b[i - 1])) {
                let start = i;
                while i < b.len() && word(b[i]) {
                    i += 1;
                }
                return Some((start, &s[start..i]));
            }
            i += 1;
        }
        None
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(s: &str) -> Node {
        Node::Line(s.to_string())
    }

    fn text(nodes: &[Node]) -> String {
        let mut out = String::new();
        print(nodes, 0, &mut out);
        out
    }

    #[test]
    fn a_labeled_block_left_from_nested_ifs_becomes_if_else() {
        // 'b1: { if a { if b { x; break 'b1; } } y; }  z;
        let body = vec![
            Node::If {
                c: "a".into(),
                then: vec![Node::If { c: "b".into(), then: vec![line("x;"), Node::Break(Label::Block(1))], els: vec![] }],
                els: vec![],
            },
            Node::Copy("y;".into()),
        ];
        let nodes = tidy(vec![Node::Block { label: 1, body }, line("z;")]);
        assert_eq!(text(&nodes), "if a && b {\n    x;\n} else {\n    y;\n}\nz;\n");
    }

    #[test]
    fn code_after_a_loop_left_by_break_moves_to_the_loops_exits() {
        // 'b1: { loop { if d { x; break 'b1; } if c { break; } } y; }
        let lp = Node::Loop {
            head: 2,
            body: vec![
                Node::If { c: "d".into(), then: vec![line("x;"), Node::Break(Label::Block(1))], els: vec![] },
                Node::If { c: "c".into(), then: vec![Node::Break(Label::Loop(2))], els: vec![] },
            ],
        };
        let nodes = tidy(vec![Node::Block { label: 1, body: vec![lp, line("y;")] }, Node::Exit("return;".into())]);
        assert_eq!(text(&nodes), "loop {\n    if d {\n        x;\n        break;\n    }\n    if c {\n        y;\n        break;\n    }\n}\nreturn;\n");
    }

    #[test]
    fn a_label_stays_where_removing_it_would_copy_too_much() {
        let big: Vec<Node> = (0..UNBREAK_COPIES + 1).map(|k| line(&format!("y{k};"))).collect();
        let body = vec![
            Node::If {
                c: "a".into(),
                then: vec![Node::If { c: "b".into(), then: vec![Node::Break(Label::Block(1))], els: vec![] }, line("x;")],
                els: vec![],
            },
        ]
        .into_iter()
        .chain(big)
        .collect();
        let nodes = tidy(vec![Node::Block { label: 1, body }, line("z;")]);
        assert!(text(&nodes).contains("'b1: {"));
    }

    #[test]
    fn a_variable_is_declared_where_it_is_assigned_only_if_its_uses_follow() {
        // v1: assigned, then used after it in the same list and in a nested `if`.
        // v2: assigned inside a loop, used after the loop. v3: assigned on two
        // edges. v4: used in a loop condition before its assignment in the body.
        let mut nodes = vec![
            line("v1 = f(); // 0x1"),
            Node::If { c: "v1 != 0".into(), then: vec![Node::Exit("return v1;".into())], els: vec![] },
            Node::Loop { head: 1, body: vec![line("v2 = g(v1);"), Node::Break(Label::Loop(1))] },
            Node::Copy("v3 = v2;".into()),
            Node::If { c: "v2 == 0".into(), then: vec![Node::Copy("v3 = 1_u64;".into())], els: vec![] },
            Node::While { head: 2, c: "v4 != 0".into(), body: vec![line("v4 = h(v3);")] },
        ];
        let ok = declare_in_place(&mut nodes, &["v1", "v2", "v3", "v4"], &vec!["u64".to_string(); 4]);
        assert_eq!(ok, [true, false, false, false]);
        assert!(matches!(&nodes[0], Node::Line(l) if l == "let v1: u64 = f(); // 0x1"));
        assert!(matches!(&nodes[2], Node::Loop { body, .. } if matches!(&body[0], Node::Line(l) if l == "v2 = g(v1);")));
    }
}
