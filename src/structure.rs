//! Control-flow structuring: turn a CFG into nested `if`/`else`, `while`, `loop`,
//! labeled blocks, `break` and `continue`, instead of a `loop { match bb { .. } }`
//! state machine.
//!
//! It works on its own copy of the CFG (`Graph`), which it rewrites in two ways
//! before structuring:
//!
//! 1. **Short-circuit conditions.** A block that does nothing but test a condition
//!    (the emitter inlined all its statements into the test), reached only from
//!    another branch that shares one of its targets, is folded into that branch:
//!    `if a { X } else if b { X } else { Y }` becomes `if a || b { X } else { Y }`,
//!    and likewise for `&&`. Without this, `X` has two forward in-edges and needs
//!    a labeled block.
//! 2. **Irreducible regions.** A cycle with more than one entry has no nesting.
//!    Each such strongly connected component gets a dispatcher: every edge into one
//!    of its entries sets a state variable and goes to a new header that tests it
//!    (`if bb == 4 { .. } else if bb == 9 { .. }`). That makes the graph reducible
//!    while the state machine covers only that region; the rest of the function,
//!    and the inside of the region, stay structured.
//!
//! The construction is then Ramsey's "Beyond Relooper" (ICFP 2022): walk the
//! dominator tree; a block's dominator-tree children that are join points (two or
//! more forward in-edges) are emitted after it, each wrapped in a labeled block
//! that branches `break` out of; a loop header wraps its subtree in a `loop` that
//! back edges `continue`; every other edge goes to a block with a single
//! predecessor, which is emitted inline at the branch. That is correct for any
//! reducible CFG. Rust's labeled blocks (`'b: { .. break 'b; }`) make it a direct
//! translation.
//!
//! The raw result is correct but noisy, so `tidy` then removes what isn't needed:
//! a `break`/`continue` that control would reach anyway by falling off the end,
//! labeled blocks nothing breaks out of, and `if c { A } else { B }` where `A`
//! always leaves (`if c { A } B`). A `loop` that starts with `if !c { break; }`
//! becomes `while c { .. }`. Labels are printed only where a plain
//! `break`/`continue` would not mean the same thing.
use crate::cfg::Cfg;
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
    /// A statement that never continues (`return`, `panic!`, `todo!`).
    Exit(String),
    If { c: Cond, then: Vec<Node>, els: Vec<Node> },
    Loop { head: u32, body: Vec<Node> },
    /// `while c { body }`: a `loop` whose first statement is `if !c { break; }`.
    While { head: u32, c: Cond, body: Vec<Node> },
    Block { label: u32, body: Vec<Node> },
    /// Leave the labeled block or loop.
    Break(Label),
    /// Next iteration of the loop headed by this block.
    Continue(u32),
}

/// A branch condition: an expression from the emitter, or a short-circuit
/// combination of them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cond {
    Atom(String),
    Not(Box<Cond>),
    And(Box<Cond>, Box<Cond>),
    Or(Box<Cond>, Box<Cond>),
}

impl Cond {
    fn and(a: Cond, b: Cond) -> Cond {
        Cond::And(Box::new(a), Box::new(b))
    }

    fn or(a: Cond, b: Cond) -> Cond {
        Cond::Or(Box::new(a), Box::new(b))
    }

    /// Print with just the parentheses needed inside an operator of precedence
    /// `min` (0: none, 1: `||`, 2: `&&`, 3: `!`).
    fn show(&self, min: u8, out: &mut String) {
        let (prec, l, r, op) = match self {
            Cond::Atom(s) => {
                let bare = match min {
                    0 => !s.starts_with("if "),
                    3 => atomic(s),
                    _ => !s.starts_with("if ") && !top_level(s).iter().any(|t| matches!(*t, "&&" | "||" | "..")),
                };
                if bare {
                    out.push_str(s);
                } else {
                    let _ = write!(out, "({s})");
                }
                return;
            }
            Cond::Not(c) => {
                out.push('!');
                c.show(3, out);
                return;
            }
            Cond::Or(a, b) => (1, a, b, " || "),
            Cond::And(a, b) => (2, a, b, " && "),
        };
        if min > prec {
            out.push('(');
        }
        l.show(prec, out);
        out.push_str(op);
        r.show(prec, out);
        if min > prec {
            out.push(')');
        }
    }
}

/// The negation, pushed inwards (De Morgan) and folded into comparisons, so
/// `!(a < b || c)` comes out as `a >= b && !c`.
impl std::ops::Not for Cond {
    type Output = Cond;

    fn not(self) -> Cond {
        match self {
            Cond::Not(c) => *c,
            Cond::And(a, b) => Cond::Or(Box::new(!*a), Box::new(!*b)),
            Cond::Or(a, b) => Cond::And(Box::new(!*a), Box::new(!*b)),
            Cond::Atom(s) => match flip(&s) {
                Some(f) => Cond::Atom(f),
                None => Cond::Not(Box::new(Cond::Atom(s))),
            },
        }
    }
}

impl std::fmt::Display for Cond {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        self.show(0, &mut s);
        f.write_str(&s)
    }
}

/// `s` split at spaces outside brackets and string literals.
pub fn top_level(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut depth, mut quoted, mut escaped, mut start) = (0i32, false, false, 0);
    for (i, ch) in s.char_indices() {
        if quoted {
            match ch {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => quoted = false,
                _ => {}
            }
            continue;
        }
        match ch {
            '"' => quoted = true,
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ' ' if depth == 0 => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&s[start..]);
    out
}

/// Can `s` be used as an operand of any operator (or as a method receiver)
/// without parentheses? Identifiers, literals, paths, calls and method chains can.
pub fn atomic(s: &str) -> bool {
    !s.is_empty() && !s.starts_with(['!', '-', '&', '*']) && top_level(s).len() == 1
}

/// `a < b` negated is `a >= b`, and `!x` negated is `x`.
fn flip(s: &str) -> Option<String> {
    if let Some(inner) = s.strip_prefix('!') {
        return atomic(inner).then(|| inner.to_string());
    }
    let t = top_level(s);
    let [a, op, b] = t.as_slice() else { return None };
    let op = match *op {
        "==" => "!=",
        "!=" => "==",
        "<" => ">=",
        ">=" => "<",
        ">" => "<=",
        "<=" => ">",
        _ => return None,
    };
    Some(format!("{a} {op} {b}"))
}

/// What the structurer needs from the emitter, per block.
pub trait Source {
    /// The block's statements, without its terminator.
    fn stmts(&mut self, b: BlockId) -> Vec<Node>;
    /// The block's statements are few and pure (no memory access, call or
    /// possible panic), so they can run early, before the branch that leads to
    /// the block is decided. Then the block's condition can join that branch's
    /// in a short-circuit `||`/`&&`.
    fn speculate(&self, b: BlockId) -> bool;
    /// `b` only jumps on: it is quiet except for pure values inlined into its
    /// edge arguments, and each of its parameters is used exactly once, as one
    /// of those arguments. Edges into it can go straight to its successor.
    fn forwards(&self, b: BlockId) -> bool;
    /// Statements that pass `args` to `to`'s parameters.
    fn edge(&mut self, to: BlockId, args: &[ValueId]) -> Vec<Node>;
    /// The condition of a `Branch`, as an expression.
    fn cond(&mut self, c: ValueId) -> String;
    /// A terminator without successors (return, tail call, ...), as a statement.
    fn exit(&mut self, b: BlockId) -> Node;
}

/// Deepest dominator tree the structurer will recurse into. Past this the caller
/// keeps the state machine, which uses no recursion.
const MAX_DEPTH: usize = 512;

/// Is every cycle entered only through its header? Then every retreating edge
/// (to a block no later in reverse postorder) goes to a block that dominates its
/// source, and is a back edge of a natural loop.
pub fn reducible(f: &Function, cfg: &Cfg) -> bool {
    reducible_by(cfg, |b| f.blocks[b].term.successors())
}

fn reducible_by(cfg: &Cfg, succ: impl Fn(BlockId) -> [Option<BlockId>; 2]) -> bool {
    cfg.rpo.iter().all(|&b| {
        succ(b).into_iter().flatten().all(|s| cfg.rpo_index[s.index()] > cfg.rpo_index[b.index()] || cfg.dominates(s, b))
    })
}

/// A structured function body.
pub struct Structured {
    pub body: Vec<Node>,
    /// Irreducible regions that needed a dispatcher (a local state machine).
    pub dispatchers: usize,
}

/// Structure `f`, or `None` if it nests too deeply.
pub fn structure(f: &Function, cfg: &Cfg, src: &mut dyn Source) -> Option<Structured> {
    let mut g = Graph::new(f, cfg, src);
    g.thread(f, src);
    g.short_circuit(src);
    let dispatchers = g.fix_irreducible(cfg);
    let cfg = g.cfg();
    if !reducible_by(&cfg, |b| g.succs(b)) {
        debug_assert!(false, "dispatchers left the graph irreducible");
        return None;
    }

    let n = g.term.len();
    let mut forward_in = vec![0u32; n];
    let mut loop_head = vec![false; n];
    let mut children: Vec<Vec<BlockId>> = vec![Vec::new(); n];
    let mut depth = vec![0usize; n];
    for &b in &cfg.rpo {
        if let Some(d) = cfg.idom[b.index()] {
            children[d.index()].push(b);
            depth[b.index()] = depth[d.index()] + 1;
            if depth[b.index()] > MAX_DEPTH {
                return None;
            }
        }
        for s in g.succs(b).into_iter().flatten() {
            if cfg.rpo_index[s.index()] > cfg.rpo_index[b.index()] {
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
    let follows = vec![false; n];
    let entry = g.entry;
    let mut s = Structurer { g: &g, cfg: &cfg, src, forward_in, loop_head, children, follows };
    let mut body: Vec<Node> = g.states.iter().enumerate().map(|(k, &init)| Node::Line(format!("let mut {}: u32 = {init};", state_var(k as u32)))).collect();
    body.extend(whiles(tidy(s.tree(entry))));
    Some(Structured { body, dispatchers })
}

// ---------------------------------------------------------------------------
// The graph being structured

/// An edge: go to `to`, passing `args` to the parameters of `params_of` (the
/// original target, which differs from `to` when a dispatcher stands between).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Edge {
    to: BlockId,
    params_of: BlockId,
    args: Vec<ValueId>,
    /// `(dispatcher, value)`: set that dispatcher's state variable on the way.
    state: Option<(u32, u32)>,
}

#[derive(Clone, Debug)]
enum Term {
    Jump(Edge),
    Branch(Cond, Edge, Edge),
    /// No successors: the emitter prints the block's terminator.
    Exit,
}

/// The CFG being structured: the function's blocks, then dispatcher blocks.
struct Graph {
    term: Vec<Term>,
    /// Blocks `0..real` are the function's; the rest are dispatchers.
    real: usize,
    entry: BlockId,
    /// Initial value of each dispatcher's state variable.
    states: Vec<u32>,
    /// Blocks folded into a condition (`fold`), whose statements print after
    /// this block's.
    early: Vec<Vec<BlockId>>,
}

fn state_var(k: u32) -> String {
    if k == 0 {
        "bb".to_string()
    } else {
        format!("bb{}", k + 1)
    }
}

impl Graph {
    fn new(f: &Function, cfg: &Cfg, src: &mut dyn Source) -> Graph {
        let edge = |to: BlockId, args: &[ValueId]| Edge { to, params_of: to, args: args.to_vec(), state: None };
        let mut term = vec![Term::Exit; f.blocks.len()];
        for &b in &cfg.rpo {
            term[b.index()] = match f.blocks[b].term {
                Terminator::Jump { to, args } => Term::Jump(edge(to, args.get(&f.value_pool))),
                Terminator::Branch { c, t, f: e, args } => {
                    let a = args.get(&f.value_pool);
                    let nt = f.blocks[t].params.len as usize;
                    Term::Branch(Cond::Atom(src.cond(c)), edge(t, &a[..nt]), edge(e, &a[nt..]))
                }
                _ => Term::Exit,
            };
        }
        Graph { term, real: f.blocks.len(), entry: f.entry, states: Vec::new(), early: vec![Vec::new(); f.blocks.len()] }
    }

    fn succs(&self, b: BlockId) -> [Option<BlockId>; 2] {
        match &self.term[b.index()] {
            Term::Jump(e) => [Some(e.to), None],
            Term::Branch(_, t, e) => [Some(t.to), Some(e.to)],
            Term::Exit => [None, None],
        }
    }

    /// Send edges into blocks that only jump on (`Source::forwards`) straight to
    /// where they jump, substituting the arguments: `x -> b(a)`, `b(p) -> c(p)`
    /// becomes `x -> c(a)`.
    fn thread(&mut self, f: &Function, src: &dyn Source) {
        for b in (0..self.real).map(BlockId::new) {
            let Term::Jump(out) = &self.term[b.index()] else { continue };
            if b == self.entry || out.to == b || !src.forwards(b) {
                continue;
            }
            let out = out.clone();
            let params = f.blocks[b].params.get(&f.value_pool);
            let through = |e: &mut Edge| {
                if e.to != b {
                    return;
                }
                let args: Vec<ValueId> = out.args.iter().map(|&v| params.iter().position(|&p| p == v).map_or(v, |i| e.args[i])).collect();
                *e = Edge { args, ..out.clone() };
            };
            for t in &mut self.term {
                match t {
                    Term::Jump(e) => through(e),
                    Term::Branch(_, x, y) => {
                        through(x);
                        through(y);
                    }
                    Term::Exit => {}
                }
            }
            // Unreachable now.
            self.term[b.index()] = Term::Exit;
        }
    }

    fn cfg(&self) -> Cfg {
        Cfg::from_succs(self.term.len(), self.entry, |b| self.succs(b))
    }

    /// Fold condition-only blocks into the branch before them (module docs).
    fn short_circuit(&mut self, src: &dyn Source) {
        loop {
            let cfg = self.cfg();
            let mut changed = false;
            for &a in &cfg.rpo {
                // Folding can expose another fold at the same block: `a || b || c`.
                while self.fold(a, &cfg, src) {
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }

    /// Fold `a`'s successor into `a` if it is a condition-only block. `cfg` may be
    /// stale, but only in ways that hide folds: in-edge counts only go down.
    fn fold(&mut self, a: BlockId, cfg: &Cfg, src: &dyn Source) -> bool {
        let Term::Branch(c1, at, ae) = &self.term[a.index()] else { return false };
        for b_is_then in [true, false] {
            let (eb, other) = if b_is_then { (at, ae) } else { (ae, at) };
            let b = eb.to;
            let single = cfg.preds(b).iter().filter(|&&p| cfg.reachable(p)).count() == 1;
            if b == a || b == self.entry || b.index() >= self.real || !single || !eb.args.is_empty() || !src.speculate(b) {
                continue;
            }
            let Term::Branch(c2, bt, be) = &self.term[b.index()] else { continue };
            let (c1, c2) = (c1.clone(), c2.clone());
            let new = match (b_is_then, bt == other, be == other) {
                // a ? b : o, b = c2 ? x : o   =>  a && c2 ? x : o
                (true, false, true) => Term::Branch(Cond::and(c1, c2), bt.clone(), other.clone()),
                // a ? b : o, b = c2 ? o : x   =>  a && !c2 ? x : o
                (true, true, false) => Term::Branch(Cond::and(c1, !c2), be.clone(), other.clone()),
                // a ? o : b, b = c2 ? o : y   =>  a || c2 ? o : y
                (false, true, false) => Term::Branch(Cond::or(c1, c2), other.clone(), be.clone()),
                // a ? o : b, b = c2 ? y : o   =>  a || !c2 ? o : y
                (false, false, true) => Term::Branch(Cond::or(c1, !c2), other.clone(), bt.clone()),
                _ => continue,
            };
            self.term[a.index()] = new;
            // `b` is now unreachable; drop its edges so in-edge counts stay right.
            self.term[b.index()] = Term::Exit;
            // Its statements run right after `a`'s.
            let mut moved = std::mem::take(&mut self.early[b.index()]);
            moved.insert(0, b);
            self.early[a.index()].extend(moved);
            return true;
        }
        false
    }

    /// Give every multiple-entry cycle a dispatcher (module docs). Returns how
    /// many it added. `order` ranks blocks so the output is deterministic.
    fn fix_irreducible(&mut self, order: &Cfg) -> usize {
        let rank = |b: BlockId| order.rpo_index.get(b.index()).copied().unwrap_or(u32::MAX - 1);
        let cfg = self.cfg();
        let mut work: Vec<Vec<BlockId>> = vec![cfg.rpo.clone()];
        let mut added = 0;
        while let Some(region) = work.pop() {
            for scc in self.sccs(&region) {
                let mut inside = vec![false; self.term.len()];
                for &b in &scc {
                    inside[b.index()] = true;
                }
                let preds = self.cfg();
                let mut entries: Vec<BlockId> = scc
                    .iter()
                    .copied()
                    .filter(|&b| b == self.entry || preds.preds(b).iter().any(|&p| preds.reachable(p) && !inside[p.index()]))
                    .collect();
                entries.sort_by_key(|&b| rank(b));
                // Inside, look for cycles that don't pass through the header.
                let mut inner: Vec<BlockId> = scc.into_iter().filter(|&b| b != entries[0]).collect();
                if entries.len() > 1 {
                    added += 1;
                    let first = self.term.len();
                    self.dispatch(&entries);
                    inner.push(entries[0]);
                    inner.extend((first + 1..self.term.len()).map(BlockId::new));
                }
                work.push(inner);
            }
        }
        added
    }

    /// Route every edge into `entries` through a new chain of tests on a state
    /// variable, which starts at block `self.term.len()`.
    fn dispatch(&mut self, entries: &[BlockId]) {
        let k = self.states.len() as u32;
        let var = state_var(k);
        let first = BlockId::new(self.term.len());
        for b in 0..self.term.len() {
            let redirect = |e: &mut Edge| {
                if entries.contains(&e.to) {
                    e.state = Some((k, e.to.index() as u32));
                    e.to = first;
                }
            };
            match &mut self.term[b] {
                Term::Jump(e) => redirect(e),
                Term::Branch(_, t, e) => {
                    redirect(t);
                    redirect(e);
                }
                Term::Exit => {}
            }
        }
        let plain = |to: BlockId| Edge { to, params_of: to, args: Vec::new(), state: None };
        let last = entries.len() - 1;
        for (i, &e) in entries[..last].iter().enumerate() {
            let next = if i + 1 == last { entries[last] } else { BlockId::new(first.index() + i + 1) };
            let test = Cond::Atom(format!("{var} == {}", e.index()));
            self.term.push(Term::Branch(test, plain(e), plain(next)));
        }
        // The function starts inside the region: start at the dispatcher.
        let init = if entries.contains(&self.entry) {
            let init = self.entry.index() as u32;
            self.entry = first;
            init
        } else {
            0
        };
        self.states.push(init);
    }

    /// Strongly connected components of the subgraph on `nodes` that contain a
    /// cycle (Tarjan, iterative).
    fn sccs(&self, nodes: &[BlockId]) -> Vec<Vec<BlockId>> {
        let n = self.term.len();
        let mut member = vec![false; n];
        for &b in nodes {
            member[b.index()] = true;
        }
        let mut index = vec![u32::MAX; n];
        let mut low = vec![0u32; n];
        let mut on_stack = vec![false; n];
        let mut stack = Vec::new();
        let mut out = Vec::new();
        let mut next = 0u32;
        for &root in nodes {
            if index[root.index()] != u32::MAX {
                continue;
            }
            let mut call: Vec<(BlockId, usize)> = vec![(root, 0)];
            index[root.index()] = next;
            low[root.index()] = next;
            next += 1;
            stack.push(root);
            on_stack[root.index()] = true;
            while let Some(&mut (b, ref mut i)) = call.last_mut() {
                let succ = self.succs(b);
                if *i < 2 {
                    let s = succ[*i];
                    *i += 1;
                    let Some(s) = s.filter(|s| member[s.index()]) else { continue };
                    if index[s.index()] == u32::MAX {
                        index[s.index()] = next;
                        low[s.index()] = next;
                        next += 1;
                        stack.push(s);
                        on_stack[s.index()] = true;
                        call.push((s, 0));
                    } else if on_stack[s.index()] {
                        low[b.index()] = low[b.index()].min(index[s.index()]);
                    }
                    continue;
                }
                call.pop();
                if let Some(&(p, _)) = call.last() {
                    low[p.index()] = low[p.index()].min(low[b.index()]);
                }
                if low[b.index()] == index[b.index()] {
                    let mut scc = Vec::new();
                    loop {
                        let x = stack.pop().unwrap();
                        on_stack[x.index()] = false;
                        scc.push(x);
                        if x == b {
                            break;
                        }
                    }
                    let cyclic = scc.len() > 1 || succ.contains(&Some(b));
                    if cyclic {
                        out.push(scc);
                    }
                }
            }
        }
        out
    }
}

struct Structurer<'a, 'b> {
    g: &'a Graph,
    cfg: &'a Cfg,
    src: &'b mut dyn Source,
    forward_in: Vec<u32>,
    loop_head: Vec<bool>,
    /// Dominator-tree children, latest in reverse postorder first.
    children: Vec<Vec<BlockId>>,
    /// Loop exits emitted after their loop: edges to them are `break`s.
    follows: Vec<bool>,
}

impl Structurer<'_, '_> {
    fn is_join(&self, b: BlockId) -> bool {
        self.forward_in[b.index()] >= 2
    }

    /// Blocks of the natural loop headed by `h`: those that reach a back edge to
    /// `h` without passing through `h`.
    fn loop_body(&self, h: BlockId) -> Vec<bool> {
        let mut inside = vec![false; self.g.term.len()];
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
        match joins.split_first() {
            Some((&y, rest)) => {
                let inner = self.within(b, rest);
                let mut out = vec![Node::Block { label: y.index() as u32, body: inner }];
                out.extend(self.tree(y));
                out
            }
            None => {
                let g = self.g;
                let mut out = if b.index() < g.real { self.src.stmts(b) } else { Vec::new() };
                for &x in g.early.get(b.index()).into_iter().flatten() {
                    out.extend(self.src.stmts(x));
                }
                match &g.term[b.index()] {
                    Term::Jump(e) => out.extend(self.branch(b, e)),
                    Term::Branch(c, t, e) => {
                        let then = self.branch(b, t);
                        let els = self.branch(b, e);
                        out.push(Node::If { c: c.clone(), then, els });
                    }
                    Term::Exit => out.push(self.src.exit(b)),
                }
                out
            }
        }
    }

    /// Control passes from `from` along `e`.
    fn branch(&mut self, from: BlockId, e: &Edge) -> Vec<Node> {
        let to = e.to;
        let mut out = if e.args.is_empty() { Vec::new() } else { self.src.edge(e.params_of, &e.args) };
        if let Some((k, v)) = e.state {
            out.push(Node::Line(format!("{} = {v};", state_var(k))));
        }
        if self.cfg.rpo_index[to.index()] <= self.cfg.rpo_index[from.index()] {
            out.push(Node::Continue(to.index() as u32));
        } else if self.is_join(to) || self.follows[to.index()] {
            out.push(Node::Break(Label::Block(to.index() as u32)));
        } else {
            out.extend(self.tree(to));
        }
        out
    }
}

/// `loop { if !c { break; } .. }` is `while c { .. }`. Runs after `tidy`, once
/// `break`s that leave a loop target the loop. When that `break` is the loop's
/// only exit, statements before it move after the loop:
/// `loop { if !c { x = 1; break; } .. }` is `while c { .. } x = 1;`.
fn whiles(nodes: Vec<Node>) -> Vec<Node> {
    let mut out = Vec::with_capacity(nodes.len());
    for n in nodes {
        match n {
            Node::If { c, then, els } => out.push(Node::If { c, then: whiles(then), els: whiles(els) }),
            Node::Block { label, body } => out.push(Node::Block { label, body: whiles(body) }),
            Node::While { head, c, body } => out.push(Node::While { head, c, body: whiles(body) }),
            Node::Loop { head, body } => {
                let mut body = whiles(body);
                let exit = Node::Break(Label::Loop(head));
                let test = match body.first() {
                    Some(Node::If { then, els, .. }) if els.is_empty() => match then.split_last() {
                        Some((last, [])) => same_jump(last, &exit),
                        Some((last, lines)) => {
                            same_jump(last, &exit)
                                && lines.iter().all(|l| matches!(l, Node::Line(_)))
                                && !uses_break(&body[1..], Label::Loop(head))
                        }
                        None => false,
                    },
                    _ => false,
                };
                if !test {
                    out.push(Node::Loop { head, body });
                    continue;
                }
                let Node::If { c, mut then, .. } = body.remove(0) else { unreachable!() };
                then.pop();
                out.push(Node::While { head, c: !c, body });
                out.extend(then);
            }
            n => out.push(n),
        }
    }
    out
}

fn same_jump(a: &Node, b: &Node) -> bool {
    match (a, b) {
        (Node::Break(x), Node::Break(y)) => x == y,
        (Node::Continue(x), Node::Continue(y)) => x == y,
        _ => false,
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
        Node::Line(_) | Node::Exit(_) => false,
    })
}

/// Control never reaches the end of `nodes`.
fn diverges(nodes: &[Node]) -> bool {
    match nodes.last() {
        Some(Node::Exit(_) | Node::Break(_) | Node::Continue(_)) => true,
        Some(Node::If { then, els, .. }) => diverges(then) && diverges(els),
        Some(Node::Block { label, body }) => diverges(body) && !uses(body, Label::Block(*label)),
        Some(Node::Loop { head, body }) => !uses_break(body, Label::Loop(*head)),
        Some(Node::While { .. } | Node::Line(_)) | None => false,
    }
}

/// Like `uses`, counting only `break`s (a `continue` stays inside the loop).
fn uses_break(nodes: &[Node], l: Label) -> bool {
    nodes.iter().any(|n| match n {
        Node::Break(x) => *x == l,
        Node::If { then, els, .. } => uses_break(then, l) || uses_break(els, l),
        Node::Loop { body, .. } | Node::While { body, .. } | Node::Block { body, .. } => uses_break(body, l),
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
        _ => false,
    };
    if changed {
        *nodes = level(std::mem::take(nodes));
    }
    changed
}

fn size(nodes: &[Node]) -> usize {
    nodes
        .iter()
        .map(|n| match n {
            Node::If { then, els, .. } => 1 + size(then) + size(els),
            Node::Loop { body, .. } | Node::While { body, .. } | Node::Block { body, .. } => 1 + size(body),
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
                Node::Loop { head, body }
            }
            Node::Block { label, mut body } => {
                drop_tail(&mut body, &Node::Break(Label::Block(label)));
                body = tidy(body);
                // `'b: { ..; 'l: loop { .. break 'b .. } }`: leaving the loop
                // lands at the end of the block too, so `break 'b` there is `break 'l`.
                tail_loops(&mut body, Label::Block(label));
                // `'b: { loop { .. break; .. break 'b; } C }` with one plain `break`:
                // run `C` before that `break` instead, and `break 'b` is `break`.
                let l = Label::Block(label);
                sink_loop_tail(&mut body, l);
                // `'b: { if c { A; break 'b } B }` is `if c { A } else { B }`.
                if uses(&body, l) {
                    let mut flat = absorb(body.clone(), l);
                    drop_tail(&mut flat, &Node::Break(l));
                    tail_loops(&mut flat, l);
                    if !uses(&flat, l) {
                        body = flat;
                    }
                }
                Node::Block { label, body }
            }
            n => n,
        })
        .collect();
    level(nodes)
}

/// `if c { A; break l } B` (`B` the rest of the list) is `if c { A } else { B }`,
/// and more generally, when an `if` has a single path that falls through and
/// others that `break l`, the rest of the list can move to the end of that path.
/// Applied from the first such `if` on; the caller drops the `break`s.
fn absorb(mut nodes: Vec<Node>, l: Label) -> Vec<Node> {
    for i in 0..nodes.len().saturating_sub(1) {
        if matches!(nodes[i], Node::If { .. }) && falls(std::slice::from_ref(&nodes[i])) == 1 && uses(std::slice::from_ref(&nodes[i]), l) {
            let rest = absorb(nodes.split_off(i + 1), l);
            append(&mut nodes, rest);
            return nodes;
        }
    }
    nodes
}

/// How many paths through `nodes` reach its end (an `if`'s two arms count
/// separately; anything else that can fall through counts once).
fn falls(nodes: &[Node]) -> usize {
    match nodes.last() {
        None => 1,
        Some(Node::If { then, els, .. }) => falls(then) + falls(els),
        Some(_) if diverges(nodes) => 0,
        Some(_) => 1,
    }
}

/// Add `rest` at the end of the one path through `nodes` that falls through.
fn append(nodes: &mut Vec<Node>, rest: Vec<Node>) {
    if let Some(Node::If { then, els, .. }) = nodes.last_mut() {
        if falls(then) > 0 {
            return append(then, rest);
        }
        return append(els, rest);
    }
    nodes.extend(rest);
}

/// In a loop at the end of `nodes` (or at the end of an arm of an `if` at the
/// end, and so on), leaving the loop lands where leaving `nodes` does: a
/// `break l` there is a `break` of the loop.
fn tail_loops(nodes: &mut [Node], l: Label) {
    match nodes.last_mut() {
        Some(Node::Loop { head, body }) => {
            let h = Label::Loop(*head);
            retarget(body, l, h);
        }
        Some(Node::If { then, els, .. }) => {
            tail_loops(then, l);
            tail_loops(els, l);
        }
        Some(Node::Block { body, .. }) => tail_loops(body, l),
        _ => {}
    }
}

fn sink_loop_tail(body: &mut Vec<Node>, l: Label) {
    let Some(at) = body.iter().rposition(|n| matches!(n, Node::Loop { .. })) else { return };
    let Node::Loop { head, body: lb } = &body[at] else { unreachable!() };
    let exit = Label::Loop(*head);
    if at + 1 == body.len() || !uses(lb, l) || count_breaks(lb, exit) != 1 {
        return;
    }
    let tail = body.split_off(at + 1);
    let Some(Node::Loop { head, body: lb }) = body.last_mut() else { unreachable!() };
    let mut tail = Some(tail);
    insert_before_break(lb, exit, &mut tail);
    let lb_head = Label::Loop(*head);
    retarget(lb, l, lb_head);
}

/// How many `break`s target `l`.
fn count_breaks(nodes: &[Node], l: Label) -> usize {
    nodes
        .iter()
        .map(|n| match n {
            Node::Break(x) => (*x == l) as usize,
            Node::If { then, els, .. } => count_breaks(then, l) + count_breaks(els, l),
            Node::Loop { body, .. } | Node::While { body, .. } | Node::Block { body, .. } => count_breaks(body, l),
            _ => 0,
        })
        .sum()
}

/// Put `code` in front of the `break` to `l`.
fn insert_before_break(nodes: &mut Vec<Node>, l: Label, code: &mut Option<Vec<Node>>) {
    let mut i = 0;
    while i < nodes.len() && code.is_some() {
        match &mut nodes[i] {
            Node::Break(x) if *x == l => {
                let c = code.take().unwrap();
                let keep = !diverges(&c);
                let n = c.len();
                nodes.splice(i..i, c);
                if !keep {
                    nodes.remove(i + n);
                }
                return;
            }
            Node::If { then, els, .. } => {
                insert_before_break(then, l, code);
                insert_before_break(els, l, code);
            }
            Node::Loop { body, .. } | Node::While { body, .. } | Node::Block { body, .. } => insert_before_break(body, l, code),
            _ => {}
        }
        i += 1;
    }
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
                    (true, true) => {}
                    (false, true) => out.push(Node::If { c, then, els }),
                    (true, false) => todo.push(Node::If { c: !c, then: els, els: then }),
                    // Early exit: keep the branch that leaves (the shorter one if
                    // both do) under the `if`, and the other after it.
                    _ if td && (!ed || size(&then) <= size(&els)) => {
                        out.push(Node::If { c, then, els: Vec::new() });
                        todo.extend(els.into_iter().rev());
                    }
                    _ if ed => {
                        out.push(Node::If { c: !c, then: els, els: Vec::new() });
                        todo.extend(then.into_iter().rev());
                    }
                    _ => out.push(Node::If { c, then, els }),
                }
            }
            n => out.push(n),
        }
    }
    out
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
                Node::Line(s) | Node::Exit(s) => {
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
                        Node::While { c, .. } => {
                            let _ = writeln!(out, "{ind}{name}while {c} {{");
                        }
                        _ => {
                            let _ = writeln!(out, "{ind}{name}loop {{");
                        }
                    }
                    self.stack.push(l);
                    self.list(body, depth + 1, out);
                    self.stack.pop();
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
