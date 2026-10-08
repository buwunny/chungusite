//! Control-flow structuring: turn a reducible CFG into nested `if`/`else`, `loop`,
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
//! The raw result is correct but noisy, so `tidy` then removes what isn't needed:
//! a `break`/`continue` that control would reach anyway by falling off the end,
//! labeled blocks nothing breaks out of, and `if c { A } else { B }` where `A`
//! always leaves (`if c { A } B`). Labels are printed only where a plain
//! `break`/`continue` would not mean the same thing.
//!
//! Irreducible CFGs (a cycle with more than one entry) have no such nesting;
//! `reducible` says when the caller should keep the state machine instead.
use crate::cfg::Cfg;
use crate::ir::*;
use std::fmt::Write;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Label {
    /// The `loop` headed by this block.
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
    If { c: String, then: Vec<Node>, els: Vec<Node> },
    /// `match v { pattern => arm, ... }`; the last pattern is `_`.
    Match { v: String, arms: Vec<(String, Vec<Node>)> },
    Loop { head: u32, body: Vec<Node> },
    Block { label: u32, body: Vec<Node> },
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
    /// The value a `Switch` matches on, as a `u64` expression.
    fn scrutinee(&self, v: ValueId) -> String;
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
    cfg.rpo.iter().all(|&b| {
        f.blocks[b].term.successors(&f.value_pool).all(|s| {
            cfg.rpo_index[s.index()] > cfg.rpo_index[b.index()] || cfg.dominates(s, b)
        })
    })
}

/// Structure `f`, or `None` if its CFG is irreducible or nests too deeply.
pub fn structure(f: &Function, cfg: &Cfg, src: &mut dyn Source) -> Option<Vec<Node>> {
    if !reducible(f, cfg) {
        return None;
    }
    let n = f.blocks.len();
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
        // A switch is one edge per distinct target: each target is one arm.
        let mut succs: Vec<BlockId> = f.blocks[b].term.successors(&f.value_pool).collect();
        if let Terminator::Switch { .. } = f.blocks[b].term {
            succs.sort_unstable_by_key(|s| s.index());
            succs.dedup();
        }
        for s in succs {
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
    let mut s = Structurer { f, cfg, src, forward_in, loop_head, children, follows };
    let body = s.tree(f.entry);
    Some(tidy(body))
}

struct Structurer<'a, 'b> {
    f: &'a Function,
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
        let mut inside = vec![false; self.f.blocks.len()];
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
                let mut out = self.src.stmts(b);
                let f = self.f;
                match f.blocks[b].term {
                    Terminator::Jump { to, args } => {
                        out.extend(self.branch(b, to, args.get(&f.value_pool)));
                    }
                    Terminator::Branch { c, t, f: e, args } => {
                        let a = args.get(&f.value_pool);
                        let nt = f.blocks[t].params.len as usize;
                        let then = self.branch(b, t, &a[..nt]);
                        let els = self.branch(b, e, &a[nt..]);
                        out.push(Node::If { c: self.src.cond(c), then, els });
                    }
                    Terminator::Switch { v, .. } => {
                        let mut arms = Vec::new();
                        for (to, cases, args) in switch_arms(f, b) {
                            let body = self.branch(b, to, &args);
                            arms.push((cases, body));
                        }
                        out.push(Node::Match { v: self.src.scrutinee(v), arms });
                    }
                    _ => out.push(self.src.exit(b)),
                }
                out
            }
        }
    }

    /// Control passes from `from` to `to`.
    fn branch(&mut self, from: BlockId, to: BlockId, args: &[ValueId]) -> Vec<Node> {
        let mut out = self.src.edge(to, args);
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

/// A `Switch`'s table grouped by target, in order of first appearance, as
/// (target, `match` pattern, edge arguments). The target with the most cases
/// goes last, with pattern `_`.
pub fn switch_arms(f: &Function, b: BlockId) -> Vec<(BlockId, String, Vec<ValueId>)> {
    let mut groups: Vec<(BlockId, Vec<usize>, Vec<ValueId>)> = Vec::new();
    for (k, (to, args)) in f.edges(b).enumerate() {
        match groups.iter_mut().find(|g| g.0 == to) {
            Some(g) => g.1.push(k),
            None => groups.push((to, vec![k], args.to_vec())),
        }
    }
    if let Some(big) = (0..groups.len()).max_by_key(|&i| (groups[i].1.len(), std::cmp::Reverse(i))) {
        let g = groups.remove(big);
        groups.push(g);
    }
    let last = groups.len().saturating_sub(1);
    groups
        .into_iter()
        .enumerate()
        .map(|(i, (to, cases, args))| {
            let pat = if i == last { "_".to_string() } else { patterns(&cases) };
            (to, pat, args)
        })
        .collect()
}

/// `0 | 2..=5 | 9`: runs of consecutive cases as ranges.
fn patterns(cases: &[usize]) -> String {
    let mut parts = Vec::new();
    let mut i = 0;
    while i < cases.len() {
        let mut j = i;
        while j + 1 < cases.len() && cases[j + 1] == cases[j] + 1 {
            j += 1;
        }
        parts.push(match j - i {
            0 => format!("{}", cases[i]),
            1 => format!("{} | {}", cases[i], cases[j]),
            _ => format!("{}..={}", cases[i], cases[j]),
        });
        i = j + 1;
    }
    parts.join(" | ")
}

// ---------------------------------------------------------------------------
// Tidying

/// Does any `break`/`continue` in `nodes` target `l`?
fn uses(nodes: &[Node], l: Label) -> bool {
    nodes.iter().any(|n| match n {
        Node::Break(x) => *x == l,
        Node::Continue(h) => l == Label::Loop(*h),
        Node::If { then, els, .. } => uses(then, l) || uses(els, l),
        Node::Match { arms, .. } => arms.iter().any(|(_, a)| uses(a, l)),
        Node::Loop { body, .. } | Node::Block { body, .. } => uses(body, l),
        Node::Line(_) | Node::Exit(_) => false,
    })
}

/// Control never reaches the end of `nodes`.
fn diverges(nodes: &[Node]) -> bool {
    match nodes.last() {
        Some(Node::Exit(_) | Node::Break(_) | Node::Continue(_)) => true,
        Some(Node::If { then, els, .. }) => diverges(then) && diverges(els),
        Some(Node::Match { arms, .. }) => arms.iter().all(|(_, a)| diverges(a)),
        Some(Node::Block { label, body }) => diverges(body) && !uses(body, Label::Block(*label)),
        Some(Node::Loop { head, body }) => !uses_break(body, Label::Loop(*head)),
        Some(Node::Line(_)) | None => false,
    }
}

/// Like `uses`, counting only `break`s (a `continue` stays inside the loop).
fn uses_break(nodes: &[Node], l: Label) -> bool {
    nodes.iter().any(|n| match n {
        Node::Break(x) => *x == l,
        Node::If { then, els, .. } => uses_break(then, l) || uses_break(els, l),
        Node::Match { arms, .. } => arms.iter().any(|(_, a)| uses_break(a, l)),
        Node::Loop { body, .. } | Node::Block { body, .. } => uses_break(body, l),
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
        Some(Node::Match { arms, .. }) => arms.iter_mut().fold(false, |c, (_, a)| drop_tail(a, jump) | c),
        // Falling off an inner block's end also falls off ours.
        Some(Node::Block { body, .. }) => drop_tail(body, jump),
        _ => false,
    };
    if changed {
        *nodes = level(std::mem::take(nodes));
    }
    changed
}

fn not(c: &str) -> String {
    match c.strip_prefix('!') {
        Some(inner) if !inner.contains(' ') => inner.to_string(),
        _ if c.contains(' ') => format!("!({c})"),
        _ => format!("!{c}"),
    }
}

fn size(nodes: &[Node]) -> usize {
    nodes
        .iter()
        .map(|n| match n {
            Node::If { then, els, .. } => 1 + size(then) + size(els),
            Node::Match { arms, .. } => 1 + arms.iter().map(|(_, a)| size(a)).sum::<usize>(),
            Node::Loop { body, .. } | Node::Block { body, .. } => 1 + size(body),
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
            Node::Match { v, arms } => Node::Match { v, arms: arms.into_iter().map(|(p, a)| (p, tidy(a))).collect() },
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
                if let Some(Node::Loop { head, body: lb }) = body.last_mut() {
                    retarget(lb, Label::Block(label), Label::Loop(*head));
                }
                Node::Block { label, body }
            }
            n => n,
        })
        .collect();
    level(nodes)
}

fn retarget(nodes: &mut [Node], from: Label, to: Label) {
    for n in nodes {
        match n {
            Node::Break(l) if *l == from => *l = to,
            Node::If { then, els, .. } => {
                retarget(then, from, to);
                retarget(els, from, to);
            }
            Node::Match { arms, .. } => arms.iter_mut().for_each(|(_, a)| retarget(a, from, to)),
            Node::Loop { body, .. } | Node::Block { body, .. } => retarget(body, from, to),
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
            Node::Match { arms, .. } => arms.iter().for_each(|(_, a)| needs_label(a, stack, named)),
            Node::Loop { head, body } => {
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
                Node::Match { v, arms } => {
                    let _ = writeln!(out, "{ind}match {v} {{");
                    for (pat, arm) in arms {
                        let _ = writeln!(out, "{ind}    {pat} => {{");
                        self.list(arm, depth + 2, out);
                        let _ = writeln!(out, "{ind}    }}");
                    }
                    let _ = writeln!(out, "{ind}}}");
                }
                Node::Loop { head, body } => {
                    let l = Label::Loop(*head);
                    let name = if self.named.contains(&l) { format!("{}: ", label_name(l)) } else { String::new() };
                    let _ = writeln!(out, "{ind}{name}loop {{");
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
