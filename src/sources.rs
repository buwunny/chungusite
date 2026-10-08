//! Where a pointer comes from, for statistics only: the stack frame, a global, an
//! argument, or something else (a pointer loaded from memory, a call result).
//!
//! A forward dataflow over the SSA graph like `borrow::origins`, but on a fixed
//! four-bit lattice and without points-to or call summaries, so it gives the same
//! answer whatever the borrow analysis proves. `EmitStats::raw_by` uses it to say
//! which kinds of object the raw accesses that remain go through.
use crate::cfg::Cfg;
use crate::ir::*;

pub const ARG: u8 = 1;
pub const FRAME: u8 = 2;
pub const GLOBAL: u8 = 4;

/// Index into `EmitStats::raw_by`.
pub const SOURCES: [&str; 4] = ["frame", "global", "arg", "other"];

/// The `SOURCES` index for a mask: frame first, then global, then argument.
pub fn bucket(mask: u8) -> usize {
    if mask & FRAME != 0 {
        0
    } else if mask & GLOBAL != 0 {
        1
    } else if mask & ARG != 0 {
        2
    } else {
        3
    }
}

/// A mask of `ARG | FRAME | GLOBAL` for every value.
pub fn sources(f: &Function) -> Vec<u8> {
    use InstKind::*;
    let cfg = Cfg::new(f);
    let mut s = vec![0u8; f.insts.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &cfg.rpo {
            let blk = &f.blocks[b];
            for (k, &p) in blk.params.get(&f.value_pool).iter().enumerate() {
                let mut new = if b == f.entry {
                    match f.insts[p].kind {
                        InstKind::BlockParam(crate::borrow::RSP) => FRAME,
                        _ => ARG,
                    }
                } else {
                    0
                };
                for &pred in cfg.preds(b) {
                    crate::borrow::incoming(f, pred, b, k, |a| new |= s[a.index()]);
                }
                if new != s[p.index()] {
                    s[p.index()] = new;
                    changed = true;
                }
            }
            for &id in blk.insts.get(&f.value_pool) {
                let of = |v: ValueId| s[v.index()];
                let new = match f.insts[id].kind {
                    AddrOfLocal(_) => FRAME,
                    IntToPtr(v) if matches!(f.insts[v].kind, Const(_)) => GLOBAL,
                    PtrOffset { base, index, .. } => of(base) | index.map_or(0, of),
                    Bin { op: BinOp::Add | BinOp::Sub, lhs, rhs } => of(lhs) | of(rhs),
                    Select { t, f: e, .. } => of(t) | of(e),
                    Cast { kind: CastKind::Bitcast, v } | IntToPtr(v) | PtrToInt(v) => of(v),
                    _ => 0,
                };
                if new != s[id.index()] {
                    s[id.index()] = new;
                    changed = true;
                }
            }
        }
    }
    s
}
