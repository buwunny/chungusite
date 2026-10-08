//! IR -> C-like text, emitted one pre-token at a time.
//!
//! Byte-level BPE tokenizers (GPT-2 / RoBERTa / Qwen / CodeT5+ style) first split
//! text with a regex into pre-tokens (` v`, `12`, ` =`, `(`, `);`, `\n` ...) and run
//! BPE inside each one; merges never cross a pre-token boundary. So if we emit text
//! in exactly those pieces, the ids of the whole text are the concatenation of the
//! ids of each piece, and each piece's ids can be computed once and cached.
//! The rules that matter for the default regex:
//! - a word or number carries its leading space (` load`, ` 8`), so `num_sp` and
//!   `num` are different pieces;
//! - a run of punctuation is one pre-token (`);`, `):`, `();`), so it must be one
//!   piece, never `)` followed by `;`.
//!
//! `tests/exact.rs` checks the result against a real tokenizer.
use crate::ir::*;

pub trait Emit {
    /// A fixed pre-token, e.g. `" load"` or `");"`.
    fn lit(&mut self, s: &'static str);
    /// A number with no leading space (`v12` -> `v`, then `num(12)`).
    fn num(&mut self, n: u64);
    /// A number with its leading space (`* 8` -> `" *"`, then `num_sp(8)`).
    fn num_sp(&mut self, n: u64);
}

/// Debug sink: builds the text the model sees.
#[derive(Default)]
pub struct Text(pub String);

impl Emit for Text {
    fn lit(&mut self, s: &'static str) { self.0.push_str(s) }
    fn num(&mut self, n: u64) { use std::fmt::Write; let _ = write!(self.0, "{n}"); }
    fn num_sp(&mut self, n: u64) { use std::fmt::Write; let _ = write!(self.0, " {n}"); }
}

fn val(e: &mut impl Emit, sp: bool, v: ValueId) {
    e.lit(if sp { " v" } else { "v" });
    e.num(v.index() as u64);
}

/// `(v1, v2` + closer, or the single pre-token `closer_empty` when there are no values.
fn list(e: &mut impl Emit, vs: &[ValueId], closer: &'static str, closer_empty: &'static str) {
    if vs.is_empty() {
        e.lit(closer_empty);
        return;
    }
    e.lit("(");
    for (i, &v) in vs.iter().enumerate() {
        if i > 0 { e.lit(","); }
        val(e, i > 0, v);
    }
    e.lit(closer);
}

pub fn serialize(f: &Function, e: &mut impl Emit) {
    for (b, blk) in f.blocks.iter() {
        e.lit("bb");
        e.num(b.index() as u64);
        list(e, blk.params.get(&f.value_pool), "):", ":");
        e.lit("\n");
        for &id in blk.insts.get(&f.value_pool) {
            inst(f, id, e);
        }
        term(f, blk.term, e);
    }
}

fn inst(f: &Function, id: ValueId, e: &mut impl Emit) {
    let k = f.insts[id].kind;
    if let InstKind::Store { ptr, val: v, .. } = k {
        e.lit(" *");
        val(e, false, ptr);
        e.lit(" =");
        val(e, true, v);
        e.lit(";");
        e.lit("\n");
        return;
    }
    val(e, true, id);
    e.lit(" =");
    match k {
        InstKind::Const(c) => e.num_sp(f.consts[c.index()] as u64),
        InstKind::Bin { op, lhs, rhs } => {
            val(e, true, lhs);
            e.lit(match op {
                BinOp::Add => " +", BinOp::Sub => " -", BinOp::Mul | BinOp::UMulHi | BinOp::SMulHi => " *", BinOp::UDiv | BinOp::SDiv => " /",
                BinOp::URem | BinOp::SRem => " %", BinOp::And => " &", BinOp::Or => " |", BinOp::Xor => " ^",
                BinOp::Shl | BinOp::RotL => " <<", BinOp::LShr | BinOp::AShr | BinOp::RotR => " >>",
            });
            val(e, true, rhs);
        }
        InstKind::Cmp { cc, lhs, rhs } => {
            val(e, true, lhs);
            e.lit(match cc {
                Cond::Eq => " ==", Cond::Ne => " !=",
                Cond::Ult | Cond::Slt => " <", Cond::Ule | Cond::Sle => " <=",
                Cond::Ugt | Cond::Sgt => " >", Cond::Uge | Cond::Sge => " >=",
            });
            val(e, true, rhs);
        }
        InstKind::Cast { kind, v } => {
            e.lit(match kind {
                CastKind::ZExt => " zext", CastKind::SExt => " sext", CastKind::Trunc => " trunc",
                _ => " cast",
            });
            // `);` is one pre-token, so the call closes the statement itself.
            list(e, &[v], ");", "();");
            e.lit("\n");
            return;
        }
        InstKind::Load { ptr, .. } => {
            e.lit(" *");
            val(e, false, ptr);
        }
        InstKind::PtrOffset { base, index, scale, disp } => {
            val(e, true, base);
            if let Some(i) = index {
                e.lit(" +");
                val(e, true, i);
                e.lit(" *");
                e.num_sp(scale as u64);
            }
            if disp != 0 {
                e.lit(if disp < 0 { " -" } else { " +" });
                e.num_sp(disp.unsigned_abs() as u64);
            }
        }
        InstKind::IntToPtr(v) => {
            e.lit(" ptr");
            list(e, &[v], ");", "();");
            e.lit("\n");
            return;
        }
        _ => e.lit(" opaque"),
    }
    e.lit(";");
    e.lit("\n");
}

fn term(f: &Function, t: Terminator, e: &mut impl Emit) {
    let goto = |e: &mut _, b: BlockId, args: &[ValueId]| {
        Emit::lit(e, " goto");
        Emit::lit(e, " bb");
        Emit::num(e, b.index() as u64);
        list(e, args, ");", "();");
    };
    match t {
        Terminator::Jump { to, args } => goto(e, to, args.get(&f.value_pool)),
        Terminator::Branch { c, t, f: el, args } => {
            let a = args.get(&f.value_pool);
            let nt = f.blocks[t].params.len as usize;
            e.lit(" if");
            val(e, true, c);
            goto(e, t, &a[..nt]);
            e.lit(" else");
            goto(e, el, &a[nt..]);
        }
        Terminator::Return(Some(v)) => {
            e.lit(" return");
            val(e, true, v);
            e.lit(";");
        }
        Terminator::Return(None) => { e.lit(" return"); e.lit(";"); }
        _ => { e.lit(" unreachable"); e.lit(";"); }
    }
    e.lit("\n");
}
