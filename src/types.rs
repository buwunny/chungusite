//! Type recovery (docs/types.md): integer widths and signedness for every value,
//! narrow argument and return types, and what pointer arguments point to (a
//! struct, a scalar or an array), instead of `u64` everywhere.
//!
//! Types come from two places, and the first one decides what is allowed:
//!
//! 1. **Facts** from the code itself (`Facts`), which are always sound:
//!    * *Demanded bits*: how many low bits of each value anything uses. An argument
//!      whose uses only need its low 32 bits can be taken as a 32-bit integer.
//!    * *Zero-extended returns*: every return value provably fits in N bits, so
//!      the function can return an N-bit integer and callers widen it again.
//!    * *Signedness*: values that must have one Rust type (operands of the same
//!      operation, block parameters and their arguments) form classes, and each
//!      class votes: signed comparisons, `SAR`, `IDIV` and `MOVSX` for signed,
//!      unsigned comparisons, `SHR`, `DIV` and `MOVZX` for unsigned. Signedness
//!      only changes how the emitter spells a value; every operation keeps its
//!      exact x86 meaning (an unsigned compare of two `i32`s casts them).
//!    * *Accesses*: every load and store whose address is an argument plus a
//!      constant offset (from `borrow`'s origins), or plus a computed offset
//!      whose alignment is known (`residues`).
//! 2. **Proposals** (`TypeModel`): DWARF (`dwarf.rs`) and, later, the ML type
//!    model (docs/ml-runtime.md) propose C types and names for arguments and the
//!    return value. Each proposal is checked against the facts and accepted or
//!    rejected with a reason: a 32-bit `int` argument whose upper bits are used is
//!    rejected, and so is a `struct S *` whose accesses don't land on `S`'s fields.
//!
//! What isn't proposed is inferred from the facts alone: a pointer argument
//! accessed at a few constant offsets becomes a struct with fields `f0`, `f8`,
//! ...; one indexed with a single element size becomes a slice.
//!
//! The emitter looks types up in `FnTypes`; nothing here changes the IR.
use crate::abi::{Sig, STACK_ARG_BASE, SYSV_ARGS};
use crate::borrow::{analyze, Analysis, Class, Off, RSP};
use crate::cfg::Cfg;
use crate::ir::*;
use crate::verify::for_each_operand;
use std::fmt::Write as _;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Types

/// An integer type: `u8`..`u64`, `i8`..`i64`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct IntTy {
    pub bytes: u8,
    pub signed: bool,
}

impl IntTy {
    pub const U64: IntTy = IntTy { bytes: 8, signed: false };

    pub fn new(bytes: u8, signed: bool) -> IntTy {
        IntTy { bytes, signed }
    }

    pub fn rust(self) -> &'static str {
        match (self.bytes, self.signed) {
            (1, false) => "u8",
            (2, false) => "u16",
            (4, false) => "u32",
            (1, true) => "i8",
            (2, true) => "i16",
            (4, true) => "i32",
            (_, true) => "i64",
            (_, false) => "u64",
        }
    }

    /// The unsigned type of the same width.
    pub fn unsigned(self) -> &'static str {
        IntTy { signed: false, ..self }.rust()
    }

    pub fn bits(self) -> u32 {
        self.bytes as u32 * 8
    }
}

/// A C type, as debug info or a model describes it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CType {
    /// Integers, `bool` (as `u8`), enums and `char`.
    Int(IntTy),
    Float { bytes: u8 },
    /// A pointer, to `pointee` when it is known (`None` for `void *`).
    Ptr(Option<Arc<CType>>),
    Struct(Arc<StructTy>),
    Array { elem: Arc<CType>, len: u64 },
    /// Bytes nothing describes: unions, enums with data, bit-fields, incomplete types.
    Opaque { bytes: u64 },
}

/// A struct with its layout. `fields` are sorted by offset and don't overlap.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StructTy {
    pub name: String,
    pub size: u64,
    pub fields: Vec<Field>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Field {
    pub name: String,
    pub off: u64,
    pub ty: CType,
}

impl CType {
    pub fn size(&self) -> u64 {
        match self {
            CType::Int(t) => t.bytes as u64,
            CType::Float { bytes } => *bytes as u64,
            CType::Ptr(_) => 8,
            CType::Struct(s) => s.size,
            CType::Array { elem, len } => elem.size().saturating_mul(*len),
            CType::Opaque { bytes } => *bytes,
        }
    }

    /// C-like spelling, for `--emit types`.
    pub fn describe(&self) -> String {
        match self {
            CType::Int(t) => t.rust().to_string(),
            CType::Float { bytes } => format!("f{}", *bytes as u32 * 8),
            CType::Ptr(None) => "void *".to_string(),
            CType::Ptr(Some(p)) => format!("{} *", p.describe()),
            CType::Struct(s) => format!("struct {}", s.name),
            CType::Array { elem, len } => format!("{}[{len}]", elem.describe()),
            CType::Opaque { bytes } => format!("opaque[{bytes}]"),
        }
    }
}

/// The scalar a load or store reaches inside a struct.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Leaf {
    Int(IntTy),
    Float(u8),
}

impl Leaf {
    pub fn rust(self) -> &'static str {
        match self {
            Leaf::Int(t) => t.rust(),
            Leaf::Float(4) => "f32",
            Leaf::Float(_) => "f64",
        }
    }
}

/// The place `bytes` bytes at `off` inside `ty` name, as a Rust projection
/// (`.pos.x`, `.items[3]`), and its scalar type. `None` if the access doesn't
/// cover exactly one scalar.
pub fn locate(ty: &CType, off: u64, bytes: u8) -> Option<(String, Leaf)> {
    match ty {
        CType::Int(t) if off == 0 && t.bytes == bytes => Some((String::new(), Leaf::Int(*t))),
        CType::Float { bytes: b } if off == 0 && *b == bytes => Some((String::new(), Leaf::Float(*b))),
        CType::Ptr(_) if off == 0 && bytes == 8 => Some((String::new(), Leaf::Int(IntTy::U64))),
        CType::Struct(s) => {
            let f = s.fields.iter().find(|f| f.off <= off && off < f.off + f.ty.size())?;
            let (sub, leaf) = locate(&f.ty, off - f.off, bytes)?;
            Some((format!(".{}{sub}", f.name), leaf))
        }
        CType::Array { elem, len } => {
            let es = elem.size();
            if es == 0 || off / es >= *len {
                return None;
            }
            let (sub, leaf) = locate(elem, off % es, bytes)?;
            Some((format!("[{}]{sub}", off / es), leaf))
        }
        _ => None,
    }
}

/// A field type as Rust spells it inside a struct definition. Pointers are `u64`
/// (they are addresses everywhere else too) and `bool` is `u8`, so every bit
/// pattern is a valid value and a struct can be borrowed from any bytes.
fn rust_field_ty(ty: &CType, ident_of: &dyn Fn(&StructTy) -> String) -> String {
    match ty {
        CType::Int(t) => t.rust().to_string(),
        CType::Float { bytes: 4 } => "f32".to_string(),
        CType::Float { .. } => "f64".to_string(),
        CType::Ptr(_) => "u64".to_string(),
        CType::Struct(s) => ident_of(s),
        CType::Array { elem, len } => format!("[{}; {len}]", rust_field_ty(elem, ident_of)),
        CType::Opaque { bytes } => format!("[u8; {bytes}]"),
    }
}

/// `#[repr(C, packed)]` definition of `s` named `ident`, with explicit padding so
/// that the Rust layout is byte-for-byte the C one. Packed, so it has alignment 1
/// and a reference to it can be made from any address the binary used.
pub fn struct_def(s: &StructTy, ident: &str, ident_of: &dyn Fn(&StructTy) -> String, out: &mut String) {
    let _ = writeln!(out, "#[repr(C, packed)]\n#[derive(Copy, Clone)]\npub struct {ident} {{");
    let mut at = 0;
    for f in &s.fields {
        if f.off > at {
            let _ = writeln!(out, "    _pad{at}: [u8; {}],", f.off - at);
        }
        let _ = writeln!(out, "    pub {}: {},", f.name, rust_field_ty(&f.ty, ident_of));
        at = f.off + f.ty.size();
    }
    if s.size > at {
        let _ = writeln!(out, "    _pad{at}: [u8; {}],", s.size - at);
    }
    out.push_str("}\n");
}

/// Structs that `ty` contains by value, outermost first.
pub fn nested_structs(ty: &CType, out: &mut Vec<Arc<StructTy>>) {
    match ty {
        CType::Struct(s) => {
            out.push(s.clone());
            for f in &s.fields {
                nested_structs(&f.ty, out);
            }
        }
        CType::Array { elem, .. } => nested_structs(elem, out),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// The hook: whoever proposes types

/// A proposed type and name for one argument.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParamProposal {
    pub name: Option<String>,
    pub ty: Option<CType>,
}

/// What a `TypeModel` proposes for one function.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Proposal {
    /// Integer-class arguments in C order: the first goes in rdi, then rsi, rdx,
    /// rcx, r8, r9, then the stack. Floating-point arguments are left out (they
    /// travel in xmm registers).
    pub params: Vec<ParamProposal>,
    /// The return type, if the function returns an integer or a pointer.
    pub ret: Option<CType>,
}

/// The function a proposal is for.
pub struct FuncRef<'a> {
    pub name: &'a str,
    pub addr: u64,
    pub ir: &'a Function,
    pub sig: Sig,
}

/// Something that proposes types: debug info, or a trained model. It only
/// proposes; `infer` checks every proposal against what the code does and keeps
/// what the code allows. A model that proposes nothing changes nothing.
///
/// The ML type model (docs/ml-runtime.md) plugs in here: `propose` runs
/// `refine::to_ids` on `func.ir` and turns the model's answer into a `Proposal`.
pub trait TypeModel: Sync {
    /// Shown in `--emit types` next to what it proposed.
    fn name(&self) -> &str;
    fn propose(&self, func: &FuncRef) -> Option<Proposal>;
}

/// The default model: proposes nothing, so types come from the facts alone.
pub struct NoModel;

impl TypeModel for NoModel {
    fn name(&self) -> &str {
        "none"
    }
    fn propose(&self, _: &FuncRef) -> Option<Proposal> {
        None
    }
}

// ---------------------------------------------------------------------------
// Facts

/// One load or store through an argument.
#[derive(Copy, Clone, Debug)]
pub struct Access {
    /// The `Load` or `Store`.
    pub inst: ValueId,
    /// The value loaded or stored.
    pub value: ValueId,
    pub write: bool,
    /// 0 for a `MemCopy`, which no scalar type describes.
    pub bytes: u8,
    /// Byte offset from the argument, when it is a constant.
    pub off: Option<i64>,
    /// The address is this argument plus an offset, and no other argument.
    pub single: bool,
    /// For a computed offset: it is a multiple of `bytes`.
    pub aligned: bool,
}

/// What the code itself says about types. All of it is sound.
pub struct Facts {
    pub borrow: Analysis,
    /// Low bits of each value something uses (0..=64).
    pub demanded: Vec<u8>,
    /// Every return value fits in this many low bits (64 if any doesn't, or the
    /// function doesn't return a value in rax alone).
    pub ret_bits: u8,
    /// Loads and stores through each entry parameter.
    pub accesses: Vec<Vec<Access>>,
    /// Entry parameter index of each C argument (`None`: taken but unused).
    pub arg_param: Vec<Option<usize>>,
}

fn bits_of(ty: TyId) -> u8 {
    match ty {
        TyId::B1 => 8,
        TyId::B2 => 16,
        TyId::B4 => 32,
        TyId::BOOL => 1,
        TyId::UNIT => 0,
        _ => 64,
    }
}

fn is_int(ty: TyId) -> bool {
    matches!(ty, TyId::B1 | TyId::B2 | TyId::B4 | TyId::B8)
}

fn konst(f: &Function, v: ValueId) -> Option<u64> {
    match f.insts[v].kind {
        InstKind::Const(c) => Some(f.consts[c.index()] as u64),
        _ => None,
    }
}

/// Edge arguments of block `b`'s terminator, paired with the target parameters.
fn edges(f: &Function, t: Terminator, mut cb: impl FnMut(ValueId, ValueId)) {
    let mut pass = |to: BlockId, args: &[ValueId]| {
        for (&p, &a) in f.blocks[to].params.get(&f.value_pool).iter().zip(args) {
            cb(p, a);
        }
    };
    match t {
        Terminator::Jump { to, args } => pass(to, args.get(&f.value_pool)),
        Terminator::Branch { t, f: e, args, .. } => {
            let a = args.get(&f.value_pool);
            let nt = f.blocks[t].params.len as usize;
            pass(t, &a[..nt]);
            pass(e, &a[nt..]);
        }
        _ => {}
    }
}

impl Facts {
    pub fn gather(f: &Function, sig: Sig) -> Facts {
        let cfg = Cfg::new(f);
        let borrow = analyze(f);
        let params = f.blocks[f.entry].params.get(&f.value_pool);
        let find = |reg: u8| params.iter().position(|&p| matches!(f.insts[p].kind, InstKind::BlockParam(r) if r == reg));
        let mut arg_param: Vec<Option<usize>> = SYSV_ARGS[..sig.args as usize].iter().map(|&r| find(r)).collect();
        arg_param.extend((0..sig.stack_args).map(|j| find(STACK_ARG_BASE + j)));

        let ret_bits = ret_bits(f, &cfg, sig);
        let demanded = demanded(f, &cfg, if ret_bits <= 32 { ret_bits } else { 64 });
        let res = residues(f, &cfg, &borrow);
        let mut accesses = vec![Vec::new(); params.len()];
        for &b in &cfg.rpo {
            for &id in f.blocks[b].insts.get(&f.value_pool) {
                let (ptr, value, write, bytes) = match f.insts[id].kind {
                    InstKind::Load { ptr, .. } => (ptr, id, false, bits_of(f.insts[id].ty) / 8),
                    InstKind::Store { ptr, val, .. } => (ptr, val, true, bits_of(f.insts[val].ty) / 8),
                    InstKind::MemCopy { dst, src, .. } => {
                        for p in [dst, src] {
                            for k in each_root(borrow.origin[p.index()].roots) {
                                accesses[k].push(Access { inst: id, value: id, write: p == dst, bytes: 0, off: None, single: false, aligned: false });
                            }
                        }
                        continue;
                    }
                    _ => continue,
                };
                let o = borrow.origin[ptr.index()];
                let single = o.roots.count_ones() == 1;
                let off = match o.off {
                    Off::Known(x) if single => Some(x),
                    _ => None,
                };
                let aligned = match (res[ptr.index()], bytes) {
                    (_, 0) => false,
                    (Some((m, r)), w) => m >= w && r % w == 0,
                    (None, _) => false,
                };
                for k in each_root(o.roots) {
                    accesses[k].push(Access { inst: id, value, write, bytes, off, single, aligned });
                }
            }
        }
        Facts { borrow, demanded, ret_bits, accesses, arg_param }
    }

    /// Is entry parameter `k` dereferenced (a pointer)?
    pub fn is_pointer(&self, k: usize) -> bool {
        self.borrow.params[k].class != Class::NotPointer
    }
}

fn each_root(roots: u32) -> impl Iterator<Item = usize> {
    (0..32).filter(move |&k| roots & (1 << k) != 0)
}

/// Backward demanded-bits analysis: the low bits of each value that some use
/// needs. Addition, subtraction, multiplication, bitwise operations, left shifts
/// and truncation only need the low bits of their operands that their own result
/// needs; everything else needs all of them.
fn demanded(f: &Function, cfg: &Cfg, ret: u8) -> Vec<u8> {
    use InstKind::*;
    let full = |v: ValueId| bits_of(f.insts[v].ty);
    let mut need = vec![0u8; f.insts.len()];
    let mut changed = true;
    while changed {
        changed = false;
        let mut want = |need: &mut Vec<u8>, v: ValueId, d: u8| {
            let d = d.min(full(v));
            if d > need[v.index()] {
                need[v.index()] = d;
                changed = true;
            }
        };
        for &b in cfg.rpo.iter().rev() {
            let blk = &f.blocks[b];
            let t = blk.term;
            match t {
                Terminator::Return(Some(v)) => want(&mut need, v, ret),
                Terminator::Branch { c, .. } => want(&mut need, c, 64),
                Terminator::Switch { v, .. } => want(&mut need, v, 64),
                Terminator::TailCall { callee, args } => {
                    want(&mut need, callee, 64);
                    for &a in args.get(&f.value_pool) {
                        want(&mut need, a, 64);
                    }
                }
                _ => {}
            }
            let mut pairs = Vec::new();
            edges(f, t, |p, a| pairs.push((p, a)));
            for (p, a) in pairs {
                let d = need[p.index()];
                want(&mut need, a, d);
            }
            for &id in blk.insts.get(&f.value_pool).iter().rev() {
                let d = need[id.index()];
                match f.insts[id].kind {
                    Bin { op: BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::And | BinOp::Or | BinOp::Xor, lhs, rhs } => {
                        want(&mut need, lhs, d);
                        want(&mut need, rhs, d);
                    }
                    Bin { op: BinOp::Shl, lhs, rhs } => {
                        want(&mut need, lhs, d);
                        if d > 0 {
                            want(&mut need, rhs, 64);
                        }
                    }
                    Un { op: UnOp::Neg | UnOp::Not, v } => want(&mut need, v, d),
                    Cast { kind: CastKind::Trunc | CastKind::ZExt | CastKind::SExt | CastKind::Bitcast, v } => want(&mut need, v, d),
                    Select { c, t, f: e } => {
                        want(&mut need, t, d);
                        want(&mut need, e, d);
                        if d > 0 {
                            want(&mut need, c, 64);
                        }
                    }
                    PtrOffset { base, index, .. } => {
                        want(&mut need, base, d);
                        if let Some(i) = index {
                            want(&mut need, i, d);
                        }
                    }
                    // pure, and only its result's use matters
                    k @ (Bin { .. } | Un { .. } | Cmp { .. } | Cast { .. }) => {
                        if d > 0 {
                            for_each_operand(k, f, |v| want(&mut need, v, 64));
                        }
                    }
                    CallOut { .. } => {}
                    k => for_each_operand(k, f, |v| want(&mut need, v, 64)),
                }
            }
        }
    }
    need
}

/// How many low bits every return value fits in: values zero-extended from a
/// narrower one, constants, masks and narrow loads. 64 when it can't tell, or
/// when the function doesn't return exactly rax (no value, rax:rdx, or a tail call
/// that returns whatever the callee does).
fn ret_bits(f: &Function, cfg: &Cfg, sig: Sig) -> u8 {
    use InstKind::*;
    if !sig.ret || sig.ret2 {
        return 64;
    }
    if cfg.rpo.iter().any(|&b| matches!(f.blocks[b].term, Terminator::TailCall { .. } | Terminator::Return(None))) {
        return 64;
    }
    // Least fixpoint from 0, so loops carrying a narrow value stay narrow. Entry
    // parameters are whatever the caller passed.
    let mut zb = vec![0u8; f.insts.len()];
    for &p in f.blocks[f.entry].params.get(&f.value_pool) {
        zb[p.index()] = 64;
    }
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &cfg.rpo {
            let blk = &f.blocks[b];
            for &id in blk.insts.get(&f.value_pool) {
                let full = bits_of(f.insts[id].ty);
                let new = match f.insts[id].kind {
                    Cast { kind: CastKind::ZExt, v } => zb[v.index()].min(bits_of(f.insts[v].ty)),
                    Const(c) => (64 - (f.consts[c.index()] as u64).leading_zeros()) as u8,
                    Bin { op: BinOp::And, lhs, rhs } => zb[lhs.index()].min(zb[rhs.index()]),
                    Bin { op: BinOp::Or | BinOp::Xor, lhs, rhs } => zb[lhs.index()].max(zb[rhs.index()]),
                    Select { t, f: e, .. } => zb[t.index()].max(zb[e.index()]),
                    Cmp { .. } => 1,
                    _ => full,
                }
                .min(full);
                if new > zb[id.index()] {
                    zb[id.index()] = new;
                    changed = true;
                }
            }
            let mut pairs = Vec::new();
            edges(f, blk.term, |p, a| pairs.push((p, a)));
            for (p, a) in pairs {
                let new = zb[a.index()].max(zb[p.index()]);
                if new > zb[p.index()] {
                    zb[p.index()] = new;
                    changed = true;
                }
            }
        }
    }
    let mut bits = 0;
    for &b in &cfg.rpo {
        if let Terminator::Return(Some(v)) = f.blocks[b].term {
            bits = bits.max(zb[v.index()]);
        }
    }
    bits
}

/// Known low bits of each value: `v ≡ r (mod m)`, with `m` a power of two up to
/// 16. For a value derived from an argument (`borrow` origins) it describes the
/// offset from that argument, so `p + 4*i` is 4-aligned relative to `p`. `None`
/// is "not reached yet" (the top of the lattice), which keeps loops precise.
fn residues(f: &Function, cfg: &Cfg, borrow: &Analysis) -> Vec<Option<(u8, u8)>> {
    use InstKind::*;
    type Res = Option<(u8, u8)>;
    const ANY: Res = Some((1, 0));
    fn join(a: Res, b: Res) -> Res {
        match (a, b) {
            (None, x) | (x, None) => x,
            (Some((m1, r1)), Some((m2, r2))) => {
                let mut m = m1.min(m2);
                while m > 1 && r1 % m != r2 % m {
                    m /= 2;
                }
                Some((m, r1 % m))
            }
        }
    }
    let norm = |m: u64, r: u64| -> Res {
        let m = m.clamp(1, 16) as u8;
        Some((m, (r % m as u64) as u8))
    };
    let derived = |v: ValueId| borrow.origin[v.index()].roots != 0;
    let mut res: Vec<Res> = vec![None; f.insts.len()];
    for &p in f.blocks[f.entry].params.get(&f.value_pool) {
        res[p.index()] = Some((16, 0));
    }
    // Sum of two values when at most one is derived from an argument.
    let add = |a: Res, b: Res, both: bool| -> Res {
        if both {
            return ANY;
        }
        match (a, b) {
            (Some((m1, r1)), Some((m2, r2))) => {
                let m = m1.min(m2) as u64;
                norm(m, r1 as u64 + r2 as u64)
            }
            _ => None,
        }
    };
    let scale = |a: Res, c: u64| -> Res {
        let (m, r) = a?;
        if c == 0 {
            return Some((16, 0));
        }
        let m = (m as u64).saturating_mul(1u64 << c.trailing_zeros().min(4));
        norm(m, (r as u64).wrapping_mul(c))
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &cfg.rpo {
            let blk = &f.blocks[b];
            for &id in blk.insts.get(&f.value_pool) {
                let r = |v: ValueId| res[v.index()];
                let new = match f.insts[id].kind {
                    Const(c) => norm(16, f.consts[c.index()] as u64),
                    PtrOffset { base, index, scale: s, disp } => {
                        let d = norm(16, disp as i64 as u64);
                        let with_disp = add(r(base), d, false);
                        match index {
                            None => with_disp,
                            Some(i) => add(with_disp, scale(r(i), s as u64), derived(base) && derived(i)),
                        }
                    }
                    Bin { op: BinOp::Add, lhs, rhs } => add(r(lhs), r(rhs), derived(lhs) && derived(rhs)),
                    Bin { op: BinOp::Sub, lhs, rhs } if !derived(rhs) => {
                        let neg = r(rhs).map(|(m, x)| ((m as u64), (m as u64 - x as u64)));
                        match (r(lhs), neg) {
                            (Some((m1, r1)), Some((m2, r2))) => norm((m1 as u64).min(m2), r1 as u64 + r2),
                            _ => None,
                        }
                    }
                    Bin { op: BinOp::Mul, lhs, rhs } if !derived(lhs) && !derived(rhs) => match (konst(f, lhs), konst(f, rhs)) {
                        (_, Some(c)) => scale(r(lhs), c),
                        (Some(c), _) => scale(r(rhs), c),
                        _ => ANY,
                    },
                    Bin { op: BinOp::Shl, lhs, rhs } if !derived(lhs) => match konst(f, rhs) {
                        Some(s) if s < 64 => scale(r(lhs), 1u64 << s),
                        _ => ANY,
                    },
                    Bin { op: BinOp::And, lhs, rhs } if !derived(lhs) => match (r(lhs), konst(f, rhs)) {
                        (Some((m, x)), Some(c)) => {
                            let k = (m.trailing_zeros()).max(c.trailing_zeros().min(4));
                            norm(1 << k, (x as u64) & c)
                        }
                        _ => ANY,
                    },
                    Cast { v, .. } => r(v),
                    Select { t, f: e, .. } => join(r(t), r(e)),
                    _ => ANY,
                };
                let new = join(res[id.index()], new);
                if new != res[id.index()] {
                    res[id.index()] = new;
                    changed = true;
                }
            }
            let mut pairs = Vec::new();
            edges(f, blk.term, |p, a| pairs.push((p, a)));
            for (p, a) in pairs {
                // A parameter mixing pointers and integers has no meaningful residue.
                let incoming = if derived(p) != derived(a) { ANY } else { res[a.index()] };
                let new = join(res[p.index()], incoming);
                if new != res[p.index()] {
                    res[p.index()] = new;
                    changed = true;
                }
            }
        }
    }
    res
}

// ---------------------------------------------------------------------------
// The result

/// What a pointer argument points to.
#[derive(Clone, Debug, PartialEq)]
pub enum PointeeTy {
    Struct(Arc<StructTy>),
    /// One scalar, at offset 0.
    Scalar(IntTy),
    /// An array of scalars.
    Slice(IntTy),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pointee {
    pub ty: PointeeTy,
    /// Every load and store through the argument is described by `ty`, so safe
    /// mode can take it as `&S`, `&T` or `&[T]` instead of `&[u8]`.
    pub complete: bool,
    /// The Rust type's name: the struct's identifier (set by `Program`), or the
    /// element type.
    pub ident: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParamType {
    /// From debug info or a model.
    pub name: Option<String>,
    /// The integer type the argument is taken as when it isn't a pointer:
    /// narrower than `u64` when only its low bits are used.
    pub int: IntTy,
    pub pointee: Option<Pointee>,
}

/// Where a load or store goes, relative to the argument it is through.
#[derive(Clone, Debug, PartialEq)]
pub enum Path {
    /// `.field` (or `.a.b[2]`), and the field's type.
    Field(String, Leaf),
    /// `*p`
    Deref,
    /// `p[i]`: a constant index, or `None` to compute it from the address.
    Index(Option<u64>),
}

/// Types for one function, which the emitter looks up.
#[derive(Clone, Debug, Default)]
pub struct FnTypes {
    /// One per entry parameter, in entry-block order.
    pub params: Vec<ParamType>,
    /// The return type, when it is narrower than `u64`.
    pub ret: Option<IntTy>,
    signed: Vec<bool>,
    /// `Trunc` of an argument that the narrowed argument is (entry parameter index).
    alias: Vec<Option<u8>>,
    /// Load or store -> entry parameter index and place.
    paths: Vec<Option<(u8, Path)>>,
    /// What was proposed and decided, for `--emit types`.
    pub notes: Vec<String>,
}

impl FnTypes {
    /// The Rust integer type of value `v` (an integer-typed value).
    pub fn int(&self, f: &Function, v: ValueId) -> Option<IntTy> {
        let ty = f.insts[v].ty;
        is_int(ty).then(|| IntTy { bytes: bits_of(ty) / 8, signed: self.signed.get(v.index()).copied().unwrap_or(false) })
    }

    pub fn is_signed(&self, v: ValueId) -> bool {
        self.signed.get(v.index()).copied().unwrap_or(false)
    }

    /// The entry parameter whose narrowed value `v` is.
    pub fn alias(&self, v: ValueId) -> Option<usize> {
        self.alias.get(v.index()).copied().flatten().map(usize::from)
    }

    /// The argument and place a load or store accesses.
    pub fn path(&self, inst: ValueId) -> Option<(usize, &Path)> {
        self.paths.get(inst.index())?.as_ref().map(|(k, p)| (*k as usize, p))
    }

    /// Struct types the signature uses (pointees), for naming and definitions.
    pub fn structs_mut(&mut self) -> impl Iterator<Item = &mut Pointee> {
        self.params.iter_mut().filter_map(|p| p.pointee.as_mut()).filter(|p| matches!(p.ty, PointeeTy::Struct(_)))
    }
}

// ---------------------------------------------------------------------------
// Inference

/// Union-find over values, plus one node per entry parameter for its narrowed type.
struct Classes {
    parent: Vec<u32>,
}

impl Classes {
    fn find(&mut self, x: usize) -> usize {
        let mut x = x;
        while self.parent[x] as usize != x {
            let p = self.parent[x] as usize;
            self.parent[x] = self.parent[p];
            x = p;
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.parent[a.max(b)] = a.min(b) as u32;
        }
    }
}

fn round_bytes(bits: u8) -> u8 {
    match bits {
        0..=8 => 1,
        9..=16 => 2,
        17..=32 => 4,
        _ => 8,
    }
}

const REG32: [&str; 16] =
    ["eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi", "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d"];
const REG16: [&str; 16] =
    ["ax", "cx", "dx", "bx", "sp", "bp", "si", "di", "r8w", "r9w", "r10w", "r11w", "r12w", "r13w", "r14w", "r15w"];
const REG8: [&str; 16] =
    ["al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil", "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b"];
const REG: [&str; 16] =
    ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"];

/// The register an entry parameter arrives in, named for `bytes` (`edi`, `dil`),
/// or `arg6` for a stack argument.
pub fn reg_name(reg: u8, bytes: u8) -> String {
    if reg >= STACK_ARG_BASE {
        let n = 6 + (reg - STACK_ARG_BASE) as usize;
        return if bytes == 8 { format!("arg{n}") } else { format!("arg{n}_lo") };
    }
    let r = reg as usize & 15;
    match bytes {
        1 => REG8[r],
        2 => REG16[r],
        4 => REG32[r],
        _ => REG[r],
    }
    .to_string()
}

/// Infer `f`'s types. `func.sig` must be its signature after `abi::apply`;
/// `models` propose, in order of trust (debug info first).
pub fn infer(func: &FuncRef, ident: &str, models: &[&dyn TypeModel]) -> FnTypes {
    let f = func.ir;
    let facts = Facts::gather(f, func.sig);
    let cfg = Cfg::new(f);
    let n = f.insts.len();
    let params = f.blocks[f.entry].params.get(&f.value_pool).to_vec();
    let np = params.len();
    let mut notes = Vec::new();
    let reg = |k: usize| match f.insts[params[k]].kind {
        InstKind::BlockParam(r) => r,
        _ => u8::MAX,
    };

    // Proposals, per entry parameter and for the return value; the first model
    // to propose something for a slot wins.
    let mut prop_param: Vec<Option<(String, ParamProposal)>> = vec![None; np];
    let mut prop_ret: Option<(String, CType)> = None;
    for m in models {
        let Some(p) = m.propose(func) else { continue };
        if !p.params.is_empty() && p.params.len() < facts.arg_param.len() {
            notes.push(format!(
                "{}: rejected all arguments: it lists {} integer arguments, the code takes {}",
                m.name(),
                p.params.len(),
                facts.arg_param.len()
            ));
        } else {
            for (j, pp) in p.params.into_iter().enumerate().take(facts.arg_param.len()) {
                if let Some(k) = facts.arg_param[j] {
                    if prop_param[k].is_none() && (pp.name.is_some() || pp.ty.is_some()) {
                        prop_param[k] = Some((m.name().to_string(), pp));
                    }
                }
            }
        }
        if prop_ret.is_none() {
            prop_ret = p.ret.map(|t| (m.name().to_string(), t));
        }
    }

    // 1. Integer widths of arguments: from demanded bits, or the proposal when the
    //    demanded bits fit in it.
    let mut ptypes: Vec<ParamType> = (0..np).map(|_| ParamType { name: None, int: IntTy::U64, pointee: None }).collect();
    let mut seed: Vec<Option<bool>> = vec![None; np]; // signedness from a proposal
    let mut rejected = vec![false; np];
    for k in 0..np {
        let r = reg(k);
        if r == RSP || r == u8::MAX || !facts.arg_param.contains(&Some(k)) {
            continue;
        }
        let need = facts.demanded[params[k].index()];
        if !facts.is_pointer(k) && need > 0 && need <= 32 {
            ptypes[k].int = IntTy::new(round_bytes(need), false);
        }
        let Some((src, pp)) = &prop_param[k] else { continue };
        match &pp.ty {
            Some(CType::Int(t)) => {
                if facts.is_pointer(k) {
                    notes.push(format!("{src}: rejected {} for {}: it is dereferenced", t.rust(), reg_name(r, 8)));
                    rejected[k] = true;
                } else if need as u32 > t.bits() {
                    notes.push(format!("{src}: rejected {} for {}: the code uses {need} bits of it", t.rust(), reg_name(r, 8)));
                    rejected[k] = true;
                } else {
                    ptypes[k].int = IntTy::new(t.bytes, false);
                    seed[k] = Some(t.signed);
                }
            }
            Some(CType::Float { .. } | CType::Struct(_) | CType::Array { .. } | CType::Opaque { .. }) => {
                notes.push(format!("{src}: rejected {} for {}: not passed in an integer register", pp.ty.as_ref().unwrap().describe(), reg_name(r, 8)));
                rejected[k] = true;
            }
            Some(CType::Ptr(_)) | None => {}
        }
    }

    // 2. Return width.
    let mut ret: Option<IntTy> = None;
    let mut ret_seed: Option<bool> = None;
    if facts.ret_bits <= 32 {
        ret = Some(IntTy::new(round_bytes(facts.ret_bits.max(1)), false));
    }
    if let Some((src, t)) = &prop_ret {
        match t {
            CType::Int(t) if ret.is_some() && facts.ret_bits as u32 <= t.bits() => {
                ret = Some(IntTy::new(t.bytes, false));
                ret_seed = Some(t.signed);
            }
            CType::Int(t) if t.bytes < 8 => notes.push(format!(
                "{src}: rejected {} for the return value: the code returns {} bits",
                t.rust(),
                if facts.ret_bits <= 32 { round_bytes(facts.ret_bits.max(1)) as u32 * 8 } else { 64 }
            )),
            _ => {}
        }
    }

    // 3. Signedness classes.
    let mut cls = Classes { parent: (0..(n + np) as u32).collect() };
    let same = |cls: &mut Classes, a: ValueId, b: ValueId| {
        if f.insts[a].ty == f.insts[b].ty && is_int(f.insts[a].ty) {
            cls.union(a.index(), b.index());
        }
    };
    let mut alias: Vec<Option<u8>> = vec![None; n];
    let mut votes = vec![0i32; n + np];
    let mut forced = vec![false; n + np];
    let vote = |votes: &mut Vec<i32>, v: ValueId, w: i32| votes[v.index()] += w;
    for &b in &cfg.rpo {
        let blk = &f.blocks[b];
        for &id in blk.insts.get(&f.value_pool) {
            use InstKind::*;
            match f.insts[id].kind {
                Bin { op, lhs, rhs } => {
                    match op {
                        BinOp::Shl | BinOp::LShr | BinOp::AShr | BinOp::RotL | BinOp::RotR => same(&mut cls, lhs, id),
                        _ => {
                            same(&mut cls, lhs, rhs);
                            same(&mut cls, lhs, id);
                        }
                    }
                    match op {
                        BinOp::SDiv | BinOp::SRem => {
                            vote(&mut votes, lhs, 2);
                            vote(&mut votes, rhs, 2);
                        }
                        BinOp::UDiv | BinOp::URem => {
                            vote(&mut votes, lhs, -2);
                            vote(&mut votes, rhs, -2);
                        }
                        BinOp::AShr => vote(&mut votes, lhs, 2),
                        BinOp::LShr => vote(&mut votes, lhs, -1),
                        _ => {}
                    }
                }
                Un { op: UnOp::Neg | UnOp::Not | UnOp::Bswap, v } => same(&mut cls, v, id),
                Cmp { cc, lhs, rhs } => {
                    same(&mut cls, lhs, rhs);
                    let w = match cc {
                        Cond::Slt | Cond::Sle | Cond::Sgt | Cond::Sge => 2,
                        Cond::Ult | Cond::Ule | Cond::Ugt | Cond::Uge => -2,
                        Cond::Eq | Cond::Ne => 0,
                    };
                    vote(&mut votes, lhs, w);
                }
                Select { t, f: e, .. } => {
                    same(&mut cls, t, e);
                    same(&mut cls, t, id);
                }
                Cast { kind: CastKind::SExt, v } => {
                    vote(&mut votes, v, 3);
                    vote(&mut votes, id, 1);
                }
                // `mov eax, x` zero-extends every 32-bit result: no evidence.
                Cast { kind: CastKind::ZExt, v } if f.insts[v].ty != TyId::B4 => vote(&mut votes, v, -3),
                Cast { kind: CastKind::Trunc, v } => {
                    if let Some(k) = params.iter().position(|&p| p == v) {
                        if ptypes[k].int.bytes < 8 && bits_of(f.insts[id].ty) / 8 == ptypes[k].int.bytes {
                            alias[id.index()] = Some(k as u8);
                            cls.union(id.index(), n + k);
                        }
                    }
                }
                Call { .. } | CallOut { .. } | PtrOffset { .. } | IntToPtr(_) | PtrToInt(_) | AddrOfLocal(_)
                | AddrOfGlobal(_) | Aggregate { .. } | Opaque { .. } | Param(_) => forced[id.index()] = true,
                _ => {}
            }
        }
        let mut pairs = Vec::new();
        edges(f, blk.term, |p, a| pairs.push((p, a)));
        for (p, a) in pairs {
            same(&mut cls, p, a);
        }
    }
    for &p in &params {
        forced[p.index()] = true;
    }
    // Values derived from a pointer argument are addresses.
    for (v, o) in facts.borrow.origin.iter().enumerate() {
        if each_root(o.roots).any(|k| facts.is_pointer(k)) {
            forced[v] = true;
        }
    }
    for k in 0..np {
        if let Some(s) = seed[k] {
            votes[n + k] += if s { 100 } else { -100 };
        }
    }
    if let Some(s) = ret_seed {
        // The returned value is `ZExt(x)`: x carries the declared signedness.
        for &b in &cfg.rpo {
            if let Terminator::Return(Some(v)) = f.blocks[b].term {
                if let InstKind::Cast { kind: CastKind::ZExt, v: x } = f.insts[v].kind {
                    votes[x.index()] += if s { 100 } else { -100 };
                }
            }
        }
    }

    // 4. Pointees (needs field signedness, so solve once without them first).
    let solve = |cls: &mut Classes, votes: &[i32], forced: &[bool]| -> Vec<bool> {
        let mut sum = vec![0i32; n + np];
        let mut force = vec![false; n + np];
        for x in 0..n + np {
            let r = cls.find(x);
            sum[r] += votes[x];
            force[r] |= forced[x];
        }
        (0..n + np)
            .map(|x| {
                let r = cls.find(x);
                !force[r] && sum[r] > 0
            })
            .collect()
    };
    let signed0 = solve(&mut cls, &votes, &forced);
    let mut paths: Vec<Option<(u8, Path)>> = vec![None; n];
    for k in 0..np {
        let r = reg(k);
        if r == RSP || facts.accesses[k].is_empty() || !facts.is_pointer(k) {
            continue;
        }
        let acc = &facts.accesses[k];
        let proposed = prop_param[k].as_ref().filter(|_| !rejected[k]).and_then(|(src, pp)| match &pp.ty {
            Some(CType::Ptr(Some(t))) => Some((src.clone(), t.clone())),
            _ => None,
        });
        let mut chosen = None;
        if let Some((src, t)) = proposed {
            match check_pointee(&t, acc) {
                Ok(p) => {
                    for a in acc {
                        if let Some(path) = path_for(&p, &t, a) {
                            if let Leaf::Int(lt) = leaf_of(&p, &path) {
                                votes[a.value.index()] += if lt.signed { 50 } else { -50 };
                            }
                            paths[a.inst.index()] = Some((k as u8, path));
                        }
                    }
                    chosen = Some(p);
                }
                Err(why) => {
                    notes.push(format!("{src}: rejected {} * for {}: {why}", t.describe(), reg_name(r, 8)));
                    rejected[k] = true;
                }
            }
        }
        if chosen.is_none() {
            if let Some((p, t)) = infer_pointee(acc, &signed0, ident, &reg_name(r, 8)) {
                for a in acc {
                    if let Some(path) = path_for(&p, &t, a) {
                        paths[a.inst.index()] = Some((k as u8, path));
                    }
                }
                chosen = Some(p);
            }
        }
        ptypes[k].pointee = chosen;
    }
    let signed = solve(&mut cls, &votes, &forced);

    // Names: accepted unless the same proposal's type was rejected.
    for k in 0..np {
        if let Some((_, pp)) = &prop_param[k] {
            if !rejected[k] {
                ptypes[k].name = pp.name.clone();
            }
        }
        ptypes[k].int.signed = signed[n + k];
    }
    if let Some(t) = &mut ret {
        t.signed = match ret_seed {
            Some(s) => s,
            None => {
                // the signedness of what is zero-extended into the return value
                let mut s = None;
                for &b in &cfg.rpo {
                    if let Terminator::Return(Some(v)) = f.blocks[b].term {
                        if let InstKind::Cast { kind: CastKind::ZExt, v: x } = f.insts[v].kind {
                            if bits_of(f.insts[x].ty) / 8 == t.bytes {
                                s = Some(s.unwrap_or(true) && signed[x.index()]);
                            }
                        }
                    }
                }
                s.unwrap_or(false)
            }
        };
    }
    let mut order: Vec<usize> = facts.arg_param.iter().flatten().copied().collect();
    let rest: Vec<usize> = (0..np).filter(|k| !order.contains(k)).collect();
    order.extend(rest);
    for k in order {
        let p = &ptypes[k];
        let r = reg(k);
        if r == RSP || r == u8::MAX {
            continue;
        }
        let mut line = format!("{}: ", reg_name(r, 8));
        match &p.pointee {
            Some(pt) => {
                let what = match &pt.ty {
                    PointeeTy::Struct(s) => format!("struct {}", s.name),
                    PointeeTy::Scalar(t) => t.rust().to_string(),
                    PointeeTy::Slice(t) => format!("[{}]", t.rust()),
                };
                let _ = write!(line, "*{what}{}", if pt.complete { "" } else { " (some accesses aren't covered)" });
            }
            None => line.push_str(p.int.rust()),
        }
        if let Some(nm) = &p.name {
            let _ = write!(line, " named {nm}");
        }
        if let Some((src, _)) = &prop_param[k] {
            let _ = write!(line, " [{src}{}]", if rejected[k] { ", rejected" } else { "" });
        }
        notes.push(line);
    }
    if let Some(t) = ret {
        notes.push(format!("returns {}{}", t.rust(), if ret_seed.is_some() { format!(" [{}]", prop_ret.as_ref().unwrap().0) } else { String::new() }));
    }
    let _ = rejected;
    FnTypes { params: ptypes, ret, signed: signed[..n].to_vec(), alias, paths, notes }
}

/// The scalar type an access path reaches.
fn leaf_of(p: &Pointee, path: &Path) -> Leaf {
    match (path, &p.ty) {
        (Path::Field(_, l), _) => *l,
        (_, PointeeTy::Scalar(t) | PointeeTy::Slice(t)) => Leaf::Int(*t),
        _ => Leaf::Int(IntTy::U64),
    }
}

/// Where access `a` goes in pointee `p` (whose C type is `t`).
fn path_for(p: &Pointee, t: &CType, a: &Access) -> Option<Path> {
    if !a.single || a.bytes == 0 {
        return None;
    }
    match &p.ty {
        PointeeTy::Struct(_) => {
            let off = u64::try_from(a.off?).ok()?;
            let (path, leaf) = locate(t, off, a.bytes)?;
            Some(Path::Field(path, leaf))
        }
        PointeeTy::Scalar(e) => (a.off == Some(0) && a.bytes == e.bytes).then_some(Path::Deref),
        PointeeTy::Slice(e) => {
            if a.bytes != e.bytes {
                return None;
            }
            match a.off {
                Some(o) if o >= 0 && o % e.bytes as i64 == 0 => Some(Path::Index(Some(o as u64 / e.bytes as u64))),
                Some(_) => None,
                None => a.aligned.then_some(Path::Index(None)),
            }
        }
    }
}

/// Does `t` (a proposed pointee) describe the accesses? A struct must have a
/// scalar field exactly where each constant-offset access lands; a scalar or
/// array element must have the size of every access, at aligned offsets.
fn check_pointee(t: &CType, acc: &[Access]) -> Result<Pointee, String> {
    match t {
        CType::Struct(s) => {
            let mut complete = true;
            for a in acc {
                match (a.single, a.off, a.bytes) {
                    (true, Some(o), w) if w > 0 => {
                        if o < 0 || locate(t, o as u64, w).is_none() {
                            return Err(format!("{w}-byte access at {o:+} matches no field"));
                        }
                    }
                    _ => complete = false,
                }
            }
            Ok(Pointee { ty: PointeeTy::Struct(s.clone()), complete, ident: s.name.clone() })
        }
        CType::Int(_) | CType::Ptr(_) => {
            let e = match t {
                CType::Int(e) => *e,
                _ => IntTy::U64,
            };
            let mut complete = true;
            let mut indexed = false;
            for a in acc {
                if !a.single || a.bytes == 0 {
                    complete = false;
                    continue;
                }
                if a.bytes != e.bytes {
                    return Err(format!("{}-byte access to {}-byte elements", a.bytes, e.bytes));
                }
                match a.off {
                    Some(0) => {}
                    Some(o) if o > 0 && o % e.bytes as i64 == 0 => indexed = true,
                    Some(o) => return Err(format!("access at {o:+} is between elements")),
                    None if a.aligned => indexed = true,
                    None => complete = false,
                }
            }
            let ty = if indexed { PointeeTy::Slice(e) } else { PointeeTy::Scalar(e) };
            Ok(Pointee { ty, complete, ident: e.rust().to_string() })
        }
        _ => Err("no layout to check".to_string()),
    }
}

/// A pointee from the accesses alone: a slice when it is indexed with one element
/// size, one scalar when only offset 0 is used, otherwise a struct with a field at
/// each accessed offset. Returns the C type too, for paths.
fn infer_pointee(acc: &[Access], signed: &[bool], ident: &str, reg: &str) -> Option<(Pointee, CType)> {
    let sign_of = |pred: &dyn Fn(&Access) -> bool| -> bool {
        let (mut s, mut u) = (0, 0);
        for a in acc.iter().filter(|a| pred(a)) {
            if signed[a.value.index()] {
                s += 1
            } else {
                u += 1
            }
        }
        s > u
    };
    let usable: Vec<&Access> = acc.iter().filter(|a| a.single && a.bytes > 0).collect();
    if usable.is_empty() {
        return None;
    }
    let complete = usable.len() == acc.len();
    let w = usable[0].bytes;
    let indexed = usable.iter().any(|a| a.off.is_none());
    if indexed {
        let ok = usable.iter().all(|a| {
            a.bytes == w
                && match a.off {
                    Some(o) => o >= 0 && o % w as i64 == 0,
                    None => a.aligned,
                }
        });
        if !ok {
            return None;
        }
        let e = IntTy::new(w, sign_of(&|_| true));
        return Some((Pointee { ty: PointeeTy::Slice(e), complete, ident: e.rust().to_string() }, CType::Int(e)));
    }
    let mut fields: Vec<(i64, u8)> = usable.iter().map(|a| (a.off.unwrap(), a.bytes)).collect();
    fields.sort_unstable();
    fields.dedup();
    if fields[0].0 < 0 || fields.windows(2).any(|p| p[0].0 + p[0].1 as i64 > p[1].0) {
        return None;
    }
    if fields.len() == 1 && fields[0].0 == 0 {
        let e = IntTy::new(w, sign_of(&|_| true));
        return Some((Pointee { ty: PointeeTy::Scalar(e), complete, ident: e.rust().to_string() }, CType::Int(e)));
    }
    let s = StructTy {
        name: format!("S_{ident}_{reg}"),
        size: fields.iter().map(|&(o, w)| o as u64 + w as u64).max().unwrap_or(0),
        fields: fields
            .iter()
            .map(|&(o, w)| Field {
                name: format!("f{o}"),
                off: o as u64,
                ty: CType::Int(IntTy::new(w, sign_of(&|a: &Access| a.off == Some(o) && a.bytes == w))),
            })
            .collect(),
    };
    let s = Arc::new(s);
    let ident = s.name.clone();
    Some((Pointee { ty: PointeeTy::Struct(s.clone()), complete, ident }, CType::Struct(s)))
}

// ---------------------------------------------------------------------------
// Program-wide naming

/// `name` as a Rust identifier, with keywords and reserved words suffixed by `_`
/// (`type` becomes `type_`, `self` becomes `self_`).
pub fn ident_fix(name: &str) -> String {
    let s = crate::names::sanitize(name);
    match s.strip_prefix("f_") {
        Some(rest) if !name.starts_with("f_") && !rest.is_empty() && !rest.starts_with(|c: char| c.is_ascii_digit()) => {
            format!("{rest}_")
        }
        _ => s,
    }
}

/// Would `s` clash with a name the emitter makes up (`v12`, `rdi`, `frame`, ...)?
fn emitter_name(s: &str) -> bool {
    let digits = |p: &str| s.strip_prefix(p).is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()));
    digits("v")
        || digits("arg")
        || s.strip_suffix("_lo").is_some_and(|a| a.strip_prefix("arg").is_some_and(|d| d.bytes().all(|b| b.is_ascii_digit())))
        || [REG, REG32, REG16, REG8].iter().any(|t| t.contains(&s))
        || matches!(s, "frame" | "bb" | "ffi" | "core" | "std" | "Some" | "None" | "Ok" | "Err")
        || s.ends_with("_ref")
        || s.ends_with("_base")
        || s.ends_with("_pair")
}

impl FnTypes {
    /// Make argument names Rust identifiers that clash with nothing in `taken`
    /// (function and static names), the emitter's own names, or each other.
    pub fn fix_names(&mut self, taken: &std::collections::HashSet<String>) {
        let mut used = std::collections::HashSet::new();
        for p in &mut self.params {
            let Some(n) = &p.name else { continue };
            let mut s = ident_fix(n);
            while emitter_name(&s) || taken.contains(&s) || !used.insert(s.clone()) {
                s.push('_');
            }
            p.name = Some(s);
        }
    }

    /// The Rust type of each argument (C order) as callers pass it: `*mut S` for a
    /// struct pointer, else its integer type. Only `Mode::Safe` functions that no
    /// decompiled function calls take references, so callers never see those.
    pub fn arg_types(&self, f: &Function, sig: Sig) -> Vec<String> {
        let params = f.blocks[f.entry].params.get(&f.value_pool);
        let find = |reg: u8| params.iter().position(|&p| matches!(f.insts[p].kind, InstKind::BlockParam(r) if r == reg));
        let regs = SYSV_ARGS[..sig.args as usize].iter().copied().chain((0..sig.stack_args).map(|j| STACK_ARG_BASE + j));
        regs.map(|r| match find(r).and_then(|k| self.params.get(k)) {
            Some(ParamType { pointee: Some(Pointee { ty: PointeeTy::Struct(_), ident, .. }), .. }) => format!("*mut {ident}"),
            Some(p) => p.int.rust().to_string(),
            None => "u64".to_string(),
        })
        .collect()
    }
}

/// `name` without module paths: `Vec<alloc::string::String, alloc::alloc::Global>`
/// becomes `Vec<String, Global>`.
fn short_name(name: &str) -> String {
    let mut out = String::new();
    let mut seg = 0; // where the current path segment starts in `out`
    let mut chars = name.chars().peekable();
    while let Some(c) = chars.next() {
        if c == ':' && chars.peek() == Some(&':') {
            chars.next();
            out.truncate(seg);
            continue;
        }
        out.push(c);
        if !(c.is_alphanumeric() || c == '_') {
            seg = out.len();
        }
    }
    out
}

/// Rust identifiers for structs, shared by the whole output: one definition per
/// distinct layout, named after the C struct (or `S_<function>_<register>` when
/// inferred), suffixed when two different layouts share a name.
#[derive(Default)]
pub struct StructNames {
    map: std::collections::HashMap<Arc<StructTy>, String>,
    used: std::collections::HashSet<String>,
    order: Vec<Arc<StructTy>>,
}

impl StructNames {
    /// The identifier for `s`, registering it and the structs it contains.
    pub fn name(&mut self, s: &Arc<StructTy>) -> String {
        if let Some(n) = self.map.get(s) {
            return n.clone();
        }
        let mut base = crate::names::sanitize(&short_name(&s.name));
        if matches!(
            base.as_str(),
            "Option" | "Some" | "None" | "Vec" | "Box" | "String" | "Bytes" | "Word" | "Words" | "Pair" | "ffi" | "core"
                | "std" | "str" | "char" | "bool" | "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "i8" | "i16"
                | "i32" | "i64" | "i128" | "isize" | "f32" | "f64" | "Self"
        ) {
            base.push('_');
        }
        let mut n = base.clone();
        let mut k = 1;
        while self.used.contains(&n) {
            k += 1;
            n = format!("{base}_{k}");
        }
        self.used.insert(n.clone());
        self.map.insert(s.clone(), n.clone());
        self.order.push(s.clone());
        for f in &s.fields {
            let mut inner = Vec::new();
            nested_structs(&f.ty, &mut inner);
            for t in inner {
                self.name(&t);
            }
        }
        n
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Every definition, sorted by name.
    pub fn defs(&self) -> String {
        let mut v: Vec<(&String, &Arc<StructTy>)> = self.order.iter().map(|s| (&self.map[s], s)).collect();
        v.sort_by(|a, b| a.0.cmp(b.0));
        let mut out = String::new();
        let ident_of = |s: &StructTy| self.map.iter().find(|(k, _)| k.as_ref() == s).map(|(_, n)| n.clone()).unwrap_or_default();
        for (n, s) in v {
            out.push('\n');
            struct_def(s, n, &ident_of, &mut out);
        }
        out
    }
}
