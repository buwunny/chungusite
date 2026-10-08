//! Type recovery (roadmap step 5): integer widths and signedness, pointers versus
//! integers, and structs, for the emitter to print instead of `u64` everywhere.
//!
//! The IR keeps the lifter's storage types (`B1`..`B8`). What this pass recovers
//! is a side table per function (`FnTypes`) plus the program's `TyTable`, which
//! holds the struct definitions. Three sources, strongest first:
//!
//! 1. **DWARF** (`dwarf.rs`), when the binary has it: parameter names and types,
//!    return types, struct layouts with field names.
//! 2. **A type model** (`TypeModel`), optional. It proposes a C type for each
//!    argument and the return value; a proposal is used only if it agrees with
//!    the facts below (`accept`). Nothing in this crate implements it yet: it is
//!    the hook for the fine-tuned models (docs/types.md).
//! 3. **Inference** from how values are used, always on:
//!    * *Signedness.* Values that flow into each other (arithmetic operands and
//!      results, block parameters and their edge arguments, select arms, compare
//!      operands) share a class. Signed compares, `SDiv`/`SRem`, `SAR`, `MOVSX`
//!      and small negative constants vote signed; unsigned compares, `UDiv`/`URem`
//!      and `SHR` vote unsigned; a class used as an address is unsigned.
//!    * *Narrow arguments and returns.* An argument whose every use truncates it
//!      to 32 bits is an `i32`/`u32`. A return value that is always a zero-extended
//!      32-bit value, or a `SETcc` result, returns `i32`/`u32` or `bool`.
//!    * *Pointee types* (Steensgaard-style unification). Values that flow into
//!      each other point at the same type. Each load or store `*(base + d)` adds a
//!      field at offset `d` to `base`'s type, and the value loaded or stored there
//!      is unified with the field's other values, so `p = p->next` makes `next` a
//!      pointer to the same struct. Non-overlapping fields make a struct
//!      (`#[repr(C, packed)]`, since the real alignment isn't known); one field at
//!      offset 0, or indexed accesses of one width, make a scalar pointee.
//!
//! Every type here is a claim the emitted code relies on, so each one preserves
//! what the machine code does: a narrowed argument is only ever used truncated,
//! a narrowed return value has zero upper bits, and a field access happens at
//! exactly the offset and width the instruction used.
use crate::abi::{Sig, STACK_ARG_BASE, SYSV_ARGS};
use crate::borrow::RSP;
use crate::cfg::Cfg;
use crate::dwarf::{DebugFn, DebugInfo};
use crate::ir::*;
use crate::verify::for_each_operand;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};

// ---------------------------------------------------------------------------
// Results

/// Recovered types for one function.
#[derive(Clone, Debug, Default)]
pub struct FnTypes {
    /// Rust type of each value, indexed by `ValueId`: `Int` with its signedness
    /// for integers, `BOOL`, or the storage type for anything else (`PAIR`, `UNIT`).
    pub vals: Vec<TyId>,
    /// What each value points at, when it is used as a pointer and that is known
    /// (a struct, or a scalar).
    pub pointee: Vec<Option<TyId>>,
    /// The arguments, in signature order (rdi, rsi, ... then stack arguments).
    pub args: Vec<ArgTy>,
    /// The return type, if it is something other than `u64`.
    pub ret: Option<TyId>,
}

/// One argument of the signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArgTy {
    /// The x86 register it arrives in (`STACK_ARG_BASE + j` for stack argument j).
    pub reg: u8,
    /// `None`: `u64`.
    pub ty: Option<TyId>,
    /// Its name from debug info, as a Rust identifier.
    pub name: Option<String>,
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct TypeStats {
    /// Functions whose prototype came from DWARF.
    pub debug_fns: usize,
    /// Structs inferred from access patterns.
    pub inferred_structs: usize,
    /// Arguments, and how many got a type other than `u64`.
    pub args: usize,
    pub typed_args: usize,
    /// Model proposals used, and turned down by the facts.
    pub accepted: usize,
    pub rejected: usize,
}

// ---------------------------------------------------------------------------
// The hook for a type model

/// What a type model is asked about.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Var {
    /// Argument `j` in signature order; `value` is its entry parameter (or `None`
    /// if the function takes it but never reads it).
    Arg { j: usize, value: Option<ValueId> },
    /// The return value; `value` is one of the values returned.
    Ret { value: ValueId },
}

/// A model's answer for one `Var`: a C type spelled like the labels of
/// `train_types.py` (normalised DWARF names: `int`, `unsigned char`, `char *`,
/// `size_t` ...). `score` is the model's confidence, for the caller's threshold.
#[derive(Clone, Debug, PartialEq)]
pub struct Proposal {
    pub label: String,
    pub score: f32,
}

/// A source of type proposals, such as an ONNX classifier over `refine::to_ids`.
/// It only proposes: `accept` checks each proposal against what the code does,
/// and a rejected one changes nothing.
pub trait TypeModel: Sync {
    /// One answer (or `None`) per entry of `vars`, in order.
    fn propose(&self, f: &Function, vars: &[Var]) -> Vec<Option<Proposal>>;
}

/// What a C type label means, as far as the gate is concerned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Label {
    Int { bytes: u8, signed: bool },
    Bool,
    Ptr(Option<Box<Label>>),
    Void,
    Float,
}

/// Parses a C type label; `None` for names the gate doesn't know (struct and
/// typedef names other than the standard ones).
pub fn parse_label(s: &str) -> Option<Label> {
    let s = s.trim();
    if let Some(inner) = s.strip_suffix('*') {
        return Some(Label::Ptr(parse_label(inner).filter(|l| *l != Label::Void).map(Box::new)));
    }
    let words: Vec<&str> = s.split_whitespace().filter(|w| !matches!(*w, "const" | "volatile" | "restrict")).collect();
    let int = |bytes, signed| Some(Label::Int { bytes, signed });
    if words.iter().all(|w| matches!(*w, "unsigned" | "signed" | "int" | "long" | "short" | "char")) && !words.is_empty() {
        let signed = !words.contains(&"unsigned");
        let bytes = if words.contains(&"char") {
            1
        } else if words.contains(&"short") {
            2
        } else if words.contains(&"long") {
            8
        } else {
            4
        };
        return int(bytes, signed);
    }
    match words.join(" ").as_str() {
        "_Bool" | "bool" => Some(Label::Bool),
        "void" => Some(Label::Void),
        "float" | "double" | "long double" => Some(Label::Float),
        "int8_t" => int(1, true),
        "uint8_t" => int(1, false),
        "int16_t" => int(2, true),
        "uint16_t" => int(2, false),
        "int32_t" => int(4, true),
        "uint32_t" => int(4, false),
        "int64_t" | "ssize_t" | "ptrdiff_t" | "intptr_t" | "off_t" => int(8, true),
        "uint64_t" | "size_t" | "uintptr_t" => int(8, false),
        _ => None,
    }
}

/// What the code says about a variable, for `accept`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct VarFacts {
    /// Dereferenced (or derived from something that is).
    pub pointer: bool,
    /// The fewest bytes that hold every use (1, 2, 4 or 8).
    pub bytes: u8,
    /// Signed votes minus unsigned votes.
    pub evidence: i32,
    /// Always 0 or 1.
    pub boolish: bool,
    /// What it points at, if inferred.
    pub pointee: Option<TyId>,
}

/// The type a proposal turns into, if the facts allow it. A model may say more
/// than the code shows (an `int` argument the code only uses as a byte), but
/// never less (a byte for a value used at 32 bits) or something else (an
/// integer for a pointer, unsigned for a value compared as signed).
pub fn accept(label: &Label, facts: &VarFacts, table: &mut TyTable) -> Option<TyId> {
    match label {
        Label::Int { bytes, signed } => {
            let contradicts = if *signed { facts.evidence < 0 } else { facts.evidence > 0 };
            (!facts.pointer && *bytes >= facts.bytes && !contradicts).then(|| table.int(*bytes as usize, *signed))
        }
        Label::Bool => (!facts.pointer && facts.bytes <= 1 && facts.boolish).then_some(TyId::BOOL),
        Label::Ptr(inner) => {
            if !facts.pointer {
                return None;
            }
            let pointee = facts.pointee.or_else(|| match inner.as_deref() {
                Some(Label::Int { bytes, signed }) => Some(table.int(*bytes as usize, *signed)),
                Some(Label::Bool) => Some(TyId::BOOL),
                _ => None,
            });
            Some(table.ptr(pointee.unwrap_or(TyId::B1), Mutbl::Mut))
        }
        Label::Void | Label::Float => None,
    }
}

// ---------------------------------------------------------------------------
// Helpers shared with the emitter

/// Storage width in bytes of an integer-like IR type (`PTR` is 8).
pub fn width(ty: TyId) -> Option<u8> {
    match ty {
        TyId::B1 => Some(1),
        TyId::B2 => Some(2),
        TyId::B4 => Some(4),
        TyId::B8 | TyId::PTR => Some(8),
        _ => None,
    }
}

fn konst(f: &Function, v: ValueId) -> Option<i64> {
    match f.insts[v].kind {
        InstKind::Const(c) => Some(f.consts[c.index()] as u64 as i64),
        _ => None,
    }
}

/// `v` as `base + d` for a constant `d`, following `PtrOffset`s without an index,
/// additions and subtractions of constants, and pointer/integer casts.
pub fn decompose(f: &Function, mut v: ValueId) -> (ValueId, i64) {
    use InstKind::*;
    let mut d = 0i64;
    for _ in 0..64 {
        let wide = width(f.insts[v].ty) == Some(8);
        match f.insts[v].kind {
            PtrOffset { base, index: None, disp, .. } => {
                d = d.wrapping_add(disp as i64);
                v = base;
            }
            Bin { op: BinOp::Add, lhs, rhs } if wide => match (konst(f, lhs), konst(f, rhs)) {
                (_, Some(c)) => {
                    d = d.wrapping_add(c);
                    v = lhs;
                }
                (Some(c), None) => {
                    d = d.wrapping_add(c);
                    v = rhs;
                }
                _ => break,
            },
            Bin { op: BinOp::Sub, lhs, rhs } if wide => match konst(f, rhs) {
                Some(c) => {
                    d = d.wrapping_sub(c);
                    v = lhs;
                }
                None => break,
            },
            IntToPtr(x) | PtrToInt(x) | Cast { kind: CastKind::Bitcast, v: x } => v = x,
            _ => break,
        }
    }
    (v, d)
}

/// The scalar inside `ty` at byte `off` that is exactly `bytes` wide, as a Rust
/// place projection (`.next`, `.items[2].len`) and its type.
pub fn leaf(table: &TyTable, ty: TyId, off: i64, bytes: u32) -> Option<(String, TyId)> {
    fn go(t: &TyTable, ty: TyId, off: u32, bytes: u32, packed: bool, path: &mut String) -> Option<TyId> {
        if path.len() > 512 {
            return None; // a struct that contains itself: bad debug info
        }
        match t.tys[ty] {
            Ty::Struct(s) => {
                let def = &t.structs[s];
                let f = def.fields.iter().find(|f| off >= f.off && off < f.off + t.size_of(f.ty))?;
                path.push('.');
                path.push_str(&f.name);
                go(t, f.ty, off - f.off, bytes, packed || def.packed, path)
            }
            // Indexing an array in a packed struct would take an unaligned reference.
            Ty::Array { elem, len } if !packed => {
                let es = t.size_of(elem);
                if es == 0 || off / es >= len {
                    return None;
                }
                path.push_str(&format!("[{}]", off / es));
                go(t, elem, off % es, bytes, packed, path)
            }
            Ty::Array { .. } | Ty::Slice { .. } | Ty::Ref { .. } => None,
            _ => (off == 0 && t.size_of(ty) == bytes).then_some(ty),
        }
    }
    let off = u32::try_from(off).ok()?;
    let mut path = String::new();
    go(table, ty, off, bytes, false, &mut path).map(|t| (path, t))
}

/// The Rust spelling of a type.
pub fn render(table: &TyTable, ty: TyId) -> String {
    match table.tys[ty] {
        Ty::Int { bits, signed } => format!("{}{bits}", if signed { 'i' } else { 'u' }),
        Ty::Bool => "bool".into(),
        Ty::F32 => "f32".into(),
        Ty::F64 => "f64".into(),
        Ty::Unknown { bytes: b @ (1 | 2 | 4 | 8) } => format!("u{}", b * 8),
        Ty::Unknown { bytes } => format!("[u8; {bytes}]"),
        Ty::RawPtr { pointee, mutbl } => {
            let p = if table.size_of(pointee) == 0 { "u8".into() } else { render(table, pointee) };
            format!("*{} {p}", if mutbl == Mutbl::Mut { "mut" } else { "const" })
        }
        Ty::Ref { pointee, mutbl, .. } => format!("&{}{}", if mutbl == Mutbl::Mut { "mut " } else { "" }, render(table, pointee)),
        Ty::Array { elem, len } => format!("[{}; {len}]", render(table, elem)),
        Ty::Slice { elem } => format!("[{}]", render(table, elem)),
        Ty::Struct(s) => table.structs[s].name.clone(),
        Ty::Fn(_) => "*const u8".into(),
    }
}

/// `#[repr(C)] struct` definitions for `roots` and every struct they contain by
/// value. A pointer field to a struct outside that set is printed `*mut u8`.
pub fn render_structs(table: &TyTable, roots: impl IntoIterator<Item = TyId>) -> String {
    use std::fmt::Write;
    let mut need = vec![false; table.structs.len()];
    // what a root points at is needed too (`-> *mut Node` prints `Node`)
    let mut work: Vec<TyId> = roots
        .into_iter()
        .map(|mut t| {
            while let Ty::RawPtr { pointee, .. } | Ty::Ref { pointee, .. } = table.tys[t] {
                t = pointee;
            }
            t
        })
        .collect();
    while let Some(t) = work.pop() {
        match table.tys[t] {
            Ty::Struct(s) if !need[s.index()] => {
                need[s.index()] = true;
                work.extend(table.structs[s].fields.iter().map(|f| f.ty));
            }
            Ty::Array { elem, .. } | Ty::Slice { elem } => work.push(elem),
            Ty::RawPtr { pointee, .. } | Ty::Ref { pointee, .. } if !matches!(table.tys[pointee], Ty::Struct(_)) => work.push(pointee),
            _ => {}
        }
    }
    let field_ty = |t: TyId| -> String {
        fn go(table: &TyTable, need: &[bool], t: TyId) -> String {
            match table.tys[t] {
                Ty::RawPtr { pointee, mutbl } => match table.tys[pointee] {
                    Ty::Struct(s) if !need[s.index()] => format!("*{} u8", if mutbl == Mutbl::Mut { "mut" } else { "const" }),
                    _ => format!("*{} {}", if mutbl == Mutbl::Mut { "mut" } else { "const" }, go(table, need, pointee)),
                },
                Ty::Array { elem, len } => format!("[{}; {len}]", go(table, need, elem)),
                _ => render(table, t),
            }
        }
        go(table, &need, t)
    };
    let mut out = String::new();
    for (s, def) in table.structs.iter() {
        if !need[s.index()] {
            continue;
        }
        let what = if def.debug { "from debug info" } else { "inferred from field accesses" };
        let _ = writeln!(out, "/// {} bytes, {what}.", def.size);
        let _ = writeln!(out, "#[repr(C{})]", if def.packed { ", packed" } else { "" });
        let _ = writeln!(out, "pub struct {} {{", def.name);
        let mut at = 0;
        for f in &def.fields {
            if f.off > at {
                let _ = writeln!(out, "    pub _pad{at}: [u8; {}],", f.off - at);
            }
            let _ = writeln!(out, "    pub {}: {},", f.name, field_ty(f.ty));
            at = f.off + table.size_of(f.ty);
        }
        if def.size > at {
            let _ = writeln!(out, "    pub _pad{at}: [u8; {}],", def.size - at);
        }
        out.push_str("}\n");
        let _ = writeln!(out, "const _: () = assert!(core::mem::size_of::<{}>() == {});\n", def.name, def.size);
    }
    out
}

// ---------------------------------------------------------------------------
// Facts about one function

struct Uf(Vec<u32>);

impl Uf {
    fn new(n: usize) -> Uf {
        Uf((0..n as u32).collect())
    }
    fn find(&mut self, x: usize) -> usize {
        let mut r = x;
        while self.0[r] as usize != r {
            r = self.0[r] as usize;
        }
        let mut x = x;
        while self.0[x] as usize != r {
            let next = self.0[x] as usize;
            self.0[x] = r as u32;
            x = next;
        }
        r
    }
}

/// A field slot of an inferred pointee type.
#[derive(Clone, Debug)]
struct Slot {
    bytes: u8,
    /// Signed minus unsigned evidence of the values stored or loaded here.
    votes: i32,
    /// A value loaded or stored here (its pointee class is the field's pointee).
    link: Option<u32>,
    /// Those values are used as addresses.
    addr: bool,
}

/// What the accesses through one pointee class say about its type.
#[derive(Clone, Debug, Default)]
struct Shape {
    fields: BTreeMap<i64, Slot>,
    /// Accessed as `base[i]` with elements of this many bytes.
    elem: Option<u8>,
    /// Overlapping fields, negative offsets, different element sizes...: no type.
    conflict: bool,
    written: bool,
    /// Contains a stack, global or constant address: not a heap object to type.
    tainted: bool,
    /// A pointer of the class is advanced by this many bytes (the gcd, if by
    /// several) to another pointer of the class: it walks an array.
    stride: u64,
}

/// Pointee classes: union-find over values with a `Shape` per root.
struct Pts {
    uf: Uf,
    shape: Vec<Shape>,
    pending: Vec<(u32, u32)>,
}

impl Pts {
    fn find(&mut self, x: usize) -> usize {
        self.uf.find(x)
    }

    fn union(&mut self, a: usize, b: usize) {
        self.pending.push((a as u32, b as u32));
        while let Some((a, b)) = self.pending.pop() {
            let (ra, rb) = (self.find(a as usize), self.find(b as usize));
            if ra == rb {
                continue;
            }
            self.uf.0[rb] = ra as u32;
            let sb = std::mem::take(&mut self.shape[rb]);
            let sa = &mut self.shape[ra];
            sa.conflict |= sb.conflict;
            sa.written |= sb.written;
            sa.tainted |= sb.tainted;
            sa.stride = gcd(sa.stride, sb.stride);
            sa.elem = match (sa.elem, sb.elem) {
                (Some(x), Some(y)) if x != y => {
                    sa.conflict = true;
                    Some(x)
                }
                (x, y) => x.or(y),
            };
            for (off, slot) in sb.fields {
                self.add_slot(ra, off, slot);
            }
        }
    }

    /// Adds a field slot to root `c`'s shape, queueing link unions.
    fn add_slot(&mut self, c: usize, off: i64, slot: Slot) {
        let s = &mut self.shape[c];
        if off < 0 {
            s.conflict = true;
            return;
        }
        if let Some(old) = s.fields.get_mut(&off) {
            if old.bytes != slot.bytes {
                s.conflict = true;
                return;
            }
            old.votes += slot.votes;
            old.addr |= slot.addr;
            match (old.link, slot.link) {
                (Some(a), Some(b)) => self.pending.push((a, b)),
                (None, l) => old.link = l,
                _ => {}
            }
            return;
        }
        let end = off + slot.bytes as i64;
        let before = s.fields.range(..off).next_back().is_some_and(|(o, x)| o + x.bytes as i64 > off);
        let after = s.fields.range(off + 1..).next().is_some_and(|(o, _)| *o < end);
        if before || after {
            s.conflict = true;
            return;
        }
        s.fields.insert(off, slot);
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

struct ParamFacts {
    value: ValueId,
    reg: u8,
    used: bool,
    /// Every use truncates it to this many bytes.
    narrow: Option<u8>,
    /// Signedness evidence of its uses (of the truncated values if narrow).
    evidence: i32,
}

struct RetFacts {
    value: ValueId,
    bytes: u8,
    boolish: bool,
    evidence: i32,
    addr: bool,
}

struct Facts {
    /// Signedness class evidence of each value (signed votes minus unsigned).
    evidence: Vec<i32>,
    /// The value's signedness class is used as an address.
    addr: Vec<bool>,
    /// Pointee class root of each value.
    class: Vec<u32>,
    shape: Vec<Shape>,
    params: Vec<ParamFacts>,
    ret: Option<RetFacts>,
}

/// Edges `(block param, argument)` of every reachable terminator.
fn edges(f: &Function, cfg: &Cfg, mut cb: impl FnMut(ValueId, ValueId)) {
    for &b in &cfg.rpo {
        let mut pass = |to: BlockId, args: &[ValueId]| {
            for (&p, &a) in f.blocks[to].params.get(&f.value_pool).iter().zip(args) {
                cb(p, a);
            }
        };
        match f.blocks[b].term {
            Terminator::Jump { to, args } => pass(to, args.get(&f.value_pool)),
            Terminator::Branch { t, f: e, args, .. } => {
                let a = args.get(&f.value_pool);
                let nt = (f.blocks[t].params.len as usize).min(a.len());
                pass(t, &a[..nt]);
                pass(e, &a[nt..]);
            }
            _ => {}
        }
    }
}

fn term_uses(f: &Function, t: Terminator, mut cb: impl FnMut(ValueId)) {
    let list = |l: ListRef, cb: &mut dyn FnMut(ValueId)| l.get(&f.value_pool).iter().for_each(|&v| cb(v));
    match t {
        Terminator::Jump { args, .. } => list(args, &mut cb),
        Terminator::Branch { c, args, .. } => {
            cb(c);
            list(args, &mut cb)
        }
        Terminator::Return(Some(v)) | Terminator::Switch { v, .. } => cb(v),
        Terminator::TailCall { callee, args } => {
            cb(callee);
            list(args, &mut cb)
        }
        Terminator::Return(None) | Terminator::Unreachable => {}
    }
}

fn facts(f: &Function) -> Facts {
    use InstKind::*;
    let n = f.insts.len();
    let cfg = Cfg::new(f);
    let wid = |v: ValueId| width(f.insts[v].ty);
    let insts: Vec<ValueId> = cfg.rpo.iter().flat_map(|&b| f.blocks[b].insts.get(&f.value_pool).iter().copied()).collect();
    let returned: Vec<ValueId> = cfg
        .rpo
        .iter()
        .filter_map(|&b| match f.blocks[b].term {
            Terminator::Return(Some(v)) => Some(v),
            _ => None,
        })
        .collect();

    // 1. Signedness classes.
    let mut uf = Uf::new(n);
    let join = |uf: &mut Uf, a: ValueId, b: ValueId| {
        if wid(a).is_some() && wid(a) == wid(b) {
            let (ra, rb) = (uf.find(a.index()), uf.find(b.index()));
            uf.0[rb] = ra as u32;
        }
    };
    let mut votes: Vec<(ValueId, i32)> = Vec::new();
    let mut addr_use: Vec<ValueId> = Vec::new();
    for &id in &insts {
        match f.insts[id].kind {
            Bin { op, lhs, rhs } => {
                match op {
                    BinOp::Shl | BinOp::LShr | BinOp::AShr | BinOp::RotL | BinOp::RotR => join(&mut uf, id, lhs),
                    // the difference of two pointers is an integer
                    BinOp::Sub if konst(f, rhs).is_none() => join(&mut uf, lhs, rhs),
                    _ => {
                        join(&mut uf, id, lhs);
                        join(&mut uf, id, rhs);
                    }
                }
                match op {
                    BinOp::SDiv | BinOp::SRem | BinOp::AShr => votes.push((lhs, 2)),
                    BinOp::UDiv | BinOp::URem | BinOp::LShr => votes.push((lhs, -1)),
                    _ => {}
                }
            }
            Un { op: UnOp::Neg | UnOp::Not | UnOp::Bswap, v } => join(&mut uf, id, v),
            Select { t, f: e, .. } => {
                join(&mut uf, id, t);
                join(&mut uf, id, e);
            }
            Cmp { cc, lhs, rhs } => {
                join(&mut uf, lhs, rhs);
                match cc {
                    Cond::Slt | Cond::Sle | Cond::Sgt | Cond::Sge => votes.push((lhs, 2)),
                    Cond::Ult | Cond::Ule | Cond::Ugt | Cond::Uge => votes.push((lhs, -2)),
                    _ => {}
                }
            }
            Cast { kind: CastKind::SExt, v } => {
                votes.push((v, 2));
                votes.push((id, 1));
            }
            Const(c) => {
                if let Some(w) = wid(id) {
                    let bits = w as u32 * 8;
                    let x = f.consts[c.index()] as u64;
                    let s = if bits == 64 { x as i64 } else { ((x << (64 - bits)) as i64) >> (64 - bits) };
                    if (-128..0).contains(&s) && s != -1 {
                        votes.push((id, 1));
                    }
                }
            }
            // `lea` is arithmetic as often as it is an address: only a dereference
            // makes something a pointer.
            Load { ptr, .. } => addr_use.push(ptr),
            Store { ptr, .. } => addr_use.push(ptr),
            MemCopy { dst, src, .. } => {
                addr_use.push(dst);
                addr_use.push(src);
            }
            _ => {}
        }
    }
    edges(f, &cfg, |p, a| join(&mut uf, p, a));
    for w in returned.windows(2) {
        join(&mut uf, w[0], w[1]);
    }
    let mut ev_root = vec![0i32; n];
    let mut addr_root = vec![false; n];
    for (v, k) in votes {
        let r = uf.find(v.index());
        ev_root[r] += k;
    }
    for v in addr_use {
        // through the casts and offsets to the base the address was computed from
        let (mut b, _) = decompose(f, v);
        if let PtrOffset { base, index: Some(_), .. } = f.insts[b].kind {
            b = decompose(f, base).0;
        }
        for x in [v, b] {
            let r = uf.find(x.index());
            addr_root[r] = true;
        }
    }
    let mut evidence = vec![0i32; n];
    let mut addr = vec![false; n];
    for v in 0..n {
        let r = uf.find(v);
        addr[v] = addr_root[r];
        evidence[v] = if addr[v] { 0 } else { ev_root[r] };
    }

    // 2. Pointee classes.
    let mut pts = Pts { uf: Uf::new(n), shape: vec![Shape::default(); n], pending: Vec::new() };
    let entry_params = f.blocks[f.entry].params.get(&f.value_pool);
    for &p in entry_params {
        if matches!(f.insts[p].kind, BlockParam(RSP)) {
            pts.shape[p.index()].tainted = true;
        }
    }
    for &id in &insts {
        match f.insts[id].kind {
            AddrOfLocal(_) | AddrOfGlobal(_) | Const(_) => pts.shape[id.index()].tainted = true,
            PtrOffset { base, index: None, disp: 0, .. } => pts.union(id.index(), base.index()),
            IntToPtr(x) | PtrToInt(x) | Cast { kind: CastKind::Bitcast, v: x } => pts.union(id.index(), x.index()),
            Select { t, f: e, .. } => {
                pts.union(id.index(), t.index());
                pts.union(id.index(), e.index());
            }
            _ => {}
        }
    }
    edges(f, &cfg, |p, a| pts.union(p.index(), a.index()));
    for &id in &insts {
        let (ptr, val, bytes, write) = match f.insts[id].kind {
            Load { ptr, .. } => (ptr, id, wid(id), false),
            Store { ptr, val, .. } => (ptr, val, wid(val), true),
            _ => continue,
        };
        let Some(bytes) = bytes else { continue };
        let (root, d) = decompose(f, ptr);
        if let PtrOffset { base, index: Some(_), scale, .. } = f.insts[root].kind {
            let c = pts.find(base.index());
            let s = &mut pts.shape[c];
            s.written |= write;
            if scale != bytes || s.elem.is_some_and(|e| e != bytes) {
                s.conflict = true;
            } else {
                s.elem = Some(bytes);
            }
            continue;
        }
        let c = pts.find(root.index());
        pts.shape[c].written |= write;
        let slot = Slot { bytes, votes: evidence[val.index()].signum(), link: Some(val.index() as u32), addr: addr[val.index()] };
        pts.add_slot(c, d, slot);
        while let Some((a, b)) = pts.pending.pop() {
            pts.union(a as usize, b as usize);
        }
    }
    // Pointers advanced within their own class: array walks.
    for &id in &insts {
        if !matches!(f.insts[id].kind, PtrOffset { index: None, .. } | Bin { op: BinOp::Add | BinOp::Sub, .. }) {
            continue;
        }
        let (b, d) = decompose(f, id);
        if d != 0 && b != id && pts.find(b.index()) == pts.find(id.index()) {
            let c = pts.find(id.index());
            pts.shape[c].stride = gcd(pts.shape[c].stride, d.unsigned_abs());
        }
    }
    let class: Vec<u32> = (0..n).map(|v| pts.find(v) as u32).collect();

    // 3. Arguments: how they are used.
    let mut use_ok = vec![true; n]; // every use so far is a `Trunc`
    let mut used = vec![false; n];
    let mut narrow = vec![0u8; n];
    let mut trunc_ev = vec![0i32; n];
    for &id in &insts {
        let k = f.insts[id].kind;
        for_each_operand(k, f, |v| {
            used[v.index()] = true;
            match k {
                Cast { kind: CastKind::Trunc, .. } => {
                    narrow[v.index()] = narrow[v.index()].max(wid(id).unwrap_or(8));
                    trunc_ev[v.index()] += evidence[id.index()];
                }
                _ => use_ok[v.index()] = false,
            }
        });
    }
    for &b in &cfg.rpo {
        term_uses(f, f.blocks[b].term, |v| {
            used[v.index()] = true;
            use_ok[v.index()] = false;
        });
    }
    let params = entry_params
        .iter()
        .map(|&p| {
            let reg = match f.insts[p].kind {
                BlockParam(r) => r,
                _ => u8::MAX,
            };
            let i = p.index();
            let nar = (used[i] && use_ok[i]).then_some(narrow[i]).filter(|&w| w > 0 && w < 8);
            let evidence = if nar.is_some() { trunc_ev[i] } else { evidence[i] };
            ParamFacts { value: p, reg, used: used[i], narrow: nar, evidence }
        })
        .collect();

    // 4. Return value: how wide it really is.
    let ret = (!returned.is_empty()).then(|| ret_facts(f, &cfg, &returned, &evidence, &addr));
    Facts { evidence, addr, class, shape: pts.shape, params, ret }
}

/// How many low bytes of `v` can be non-zero, whether it is always 0 or 1, and
/// whether that comes from a non-constant.
#[derive(Copy, Clone, PartialEq, Eq)]
struct Nar {
    bytes: u8,
    konst: u8,
    boolish: bool,
}

fn ret_facts(f: &Function, cfg: &Cfg, returned: &[ValueId], evidence: &[i32], addr: &[bool]) -> RetFacts {
    use InstKind::*;
    let n = f.insts.len();
    let mut nar = vec![Nar { bytes: 0, konst: 0, boolish: true }; n];
    let mut incoming: Vec<Vec<ValueId>> = vec![Vec::new(); n];
    edges(f, cfg, |p, a| incoming[p.index()].push(a));
    let entry: Vec<ValueId> = f.blocks[f.entry].params.get(&f.value_pool).to_vec();
    let order: Vec<ValueId> = cfg
        .rpo
        .iter()
        .flat_map(|&b| f.blocks[b].params.get(&f.value_pool).iter().chain(f.blocks[b].insts.get(&f.value_pool)).copied())
        .collect();
    let full = |v: ValueId| Nar { bytes: width(f.insts[v].ty).unwrap_or(8), konst: 0, boolish: f.insts[v].ty == TyId::BOOL };
    let join = |a: Nar, b: Nar| Nar { bytes: a.bytes.max(b.bytes), konst: a.konst.max(b.konst), boolish: a.boolish && b.boolish };
    for _ in 0..32 {
        let mut changed = false;
        for &v in &order {
            let new = match f.insts[v].kind {
                Const(c) => {
                    let x = f.consts[c.index()] as u64;
                    let need = match x {
                        0..=0xff => 1,
                        0x100..=0xffff => 2,
                        0x1_0000..=0xffff_ffff => 4,
                        _ => 8,
                    };
                    Nar { bytes: 0, konst: need.min(width(f.insts[v].ty).unwrap_or(8)), boolish: x <= 1 }
                }
                Cast { kind: CastKind::ZExt, v: x } => {
                    if f.insts[x].ty == TyId::BOOL {
                        Nar { bytes: 1, konst: 0, boolish: true }
                    } else {
                        let inner = nar[x.index()];
                        let w = width(f.insts[x].ty).unwrap_or(8);
                        // a zero-extended narrow constant stays a constant
                        if matches!(f.insts[x].kind, Const(_)) { inner } else { Nar { bytes: w, konst: 0, boolish: false } }
                    }
                }
                BlockParam(_) if !entry.contains(&v) => incoming[v.index()]
                    .iter()
                    .map(|a| nar[a.index()])
                    .fold(Nar { bytes: 0, konst: 0, boolish: true }, join),
                Select { t, f: e, .. } => join(nar[t.index()], nar[e.index()]),
                _ => full(v),
            };
            if new != nar[v.index()] {
                nar[v.index()] = new;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let all = returned.iter().map(|v| nar[v.index()]).fold(Nar { bytes: 0, konst: 0, boolish: true }, join);
    // Signedness of what is returned: from the values zero-extended into it.
    let mut ev = 0;
    let mut seen = vec![false; n];
    let mut work: Vec<ValueId> = returned.to_vec();
    while let Some(v) = work.pop() {
        if std::mem::replace(&mut seen[v.index()], true) {
            continue;
        }
        match f.insts[v].kind {
            Cast { kind: CastKind::ZExt, v: x } => ev += evidence[x.index()].signum(),
            BlockParam(_) if !entry.contains(&v) => work.extend(&incoming[v.index()]),
            Select { t, f: e, .. } => work.extend([t, e]),
            Const(_) => {}
            _ => ev += evidence[v.index()].signum(),
        }
    }
    let bytes = if all.bytes == 0 {
        // only constants: an `int` unless they need more
        if all.konst <= 4 { 4 } else { 8 }
    } else {
        all.bytes.max(all.konst)
    };
    let boolish = all.boolish && all.bytes == 1;
    RetFacts { value: returned[0], bytes, boolish, evidence: ev, addr: returned.iter().any(|v| addr[v.index()]) }
}

// ---------------------------------------------------------------------------
// The whole program

/// One function to recover types for.
pub struct Input<'a> {
    pub f: &'a Function,
    pub sig: Sig,
    pub addr: u64,
    /// Its symbol, to match against the debug info's name.
    pub name: &'a str,
}

/// Recover types for every function. `table` receives the structs and must be
/// the one `debug` was read into.
pub fn recover(
    inputs: &[Option<Input>],
    debug: Option<&DebugInfo>,
    model: Option<&dyn TypeModel>,
    table: &mut TyTable,
) -> (Vec<Option<FnTypes>>, TypeStats) {
    let facts: Vec<Option<Facts>> = inputs.par_iter().map(|x| x.as_ref().map(|x| facts(x.f))).collect();
    for b in [1, 2, 4, 8] {
        table.int(b, false);
        table.int(b, true);
    }
    let mut stats = TypeStats::default();
    let mut layouts: HashMap<String, TyId> = HashMap::new();
    // Structs are shared between functions, so this part is sequential...
    let sigs: Vec<Option<(FnTypes, HashMap<u32, TyId>)>> = inputs
        .iter()
        .zip(&facts)
        .map(|(x, fa)| {
            let (x, fa) = (x.as_ref()?, fa.as_ref()?);
            let dfn = debug.and_then(|d| d.funcs.get(&x.addr)).filter(|d| usable(d, x, table));
            Some(one(x, fa, dfn, model, table, &mut layouts, &mut stats))
        })
        .collect();
    // ... and the per-value tables are filled in parallel.
    let ints: Vec<TyId> = (0..8).map(|i| table.get(&Ty::Int { bits: 8 << (i / 2), signed: i % 2 == 1 }).unwrap()).collect();
    let int = |w: u8, signed: bool| ints[2 * w.trailing_zeros() as usize + signed as usize];
    let out = sigs
        .into_par_iter()
        .zip(inputs.par_iter().zip(&facts))
        .map(|(sig, (x, fa))| {
            let (mut t, classes) = sig?;
            let (f, fa) = (x.as_ref()?.f, fa.as_ref()?);
            let n = f.insts.len();
            t.vals = (0..n)
                .map(|v| {
                    let ty = f.insts[ValueId::new(v)].ty;
                    width(ty).map_or(ty, |w| int(w, fa.evidence[v] > 0 && !fa.addr[v]))
                })
                .collect();
            t.pointee = (0..n).map(|v| classes.get(&fa.class[v]).copied()).collect();
            Some(t)
        })
        .collect();
    (out, stats)
}

/// Can the debug info's prototype be matched to registers?
fn usable(d: &DebugFn, x: &Input, table: &TyTable) -> bool {
    let named = d.name == x.name || d.linkage.as_deref() == Some(x.name);
    let regs = d.params.iter().all(|&(_, t)| scalar(table, t));
    let ret = d.ret.is_none_or(|t| scalar(table, t) || matches!(table.tys[t], Ty::F32 | Ty::F64));
    let enough = d.params.len() >= x.sig.args as usize && (x.sig.stack_args == 0 || d.params.len() == 6 + x.sig.stack_args as usize);
    named && regs && ret && enough && !d.variadic
}

/// Fits in one integer register.
fn scalar(table: &TyTable, t: TyId) -> bool {
    match table.tys[t] {
        Ty::Int { .. } | Ty::Bool | Ty::RawPtr { .. } => true,
        Ty::Unknown { bytes } => matches!(bytes, 1 | 2 | 4 | 8),
        _ => false,
    }
}

/// Struct and scalar types for one function's pointee classes.
struct Classes<'a> {
    fa: &'a Facts,
    memo: HashMap<u32, Option<TyId>>,
}

impl Classes<'_> {
    fn typable(&self, c: u32) -> bool {
        let s = &self.fa.shape[c as usize];
        !s.conflict && !s.tainted && (!s.fields.is_empty() || s.elem.is_some())
    }

    /// A scalar pointee: one width, at offset 0 or indexed, or every field one
    /// width in an array walked in steps of that width.
    fn scalar(&self, c: u32) -> Option<(u8, Option<Slot>)> {
        let s = &self.fa.shape[c as usize];
        let w = s.fields.values().next().map(|x| x.bytes);
        let walk = w.filter(|&w| s.stride != 0 && s.stride.is_multiple_of(w as u64) && s.fields.values().all(|x| x.bytes == w));
        match s.elem.or(walk) {
            Some(w) => s.fields.iter().all(|(o, x)| x.bytes == w && o % w as i64 == 0).then(|| (w, s.fields.get(&0).cloned())),
            None if s.fields.len() == 1 => s.fields.get(&0).map(|x| (x.bytes, Some(x.clone()))),
            None => None,
        }
    }

    fn link(&self, s: &Slot) -> Option<u32> {
        s.link.map(|l| self.fa.class[l as usize]).filter(|&l| self.typable(l))
    }

    /// A key for the layout, to share one struct between functions.
    fn key(&self, c: u32, stack: &mut Vec<u32>) -> String {
        use std::fmt::Write;
        if let Some(d) = stack.iter().position(|&x| x == c) {
            return format!("^{d}");
        }
        if stack.len() > 3 {
            return "~".into();
        }
        stack.push(c);
        let s = &self.fa.shape[c as usize];
        let mut k = format!("e{:?}{{", s.elem);
        for (o, x) in &s.fields {
            let _ = write!(k, "{o}:{}:{}:{}:", x.bytes, x.votes > 0, x.addr);
            if x.bytes == 8 {
                if let Some(l) = self.link(x) {
                    k += &self.key(l, stack);
                }
            }
            k.push(';');
        }
        stack.pop();
        k + "}"
    }

    fn slot_ty(&mut self, x: &Slot, table: &mut TyTable, layouts: &mut HashMap<String, TyId>, stats: &mut TypeStats) -> TyId {
        if x.bytes == 8 {
            if let Some(l) = self.link(x) {
                if let Some(t) = self.ty(l, table, layouts, stats) {
                    return table.ptr(t, Mutbl::Mut);
                }
            }
            if x.addr {
                return table.ptr(TyId::B1, Mutbl::Mut);
            }
        }
        table.int(x.bytes as usize, x.votes > 0)
    }

    fn ty(&mut self, c: u32, table: &mut TyTable, layouts: &mut HashMap<String, TyId>, stats: &mut TypeStats) -> Option<TyId> {
        if let Some(&t) = self.memo.get(&c) {
            return t;
        }
        if !self.typable(c) {
            self.memo.insert(c, None);
            return None;
        }
        let shape = self.fa.shape[c as usize].clone();
        if let Some((w, slot)) = self.scalar(c) {
            self.memo.insert(c, None); // a pointer to itself is a byte pointer
            let t = match slot {
                Some(x) => self.slot_ty(&x, table, layouts, stats),
                None => table.int(w as usize, false),
            };
            self.memo.insert(c, Some(t));
            return Some(t);
        }
        if shape.elem.is_some() {
            self.memo.insert(c, None);
            return None;
        }
        let key = self.key(c, &mut Vec::new());
        if let Some(&t) = layouts.get(&key) {
            self.memo.insert(c, Some(t));
            return Some(t);
        }
        let size = shape.fields.iter().map(|(o, x)| *o as u32 + x.bytes as u32).max().unwrap_or(0);
        let name = table.fresh_struct_name(&format!("s{}", stats.inferred_structs + 1));
        stats.inferred_structs += 1;
        let (sid, t) = table.add_struct(StructDef { name, fields: Vec::new(), size, packed: true, debug: false });
        layouts.insert(key, t);
        self.memo.insert(c, Some(t));
        let mut fields = Vec::new();
        for (o, x) in &shape.fields {
            let ty = self.slot_ty(x, table, layouts, stats);
            fields.push(Field { name: format!("f{o}"), off: *o as u32, ty });
        }
        table.structs[sid].fields = fields;
        Some(t)
    }
}

/// A Rust identifier for a parameter name from debug info that can't clash with
/// the emitter's own names (registers, `vN`, `argN`, `frame`, `bb`).
fn param_ident(name: &str, used: &mut std::collections::HashSet<String>) -> String {
    const REGS: [&str; 16] = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"];
    let mut s = crate::names::sanitize(name);
    let numbered = |p: &str| s.strip_prefix(p).is_some_and(|r| !r.is_empty() && r.bytes().all(|c| c.is_ascii_digit()));
    if REGS.contains(&s.as_str()) || numbered("v") || numbered("arg") || matches!(s.as_str(), "frame" | "bb" | "pair_") || s.starts_with('_') {
        s.push('_');
    }
    while !used.insert(s.clone()) {
        s.push('_');
    }
    s
}

fn one(
    x: &Input,
    fa: &Facts,
    dfn: Option<&DebugFn>,
    model: Option<&dyn TypeModel>,
    table: &mut TyTable,
    layouts: &mut HashMap<String, TyId>,
    stats: &mut TypeStats,
) -> (FnTypes, HashMap<u32, TyId>) {
    let f = x.f;
    let n = f.insts.len();
    let mut cl = Classes { fa, memo: HashMap::new() };
    // Inferred pointee types, in value order so struct names are deterministic.
    let mut pointee_of: HashMap<u32, TyId> = HashMap::new();
    for v in 0..n {
        let c = fa.class[v];
        if c as usize == v {
            if let Some(t) = cl.ty(c, table, layouts, stats) {
                pointee_of.insert(c, t);
            }
        }
    }

    // Signature order: rdi, rsi, ... then stack arguments.
    let regs: Vec<u8> = SYSV_ARGS[..x.sig.args as usize]
        .iter()
        .copied()
        .chain((0..x.sig.stack_args).map(|j| STACK_ARG_BASE + j))
        .collect();
    let param = |reg: u8| fa.params.iter().find(|p| p.reg == reg);

    let mut args: Vec<ArgTy> = Vec::new();
    let mut ret = None;
    let mut debug_class: HashMap<u32, TyId> = HashMap::new();
    if let Some(d) = dfn {
        stats.debug_fns += 1;
        let mut used = std::collections::HashSet::new();
        for (j, &reg) in regs.iter().enumerate() {
            let (name, ty) = &d.params[j];
            let mut ty = *ty;
            if let (Ty::RawPtr { pointee, mutbl }, Some(p)) = (table.tys[ty], param(reg)) {
                let c = fa.class[p.value.index()];
                // `void *` and `char *` say less than an inferred struct
                let vague = matches!(table.tys[pointee], Ty::Unknown { .. } | Ty::Int { bits: 8, .. });
                match pointee_of.get(&c) {
                    Some(&inferred) if vague => ty = table.ptr(inferred, mutbl),
                    _ => {
                        debug_class.insert(c, pointee);
                    }
                }
            }
            args.push(ArgTy { reg, ty: Some(ty), name: Some(param_ident(name, &mut used)) });
        }
        if x.sig.ret && !x.sig.ret2 {
            ret = d.ret.filter(|&t| scalar(table, t));
        }
        // Pointer fields of debug-info structs type the values loaded from them.
        let mut work: Vec<(u32, TyId)> = debug_class.drain().collect();
        while let Some((c, t)) = work.pop() {
            if debug_class.contains_key(&c) {
                continue;
            }
            debug_class.insert(c, t);
            for (o, slot) in &fa.shape[c as usize].fields {
                let Some(l) = slot.link else { continue };
                if let Some((_, lt)) = leaf(table, t, *o, slot.bytes as u32) {
                    if let Ty::RawPtr { pointee, .. } = table.tys[lt] {
                        if table.size_of(pointee) > 0 {
                            work.push((fa.class[l as usize], pointee));
                        }
                    }
                }
            }
        }
    } else {
        for &reg in &regs {
            let ty = param(reg).and_then(|p| infer_arg(p, fa, &pointee_of, table));
            args.push(ArgTy { reg, ty, name: None });
        }
        if x.sig.ret && !x.sig.ret2 {
            ret = fa.ret.as_ref().and_then(|r| infer_ret(r, fa, &pointee_of, table));
        }
        if let Some(m) = model {
            ask(m, x, fa, &pointee_of, &regs, &mut args, &mut ret, table, stats);
        }
    }
    stats.args += args.len();
    stats.typed_args += args.iter().filter(|a| a.ty.is_some()).count();

    let mut classes = pointee_of;
    classes.extend(debug_class);
    (FnTypes { vals: Vec::new(), pointee: Vec::new(), args, ret }, classes)
}

fn infer_arg(p: &ParamFacts, fa: &Facts, pointee_of: &HashMap<u32, TyId>, table: &mut TyTable) -> Option<TyId> {
    if !p.used || p.reg == RSP {
        return None;
    }
    let c = fa.class[p.value.index()];
    if let Some(&t) = pointee_of.get(&c) {
        let m = if fa.shape[c as usize].written { Mutbl::Mut } else { Mutbl::Not };
        return Some(table.ptr(t, m));
    }
    if fa.addr[p.value.index()] {
        return Some(table.ptr(TyId::B1, Mutbl::Mut));
    }
    let (bytes, signed) = match p.narrow {
        Some(w) => (w, p.evidence > 0),
        None => (8, fa.evidence[p.value.index()] > 0),
    };
    (bytes != 8 || signed).then(|| table.int(bytes as usize, signed))
}

fn infer_ret(r: &RetFacts, fa: &Facts, pointee_of: &HashMap<u32, TyId>, table: &mut TyTable) -> Option<TyId> {
    if r.addr {
        let t = pointee_of.get(&fa.class[r.value.index()]).copied().unwrap_or(TyId::B1);
        return Some(table.ptr(t, Mutbl::Mut));
    }
    if r.boolish {
        return Some(TyId::BOOL);
    }
    let signed = r.evidence > 0;
    (r.bytes != 8 || signed).then(|| table.int(r.bytes as usize, signed))
}

#[allow(clippy::too_many_arguments)]
fn ask(
    m: &dyn TypeModel,
    x: &Input,
    fa: &Facts,
    pointee_of: &HashMap<u32, TyId>,
    regs: &[u8],
    args: &mut [ArgTy],
    ret: &mut Option<TyId>,
    table: &mut TyTable,
    stats: &mut TypeStats,
) {
    let param = |reg: u8| fa.params.iter().find(|p| p.reg == reg);
    let mut vars: Vec<Var> = regs.iter().enumerate().map(|(j, &r)| Var::Arg { j, value: param(r).map(|p| p.value) }).collect();
    let want_ret = x.sig.ret && !x.sig.ret2;
    if let (true, Some(r)) = (want_ret, &fa.ret) {
        vars.push(Var::Ret { value: r.value });
    }
    let answers = m.propose(x.f, &vars);
    for (var, ans) in vars.iter().zip(answers) {
        let Some(p) = ans else { continue };
        let facts = match *var {
            Var::Arg { value: Some(v), .. } => {
                let pf = param(match f_reg(x.f, v) {
                    Some(r) => r,
                    None => continue,
                })
                .unwrap();
                let c = fa.class[v.index()];
                VarFacts {
                    pointer: pointee_of.contains_key(&c) || fa.addr[v.index()],
                    bytes: pf.narrow.unwrap_or(8),
                    evidence: pf.evidence,
                    boolish: false,
                    pointee: pointee_of.get(&c).copied(),
                }
            }
            // never read: anything the right size goes
            Var::Arg { value: None, .. } => VarFacts { pointer: true, bytes: 1, evidence: 0, boolish: true, pointee: None },
            Var::Ret { value } => {
                let r = fa.ret.as_ref().unwrap();
                let c = fa.class[value.index()];
                VarFacts { pointer: r.addr, bytes: r.bytes, evidence: r.evidence, boolish: r.boolish, pointee: pointee_of.get(&c).copied() }
            }
        };
        match parse_label(&p.label).and_then(|l| accept(&l, &facts, table)) {
            Some(t) => {
                stats.accepted += 1;
                match *var {
                    Var::Arg { j, .. } => args[j].ty = Some(t),
                    Var::Ret { .. } => *ret = Some(t),
                }
            }
            None => stats.rejected += 1,
        }
    }
}

fn f_reg(f: &Function, v: ValueId) -> Option<u8> {
    match f.insts[v].kind {
        InstKind::BlockParam(r) => Some(r),
        _ => None,
    }
}

/// For each value, the value it always equals: itself, or for a block parameter
/// whose every incoming argument is (after `decompose` with offset 0) the same
/// value, that value.
pub fn aliases(f: &Function) -> Vec<ValueId> {
    let cfg = Cfg::new(f);
    let n = f.insts.len();
    let mut alias: Vec<ValueId> = (0..n).map(ValueId::new).collect();
    let mut incoming: Vec<Vec<ValueId>> = vec![Vec::new(); n];
    edges(f, &cfg, |p, a| incoming[p.index()].push(a));
    let resolve = |alias: &[ValueId], mut v: ValueId| {
        for _ in 0..64 {
            let (b, d) = decompose(f, v);
            if d != 0 {
                return v;
            }
            if alias[b.index()] == b {
                return b;
            }
            v = alias[b.index()];
        }
        v
    };
    for _ in 0..16 {
        let mut changed = false;
        for p in 0..n {
            if incoming[p].is_empty() {
                continue;
            }
            let me = ValueId::new(p);
            let mut same: Option<ValueId> = None;
            let mut ok = true;
            for &a in &incoming[p] {
                let r = resolve(&alias, a);
                if r == me {
                    continue;
                }
                match same {
                    None => same = Some(r),
                    Some(x) if x == r => {}
                    _ => ok = false,
                }
            }
            if let (true, Some(x)) = (ok, same) {
                if alias[p] != x {
                    alias[p] = x;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    // Resolve chains so every entry points at its final value.
    (0..n).map(|v| resolve(&alias, ValueId::new(v))).collect()
}

/// `v` as `base + d`, looking through `aliases` as well as `decompose`.
pub fn base_of(f: &Function, alias: &[ValueId], mut v: ValueId) -> (ValueId, i64) {
    let mut d = 0i64;
    for _ in 0..64 {
        let (b, x) = decompose(f, v);
        d = d.wrapping_add(x);
        let a = alias[b.index()];
        if a == b {
            return (b, d);
        }
        v = a;
    }
    (v, d)
}
