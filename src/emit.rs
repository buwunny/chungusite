//! IR -> Rust source. One emitter for both modes (docs/ir.md).
//!
//! Every IR value becomes a Rust integer (`u8`..`u64`, or `bool` for comparisons).
//! Pointers are `u64` addresses, so the output always type-checks no matter how a
//! register is used, and each memory access becomes an explicit read or write.
//!
//! * **fast**: every load and store is a raw, unaligned access inside `unsafe { }`,
//!   and the function is an `unsafe fn` taking every argument as `u64`.
//! * **safe**: arguments that `borrow::analyze` classifies as `&T` / `&mut T`
//!   arrive as byte slices (`&[u8]`, `&mut [u8]`, or `Option<..>` when nullable).
//!   An access whose pointer derives from exactly one such argument becomes a
//!   bounds-checked slice read or write; everything else falls back to the
//!   fast-mode raw access. The function is only `unsafe` if a raw access remains.
//!
//! Control flow is emitted as-is: a single-block function is straight-line code,
//! anything with edges is a `loop { match bb { .. } }` state machine with block
//! parameters as mutable variables. Recovering `if`/`while` is future work.
//!
//! The input must be clean SSA (`opt::clean`), which is what the borrow analysis
//! expects too.
use crate::borrow::{analyze, Class, ParamBorrow, RSP};
use crate::cfg::Cfg;
use crate::ir::*;
use crate::verify::for_each_operand;
use std::fmt::Write;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Mode {
    Fast,
    Safe,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmitStats {
    /// Loads and stores emitted as bounds-checked slice accesses.
    pub checked: usize,
    /// Loads and stores emitted as raw pointer accesses inside `unsafe`.
    pub raw: usize,
    /// Instructions or terminators the emitter can't express yet (`todo!()`).
    pub todo: usize,
}

const REG: [&str; 16] = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
];

/// System V argument registers, in order: rdi, rsi, rdx, rcx, r8, r9.
const SYSV_ARGS: [u8; 6] = [7, 6, 2, 1, 8, 9];

/// How a safe-mode argument arrives.
#[derive(Copy, Clone, PartialEq, Eq)]
enum ArgKind {
    Int,
    Slice { mutbl: bool, nullable: bool },
}

struct Emitter<'a> {
    f: &'a Function,
    /// Borrow analysis (safe mode only).
    borrow: Option<crate::borrow::Analysis>,
    args: Vec<ArgKind>,
    /// Entry parameter index of each value, if it is one.
    entry_param: Vec<Option<usize>>,
    /// Values that need a variable declared up front (block params, and values
    /// used outside the block that defines them).
    hoisted: Vec<bool>,
    tail_name: &'a dyn Fn(u64) -> Option<String>,
    /// Rust expression for a constant address that points into the binary's data.
    global_of: &'a dyn Fn(u64) -> Option<String>,
    stats: EmitStats,
}

/// Emit `f` as a Rust function named `name` into `out`. `name_of` names call and
/// tail-call targets by address, for comments.
pub fn emit_function(
    f: &Function,
    name: &str,
    mode: Mode,
    name_of: &dyn Fn(u64) -> Option<String>,
    out: &mut String,
) -> EmitStats {
    emit_function_with_globals(f, name, mode, name_of, &|_| None, out)
}

/// `emit_function`, plus `global_of`, which turns a constant address used as a
/// pointer into a Rust expression (the address of a `static`), or `None` to keep
/// the raw address.
pub fn emit_function_with_globals(
    f: &Function,
    name: &str,
    mode: Mode,
    name_of: &dyn Fn(u64) -> Option<String>,
    global_of: &dyn Fn(u64) -> Option<String>,
    out: &mut String,
) -> EmitStats {
    let cfg = Cfg::new(f);
    let entry_params = f.blocks[f.entry].params.get(&f.value_pool);
    let mut entry_param = vec![None; f.insts.len()];
    for (k, &p) in entry_params.iter().enumerate() {
        entry_param[p.index()] = Some(k);
    }

    let borrow = (mode == Mode::Safe).then(|| analyze(f));
    let args = entry_params
        .iter()
        .enumerate()
        .map(|(k, _)| match &borrow {
            Some(a) => arg_kind(&a.params[k]),
            None => ArgKind::Int,
        })
        .collect();

    let mut e = Emitter {
        f,
        borrow,
        args,
        entry_param,
        hoisted: hoisted(f, &cfg),
        tail_name: name_of,
        global_of,
        stats: EmitStats::default(),
    };

    let mut body = String::new();
    e.body(&cfg, &mut body);
    e.signature(name, out);
    out.push_str(&body);
    out.push_str("}\n");
    e.stats
}

fn arg_kind(p: &ParamBorrow) -> ArgKind {
    match p.class {
        _ if p.reg == RSP => ArgKind::Int,
        Class::Shared => ArgKind::Slice { mutbl: false, nullable: p.nullable },
        Class::Mut => ArgKind::Slice { mutbl: true, nullable: p.nullable },
        Class::NotPointer | Class::Raw => ArgKind::Int,
    }
}

/// Block of definition for every value, then mark values used in another block.
fn hoisted(f: &Function, cfg: &Cfg) -> Vec<bool> {
    let n = f.insts.len();
    let mut def = vec![u32::MAX; n];
    let mut hoist = vec![false; n];
    for &b in &cfg.rpo {
        let blk = &f.blocks[b];
        for &p in blk.params.get(&f.value_pool) {
            def[p.index()] = b.index() as u32;
            hoist[p.index()] = b != f.entry;
        }
        for &id in blk.insts.get(&f.value_pool) {
            def[id.index()] = b.index() as u32;
        }
    }
    for &b in &cfg.rpo {
        let blk = &f.blocks[b];
        let mut use_in = |v: ValueId| {
            if def[v.index()] != b.index() as u32 {
                hoist[v.index()] = true;
            }
        };
        for &id in blk.insts.get(&f.value_pool) {
            for_each_operand(f.insts[id].kind, f, &mut use_in);
        }
        term_uses(f, blk.term, &mut use_in);
    }
    // Entry params are function arguments, already in scope everywhere.
    for &p in f.blocks[f.entry].params.get(&f.value_pool) {
        hoist[p.index()] = false;
    }
    hoist
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

fn rty(ty: TyId) -> &'static str {
    match ty {
        TyId::B1 => "u8",
        TyId::B2 => "u16",
        TyId::B4 => "u32",
        TyId::BOOL => "bool",
        TyId::UNIT => "()",
        _ => "u64", // B8, PTR, and anything richer the IR grows later
    }
}

fn bytes(ty: TyId) -> usize {
    match ty {
        TyId::B1 => 1,
        TyId::B2 => 2,
        TyId::B4 => 4,
        _ => 8,
    }
}

fn signed(ty: TyId) -> &'static str {
    match bytes(ty) {
        1 => "i8",
        2 => "i16",
        4 => "i32",
        _ => "i64",
    }
}

impl Emitter<'_> {
    fn name(&self, v: ValueId) -> String {
        match (self.entry_param[v.index()], self.f.insts[v].kind) {
            (Some(_), InstKind::BlockParam(r)) => REG[r as usize & 15].to_string(),
            _ => format!("v{}", v.index()),
        }
    }

    fn ty(&self, v: ValueId) -> TyId {
        self.f.insts[v].ty
    }

    /// Entry parameters, ordered for the signature: SysV argument registers first,
    /// then any other register the function reads on entry (callee-saved, rsp...).
    fn sig_order(&self) -> Vec<usize> {
        let params = self.f.blocks[self.f.entry].params.get(&self.f.value_pool);
        let reg = |k: usize| match self.f.insts[params[k]].kind {
            InstKind::BlockParam(r) => r,
            _ => u8::MAX,
        };
        let rank = |r: u8| SYSV_ARGS.iter().position(|&a| a == r).unwrap_or(6 + r as usize);
        let mut order: Vec<usize> = (0..params.len()).collect();
        order.sort_by_key(|&k| rank(reg(k)));
        order
    }

    fn signature(&self, name: &str, out: &mut String) {
        let params = self.f.blocks[self.f.entry].params.get(&self.f.value_pool);
        let unsafety = if self.stats.raw > 0 { "unsafe " } else { "" };
        let mut sig = Vec::new();
        let mut prologue = String::new();
        for k in self.sig_order() {
            let n = self.name(params[k]);
            match self.args[k] {
                ArgKind::Int => sig.push(format!("mut {n}: u64")),
                ArgKind::Slice { mutbl, nullable } => {
                    let r = if mutbl { "&mut [u8]" } else { "&[u8]" };
                    let (m, t) = if nullable { (if mutbl { "mut " } else { "" }, format!("Option<{r}>")) } else { ("", r.to_string()) };
                    sig.push(format!("{m}{n}_ref: {t}"));
                    let addr = if nullable {
                        format!("{n}_ref.as_deref().map_or(0, |s| s.as_ptr() as u64)")
                    } else {
                        format!("{n}_ref.as_ptr() as u64")
                    };
                    let _ = writeln!(prologue, "    let {n}_base: u64 = {addr};");
                    let _ = writeln!(prologue, "    let mut {n}: u64 = {n}_base;");
                }
            }
        }
        let _ = writeln!(out, "pub {unsafety}fn {name}({}) -> u64 {{", sig.join(", "));
        out.push_str(&prologue);
    }

    fn body(&mut self, cfg: &Cfg, out: &mut String) {
        let f = self.f;
        // Up-front declarations for block params and cross-block values.
        for &b in &cfg.rpo {
            let blk = &f.blocks[b];
            for &v in blk.params.get(&f.value_pool).iter().chain(blk.insts.get(&f.value_pool)) {
                if self.hoisted[v.index()] {
                    let zero = if self.ty(v) == TyId::BOOL { "false" } else { "0" };
                    let _ = writeln!(out, "    let mut {}: {} = {zero};", self.name(v), rty(self.ty(v)));
                }
            }
        }
        let has_edges = cfg.rpo.iter().any(|&b| f.blocks[b].term.successors()[0].is_some());
        if !has_edges {
            self.block(f.entry, "    ", out);
            return;
        }
        let _ = writeln!(out, "    let mut bb: u32 = {};", f.entry.index());
        out.push_str("    loop {\n        match bb {\n");
        for &b in &cfg.rpo {
            let _ = writeln!(out, "            {} => {{", b.index());
            self.block(b, "                ", out);
            out.push_str("            }\n");
        }
        out.push_str("            _ => unreachable!(),\n        }\n    }\n");
    }

    fn block(&mut self, b: BlockId, ind: &str, out: &mut String) {
        let f = self.f;
        let blk = &f.blocks[b];
        for &id in blk.insts.get(&f.value_pool) {
            let at = f.origin.get(id.index()).copied().unwrap_or(0);
            match self.inst(id) {
                Stmt::Value(e) if self.hoisted[id.index()] => {
                    let _ = writeln!(out, "{ind}{} = {e}; // {at:#x}", self.name(id));
                }
                Stmt::Value(e) => {
                    let _ = writeln!(out, "{ind}let {}: {} = {e}; // {at:#x}", self.name(id), rty(self.ty(id)));
                }
                Stmt::Effect(e) => {
                    let _ = writeln!(out, "{ind}{e}; // {at:#x}");
                }
            }
        }
        match blk.term {
            Terminator::Jump { to, args } => {
                self.edge(to, args.get(&f.value_pool), ind, out);
            }
            Terminator::Branch { c, t, f: e, args } => {
                let a = args.get(&f.value_pool);
                let nt = f.blocks[t].params.len as usize;
                let inner = format!("{ind}    ");
                let _ = writeln!(out, "{ind}if {} {{", self.name(c));
                self.edge(t, &a[..nt], &inner, out);
                let _ = writeln!(out, "{ind}}} else {{");
                self.edge(e, &a[nt..], &inner, out);
                let _ = writeln!(out, "{ind}}}");
            }
            Terminator::Return(Some(v)) => {
                let _ = writeln!(out, "{ind}return {} as u64;", self.name(v));
            }
            Terminator::Return(None) => {
                let _ = writeln!(out, "{ind}return 0;");
            }
            Terminator::TailCall { callee, .. } => {
                self.stats.todo += 1;
                let target = match f.insts[callee].kind {
                    InstKind::IntToPtr(a) => match f.insts[a].kind {
                        InstKind::Const(c) => {
                            let addr = f.consts[c.index()] as u64;
                            (self.tail_name)(addr).unwrap_or_else(|| format!("{addr:#x}"))
                        }
                        _ => self.name(callee),
                    },
                    _ => self.name(callee),
                };
                let _ = writeln!(out, "{ind}todo!(\"tail call to {target}: argument recovery not implemented\");");
            }
            Terminator::Switch { .. } => {
                self.stats.todo += 1;
                let _ = writeln!(out, "{ind}todo!(\"switch\");");
            }
            Terminator::Unreachable => {
                let _ = writeln!(out, "{ind}panic!(\"execution ran past the end of the lifted code\");");
            }
        }
    }

    /// Pass edge arguments to `to`'s parameters (a parallel assignment) and jump.
    fn edge(&self, to: BlockId, args: &[ValueId], ind: &str, out: &mut String) {
        let params = self.f.blocks[to].params.get(&self.f.value_pool);
        let pairs: Vec<(String, String)> = params
            .iter()
            .zip(args)
            .map(|(&p, &a)| (self.name(p), self.name(a)))
            .filter(|(p, a)| p != a)
            .collect();
        match pairs.len() {
            0 => {}
            1 => {
                let _ = writeln!(out, "{ind}{} = {};", pairs[0].0, pairs[0].1);
            }
            _ => {
                let (l, r): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
                let _ = writeln!(out, "{ind}({}) = ({});", l.join(", "), r.join(", "));
            }
        }
        let _ = writeln!(out, "{ind}bb = {};", to.index());
    }

    fn inst(&mut self, id: ValueId) -> Stmt {
        use InstKind::*;
        let f = self.f;
        let ty = self.ty(id);
        let t = rty(ty);
        let n = |v: ValueId| self.name(v);
        let e = match f.insts[id].kind {
            Const(c) => {
                let v = f.consts[c.index()];
                match ty {
                    TyId::BOOL => (v != 0).to_string(),
                    _ => {
                        let mask = if bytes(ty) >= 8 { u64::MAX as u128 } else { (1u128 << (bytes(ty) * 8)) - 1 };
                        format!("{:#x}_{t}", v & mask)
                    }
                }
            }
            Param(i) => format!("todo!(\"param {i}\")"),
            BlockParam(_) => n(id),
            Bin { op, lhs, rhs } => self.bin(op, lhs, rhs),
            Un { op, v } => match op {
                UnOp::Neg => format!("{}.wrapping_neg()", n(v)),
                UnOp::Not => format!("!{}", n(v)),
                UnOp::Bswap => format!("{}.swap_bytes()", n(v)),
                UnOp::Popcnt => format!("{}.count_ones() as {t}", n(v)),
                UnOp::Ctz => format!("{}.trailing_zeros() as {t}", n(v)),
                UnOp::Clz => format!("{}.leading_zeros() as {t}", n(v)),
            },
            Cmp { cc, lhs, rhs } => {
                let (a, b) = (n(lhs), n(rhs));
                let s = signed(self.ty(lhs));
                match cc {
                    Cond::Eq => format!("{a} == {b}"),
                    Cond::Ne => format!("{a} != {b}"),
                    Cond::Ult => format!("{a} < {b}"),
                    Cond::Ule => format!("{a} <= {b}"),
                    Cond::Ugt => format!("{a} > {b}"),
                    Cond::Uge => format!("{a} >= {b}"),
                    Cond::Slt => format!("({a} as {s}) < ({b} as {s})"),
                    Cond::Sle => format!("({a} as {s}) <= ({b} as {s})"),
                    Cond::Sgt => format!("({a} as {s}) > ({b} as {s})"),
                    Cond::Sge => format!("({a} as {s}) >= ({b} as {s})"),
                }
            }
            Cast { kind, v } => match kind {
                CastKind::SExt => format!("{} as {} as {t}", n(v), signed(self.ty(v))),
                _ => format!("{} as {t}", n(v)),
            },
            Select { c, t: a, f: b } => format!("if {} {{ {} }} else {{ {} }}", n(c), n(a), n(b)),
            PtrOffset { base, index, scale, disp } => {
                let mut s = n(base);
                if let Some(i) = index {
                    s = if scale == 1 {
                        format!("{s}.wrapping_add({})", n(i))
                    } else {
                        format!("{s}.wrapping_add({}.wrapping_mul({scale}))", n(i))
                    };
                }
                match disp {
                    0 => s,
                    d if d < 0 => format!("{s}.wrapping_sub({:#x})", -(d as i64)),
                    d => format!("{s}.wrapping_add({d:#x})"),
                }
            }
            IntToPtr(v) => match f.insts[v].kind {
                Const(c) => (self.global_of)(f.consts[c.index()] as u64).unwrap_or_else(|| n(v)),
                _ => n(v),
            },
            PtrToInt(v) => n(v),
            Load { ptr, .. } => self.load(ptr, ty),
            Store { ptr, val, .. } => return Stmt::Effect(self.store(ptr, val)),
            MemCopy { dst, src, len } => {
                let s = format!(
                    "unsafe {{ core::ptr::copy({} as *const u8, {} as *mut u8, {} as usize) }}",
                    n(src),
                    n(dst),
                    n(len)
                );
                self.stats.raw += 1;
                return Stmt::Effect(s);
            }
            other => {
                self.stats.todo += 1;
                let s = format!("{other:?}").replace('"', "'");
                if ty == TyId::UNIT {
                    return Stmt::Effect(format!("todo!(\"{s}\")"));
                }
                format!("todo!(\"{s}\")")
            }
        };
        Stmt::Value(e)
    }

    fn bin(&self, op: BinOp, lhs: ValueId, rhs: ValueId) -> String {
        let (a, b) = (self.name(lhs), self.name(rhs));
        let ty = self.ty(lhs);
        let (t, s) = (rty(ty), signed(ty));
        if ty == TyId::BOOL {
            let o = match op {
                BinOp::And => "&",
                BinOp::Or => "|",
                _ => "^",
            };
            return format!("{a} {o} {b}");
        }
        match op {
            BinOp::Add => format!("{a}.wrapping_add({b})"),
            BinOp::Sub => format!("{a}.wrapping_sub({b})"),
            BinOp::Mul => format!("{a}.wrapping_mul({b})"),
            BinOp::UDiv => format!("{a} / {b}"),
            BinOp::URem => format!("{a} % {b}"),
            BinOp::SDiv => format!("({a} as {s}).wrapping_div({b} as {s}) as {t}"),
            BinOp::SRem => format!("({a} as {s}).wrapping_rem({b} as {s}) as {t}"),
            BinOp::And => format!("{a} & {b}"),
            BinOp::Or => format!("{a} | {b}"),
            BinOp::Xor => format!("{a} ^ {b}"),
            BinOp::Shl => format!("{a}.wrapping_shl({b} as u32)"),
            BinOp::LShr => format!("{a}.wrapping_shr({b} as u32)"),
            BinOp::AShr => format!("({a} as {s}).wrapping_shr({b} as u32) as {t}"),
            BinOp::RotL => format!("{a}.rotate_left({b} as u32)"),
            BinOp::RotR => format!("{a}.rotate_right({b} as u32)"),
        }
    }

    /// The safe-mode slice an access through `ptr` can use: the argument's name,
    /// if `ptr` derives from exactly one slice argument.
    fn slice_root(&self, ptr: ValueId, write: bool) -> Option<(String, ArgKind)> {
        let a = self.borrow.as_ref()?;
        let o = a.origin[ptr.index()];
        if o.roots.count_ones() != 1 {
            return None;
        }
        let k = o.roots.trailing_zeros() as usize;
        let kind = self.args[k];
        match kind {
            ArgKind::Slice { mutbl, .. } if mutbl || !write => {
                let p = self.f.blocks[self.f.entry].params.get(&self.f.value_pool)[k];
                Some((self.name(p), kind))
            }
            _ => None,
        }
    }

    fn load(&mut self, ptr: ValueId, ty: TyId) -> String {
        let (t, len, p) = (rty(ty), bytes(ty), self.name(ptr));
        if let Some((root, ArgKind::Slice { nullable, mutbl })) = self.slice_root(ptr, false) {
            self.stats.checked += 1;
            let s = match (nullable, mutbl) {
                (false, _) => format!("{root}_ref"),
                (true, false) => format!("{root}_ref.unwrap()"),
                (true, true) => format!("{root}_ref.as_deref().unwrap()"),
            };
            return format!("{t}::from_le_bytes({s}[{p}.wrapping_sub({root}_base) as usize..][..{len}].try_into().unwrap())");
        }
        self.stats.raw += 1;
        format!("unsafe {{ ({p} as *const {t}).read_unaligned() }}")
    }

    fn store(&mut self, ptr: ValueId, val: ValueId) -> String {
        let vt = self.ty(val);
        let (t, len, p, v) = (rty(vt), bytes(vt), self.name(ptr), self.name(val));
        if let Some((root, ArgKind::Slice { nullable, .. })) = self.slice_root(ptr, true) {
            self.stats.checked += 1;
            let s = if nullable { format!("{root}_ref.as_deref_mut().unwrap()") } else { format!("{root}_ref") };
            return format!("{s}[{p}.wrapping_sub({root}_base) as usize..][..{len}].copy_from_slice(&{v}.to_le_bytes())");
        }
        self.stats.raw += 1;
        format!("unsafe {{ ({p} as *mut {t}).write_unaligned({v}) }}")
    }
}

enum Stmt {
    /// Defines the value: `let vN: T = expr;`
    Value(String),
    /// Side effect only.
    Effect(String),
}
