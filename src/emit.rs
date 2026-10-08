//! IR -> Rust source. One emitter for both modes (docs/ir.md).
//!
//! Every IR value becomes a Rust integer (`u8`..`u64`, or `bool` for comparisons).
//! Pointers are `u64` addresses, so the output always type-checks no matter how a
//! register is used, and each memory access becomes an explicit read or write.
//!
//! With recovered types (`Env::types`, from `types.rs`) integers get their
//! signedness (`i32` ...), the signature gets real argument and return types
//! (`n: i32`, `p: *mut Node`, `-> bool`), and a load or store at a known offset
//! from a pointer to a struct becomes a field access (`(*p).count`). Arguments
//! are converted to the `u64` registers the body works on in a short prologue.
//!
//! * **fast**: every load and store is a raw, unaligned access inside `unsafe { }`,
//!   and the function is an `unsafe fn` taking every argument as `u64`.
//! * **safe**: arguments that `borrow::analyze` classifies as `&T` / `&mut T`
//!   arrive as byte slices (`&[u8]`, `&mut [u8]`, or `Option<..>` when nullable).
//!   An access whose pointer derives from exactly one such argument becomes a
//!   bounds-checked slice read or write; everything else falls back to the
//!   fast-mode raw access. The function is only `unsafe` if a raw access remains.
//!
//! Control flow is structured (`structure.rs`): `if`/`else`, `loop` with `break`
//! and `continue`, and early `return`, with block parameters as mutable variables.
//! An irreducible CFG falls back to a `loop { match bb { .. } }` state machine,
//! which is correct for any CFG.
//!
//! The input must be clean SSA (`opt::clean`), which is what the borrow analysis
//! expects too.
use crate::abi::{Sig, Site, STACK_ARG_BASE, SYSV_ARGS};
use crate::borrow::{analyze, Class, ParamBorrow, RSP};
use crate::cfg::Cfg;
use crate::structure::{print, structure, Node, Source};
use crate::ir::*;
use crate::types::{base_of, leaf, render, width, FnTypes};
use crate::verify::for_each_operand;
use std::fmt::Write;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Mode {
    Fast,
    Safe,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmitStats {
    /// Loads and stores emitted safely: bounds-checked slice accesses, or fields
    /// of a `&S` / `&mut S` argument.
    pub checked: usize,
    /// Loads and stores emitted as raw pointer accesses inside `unsafe`.
    pub raw: usize,
    /// Instructions or terminators the emitter can't express yet (`todo!()`).
    pub todo: usize,
    /// Functions whose control flow is irreducible, emitted as a
    /// `loop { match bb { .. } }` state machine instead of structured code.
    pub state_machines: usize,
}

const REG: [&str; 16] = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
];


/// How a safe-mode argument arrives.
#[derive(Copy, Clone, PartialEq, Eq)]
enum ArgKind {
    Int,
    Slice { mutbl: bool, nullable: bool },
    /// `&S` / `&mut S` for a recovered struct `S`: every access through it is a
    /// field access.
    Struct { mutbl: bool, nullable: bool, ty: TyId },
}

/// What a call site calls, as `program.rs` resolved it.
#[derive(Clone, Debug)]
pub struct CallInfo {
    /// The Rust path to call: a decompiled function's name, or `ffi::name` for an
    /// extern. `None` calls the callee value as a C function pointer.
    pub path: Option<String>,
    /// The callee returns a value in rax.
    pub ret: bool,
    /// ... and another in rdx: the call is a `(u64, u64)`.
    pub ret2: bool,
    /// Calling it is an FFI call (`unsafe` in any mode).
    pub foreign: bool,
    /// Recovered argument types, in signature order (`None`: `u64`). Empty if
    /// the callee takes every argument as `u64`.
    pub args: Vec<Option<TyId>>,
    /// Recovered return type (`None`: `u64`).
    pub ret_ty: Option<TyId>,
}

/// The whole-program context a function is emitted in (`program.rs`).
pub struct Env<'a> {
    /// The function's signature after `abi::apply`; `None` for IR straight from
    /// the lifter, where every register read on entry is an argument.
    pub sig: Option<Sig>,
    /// Each call site's callee.
    pub call: &'a dyn Fn(Site) -> Option<CallInfo>,
    /// Safe mode only: take every argument as an integer, because decompiled
    /// callers pass addresses, not slices.
    pub demote: bool,
    /// Try structured control flow before the state machine.
    pub structure: bool,
    /// Rust expression for a constant address that points into the binary's data
    /// (the address of a `static`), or `None` to keep the raw address.
    pub global_of: &'a dyn Fn(u64) -> Option<String>,
    /// Recovered types (`types.rs`), and the table they refer to. `None` prints
    /// every integer unsigned and every argument as `u64`.
    pub types: Option<(&'a FnTypes, &'a TyTable)>,
}

struct Emitter<'a> {
    f: &'a Function,
    sig: Option<Sig>,
    call: &'a dyn Fn(Site) -> Option<CallInfo>,
    /// Values something uses (a void call's value usually isn't).
    used: Vec<bool>,
    /// Values only computed to call a callee that is called by name instead.
    skip: Vec<bool>,
    /// Try structured control flow before the state machine.
    structure: bool,
    /// Borrow analysis (safe mode only).
    borrow: Option<crate::borrow::Analysis>,
    args: Vec<ArgKind>,
    /// Entry parameter index of each value, if it is one.
    entry_param: Vec<Option<usize>>,
    /// Values that need a variable declared up front (block params, and values
    /// used outside the block that defines them).
    hoisted: Vec<bool>,
    /// Rust expression for a constant address that points into the binary's data.
    global_of: &'a dyn Fn(u64) -> Option<String>,
    stats: EmitStats,
    table: &'a TyTable,
    types: Option<&'a FnTypes>,
    /// Rust type of each value.
    vt: Vec<TyId>,
    /// The name each entry parameter has in the signature, when it isn't the
    /// register name (a slice, a reference, a pointer, a name from debug info).
    bind: Vec<Option<String>>,
    /// What each block parameter always equals (`types::aliases`).
    alias: Vec<ValueId>,
    u64_ty: TyId,
}

/// Emit `f`, straight from the lifter (no signature), as a Rust function named
/// `name` into `out`. Calls go through the target's address as a function pointer.
/// `name_of` is unused: calls are named by `program.rs`, which knows the callees.
pub fn emit_function(
    f: &Function,
    name: &str,
    mode: Mode,
    name_of: &dyn Fn(u64) -> Option<String>,
    out: &mut String,
) -> EmitStats {
    emit_function_with(f, name, mode, true, name_of, &|_| None, out)
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
    emit_function_with(f, name, mode, true, name_of, global_of, out)
}

/// `emit_function_with_globals`, choosing whether to structure control flow. With
/// `structure: false` every function with a branch is a state machine.
pub fn emit_function_with(
    f: &Function,
    name: &str,
    mode: Mode,
    structure: bool,
    name_of: &dyn Fn(u64) -> Option<String>,
    global_of: &dyn Fn(u64) -> Option<String>,
    out: &mut String,
) -> EmitStats {
    let _ = name_of;
    let call = |_: Site| None;
    emit_function_in(f, name, mode, &Env { sig: None, call: &call, demote: false, structure, global_of, types: None }, out)
}

/// Emit `f` as a Rust function named `name`, with its signature and callees from
/// `env`.
pub fn emit_function_in(f: &Function, name: &str, mode: Mode, env: &Env, out: &mut String) -> EmitStats {
    let cfg = Cfg::new(f);
    let entry_params = f.blocks[f.entry].params.get(&f.value_pool);
    let mut entry_param = vec![None; f.insts.len()];
    for (k, &p) in entry_params.iter().enumerate() {
        entry_param[p.index()] = Some(k);
    }

    let mut own = TyTable::new();
    let (types, table) = match env.types {
        Some((t, table)) => (Some(t), table),
        None => {
            for b in [1, 2, 4, 8] {
                own.int(b, false);
            }
            (None, &own)
        }
    };
    let int = |w: u8| table.get(&Ty::Int { bits: w * 8, signed: false }).expect("integer types are interned");
    let vt: Vec<TyId> = match types {
        Some(t) => t.vals.clone(),
        None => f.insts.iter().map(|(_, i)| width(i.ty).map_or(i.ty, int)).collect(),
    };
    let alias = crate::types::aliases(f);

    let borrow = (mode == Mode::Safe).then(|| analyze(f));
    let args: Vec<ArgKind> = entry_params
        .iter()
        .enumerate()
        .map(|(k, &p)| match &borrow {
            Some(a) if !env.demote => {
                let kind = arg_kind(&a.params[k]);
                match (kind, types.and_then(|t| t.pointee[p.index()])) {
                    (ArgKind::Slice { mutbl, nullable }, Some(s)) if matches!(table.tys[s], Ty::Struct(_)) => {
                        if struct_arg(f, &cfg, a, &alias, table, k, p, s) {
                            ArgKind::Struct { mutbl, nullable, ty: s }
                        } else {
                            kind
                        }
                    }
                    _ => kind,
                }
            }
            _ => ArgKind::Int,
        })
        .collect();

    let mut e = Emitter {
        f,
        sig: env.sig,
        call: env.call,
        used: used(f, &cfg),
        skip: callee_only(f, &cfg, env.call),
        structure: env.structure,
        borrow,
        args,
        entry_param,
        hoisted: hoisted(f, &cfg),
        global_of: env.global_of,
        stats: EmitStats::default(),
        table,
        types,
        vt,
        bind: vec![None; entry_params.len()],
        alias,
        u64_ty: int(8),
    };
    e.bind_names();

    let mut body = String::new();
    e.body(&cfg, &mut body);
    e.signature(name, out);
    out.push_str(&body);
    out.push_str("}\n");
    e.stats
}

/// Can safe-mode argument `k` (entry parameter `p`) be a `&S` for struct `s`?
/// Only if every access through it is a field access of `s` at its own address.
#[allow(clippy::too_many_arguments)]
fn struct_arg(f: &Function, cfg: &Cfg, a: &crate::borrow::Analysis, alias: &[ValueId], table: &TyTable, k: usize, p: ValueId, s: TyId) -> bool {
    let through = |v: ValueId| a.origin[v.index()].roots & (1 << k) != 0;
    let mut any = false;
    for &b in &cfg.rpo {
        for &id in f.blocks[b].insts.get(&f.value_pool) {
            let (ptr, bytes) = match f.insts[id].kind {
                InstKind::Load { ptr, .. } => (ptr, width(f.insts[id].ty)),
                InstKind::Store { ptr, val, .. } => (ptr, width(f.insts[val].ty)),
                InstKind::MemCopy { dst, src, .. } if through(dst) || through(src) => return false,
                _ => continue,
            };
            if !through(ptr) {
                continue;
            }
            let (r, d) = base_of(f, alias, ptr);
            let ok = r == p && bytes.is_some_and(|w| leaf(table, s, d, w as u32).is_some());
            if !ok {
                return false;
            }
            any = true;
        }
    }
    any
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

/// Values used by some reachable instruction or terminator.
fn used(f: &Function, cfg: &Cfg) -> Vec<bool> {
    let mut u = vec![false; f.insts.len()];
    for &b in &cfg.rpo {
        let blk = &f.blocks[b];
        for &id in blk.insts.get(&f.value_pool) {
            for_each_operand(f.insts[id].kind, f, |v| u[v.index()] = true);
        }
        term_uses(f, blk.term, |v| u[v.index()] = true);
    }
    u
}

/// Values whose only use is naming the callee of calls emitted by name: the
/// callee's address, its `IntToPtr`, and a GOT slot load.
fn callee_only(f: &Function, cfg: &Cfg, call: &dyn Fn(Site) -> Option<CallInfo>) -> Vec<bool> {
    let mut uses = vec![0u32; f.insts.len()];
    for &b in &cfg.rpo {
        let blk = &f.blocks[b];
        for &id in blk.insts.get(&f.value_pool) {
            for_each_operand(f.insts[id].kind, f, |v| uses[v.index()] += 1);
        }
        term_uses(f, blk.term, |v| uses[v.index()] += 1);
    }
    let mut work = Vec::new();
    let named = |s: Site| call(s).is_some_and(|c| c.path.is_some());
    for &b in &cfg.rpo {
        let blk = &f.blocks[b];
        for &id in blk.insts.get(&f.value_pool) {
            if let InstKind::Call { callee, .. } = f.insts[id].kind {
                if named(Site::Call(id)) {
                    work.push(callee);
                }
            }
        }
        if let Terminator::TailCall { callee, .. } = blk.term {
            if named(Site::Tail(b)) {
                work.push(callee);
            }
        }
    }
    let mut skip = vec![false; f.insts.len()];
    while let Some(v) = work.pop() {
        uses[v.index()] -= 1;
        let k = f.insts[v].kind;
        let pure = matches!(k, InstKind::Const(_) | InstKind::IntToPtr(_) | InstKind::PtrToInt(_) | InstKind::Load { volatile: false, .. });
        if uses[v.index()] == 0 && pure {
            skip[v.index()] = true;
            for_each_operand(k, f, |o| work.push(o));
        }
    }
    skip
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
            (Some(_), InstKind::BlockParam(r)) if r >= STACK_ARG_BASE => format!("arg{}", 6 + (r - STACK_ARG_BASE) as usize),
            (Some(_), InstKind::BlockParam(r)) => REG[r as usize & 15].to_string(),
            _ => format!("v{}", v.index()),
        }
    }

    fn ty(&self, v: ValueId) -> TyId {
        self.f.insts[v].ty
    }

    /// The Rust spelling of a type (`PAIR` is a tuple, `UNIT` is `()`).
    fn rs(&self, t: TyId) -> String {
        match t {
            TyId::PAIR => "(u64, u64)".into(),
            TyId::UNIT => "()".into(),
            TyId::BOOL => "bool".into(),
            _ => render(self.table, t),
        }
    }

    /// Rust type of value `v`.
    fn rt(&self, v: ValueId) -> String {
        self.rs(self.vt[v.index()])
    }

    fn is_signed(&self, v: ValueId) -> bool {
        matches!(self.table.tys[self.vt[v.index()]], Ty::Int { signed: true, .. })
    }

    /// The unsigned and signed integer types as wide as `v`.
    fn ut(&self, v: ValueId) -> String {
        format!("u{}", bytes(self.ty(v)) * 8)
    }
    fn st(&self, v: ValueId) -> String {
        signed(self.ty(v)).to_string()
    }

    /// `v` as its unsigned type, ready to be an operand.
    fn as_u(&self, v: ValueId) -> String {
        if self.is_signed(v) { format!("({} as {})", self.name(v), self.ut(v)) } else { self.name(v) }
    }

    /// `v` as its signed type, ready to be an operand.
    fn as_s(&self, v: ValueId) -> String {
        if self.is_signed(v) || self.ty(v) == TyId::BOOL { self.name(v) } else { format!("({} as {})", self.name(v), self.st(v)) }
    }

    /// `v` as type `t`, ready to be an operand.
    fn opnd(&self, v: ValueId, t: TyId) -> String {
        let c = self.conv(&self.name(v), self.vt[v.index()], t);
        if c.contains(' ') { format!("({c})") } else { c }
    }

    /// Expression `e` of type `from` as type `to`, with the bits the machine
    /// code would have: integers widen by their own signedness, so a signed
    /// value is made unsigned before it is zero-extended.
    fn conv(&self, e: &str, from: TyId, to: TyId) -> String {
        if from == to {
            return e.to_string();
        }
        // `x` and `x as u32` can take another `as` without parentheses
        let atom = |e: &str| e.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'.');
        let chain = e.split(" as ").all(atom);
        let e = if chain { e.to_string() } else { format!("({e})") };
        let method = if atom(&e) { e.clone() } else { format!("({e})") };
        let t = self.table;
        let ints = |x: TyId| match t.tys[x] {
            Ty::Int { bits, signed } => Some((bits, signed)),
            Ty::Unknown { bytes: b @ (1 | 2 | 4 | 8) } => Some(((b * 8) as u8, false)),
            _ => None,
        };
        let to_s = self.rs(to);
        match (t.tys[from], t.tys[to]) {
            (Ty::Bool, _) if ints(to).is_some() => format!("{e} as {to_s}"),
            (_, Ty::Bool) => format!("{e} != 0"),
            (Ty::RawPtr { .. }, Ty::RawPtr { .. }) => format!("{e} as {to_s}"),
            (Ty::RawPtr { .. }, _) => format!("{e} as {to_s}"),
            (_, Ty::RawPtr { .. }) => match ints(from) {
                Some((bits, true)) if bits < 64 => format!("{e} as u{bits} as {to_s}"),
                _ => format!("{e} as {to_s}"),
            },
            (Ty::F32 | Ty::F64, _) => format!("{method}.to_bits() as {to_s}"),
            (_, Ty::F32) => format!("f32::from_bits({e} as u32)"),
            (_, Ty::F64) => format!("f64::from_bits({e} as u64)"),
            _ => match (ints(from), ints(to)) {
                (Some((fb, true)), Some((tb, _))) if tb > fb => format!("{e} as u{fb} as {to_s}"),
                _ => format!("{e} as {to_s}"),
            },
        }
    }

    /// An argument of type `t` as the register value the body reads. Callers
    /// extend 8- and 16-bit arguments to 32 bits by their signedness (gcc and
    /// clang both do, and clang's code relies on it), so do the same.
    fn arg_to_reg(&self, b: &str, t: TyId, to: TyId) -> String {
        match self.table.tys[t] {
            Ty::Int { bits: bits @ (8 | 16), signed } => {
                let w = self.table.get(&Ty::Int { bits: 32, signed }).expect("integer types are interned");
                let _ = bits;
                self.conv(&format!("{b} as {}32", if signed { 'i' } else { 'u' }), w, to)
            }
            Ty::Bool => self.conv(&format!("{b} as u32"), self.table.get(&Ty::Int { bits: 32, signed: false }).unwrap(), to),
            _ => self.conv(b, t, to),
        }
    }

    /// A zero of type `t`.
    fn zero(&self, t: TyId) -> String {
        match self.table.tys[t] {
            Ty::Bool => "false".into(),
            Ty::RawPtr { mutbl: Mutbl::Mut, .. } => "core::ptr::null_mut()".into(),
            Ty::RawPtr { .. } => "core::ptr::null()".into(),
            _ => format!("0_{}", self.rs(t)),
        }
    }

    /// Signature names for the entry parameters that aren't their register.
    fn bind_names(&mut self) {
        let params: Vec<ValueId> = self.f.blocks[self.f.entry].params.get(&self.f.value_pool).to_vec();
        let typed: Vec<(u8, Option<TyId>, Option<String>)> =
            self.types.map(|t| t.args.iter().map(|a| (a.reg, a.ty, a.name.clone())).collect()).unwrap_or_default();
        for (k, &p) in params.iter().enumerate() {
            let reg = match self.f.insts[p].kind {
                InstKind::BlockParam(r) => r,
                _ => continue,
            };
            let n = self.name(p);
            let (ty, dname) = typed.iter().find(|a| a.0 == reg).map(|a| (a.1, a.2.clone())).unwrap_or((None, None));
            self.bind[k] = match self.args[k] {
                ArgKind::Slice { .. } | ArgKind::Struct { .. } => Some(dname.unwrap_or(format!("{n}_ref"))),
                ArgKind::Int => match ty.map(|t| self.table.tys[t]) {
                    Some(Ty::RawPtr { .. }) => Some(dname.unwrap_or(format!("{n}_p"))),
                    _ => dname,
                },
            };
        }
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

    /// Entry parameter indices in signature order, with `None` for an argument the
    /// function takes but never uses (and the name to give it), and the
    /// argument's recovered type.
    fn arg_order(&self) -> Vec<(Option<usize>, String, Option<TyId>)> {
        let params = self.f.blocks[self.f.entry].params.get(&self.f.value_pool);
        let Some(sig) = self.sig else {
            return self.sig_order().into_iter().map(|k| (Some(k), String::new(), None)).collect();
        };
        let find = |reg: u8| params.iter().position(|&p| matches!(self.f.insts[p].kind, InstKind::BlockParam(r) if r == reg));
        let typed = |reg: u8| self.types.and_then(|t| t.args.iter().find(|a| a.reg == reg));
        let unused = |reg: u8, fallback: String| match typed(reg).and_then(|a| a.name.as_ref()) {
            Some(n) => format!("_{n}"),
            None => fallback,
        };
        let mut order = Vec::new();
        for &reg in &SYSV_ARGS[..sig.args as usize] {
            order.push((find(reg), unused(reg, format!("_{}", REG[reg as usize])), typed(reg).and_then(|a| a.ty)));
        }
        for j in 0..sig.stack_args {
            let reg = STACK_ARG_BASE + j;
            order.push((find(reg), unused(reg, format!("_arg{}", 6 + j as usize)), typed(reg).and_then(|a| a.ty)));
        }
        // anything else on entry (there shouldn't be anything after `abi::apply`)
        for k in 0..params.len() {
            if !order.iter().any(|&(x, _, _)| x == Some(k)) {
                order.push((Some(k), String::new(), None));
            }
        }
        order
    }

    fn signature(&self, name: &str, out: &mut String) {
        let params = self.f.blocks[self.f.entry].params.get(&self.f.value_pool);
        let unsafety = if self.stats.raw > 0 { "unsafe " } else { "" };
        let mut sig = Vec::new();
        let mut prologue = String::new();
        for (k, unused, ty) in self.arg_order() {
            let Some(k) = k else {
                sig.push(format!("{unused}: {}", ty.map_or("u64".to_string(), |t| self.rs(t))));
                continue;
            };
            let p = params[k];
            let n = self.name(p);
            let pt = self.rt(p);
            let bind = self.bind[k].clone();
            match self.args[k] {
                ArgKind::Int => match (ty, bind) {
                    (None, None) => sig.push(format!("mut {n}: {pt}")),
                    (Some(t), b) => {
                        let b = b.unwrap_or_else(|| n.clone());
                        sig.push(format!("{b}: {}", self.rs(t)));
                        let _ = writeln!(prologue, "    let mut {n}: {pt} = {};", self.arg_to_reg(&b, t, self.vt[p.index()]));
                    }
                    (None, Some(b)) => {
                        sig.push(format!("{b}: {pt}"));
                        let _ = writeln!(prologue, "    let mut {n}: {pt} = {b};");
                    }
                },
                ArgKind::Slice { mutbl, nullable } => {
                    let b = bind.unwrap_or_else(|| format!("{n}_ref"));
                    let r = if mutbl { "&mut [u8]" } else { "&[u8]" };
                    let (m, t) = if nullable { (if mutbl { "mut " } else { "" }, format!("Option<{r}>")) } else { ("", r.to_string()) };
                    sig.push(format!("{m}{b}: {t}"));
                    let addr = if nullable {
                        format!("{b}.as_deref().map_or(0, |s| s.as_ptr() as u64)")
                    } else {
                        format!("{b}.as_ptr() as u64")
                    };
                    let _ = writeln!(prologue, "    let {n}_base: u64 = {addr};");
                    let _ = writeln!(prologue, "    let mut {n}: {pt} = {};", self.conv(&format!("{n}_base"), self.u64_ty, self.vt[p.index()]));
                }
                ArgKind::Struct { mutbl, nullable, ty: st } => {
                    let b = bind.unwrap_or_else(|| format!("{n}_ref"));
                    let sn = self.rs(st);
                    let r = if mutbl { format!("&mut {sn}") } else { format!("&{sn}") };
                    let (m, t) = if nullable { (if mutbl { "mut " } else { "" }, format!("Option<{r}>")) } else { ("", r) };
                    sig.push(format!("{m}{b}: {t}"));
                    let addr = if nullable {
                        format!("{b}.as_deref().map_or(0, |s| s as *const {sn} as u64)")
                    } else {
                        format!("&*{b} as *const {sn} as u64")
                    };
                    let _ = writeln!(prologue, "    let {n}_base: u64 = {addr};");
                    let _ = writeln!(prologue, "    let mut {n}: {pt} = {};", self.conv(&format!("{n}_base"), self.u64_ty, self.vt[p.index()]));
                }
            }
        }
        let ret = match self.sig {
            Some(s) if s.ret2 => " -> (u64, u64)".to_string(),
            Some(s) if !s.ret => String::new(),
            _ => match self.types.and_then(|t| t.ret) {
                Some(t) => format!(" -> {}", self.rs(t)),
                None => " -> u64".to_string(),
            },
        };
        let _ = writeln!(out, "pub {unsafety}fn {name}({}){ret} {{", sig.join(", "));
        out.push_str(&prologue);
    }

    fn body(&mut self, cfg: &Cfg, out: &mut String) {
        let f = self.f;
        // The stack frame (`frame.rs`): u128s, so it is 16-byte aligned like a real one.
        if let Some((_, l)) = f.locals.iter().next() {
            let n = (l.size as usize).div_ceil(16);
            if n * 16 <= 4096 {
                let _ = writeln!(out, "    let mut frame = [0u128; {n}];");
            } else {
                let _ = writeln!(out, "    let mut frame = vec![0u128; {n}];");
            }
        }
        // Up-front declarations for block params and cross-block values.
        for &b in &cfg.rpo {
            let blk = &f.blocks[b];
            for &v in blk.params.get(&f.value_pool).iter().chain(blk.insts.get(&f.value_pool)) {
                if self.hoisted[v.index()] && !self.skip[v.index()] {
                    let zero = match self.ty(v) {
                        TyId::BOOL => "false",
                        TyId::PAIR => "(0, 0)",
                        _ => "0",
                    };
                    let _ = writeln!(out, "    let mut {}: {} = {zero};", self.name(v), self.rt(v));
                }
            }
        }
        let has_edges = cfg.rpo.iter().any(|&b| f.blocks[b].term.successors()[0].is_some());
        if !has_edges {
            self.block(f.entry, "    ", out);
            return;
        }
        // Structured `if`/`loop` when the CFG allows it. Statements are emitted
        // again below if it doesn't, so count them only once.
        let before = self.stats;
        if self.structure {
            if let Some(nodes) = structure(f, cfg, self) {
                print(&nodes, 1, out);
                return;
            }
            self.stats = before;
        }
        self.stats.state_machines += 1;
        let _ = writeln!(out, "    let mut bb: u32 = {};", f.entry.index());
        out.push_str("    loop {\n        match bb {\n");
        for &b in &cfg.rpo {
            let _ = writeln!(out, "            {} => {{", b.index());
            self.block(b, "                ", out);
            out.push_str("            }\n");
        }
        out.push_str("            _ => unreachable!(),\n        }\n    }\n");
    }

    /// The block's statements, one per line, without its terminator.
    fn stmt_lines(&mut self, b: BlockId) -> Vec<String> {
        let f = self.f;
        let blk = &f.blocks[b];
        let mut out = Vec::new();
        for &id in blk.insts.get(&f.value_pool) {
            if self.skip[id.index()] {
                continue;
            }
            let at = f.origin.get(id.index()).copied().unwrap_or(0);
            let n = self.name(id);
            match self.inst(id) {
                Stmt::Value(e) if self.hoisted[id.index()] => out.push(format!("{n} = {e}; // {at:#x}")),
                Stmt::Value(e) => out.push(format!("let {n}: {} = {e}; // {at:#x}", self.rt(id))),
                Stmt::Effect(e) => out.push(format!("{e}; // {at:#x}")),
                Stmt::Pair(e) => {
                    // rax:rdx; the rdx half is a `CallOut` reading `vN_pair.1`
                    out.push(format!("let {n}_pair: (u64, u64) = {e}; // {at:#x}"));
                    let first = self.conv(&format!("{n}_pair.0"), self.u64_ty, self.vt[id.index()]);
                    if self.hoisted[id.index()] {
                        out.push(format!("{n} = {first};"));
                    } else {
                        out.push(format!("let {n}: {} = {first};", self.rt(id)));
                    }
                }
            }
        }
        out
    }

    /// The function's return type: `None` for `u64` (the default).
    fn ret_ty(&self) -> TyId {
        self.types.and_then(|t| t.ret).unwrap_or(self.u64_ty)
    }

    /// A terminator that leaves the function (or can't be expressed yet).
    fn exit_line(&mut self, b: BlockId) -> String {
        let t = self.f.blocks[b].term;
        match t {
            Terminator::Return(Some(v)) if self.ty(v) == TyId::PAIR => format!("return {};", self.name(v)),
            Terminator::Return(Some(v)) if self.types.is_none() => format!("return {} as u64;", self.name(v)),
            Terminator::Return(Some(v)) => format!("return {};", self.conv(&self.name(v), self.vt[v.index()], self.ret_ty())),
            Terminator::Return(None) if self.sig.is_some_and(|s| !s.ret) => "return;".to_string(),
            Terminator::Return(None) if self.types.is_none() => "return 0;".to_string(),
            Terminator::Return(None) => format!("return {};", self.zero(self.ret_ty())),
            Terminator::TailCall { callee, args } => {
                let (call, ret, ret2, rty) = self.call_expr(Site::Tail(b), callee, args);
                let me2 = self.sig.is_some_and(|s| s.ret2);
                match (self.sig.is_none_or(|s| s.ret), ret) {
                    (true, true) if ret2 && !me2 => format!("return {};", self.conv(&format!("{call}.0"), self.u64_ty, self.ret_ty())),
                    (true, true) if ret2 => format!("return {call};"),
                    (true, true) => format!("return {};", self.conv(&call, rty, self.ret_ty())),
                    (true, false) if self.types.is_none() => format!("{call}; return 0;"),
                    (true, false) => format!("{call}; return {};", self.zero(self.ret_ty())),
                    (false, _) => format!("{call}; return;"),
                }
            }
            Terminator::Switch { .. } => {
                self.stats.todo += 1;
                "todo!(\"switch\");".to_string()
            }
            Terminator::Unreachable => "panic!(\"execution ran past the end of the lifted code\");".to_string(),
            Terminator::Jump { .. } | Terminator::Branch { .. } => unreachable!("not an exit"),
        }
    }

    /// One `match` arm of the state machine.
    fn block(&mut self, b: BlockId, ind: &str, out: &mut String) {
        let f = self.f;
        for line in self.stmt_lines(b) {
            let _ = writeln!(out, "{ind}{line}");
        }
        match f.blocks[b].term {
            Terminator::Jump { to, args } => {
                self.goto(to, args.get(&f.value_pool), ind, out);
            }
            Terminator::Branch { c, t, f: e, args } => {
                let a = args.get(&f.value_pool);
                let nt = f.blocks[t].params.len as usize;
                let inner = format!("{ind}    ");
                let _ = writeln!(out, "{ind}if {} {{", self.name(c));
                self.goto(t, &a[..nt], &inner, out);
                let _ = writeln!(out, "{ind}}} else {{");
                self.goto(e, &a[nt..], &inner, out);
                let _ = writeln!(out, "{ind}}}");
            }
            _ => {
                let _ = writeln!(out, "{ind}{}", self.exit_line(b));
            }
        }
    }

    /// Pass edge arguments to `to`'s parameters, as a parallel assignment.
    fn assign(&self, to: BlockId, args: &[ValueId]) -> Option<String> {
        let params = self.f.blocks[to].params.get(&self.f.value_pool);
        let pairs: Vec<(String, String)> = params
            .iter()
            .zip(args)
            .map(|(&p, &a)| (self.name(p), self.conv(&self.name(a), self.vt[a.index()], self.vt[p.index()])))
            .filter(|(p, a)| p != a)
            .collect();
        match pairs.len() {
            0 => None,
            1 => Some(format!("{} = {};", pairs[0].0, pairs[0].1)),
            _ => {
                let (l, r): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
                Some(format!("({}) = ({});", l.join(", "), r.join(", ")))
            }
        }
    }

    /// State machine edge: assign `to`'s parameters, then jump.
    fn goto(&self, to: BlockId, args: &[ValueId], ind: &str, out: &mut String) {
        if let Some(a) = self.assign(to, args) {
            let _ = writeln!(out, "{ind}{a}");
        }
        let _ = writeln!(out, "{ind}bb = {};", to.index());
    }

    fn inst(&mut self, id: ValueId) -> Stmt {
        use InstKind::*;
        let f = self.f;
        let ty = self.ty(id);
        let vt = self.vt[id.index()];
        let t = self.rt(id);
        let n = |v: ValueId| self.name(v);
        let e = match f.insts[id].kind {
            Const(c) => {
                let v = f.consts[c.index()];
                match ty {
                    TyId::BOOL => (v != 0).to_string(),
                    _ if self.is_signed(id) => {
                        let bits = bytes(ty) * 8;
                        let x = ((v as u64) << (64 - bits)) as i64 >> (64 - bits);
                        format!("{x}_{t}")
                    }
                    _ => {
                        let mask = if bytes(ty) >= 8 { u64::MAX as u128 } else { (1u128 << (bytes(ty) * 8)) - 1 };
                        format!("{:#x}_{t}", v & mask)
                    }
                }
            }
            Undef if ty == TyId::BOOL => "false".to_string(),
            CallOut { call, reg: crate::abi::RDX } if matches!(f.insts[call].kind, Call { .. }) && self.ret2(call) => {
                self.conv(&format!("{}_pair.1", n(call)), self.u64_ty, vt)
            }
            Undef | CallOut { .. } => format!("0_{t}"),
            Aggregate { ty: TyId::PAIR, fields } => {
                let v = fields.get(&f.value_pool);
                let u = |x: ValueId| self.conv(&n(x), self.vt[x.index()], self.u64_ty);
                if self.types.is_none() {
                    format!("({} as u64, {} as u64)", n(v[0]), n(v[1]))
                } else {
                    format!("({}, {})", u(v[0]), u(v[1]))
                }
            }
            AddrOfLocal(_) => self.conv("frame.as_mut_ptr() as u64", self.u64_ty, vt),
            Call { callee, args } => {
                let (call, ret, ret2, rty) = self.call_expr(Site::Call(id), callee, args);
                if ret2 {
                    return Stmt::Pair(call);
                }
                if !ret || !self.used[id.index()] {
                    return Stmt::Effect(call);
                }
                self.conv(&call, rty, vt)
            }
            Param(i) => format!("todo!(\"param {i}\")"),
            BlockParam(_) => n(id),
            Bin { op, lhs, rhs } => self.bin(id, op, lhs, rhs),
            Un { op, v } => match op {
                UnOp::Neg => format!("{}.wrapping_neg()", self.opnd(v, vt)),
                UnOp::Not => format!("!{}", self.opnd(v, vt)),
                UnOp::Bswap => format!("{}.swap_bytes()", self.opnd(v, vt)),
                UnOp::Popcnt => format!("{}.count_ones() as {t}", n(v)),
                UnOp::Ctz => format!("{}.trailing_zeros() as {t}", n(v)),
                UnOp::Clz => format!("{}.leading_zeros() as {t}", n(v)),
            },
            Cmp { cc, lhs, rhs } => match cc {
                Cond::Eq => format!("{} == {}", n(lhs), self.opnd(rhs, self.vt[lhs.index()])),
                Cond::Ne => format!("{} != {}", n(lhs), self.opnd(rhs, self.vt[lhs.index()])),
                Cond::Ult => format!("{} < {}", self.as_u(lhs), self.as_u(rhs)),
                Cond::Ule => format!("{} <= {}", self.as_u(lhs), self.as_u(rhs)),
                Cond::Ugt => format!("{} > {}", self.as_u(lhs), self.as_u(rhs)),
                Cond::Uge => format!("{} >= {}", self.as_u(lhs), self.as_u(rhs)),
                Cond::Slt => format!("{} < {}", self.as_s(lhs), self.as_s(rhs)),
                Cond::Sle => format!("{} <= {}", self.as_s(lhs), self.as_s(rhs)),
                Cond::Sgt => format!("{} > {}", self.as_s(lhs), self.as_s(rhs)),
                Cond::Sge => format!("{} >= {}", self.as_s(lhs), self.as_s(rhs)),
            },
            Cast { kind, v } => match kind {
                CastKind::ZExt => format!("{} as {t}", self.as_u(v)),
                CastKind::SExt => format!("{} as {t}", self.as_s(v)),
                _ => format!("{} as {t}", n(v)),
            },
            Select { c, t: a, f: b } => format!("if {} {{ {} }} else {{ {} }}", n(c), self.opnd(a, vt), self.opnd(b, vt)),
            PtrOffset { base, index, scale, disp } => {
                let mut s = self.as_u(base);
                if let Some(i) = index {
                    let i = self.opnd(i, self.u64_ty);
                    s = if scale == 1 {
                        format!("{s}.wrapping_add({i})")
                    } else {
                        format!("{s}.wrapping_add({i}.wrapping_mul({scale}))")
                    };
                }
                let s = match disp {
                    0 => s,
                    d if d < 0 => format!("{s}.wrapping_sub({:#x})", -(d as i64)),
                    d => format!("{s}.wrapping_add({d:#x})"),
                };
                self.conv(&s, self.u64_ty, vt)
            }
            IntToPtr(v) => match f.insts[v].kind {
                Const(c) => match (self.global_of)(f.consts[c.index()] as u64) {
                    Some(g) => self.conv(&g, self.u64_ty, vt),
                    None => self.conv(&n(v), self.vt[v.index()], vt),
                },
                _ => self.conv(&n(v), self.vt[v.index()], vt),
            },
            PtrToInt(v) => self.conv(&n(v), self.vt[v.index()], vt),
            Load { ptr, .. } => self.load(id, ptr),
            Store { ptr, val, .. } => return Stmt::Effect(self.store(ptr, val)),
            MemCopy { dst, src, len } => {
                let s = format!(
                    "unsafe {{ core::ptr::copy({} as *const u8, {} as *mut u8, {} as usize) }}",
                    self.as_u(src),
                    self.as_u(dst),
                    n(len)
                );
                self.stats.raw += 1;
                return Stmt::Effect(s);
            }
            other => {
                self.stats.todo += 1;
                let s = format!("{other:?}").replace('"', "'").replace('{', "{{").replace('}', "}}");
                if ty == TyId::UNIT {
                    return Stmt::Effect(format!("todo!(\"{s}\")"));
                }
                format!("todo!(\"{s}\")")
            }
        };
        Stmt::Value(e)
    }

    /// Does the call `call` return rax:rdx?
    fn ret2(&self, call: ValueId) -> bool {
        (self.call)(Site::Call(call)).is_some_and(|c| c.ret2)
    }

    /// A call to the callee at `site`, whether it returns a value (`ret`) and two
    /// (`ret2`, as a `(u64, u64)`), and the Rust type of the value it returns.
    fn call_expr(&mut self, site: Site, callee: ValueId, args: ListRef) -> (String, bool, bool, TyId) {
        let args = args.get(&self.f.value_pool);
        // Straight from the lifter, a call lists all six argument registers and more.
        let args = if self.sig.is_none() { &args[..args.len().min(6)] } else { args };
        let info = (self.call)(site);
        let arg = |k: usize, v: ValueId| -> String {
            let want = info.as_ref().and_then(|c| c.args.get(k).copied().flatten());
            match want {
                Some(t) => self.conv(&self.name(v), self.vt[v.index()], t),
                None if self.types.is_none() => format!("{} as u64", self.name(v)),
                None => self.conv(&self.name(v), self.vt[v.index()], self.u64_ty),
            }
        };
        let a: Vec<String> = args.iter().enumerate().map(|(k, &v)| arg(k, v)).collect();
        let a = a.join(", ");
        match info {
            Some(CallInfo { path: Some(p), ret, ret2, foreign, ret_ty, .. }) => {
                if foreign {
                    self.stats.raw += 1;
                }
                let rty = ret_ty.unwrap_or(self.u64_ty);
                if ret2 && foreign {
                    // an extern returns `ffi::Pair`, which is FFI-safe; a tuple isn't
                    return (format!("unsafe {{ let pair_ = {p}({a}); (pair_.0, pair_.1) }}"), ret, ret2, rty);
                }
                (format!("unsafe {{ {p}({a}) }}"), ret, ret2, rty)
            }
            _ => {
                self.stats.raw += 1;
                let tys = vec!["u64"; args.len()].join(", ");
                let callee = self.as_u(callee);
                (
                    format!("unsafe {{ core::mem::transmute::<u64, unsafe extern \"C\" fn({tys}) -> u64>({callee})({a}) }}"),
                    true,
                    false,
                    self.u64_ty,
                )
            }
        }
    }

    fn bin(&self, id: ValueId, op: BinOp, lhs: ValueId, rhs: ValueId) -> String {
        let ty = self.ty(lhs);
        if ty == TyId::BOOL {
            let o = match op {
                BinOp::And => "&",
                BinOp::Or => "|",
                _ => "^",
            };
            return format!("{} {o} {}", self.name(lhs), self.name(rhs));
        }
        let vt = self.vt[id.index()];
        let ut = self.ut(lhs);
        let b = self.name(rhs);
        // in the result's own type
        let same = |o: &str| format!("{}.{o}({})", self.opnd(lhs, vt), self.opnd(rhs, vt));
        let res_signed = matches!(self.table.tys[vt], Ty::Int { signed: true, .. });
        // computed unsigned or signed, then as the result's type
        let wrap = |e: String, signed_domain: bool| if signed_domain == res_signed { e } else { format!("({e}) as {}", self.rt(id)) };
        match op {
            BinOp::Add => same("wrapping_add"),
            BinOp::Sub => same("wrapping_sub"),
            BinOp::Mul => same("wrapping_mul"),
            BinOp::And => format!("{} & {}", self.opnd(lhs, vt), self.opnd(rhs, vt)),
            BinOp::Or => format!("{} | {}", self.opnd(lhs, vt), self.opnd(rhs, vt)),
            BinOp::Xor => format!("{} ^ {}", self.opnd(lhs, vt), self.opnd(rhs, vt)),
            BinOp::UDiv => wrap(format!("{} / {}", self.as_u(lhs), self.as_u(rhs)), false),
            BinOp::URem => wrap(format!("{} % {}", self.as_u(lhs), self.as_u(rhs)), false),
            BinOp::SDiv if res_signed => format!("{}.wrapping_div({})", self.as_s(lhs), self.as_s(rhs)),
            BinOp::SRem if res_signed => format!("{}.wrapping_rem({})", self.as_s(lhs), self.as_s(rhs)),
            BinOp::SDiv => format!("{}.wrapping_div({}) as {ut}", self.as_s(lhs), self.as_s(rhs)),
            BinOp::SRem => format!("{}.wrapping_rem({}) as {ut}", self.as_s(lhs), self.as_s(rhs)),
            BinOp::Shl => format!("{}.wrapping_shl({b} as u32)", self.opnd(lhs, vt)),
            BinOp::LShr => wrap(format!("{}.wrapping_shr({b} as u32)", self.as_u(lhs)), false),
            BinOp::AShr if res_signed => format!("{}.wrapping_shr({b} as u32)", self.as_s(lhs)),
            BinOp::AShr => format!("{}.wrapping_shr({b} as u32) as {ut}", self.as_s(lhs)),
            BinOp::RotL => format!("{}.rotate_left({b} as u32)", self.opnd(lhs, vt)),
            BinOp::RotR => format!("{}.rotate_right({b} as u32)", self.opnd(lhs, vt)),
        }
    }

    /// The safe-mode slice an access through `ptr` can use: the argument's name,
    /// if `ptr` derives from exactly one slice argument.
    fn slice_root(&self, ptr: ValueId, write: bool) -> Option<(String, String, ArgKind)> {
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
                let n = self.name(p);
                let b = self.bind[k].clone().unwrap_or_else(|| format!("{n}_ref"));
                Some((n, b, kind))
            }
            _ => None,
        }
    }

    /// A field access for a load or store of `bytes` through `ptr`: the place
    /// (`p.count`, `(*p).count`), whether it is a raw dereference, and the
    /// field's type.
    fn field(&self, ptr: ValueId, bytes: usize, write: bool) -> Option<(String, bool, TyId)> {
        let types = self.types?;
        let (r, d) = base_of(self.f, &self.alias, ptr);
        let s = types.pointee[r.index()].or_else(|| types.pointee[ptr.index()].filter(|_| d == 0))?;
        if !matches!(self.table.tys[s], Ty::Struct(_)) {
            return None;
        }
        let (path, lt) = leaf(self.table, s, d, bytes as u32)?;
        let sn = self.rs(s);
        if let Some(k) = self.entry_param[r.index()] {
            let b = self.bind[k].clone();
            match (self.args[k], b) {
                (ArgKind::Struct { mutbl, nullable, ty }, Some(b)) if ty == s && (mutbl || !write) => {
                    let base = match (nullable, mutbl, write) {
                        (false, _, _) => b,
                        (true, false, _) => format!("{b}.unwrap()"),
                        (true, true, false) => format!("{b}.as_deref().unwrap()"),
                        (true, true, true) => format!("{b}.as_deref_mut().unwrap()"),
                    };
                    return Some((format!("{base}{path}"), false, lt));
                }
                (ArgKind::Int, Some(b)) => {
                    let arg = types.args.iter().find(|a| Some(a.reg) == self.reg_of(r)).and_then(|a| a.ty);
                    if let Some(Ty::RawPtr { pointee, mutbl }) = arg.map(|t| self.table.tys[t]) {
                        if pointee == s && (mutbl == Mutbl::Mut || !write) {
                            return Some((format!("(*{b}){path}"), true, lt));
                        }
                    }
                }
                _ => {}
            }
        }
        let m = if write { "mut" } else { "const" };
        Some((format!("(*({} as *{m} {sn})){path}", self.as_u(r)), true, lt))
    }

    fn reg_of(&self, v: ValueId) -> Option<u8> {
        match self.f.insts[v].kind {
            InstKind::BlockParam(r) => Some(r),
            _ => None,
        }
    }

    fn load(&mut self, id: ValueId, ptr: ValueId) -> String {
        let ty = self.ty(id);
        let (t, len) = (self.rt(id), bytes(ty));
        let slice = self.slice_root(ptr, false);
        if let Some((place, raw, lt)) = self.field(ptr, len, false).filter(|(_, raw, _)| !raw || slice.is_none()) {
            let e = self.conv(&place, lt, self.vt[id.index()]);
            if raw {
                self.stats.raw += 1;
                return format!("unsafe {{ {e} }}");
            }
            self.stats.checked += 1;
            return e;
        }
        let p = self.as_u(ptr);
        if let Some((root, b, ArgKind::Slice { nullable, mutbl })) = slice {
            self.stats.checked += 1;
            let s = match (nullable, mutbl) {
                (false, _) => b,
                (true, false) => format!("{b}.unwrap()"),
                (true, true) => format!("{b}.as_deref().unwrap()"),
            };
            return format!("{t}::from_le_bytes({s}[{p}.wrapping_sub({root}_base) as usize..][..{len}].try_into().unwrap())");
        }
        self.stats.raw += 1;
        format!("unsafe {{ ({p} as *const {t}).read_unaligned() }}")
    }

    fn store(&mut self, ptr: ValueId, val: ValueId) -> String {
        let vt = self.ty(val);
        let (t, len, v) = (self.rt(val), bytes(vt), self.name(val));
        let slice = self.slice_root(ptr, true);
        if let Some((place, raw, lt)) = self.field(ptr, len, true).filter(|(_, raw, _)| !raw || slice.is_none()) {
            let e = match self.table.tys[lt] {
                // the pointee may not be printed (`render_structs`); let Rust infer it
                Ty::RawPtr { .. } => format!("{} as _", self.opnd(val, self.u64_ty)),
                _ => self.conv(&v, self.vt[val.index()], lt),
            };
            if raw {
                self.stats.raw += 1;
                return format!("unsafe {{ {place} = {e} }}");
            }
            self.stats.checked += 1;
            return format!("{place} = {e}");
        }
        let p = self.as_u(ptr);
        if let Some((root, b, ArgKind::Slice { nullable, .. })) = slice {
            self.stats.checked += 1;
            let s = if nullable { format!("{b}.as_deref_mut().unwrap()") } else { b };
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
    /// A call returning rax:rdx: `let vN_pair = expr; let vN = vN_pair.0;`
    Pair(String),
}

impl Source for Emitter<'_> {
    fn stmts(&mut self, b: BlockId) -> Vec<Node> {
        self.stmt_lines(b).into_iter().map(Node::Line).collect()
    }

    fn edge(&mut self, to: BlockId, args: &[ValueId]) -> Vec<Node> {
        self.assign(to, args).map(Node::Line).into_iter().collect()
    }

    fn cond(&self, c: ValueId) -> String {
        self.name(c)
    }

    fn exit(&mut self, b: BlockId) -> Node {
        Node::Exit(self.exit_line(b))
    }
}
