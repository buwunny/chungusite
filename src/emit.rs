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
use crate::types::{reg_name, FnTypes, IntTy, Leaf, Path, PointeeTy};
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
    /// Functions whose control flow is irreducible, emitted as a
    /// `loop { match bb { .. } }` state machine instead of structured code.
    pub state_machines: usize,
}

const REG: [&str; 16] = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
];


/// How an argument arrives.
#[derive(Copy, Clone, PartialEq, Eq)]
enum ArgKind {
    /// An integer: `u64`, or narrower when `types` says so.
    Int,
    /// Safe mode: a byte slice.
    Slice { mutbl: bool, nullable: bool },
    /// Safe mode: `&S`, `&T` or `&[T]` as `types` describes the pointee.
    Typed { mutbl: bool, nullable: bool },
    /// A raw pointer to a struct `types` recovered (`*mut S`).
    RawStruct,
}

/// What a call site calls, as `program.rs` resolved it.
#[derive(Clone, Debug, Default)]
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
    /// The Rust type of each argument, in order, when it isn't `u64` (`types.rs`).
    pub params: Vec<String>,
    /// The return type, when the callee returns a narrower integer than `u64`.
    pub ret_ty: Option<IntTy>,
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
    /// Recovered types (`types.rs`); `None` keeps every value an unsigned integer
    /// and every pointer `u64` or a byte slice.
    pub types: Option<&'a FnTypes>,
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
    types: Option<&'a FnTypes>,
    /// Entry parameters something reads as `u64` (not only through a narrowed alias).
    view_used: Vec<bool>,
    stats: EmitStats,
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

    let borrow = (mode == Mode::Safe).then(|| analyze(f));
    let types = env.types.filter(|t| t.params.len() == entry_params.len());
    let args = entry_params
        .iter()
        .enumerate()
        .map(|(k, _)| {
            let pointee = types.and_then(|t| t.params[k].pointee.as_ref());
            let kind = match &borrow {
                Some(a) if !env.demote => arg_kind(&a.params[k]),
                _ => ArgKind::Int,
            };
            match (kind, pointee) {
                (ArgKind::Slice { mutbl, nullable }, Some(p)) if p.complete => ArgKind::Typed { mutbl, nullable },
                (ArgKind::Int, Some(p)) if matches!(p.ty, PointeeTy::Struct(_)) => ArgKind::RawStruct,
                (k, _) => k,
            }
        })
        .collect();
    let mut skip = callee_only(f, &cfg, env.call);
    if let Some(t) = types {
        for (v, s) in skip.iter_mut().enumerate() {
            *s |= t.alias(ValueId::new(v)).is_some();
        }
    }

    let mut e = Emitter {
        f,
        sig: env.sig,
        call: env.call,
        used: used(f, &cfg),
        skip,
        structure: env.structure,
        borrow,
        args,
        entry_param,
        hoisted: hoisted(f, &cfg),
        global_of: env.global_of,
        types,
        view_used: Vec::new(),
        stats: EmitStats::default(),
    };
    if types.is_some() {
        e.elide(&cfg);
    }
    e.view_used = view_used(f, &cfg, &e.skip);

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

/// Entry parameters used by an instruction that is emitted, or a terminator.
fn view_used(f: &Function, cfg: &Cfg, skip: &[bool]) -> Vec<bool> {
    let params = f.blocks[f.entry].params.get(&f.value_pool);
    let mut u = vec![false; f.insts.len()];
    for &b in &cfg.rpo {
        let blk = &f.blocks[b];
        for &id in blk.insts.get(&f.value_pool) {
            if !skip[id.index()] {
                for_each_operand(f.insts[id].kind, f, |v| u[v.index()] = true);
            }
        }
        term_uses(f, blk.term, |v| u[v.index()] = true);
    }
    params.iter().map(|p| u[p.index()]).collect()
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

fn rty(ty: TyId) -> &'static str {
    match ty {
        TyId::B1 => "u8",
        TyId::B2 => "u16",
        TyId::B4 => "u32",
        TyId::BOOL => "bool",
        TyId::UNIT => "()",
        TyId::PAIR => "(u64, u64)",
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
        if let Some(k) = self.types.and_then(|t| t.alias(v)) {
            return self.sig_name(k);
        }
        match (self.entry_param[v.index()], self.f.insts[v].kind) {
            (Some(k), _) if self.args[k] == ArgKind::Int && self.param_int(k) == IntTy::U64 && self.debug_name(k).is_some() => {
                self.debug_name(k).unwrap().to_string()
            }
            (Some(_), InstKind::BlockParam(r)) if r >= STACK_ARG_BASE => format!("arg{}", 6 + (r - STACK_ARG_BASE) as usize),
            (Some(_), InstKind::BlockParam(r)) => REG[r as usize & 15].to_string(),
            _ => format!("v{}", v.index()),
        }
    }

    fn ty(&self, v: ValueId) -> TyId {
        self.f.insts[v].ty
    }

    /// The Rust type of value `v`: its integer type from `types`, else unsigned.
    fn vty(&self, v: ValueId) -> &'static str {
        match self.types.and_then(|t| t.int(self.f, v)) {
            Some(t) => t.rust(),
            None => rty(self.ty(v)),
        }
    }

    fn int_of(&self, v: ValueId) -> IntTy {
        self.types.and_then(|t| t.int(self.f, v)).unwrap_or(IntTy { bytes: bytes(self.ty(v)) as u8, signed: false })
    }

    /// `v` as a `u64`, zero-extended like a 32-bit register write.
    fn wide(&self, v: ValueId) -> String {
        let n = self.name(v);
        let t = self.int_of(v);
        if t.signed && t.bytes < 8 && self.ty(v) != TyId::BOOL {
            format!("{n} as {} as u64", t.unsigned())
        } else {
            format!("{n} as u64")
        }
    }

    /// `v` where a `u64` is needed (addresses): its name if it is one.
    fn as_u64(&self, v: ValueId) -> String {
        match self.vty(v) {
            "u64" => self.name(v),
            _ => format!("({})", self.wide(v)),
        }
    }

    /// `v` converted to the Rust type `t` if it isn't one already.
    fn conv(&self, v: ValueId, t: &str) -> String {
        if self.vty(v) == t {
            self.name(v)
        } else {
            format!("({} as {t})", self.name(v))
        }
    }

    /// Argument name from debug info or a model.
    fn debug_name(&self, k: usize) -> Option<&str> {
        self.types?.params[k].name.as_deref()
    }

    /// The integer type entry parameter `k` is taken as (u64 unless narrowed).
    fn param_int(&self, k: usize) -> IntTy {
        self.types.map_or(IntTy::U64, |t| t.params[k].int)
    }

    /// Name of entry parameter `k` in the signature, when it isn't the `u64`
    /// variable the body uses: a narrowed integer (`edi`, or its debug name).
    fn sig_name(&self, k: usize) -> String {
        if let Some(n) = self.debug_name(k) {
            return n.to_string();
        }
        let p = self.f.blocks[self.f.entry].params.get(&self.f.value_pool)[k];
        match self.f.insts[p].kind {
            InstKind::BlockParam(r) => reg_name(r, self.param_int(k).bytes),
            _ => self.name(p),
        }
    }

    /// Name of a pointer parameter in the signature (`rdi_ref`, or its debug name).
    fn ref_name(&self, k: usize) -> String {
        match self.debug_name(k) {
            Some(n) => n.to_string(),
            None => format!("{}_ref", self.name(self.f.blocks[self.f.entry].params.get(&self.f.value_pool)[k])),
        }
    }

    /// The Rust type a typed pointer parameter points to: `S`, `T` or `[T]`.
    fn pointee_ty(&self, k: usize) -> String {
        let p = self.types.and_then(|t| t.params[k].pointee.as_ref()).expect("typed parameter");
        match &p.ty {
            PointeeTy::Slice(_) => format!("[{}]", p.ident),
            _ => p.ident.clone(),
        }
    }

    /// Skip values that typed code no longer needs: the address computations of
    /// field and element accesses (`r.count` instead of `*(rdi + 8)`), and the
    /// zero-extension of a narrowed return value.
    fn elide(&mut self, cfg: &Cfg) {
        let f = self.f;
        let mut uses = vec![0u32; f.insts.len()];
        for &b in &cfg.rpo {
            let blk = &f.blocks[b];
            for &id in blk.insts.get(&f.value_pool) {
                if !self.skip[id.index()] {
                    for_each_operand(f.insts[id].kind, f, |v| uses[v.index()] += 1);
                }
            }
            term_uses(f, blk.term, |v| uses[v.index()] += 1);
        }
        let mut work = Vec::new();
        for &b in &cfg.rpo {
            let blk = &f.blocks[b];
            for &id in blk.insts.get(&f.value_pool) {
                let (ptr, write) = match f.insts[id].kind {
                    InstKind::Load { ptr, .. } => (ptr, false),
                    InstKind::Store { ptr, .. } => (ptr, true),
                    _ => continue,
                };
                let computed = matches!(self.types.and_then(|t| t.path(id)), Some((_, Path::Index(None))));
                if !self.skip[id.index()] && !computed && self.place(id, ptr, write).is_some() {
                    work.push(ptr);
                }
            }
            if let Terminator::Return(Some(v)) = blk.term {
                if let Some(x) = self.ret_src(b, v) {
                    uses[x.index()] += 1;
                    work.push(v);
                }
            }
        }
        while let Some(v) = work.pop() {
            uses[v.index()] -= 1;
            let k = f.insts[v].kind;
            let pure = matches!(k, InstKind::PtrOffset { .. } | InstKind::Cast { .. } | InstKind::Const(_) | InstKind::Bin { .. });
            if uses[v.index()] == 0 && pure && !self.skip[v.index()] {
                self.skip[v.index()] = true;
                for_each_operand(k, f, |o| work.push(o));
            }
        }
    }

    /// `x` when a narrowed function returns `v = ZExt(x)` from block `b` and `x`
    /// has the return type's width: it returns `x` itself, if `x` is in scope there.
    fn ret_src(&self, b: BlockId, v: ValueId) -> Option<ValueId> {
        let t = self.ret_ty()?;
        let f = self.f;
        match f.insts[v].kind {
            InstKind::Cast { kind: CastKind::ZExt, v: x } if bytes(self.ty(x)) == t.bytes as usize && self.ty(x) != TyId::BOOL => {
                let here = f.blocks[b].insts.get(&f.value_pool).contains(&x);
                (here || self.hoisted[x.index()] || self.entry_param[x.index()].is_some()).then_some(x)
            }
            _ => None,
        }
    }

    /// The return type, when it is narrower than `u64`.
    fn ret_ty(&self) -> Option<IntTy> {
        self.types.and_then(|t| t.ret).filter(|_| self.sig.is_some_and(|s| s.ret && !s.ret2))
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
    /// function takes but never uses.
    fn arg_order(&self) -> Vec<(Option<usize>, String)> {
        let params = self.f.blocks[self.f.entry].params.get(&self.f.value_pool);
        let Some(sig) = self.sig else {
            return self.sig_order().into_iter().map(|k| (Some(k), String::new())).collect();
        };
        let find = |reg: u8| params.iter().position(|&p| matches!(self.f.insts[p].kind, InstKind::BlockParam(r) if r == reg));
        let mut order = Vec::new();
        for (k, &reg) in SYSV_ARGS[..sig.args as usize].iter().enumerate() {
            order.push((find(reg), format!("_{}", REG[SYSV_ARGS[k] as usize])));
        }
        for j in 0..sig.stack_args {
            order.push((find(STACK_ARG_BASE + j), format!("_arg{}", 6 + j as usize)));
        }
        // anything else on entry (there shouldn't be anything after `abi::apply`)
        for k in 0..params.len() {
            if !order.iter().any(|&(x, _)| x == Some(k)) {
                order.push((Some(k), String::new()));
            }
        }
        order
    }

    fn signature(&self, name: &str, out: &mut String) {
        let params = self.f.blocks[self.f.entry].params.get(&self.f.value_pool);
        let unsafety = if self.stats.raw > 0 { "unsafe " } else { "" };
        let mut sig = Vec::new();
        let mut prologue = String::new();
        for (k, unused) in self.arg_order() {
            let Some(k) = k else {
                sig.push(format!("{unused}: u64"));
                continue;
            };
            let n = self.name(params[k]);
            match self.args[k] {
                ArgKind::Int if self.param_int(k) == IntTy::U64 => sig.push(format!("mut {n}: u64")),
                ArgKind::Int => {
                    let (s, t) = (self.sig_name(k), self.param_int(k));
                    sig.push(format!("{s}: {}", t.rust()));
                    if self.view_used[k] {
                        let _ = writeln!(prologue, "    let mut {n}: u64 = {s} as {} as u64;", t.unsigned());
                    }
                }
                ArgKind::RawStruct => {
                    let r = self.ref_name(k);
                    sig.push(format!("{r}: *mut {}", self.pointee_ty(k)));
                    if self.view_used[k] {
                        let _ = writeln!(prologue, "    let mut {n}: u64 = {r} as u64;");
                    }
                }
                ArgKind::Slice { mutbl, nullable } | ArgKind::Typed { mutbl, nullable } => {
                    let typed = matches!(self.args[k], ArgKind::Typed { .. });
                    let pt = if typed { self.pointee_ty(k) } else { "[u8]".to_string() };
                    let r = if mutbl { format!("&mut {pt}") } else { format!("&{pt}") };
                    let rn = self.ref_name(k);
                    let (m, t) = if nullable { (if mutbl { "mut " } else { "" }, format!("Option<{r}>")) } else { ("", r.to_string()) };
                    sig.push(format!("{m}{rn}: {t}"));
                    let ptr = |s: &str| if pt.starts_with('[') { format!("{s}.as_ptr() as u64") } else { format!("{s} as *const {pt} as u64") };
                    let addr = if nullable { format!("{rn}.as_deref().map_or(0, |s| {})", ptr("s")) } else { ptr(&rn) };
                    let computed = (0..self.f.insts.len()).any(|v| {
                        matches!(self.types.and_then(|t| t.path(ValueId::new(v))), Some((j, Path::Index(None))) if j == k)
                    });
                    if !typed || computed || self.view_used[k] {
                        let _ = writeln!(prologue, "    let {n}_base: u64 = {addr};");
                    }
                    if !typed || self.view_used[k] {
                        let _ = writeln!(prologue, "    let mut {n}: u64 = {n}_base;");
                    }
                }
            }
        }
        let narrow = self.ret_ty().map(|t| format!(" -> {}", t.rust()));
        let ret = match self.sig {
            _ if narrow.is_some() => narrow.as_deref().unwrap(),
            Some(s) if s.ret2 => " -> (u64, u64)",
            Some(s) if !s.ret => "",
            _ => " -> u64",
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
                    let _ = writeln!(out, "    let mut {}: {} = {zero};", self.name(v), self.vty(v));
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
                Stmt::Value(e) => out.push(format!("let {n}: {} = {e}; // {at:#x}", self.vty(id))),
                Stmt::Effect(e) => out.push(format!("{e}; // {at:#x}")),
                Stmt::Pair(e) => {
                    // rax:rdx; the rdx half is a `CallOut` reading `vN_pair.1`
                    out.push(format!("let {n}_pair: (u64, u64) = {e}; // {at:#x}"));
                    if self.hoisted[id.index()] {
                        out.push(format!("{n} = {n}_pair.0;"));
                    } else {
                        out.push(format!("let {n}: u64 = {n}_pair.0;"));
                    }
                }
            }
        }
        out
    }

    /// A terminator that leaves the function (or can't be expressed yet).
    fn exit_line(&mut self, b: BlockId) -> String {
        let t = self.f.blocks[b].term;
        match t {
            Terminator::Return(Some(v)) if self.ty(v) == TyId::PAIR => format!("return {};", self.name(v)),
            Terminator::Return(Some(v)) => match self.ret_ty() {
                Some(t) => match self.ret_src(b, v) {
                    Some(x) => format!("return {};", self.conv(x, t.rust())),
                    None => format!("return {} as {};", self.name(v), t.rust()),
                },
                None => format!("return {};", self.wide(v)),
            },
            Terminator::Return(None) if self.sig.is_some_and(|s| !s.ret) => "return;".to_string(),
            Terminator::Return(None) => "return 0;".to_string(),
            Terminator::TailCall { callee, args } => {
                let (call, ret, ret2) = self.call_expr(Site::Tail(b), callee, args);
                let me2 = self.sig.is_some_and(|s| s.ret2);
                match (self.sig.is_none_or(|s| s.ret), ret) {
                    (true, true) if ret2 && !me2 => format!("return {call}.0;"),
                    (true, true) => format!("return {call};"),
                    (true, false) => format!("{call}; return 0;"),
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
            .map(|(&p, &a)| (self.name(p), self.conv(a, self.vty(p))))
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
        let t = self.vty(id);
        let n = |v: ValueId| self.name(v);
        let e = match f.insts[id].kind {
            Const(c) => {
                let v = f.consts[c.index()];
                let mask = if bytes(ty) >= 8 { u64::MAX as u128 } else { (1u128 << (bytes(ty) * 8)) - 1 };
                match ty {
                    TyId::BOOL => (v != 0).to_string(),
                    _ if self.int_of(id).signed => {
                        let bits = bytes(ty) as u32 * 8;
                        let x = (v & mask) as i128;
                        let x = if x >> (bits - 1) & 1 == 1 { x - (1i128 << bits) } else { x };
                        format!("{x}_{t}")
                    }
                    _ => format!("{:#x}_{t}", v & mask),
                }
            }
            Undef if ty == TyId::BOOL => "false".to_string(),
            CallOut { call, reg: crate::abi::RDX } if matches!(f.insts[call].kind, Call { .. }) && self.ret2(call) => {
                format!("{}_pair.1", n(call))
            }
            Undef | CallOut { .. } => format!("0_{t}"),
            Aggregate { ty: TyId::PAIR, fields } => {
                let v = fields.get(&f.value_pool);
                format!("({}, {})", self.wide(v[0]), self.wide(v[1]))
            }
            AddrOfLocal(_) => "frame.as_mut_ptr() as u64".to_string(),
            Call { callee, args } => {
                let (call, ret, ret2) = self.call_expr(Site::Call(id), callee, args);
                if ret2 {
                    return Stmt::Pair(call);
                }
                if !ret || !self.used[id.index()] {
                    return Stmt::Effect(call);
                }
                call
            }
            Param(i) => format!("todo!(\"param {i}\")"),
            BlockParam(_) => n(id),
            Bin { op, lhs, rhs } => self.fit(self.bin(op, lhs, rhs), lhs, id),
            Un { op, v } => match op {
                UnOp::Neg => self.fit(format!("{}.wrapping_neg()", n(v)), v, id),
                UnOp::Not => self.fit(format!("!{}", n(v)), v, id),
                UnOp::Bswap => self.fit(format!("{}.swap_bytes()", n(v)), v, id),
                UnOp::Popcnt => format!("{}.count_ones() as {t}", n(v)),
                UnOp::Ctz => format!("{}.trailing_zeros() as {t}", n(v)),
                UnOp::Clz => format!("{}.leading_zeros() as {t}", n(v)),
            },
            Cmp { cc, lhs, rhs } => {
                let lt = self.vty(lhs);
                let (a, b) = (n(lhs), self.conv(rhs, lt));
                let it = self.int_of(lhs);
                let (s, u) = (signed(self.ty(lhs)), it.unsigned());
                let (sa, sb) = if it.signed { (a.clone(), b.clone()) } else { (format!("({a} as {s})"), format!("({b} as {s})")) };
                let (ua, ub) = if it.signed { (format!("({a} as {u})"), format!("({b} as {u})")) } else { (a.clone(), b.clone()) };
                match cc {
                    Cond::Eq => format!("{a} == {b}"),
                    Cond::Ne => format!("{a} != {b}"),
                    Cond::Ult => format!("{ua} < {ub}"),
                    Cond::Ule => format!("{ua} <= {ub}"),
                    Cond::Ugt => format!("{ua} > {ub}"),
                    Cond::Uge => format!("{ua} >= {ub}"),
                    Cond::Slt => format!("{sa} < {sb}"),
                    Cond::Sle => format!("{sa} <= {sb}"),
                    Cond::Sgt => format!("{sa} > {sb}"),
                    Cond::Sge => format!("{sa} >= {sb}"),
                }
            }
            Cast { kind, v } => {
                let src = self.int_of(v);
                let bool_src = self.ty(v) == TyId::BOOL;
                match kind {
                    CastKind::SExt if !src.signed || bool_src => format!("{} as {} as {t}", n(v), signed(self.ty(v))),
                    CastKind::ZExt if src.signed && !bool_src => format!("{} as {} as {t}", n(v), src.unsigned()),
                    _ => format!("{} as {t}", n(v)),
                }
            }
            Select { c, t: a, f: b } => {
                let e = format!("if {} {{ {} }} else {{ {} }}", n(c), n(a), self.conv(b, self.vty(a)));
                self.fit(e, a, id)
            }
            PtrOffset { base, index, scale, disp } => {
                let mut s = self.as_u64(base);
                if let Some(i) = index {
                    s = if scale == 1 {
                        format!("{s}.wrapping_add({})", self.as_u64(i))
                    } else {
                        format!("{s}.wrapping_add({}.wrapping_mul({scale}))", self.as_u64(i))
                    };
                }
                match disp {
                    0 => s,
                    d if d < 0 => format!("{s}.wrapping_sub({:#x})", -(d as i64)),
                    d => format!("{s}.wrapping_add({d:#x})"),
                }
            }
            IntToPtr(v) => match f.insts[v].kind {
                Const(c) => (self.global_of)(f.consts[c.index()] as u64).unwrap_or_else(|| self.as_u64(v)),
                _ => self.as_u64(v),
            },
            PtrToInt(v) => self.as_u64(v),
            Load { ptr, .. } => self.load(id, ptr, ty),
            Store { ptr, val, .. } => return Stmt::Effect(self.store(id, ptr, val)),
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

    /// A call to the callee at `site`, and whether it returns a value (`ret`), and
    /// two (`ret2`, as a `(u64, u64)`).
    fn call_expr(&mut self, site: Site, callee: ValueId, args: ListRef) -> (String, bool, bool) {
        let args = args.get(&self.f.value_pool);
        // Straight from the lifter, a call lists all six argument registers and more.
        let args = if self.sig.is_none() { &args[..args.len().min(6)] } else { args };
        let info = (self.call)(site);
        let a: Vec<String> = args
            .iter()
            .enumerate()
            .map(|(j, &v)| match info.as_ref().and_then(|c| c.params.get(j)) {
                Some(t) if t != "u64" => format!("{} as {t}", self.name(v)),
                _ => self.wide(v),
            })
            .collect();
        let a = a.join(", ");
        match info {
            Some(CallInfo { path: Some(p), ret, ret2, foreign, ret_ty, .. }) => {
                if foreign {
                    self.stats.raw += 1;
                }
                if ret2 && foreign {
                    // an extern returns `ffi::Pair`, which is FFI-safe; a tuple isn't
                    return (format!("unsafe {{ let p = {p}({a}); (p.0, p.1) }}"), ret, ret2);
                }
                // a narrower return value is zero-extended back, like rax after `mov eax`
                let widen = match ret_ty {
                    Some(t) if ret && !ret2 => format!(" as {} as u64", t.unsigned()),
                    _ => String::new(),
                };
                (format!("unsafe {{ {p}({a}){widen} }}"), ret, ret2)
            }
            _ => {
                self.stats.raw += 1;
                let tys = vec!["u64"; args.len()].join(", ");
                let callee = self.name(callee);
                (format!("unsafe {{ core::mem::transmute::<u64, unsafe extern \"C\" fn({tys}) -> u64>({callee})({a}) }}"), true, false)
            }
        }
    }

    /// `e`, whose type is `from`'s, as `to`'s type.
    fn fit(&self, e: String, from: ValueId, to: ValueId) -> String {
        let t = self.vty(to);
        if self.vty(from) == t || self.ty(to) == TyId::BOOL {
            e
        } else {
            format!("({e}) as {t}")
        }
    }

    fn bin(&self, op: BinOp, lhs: ValueId, rhs: ValueId) -> String {
        let ty = self.ty(lhs);
        let it = self.int_of(lhs);
        let (t, s, u) = (self.vty(lhs), signed(ty), it.unsigned());
        let shift = matches!(op, BinOp::Shl | BinOp::LShr | BinOp::AShr | BinOp::RotL | BinOp::RotR);
        let (a, b) = (self.name(lhs), if shift { self.name(rhs) } else { self.conv(rhs, t) });
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
            BinOp::UDiv if it.signed => format!("(({a} as {u}) / ({b} as {u})) as {t}"),
            BinOp::URem if it.signed => format!("(({a} as {u}) % ({b} as {u})) as {t}"),
            BinOp::UDiv => format!("{a} / {b}"),
            BinOp::URem => format!("{a} % {b}"),
            BinOp::SDiv if it.signed => format!("{a}.wrapping_div({b})"),
            BinOp::SRem if it.signed => format!("{a}.wrapping_rem({b})"),
            BinOp::SDiv => format!("({a} as {s}).wrapping_div({b} as {s}) as {t}"),
            BinOp::SRem => format!("({a} as {s}).wrapping_rem({b} as {s}) as {t}"),
            BinOp::And => format!("{a} & {b}"),
            BinOp::Or => format!("{a} | {b}"),
            BinOp::Xor => format!("{a} ^ {b}"),
            BinOp::Shl => format!("{a}.wrapping_shl({b} as u32)"),
            BinOp::LShr if it.signed => format!("({a} as {u}).wrapping_shr({b} as u32) as {t}"),
            BinOp::LShr => format!("{a}.wrapping_shr({b} as u32)"),
            BinOp::AShr if it.signed => format!("{a}.wrapping_shr({b} as u32)"),
            BinOp::AShr => format!("({a} as {s}).wrapping_shr({b} as u32) as {t}"),
            BinOp::RotL => format!("{a}.rotate_left({b} as u32)"),
            BinOp::RotR => format!("{a}.rotate_right({b} as u32)"),
        }
    }

    /// The safe-mode slice an access through `ptr` can use: the argument's `u64`
    /// name and its slice's name, if `ptr` derives from exactly one slice argument.
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
                Some((self.name(p), self.ref_name(k), kind))
            }
            _ => None,
        }
    }

    /// The place the load or store `id` names through a typed argument, e.g.
    /// `p.count`, `(*p).count`, `*p.unwrap()` or `p[3]`, its scalar type, and
    /// whether it is a raw pointer access.
    fn place(&self, id: ValueId, ptr: ValueId, write: bool) -> Option<(String, Leaf, bool)> {
        let t = self.types?;
        let (k, path) = t.path(id)?;
        let pt = t.params[k].pointee.as_ref()?;
        let r = self.ref_name(k);
        let (base, raw) = match self.args[k] {
            ArgKind::Typed { mutbl, nullable } => {
                if write && !mutbl {
                    return None;
                }
                let b = match (nullable, mutbl, write) {
                    (false, _, _) => r,
                    (true, false, _) => format!("{r}.unwrap()"),
                    (true, true, false) => format!("{r}.as_deref().unwrap()"),
                    (true, true, true) => format!("{r}.as_deref_mut().unwrap()"),
                };
                (b, false)
            }
            ArgKind::RawStruct => (format!("(*{r})"), true),
            _ => return None,
        };
        let view = self.name(self.f.blocks[self.f.entry].params.get(&self.f.value_pool)[k]);
        Some(match (path, &pt.ty) {
            (Path::Field(p, leaf), PointeeTy::Struct(_)) => (format!("{base}{p}"), *leaf, raw),
            (Path::Deref, PointeeTy::Scalar(e)) if !raw => (format!("*{base}"), Leaf::Int(*e), raw),
            (Path::Index(Some(i)), PointeeTy::Slice(e)) if !raw => (format!("{base}[{i}]"), Leaf::Int(*e), raw),
            (Path::Index(None), PointeeTy::Slice(e)) if !raw => {
                let i = format!("({}.wrapping_sub({view}_base) / {}) as usize", self.as_u64(ptr), e.bytes);
                (format!("{base}[{i}]"), Leaf::Int(*e), raw)
            }
            _ => return None,
        })
    }

    fn load(&mut self, id: ValueId, ptr: ValueId, ty: TyId) -> String {
        if let Some((place, leaf, raw)) = self.place(id, ptr, false) {
            let t = self.vty(id);
            let e = match leaf {
                Leaf::Int(l) if l.rust() == t => place,
                Leaf::Int(_) => format!("{place} as {t}"),
                Leaf::Float(b) if IntTy::new(b, false).rust() == t => format!("{place}.to_bits()"),
                Leaf::Float(_) => format!("{place}.to_bits() as {t}"),
            };
            return self.counted(e, raw);
        }
        let (t, len, p) = (self.vty(id), bytes(ty), self.name(ptr));
        if let Some((root, r, ArgKind::Slice { nullable, mutbl })) = self.slice_root(ptr, false) {
            self.stats.checked += 1;
            let s = match (nullable, mutbl) {
                (false, _) => r,
                (true, false) => format!("{r}.unwrap()"),
                (true, true) => format!("{r}.as_deref().unwrap()"),
            };
            return format!("{t}::from_le_bytes({s}[{}.wrapping_sub({root}_base) as usize..][..{len}].try_into().unwrap())", self.as_u64(ptr));
        }
        self.stats.raw += 1;
        format!("unsafe {{ ({p} as *const {t}).read_unaligned() }}")
    }

    /// A typed access, counted as checked or raw (and wrapped in `unsafe`).
    fn counted(&mut self, e: String, raw: bool) -> String {
        if raw {
            self.stats.raw += 1;
            format!("unsafe {{ {e} }}")
        } else {
            self.stats.checked += 1;
            e
        }
    }

    fn store(&mut self, id: ValueId, ptr: ValueId, val: ValueId) -> String {
        if let Some((place, leaf, raw)) = self.place(id, ptr, true) {
            let v = match leaf {
                Leaf::Int(l) => self.conv(val, l.rust()),
                Leaf::Float(b) => format!("{}::from_bits({})", leaf.rust(), self.conv(val, IntTy::new(b, false).rust())),
            };
            return self.counted(format!("{place} = {v}"), raw);
        }
        let vt = self.ty(val);
        let (t, len, p, v) = (self.vty(val), bytes(vt), self.name(ptr), self.name(val));
        if let Some((root, r, ArgKind::Slice { nullable, .. })) = self.slice_root(ptr, true) {
            self.stats.checked += 1;
            let s = if nullable { format!("{r}.as_deref_mut().unwrap()") } else { r };
            return format!("{s}[{}.wrapping_sub({root}_base) as usize..][..{len}].copy_from_slice(&{v}.to_le_bytes())", self.as_u64(ptr));
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
