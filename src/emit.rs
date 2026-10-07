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
use crate::borrow::{analyze_with, Analysis, Class, Ctx, Off, ParamBorrow, Pass, Root, RSP};
use crate::cfg::Cfg;
use crate::structure::{print, structure, Node, Source};
use crate::ir::*;
use crate::sources::bucket;
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
    /// Loads, stores and copies emitted as raw pointer accesses inside `unsafe`,
    /// plus FFI and indirect calls. Zero means the body has no raw pointer.
    pub raw: usize,
    /// Raw loads, stores and copies by where the pointer comes from
    /// (`sources::SOURCES`: frame, global, argument, other).
    pub raw_by: [usize; 4],
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
    /// Safe mode: how the callee takes each argument (`Pass::Borrow`: a slice).
    /// Missing arguments are integers.
    pub args: Vec<Pass>,
    /// Safe mode: it allocates `args[a]` (times `args[b]`) bytes, like `malloc`.
    pub alloc: Option<(u8, Option<u8>)>,
    /// Safe mode: it frees its first argument, like `free`.
    pub free: bool,
    /// Safe mode: it can be written as slice operations (`memcpy`, `memset`).
    pub builtin: Option<crate::libc::Builtin>,
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
    /// Safe mode: the read-only `Bytes` static containing an address, which
    /// reads can index as a slice; `None` keeps reads through it raw.
    pub global_slice: &'a dyn Fn(u64) -> Option<String>,
    /// Safe mode: the borrow analysis, if the caller ran it with call summaries
    /// (`program.rs`); otherwise it runs here, knowing nothing about calls.
    pub analysis: Option<&'a Analysis>,
}

/// Where safe-mode code indexes a root's bytes.
struct Place {
    /// The slice to read: `rdi_ref`, `frame.0`, `heap12`, `NAME.b`.
    read: String,
    /// The slice to write, if it can be written.
    write: Option<String>,
    /// The root's address, as a `u64`.
    base: String,
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
    borrow: Option<&'a Analysis>,
    /// How each root of the analysis is indexed, if it is safe.
    places: Vec<Option<Place>>,
    args: Vec<ArgKind>,
    /// Entry parameter index of each value, if it is one.
    entry_param: Vec<Option<usize>>,
    /// Values that need a variable declared up front (block params, and values
    /// used outside the block that defines them).
    hoisted: Vec<bool>,
    /// Rust expression for a constant address that points into the binary's data.
    global_of: &'a dyn Fn(u64) -> Option<String>,
    /// Where each value points (`sources.rs`), to count raw accesses by source.
    src: Vec<u8>,
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
    let env = Env { sig: None, call: &call, demote: false, structure, global_of, global_slice: &|_| None, analysis: None };
    emit_function_in(f, name, mode, &env, out)
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

    let owned;
    let borrow = match (mode, env.analysis) {
        (Mode::Fast, _) => None,
        (Mode::Safe, Some(a)) => Some(a),
        (Mode::Safe, None) => {
            let global_ok = |c: u64| (env.global_slice)(c).is_some();
            owned = analyze_with(f, &Ctx { callee: &|_| None, demoted: &|_| false, global_ok: &global_ok });
            Some(&owned)
        }
    };
    let args = entry_params
        .iter()
        .enumerate()
        .map(|(k, _)| match borrow {
            Some(a) if !env.demote => arg_kind(&a.params[k]),
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
        places: Vec::new(),
        args,
        entry_param,
        hoisted: hoisted(f, &cfg),
        global_of: env.global_of,
        src: crate::sources::sources(f),
        stats: EmitStats::default(),
    };

    e.places = e.places(env.global_slice);
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

/// Every value a terminator uses.
pub fn term_operands(f: &Function, t: Terminator, cb: impl FnMut(ValueId)) {
    term_uses(f, t, cb)
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
        match (self.entry_param[v.index()], self.f.insts[v].kind) {
            (Some(_), InstKind::BlockParam(r)) if r >= STACK_ARG_BASE => format!("arg{}", 6 + (r - STACK_ARG_BASE) as usize),
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
        let ret = match self.sig {
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
            if self.frame_safe() {
                // bytes, so that safe code can index them; aligned like a real frame
                if n * 16 <= 4096 {
                    let _ = writeln!(out, "    #[repr(C, align(16))]\n    struct Frame([u8; {}]);", n * 16);
                    let _ = writeln!(out, "    let mut frame = Frame([0; {}]);", n * 16);
                } else {
                    let _ = writeln!(out, "    struct Frame(Vec<u8>);\n    let mut frame = Frame(vec![0; {}]);", n * 16);
                }
                let _ = writeln!(out, "    let frame_base: u64 = frame.0.as_ptr() as u64;");
            } else if n * 16 <= 4096 {
                let _ = writeln!(out, "    let mut frame = [0u128; {n}];");
            } else {
                let _ = writeln!(out, "    let mut frame = vec![0u128; {n}];");
            }
        }
        // Allocations that are `Box`es.
        if let Some(a) = self.borrow {
            for (r, root) in a.roots.iter().enumerate() {
                if let (Root::Alloc(id), Some(Some(_))) = (root, self.places.get(r)) {
                    let h = format!("heap{}", id.index());
                    let _ = writeln!(out, "    let mut {h}: Box<[u8]> = Box::default();\n    let mut {h}_base: u64 = 0;");
                }
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
                    let _ = writeln!(out, "    let mut {}: {} = {zero};", self.name(v), rty(self.ty(v)));
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
                Stmt::Value(e) => out.push(format!("let {n}: {} = {e}; // {at:#x}", rty(self.ty(id)))),
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
            Terminator::Return(Some(v)) => format!("return {} as u64;", self.name(v)),
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
            .map(|(&p, &a)| (self.name(p), self.name(a)))
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
            Undef if ty == TyId::BOOL => "false".to_string(),
            CallOut { call, reg: crate::abi::RDX } if matches!(f.insts[call].kind, Call { .. }) && self.ret2(call) => {
                format!("{}_pair.1", n(call))
            }
            Undef | CallOut { .. } => format!("0_{t}"),
            Aggregate { ty: TyId::PAIR, fields } => {
                let v = fields.get(&f.value_pool);
                format!("({} as u64, {} as u64)", n(v[0]), n(v[1]))
            }
            AddrOfLocal(_) if self.frame_safe() => "frame_base".to_string(),
            AddrOfLocal(_) => "frame.as_mut_ptr() as u64".to_string(),
            Call { args, .. } if self.boxed_alloc(id).is_some() => {
                // `malloc` of an allocation that is used, then freed or dropped, like a Box
                let h = self.boxed_alloc(id).unwrap();
                let (a, b) = (self.call)(Site::Call(id)).and_then(|c| c.alloc).unwrap_or((0, None));
                let args = args.get(&f.value_pool);
                let mut len = format!("({} as usize)", n(args[a as usize]));
                if let Some(b) = b {
                    len = format!("{len}.wrapping_mul({} as usize)", n(args[b as usize]));
                }
                format!("{{ {h} = vec![0u8; {len}].into_boxed_slice(); {h}_base = {h}.as_ptr() as u64; {h}_base }}")
            }
            Call { args, .. } if self.builtin(id).is_some() => {
                let (b, ps) = self.builtin(id).unwrap();
                let a = args.get(&f.value_pool);
                let off = |k: usize| format!("{}.wrapping_sub({}) as usize", n(a[k]), ps[k].base);
                let len = format!("{} as usize", n(a[2]));
                let dst = ps[0].write.clone().unwrap();
                let e = match b {
                    crate::libc::Builtin::Fill => format!("{dst}[{}..][..{len}].fill({} as u8)", off(0), n(a[1])),
                    crate::libc::Builtin::Copy if ps[0].read == ps[1].read => {
                        format!("let __s = {}; {dst}.copy_within(__s..__s + ({len}), {})", off(1), off(0))
                    }
                    crate::libc::Builtin::Copy => {
                        format!("{dst}[{}..][..{len}].copy_from_slice(&{}[{}..][..{len}])", off(0), ps[1].read, off(1))
                    }
                };
                if !self.used[id.index()] {
                    return Stmt::Effect(format!("{{ {e}; }}"));
                }
                format!("{{ {e}; {} }}", n(a[0]))
            }
            Call { args, .. } if self.boxed_free(id).is_some() => {
                let _ = args;
                return Stmt::Effect(format!("{} = Box::default()", self.boxed_free(id).unwrap()));
            }
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
                self.stats.raw_by[bucket(self.src[dst.index()] | self.src[src.index()])] += 1;
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
        let a: Vec<String> = args.iter().map(|&v| format!("{} as u64", self.name(v))).collect();
        let a = a.join(", ");
        match (self.call)(site) {
            Some(CallInfo { path: Some(p), ret, ret2, foreign: false, args: pass, .. }) if pass.iter().any(|x| matches!(x, Pass::Borrow { .. })) => {
                let (lets, a) = self.call_args(args, &pass);
                (format!("unsafe {{ {lets}{p}({}) }}", a.join(", ")), ret, ret2)
            }
            Some(CallInfo { path: Some(p), ret, ret2, foreign, .. }) => {
                if foreign {
                    self.stats.raw += 1;
                }
                if ret2 && foreign {
                    // an extern returns `ffi::Pair`, which is FFI-safe; a tuple isn't
                    return (format!("unsafe {{ let p = {p}({a}); (p.0, p.1) }}"), ret, ret2);
                }
                (format!("unsafe {{ {p}({a}) }}"), ret, ret2)
            }
            _ => {
                self.stats.raw += 1;
                let tys = vec!["u64"; args.len()].join(", ");
                let callee = self.name(callee);
                (format!("unsafe {{ core::mem::transmute::<u64, unsafe extern \"C\" fn({tys}) -> u64>({callee})({a}) }}"), true, false)
            }
        }
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

    /// How safe-mode code reaches each safe root of the borrow analysis.
    fn places(&self, global_slice: &dyn Fn(u64) -> Option<String>) -> Vec<Option<Place>> {
        let Some(a) = self.borrow else { return Vec::new() };
        let params = self.f.blocks[self.f.entry].params.get(&self.f.value_pool);
        a.roots
            .iter()
            .enumerate()
            .map(|(r, &root)| {
                if !a.safe[r] {
                    return None;
                }
                Some(match root {
                    Root::Param(k) => {
                        let ArgKind::Slice { mutbl, nullable } = self.args[k as usize] else { return None };
                        let n = self.name(params[k as usize]);
                        let read = match (nullable, mutbl) {
                            (false, _) => format!("{n}_ref"),
                            (true, false) => format!("{n}_ref.unwrap()"),
                            (true, true) => format!("{n}_ref.as_deref().unwrap()"),
                        };
                        let write = match nullable {
                            false => format!("{n}_ref"),
                            true => format!("{n}_ref.as_deref_mut().unwrap()"),
                        };
                        Place { read, write: mutbl.then_some(write), base: format!("{n}_base") }
                    }
                    Root::Frame => Place { read: "frame.0".into(), write: Some("frame.0".into()), base: "frame_base".into() },
                    Root::Global(c) => {
                        let g = global_slice(c)?;
                        Place { read: format!("{g}.b"), write: None, base: format!("(core::ptr::addr_of!({g}) as u64)") }
                    }
                    Root::Alloc(id) => {
                        let h = format!("heap{}", id.index());
                        Place { read: h.clone(), write: Some(h.clone()), base: format!("{h}_base") }
                    }
                })
            })
            .collect()
    }

    /// The place an access through `ptr` can bounds-check against, if `ptr`
    /// derives from exactly one safe root (that can be written, for a store).
    fn slice_root(&self, ptr: ValueId, write: bool) -> Option<(String, String)> {
        let r = self.borrow?.safe_root(ptr)?;
        let p = self.places.get(r as usize)?.as_ref()?;
        match write {
            false => Some((p.read.clone(), p.base.clone())),
            true => Some((p.write.clone()?, p.base.clone())),
        }
    }

    fn load(&mut self, ptr: ValueId, ty: TyId) -> String {
        let (t, len, p) = (rty(ty), bytes(ty), self.name(ptr));
        if let Some((s, base)) = self.slice_root(ptr, false) {
            self.stats.checked += 1;
            return format!("{t}::from_le_bytes({s}[{p}.wrapping_sub({base}) as usize..][..{len}].try_into().unwrap())");
        }
        self.stats.raw += 1;
        self.stats.raw_by[bucket(self.src[ptr.index()])] += 1;
        format!("unsafe {{ ({p} as *const {t}).read_unaligned() }}")
    }

    fn store(&mut self, ptr: ValueId, val: ValueId) -> String {
        let vt = self.ty(val);
        let (t, len, p, v) = (rty(vt), bytes(vt), self.name(ptr), self.name(val));
        if let Some((s, base)) = self.slice_root(ptr, true) {
            self.stats.checked += 1;
            return format!("{s}[{p}.wrapping_sub({base}) as usize..][..{len}].copy_from_slice(&{v}.to_le_bytes())");
        }
        self.stats.raw += 1;
        self.stats.raw_by[bucket(self.src[ptr.index()])] += 1;
        format!("unsafe {{ ({p} as *mut {t}).write_unaligned({v}) }}")
    }

    /// The frame is a safe root: an array of bytes that accesses index.
    fn frame_safe(&self) -> bool {
        self.borrow
            .and_then(|a| a.roots.iter().position(|&r| r == Root::Frame).map(|r| self.places.get(r).is_some_and(|p| p.is_some())))
            .unwrap_or(false)
    }

    /// A call to `memcpy` or `memset` whose pointers all have safe roots: what it
    /// does, and the places of its arguments (`dst`, `src`).
    fn builtin(&self, call: ValueId) -> Option<(crate::libc::Builtin, Vec<&Place>)> {
        let a = self.borrow?;
        let b = (self.call)(Site::Call(call))?.builtin?;
        if a.raw_calls.contains(&Site::Call(call)) {
            return None;
        }
        let InstKind::Call { args, .. } = self.f.insts[call].kind else { return None };
        let args = args.get(&self.f.value_pool);
        let ptrs = if b == crate::libc::Builtin::Copy { 2 } else { 1 };
        if args.len() < 3 {
            return None;
        }
        let mut ps = Vec::new();
        for &p in &args[..ptrs] {
            ps.push(self.places.get(a.safe_root(p)? as usize)?.as_ref()?);
        }
        ps[0].write.as_ref()?;
        Some((b, ps))
    }

    /// The `Box` this `free` call drops, if its argument is one.
    fn boxed_free(&self, call: ValueId) -> Option<String> {
        if !(self.call)(Site::Call(call)).is_some_and(|c| c.free) {
            return None;
        }
        let InstKind::Call { args, .. } = self.f.insts[call].kind else { return None };
        let &p = args.get(&self.f.value_pool).first()?;
        let a = self.borrow?;
        let r = a.safe_root(p)?;
        match a.roots[r as usize] {
            Root::Alloc(_) => self.places.get(r as usize)?.as_ref().map(|p| p.read.clone()),
            _ => None,
        }
    }

    /// The safe root of the allocation this call makes, if it is a `Box`.
    fn boxed_alloc(&self, call: ValueId) -> Option<String> {
        let a = self.borrow?;
        let r = a.roots.iter().position(|&x| x == Root::Alloc(call))?;
        self.places.get(r)?.as_ref().map(|p| p.read.clone())
    }

    /// Safe-mode arguments for a call to a decompiled function: slices for the
    /// arguments it borrows, reborrowed from the root each one points into. Two
    /// borrows of one root, one of them mutable, start at different known offsets
    /// (the borrow analysis downgrades the others); they are split apart with
    /// `split_at_mut`. Returns the statements that do the splitting, and the
    /// argument expressions.
    fn call_args(&mut self, args: &[ValueId], pass: &[Pass]) -> (String, Vec<String>) {
        let mut out: Vec<String> = args.iter().map(|&v| format!("{} as u64", self.name(v))).collect();
        let mut lets = String::new();
        let Some(a) = self.borrow else { return (lets, out) };
        // (root, argument) for every borrowed argument that points into a root
        let mut groups: Vec<(u8, Vec<usize>)> = Vec::new();
        for (k, &v) in args.iter().enumerate() {
            let Some(&Pass::Borrow { nullable, .. }) = pass.get(k) else { continue };
            match self.borrow.and_then(|a| a.safe_root(v)) {
                Some(r) if self.places.get(r as usize).is_some_and(|p| p.is_some()) => {
                    match groups.iter_mut().find(|g| g.0 == r) {
                        Some(g) => g.1.push(k),
                        None => groups.push((r, vec![k])),
                    }
                }
                _ if nullable && self.borrow.is_some_and(|a| a.origin[v.index()].is_none()) => out[k] = "None".into(),
                _ => {
                    self.stats.todo += 1;
                    out[k] = "todo!(\"a slice the borrow analysis couldn't prove\")".into();
                }
            }
        }
        for (r, mut ks) in groups {
            let place = self.places[r as usize].as_ref().unwrap();
            let mutbl = |k: usize| matches!(pass[k], Pass::Borrow { mutbl: true, .. });
            let nullable = |k: usize| matches!(pass[k], Pass::Borrow { nullable: true, .. });
            let off = |k: usize| self.name(args[k]);
            let mut exprs: Vec<(usize, String)> = Vec::new();
            if ks.len() == 1 || !ks.iter().any(|&k| mutbl(k)) {
                for &k in &ks {
                    let v = off(k);
                    let e = match (mutbl(k), &place.write) {
                        (true, Some(w)) => format!("&mut {w}[{v}.wrapping_sub({}) as usize..]", place.base),
                        (true, None) => "todo!(\"a mutable borrow of a read-only root\")".to_string(),
                        (false, _) => format!("&{}[{v}.wrapping_sub({}) as usize..]", place.read, place.base),
                    };
                    exprs.push((k, e));
                }
            } else {
                // distinct known offsets, in order
                let at = |k: usize| match a.origin[args[k].index()].off {
                    Off::Known(x) => x,
                    Off::Unknown => i64::MAX,
                };
                ks.sort_by_key(|&k| at(k));
                let w = place.write.clone().unwrap_or_default();
                let _ = write!(lets, "let __s = &mut {w}[{}.wrapping_sub({}) as usize..]; ", off(ks[0]), place.base);
                for i in 0..ks.len() {
                    let name = format!("__s{r}_{i}");
                    if i + 1 < ks.len() {
                        let _ = write!(lets, "let ({name}, __s) = __s.split_at_mut({}.wrapping_sub({}) as usize); ", off(ks[i + 1]), off(ks[i]));
                    } else {
                        let _ = write!(lets, "let {name} = __s; ");
                    }
                    let k = ks[i];
                    exprs.push((k, if mutbl(k) { format!("&mut *{name}") } else { format!("&*{name}") }));
                }
            }
            for (k, e) in exprs {
                out[k] = if nullable(k) { format!("if {} == 0 {{ None }} else {{ Some({e}) }}", off(k)) } else { e };
            }
        }
        (lets, out)
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
