//! Function signatures (System V x86_64): which argument registers a function
//! really takes, how many stack arguments, what it returns, and which
//! caller-saved registers it leaves alone.
//!
//! The lifter can't know any of that one function at a time. It passes every
//! register a callee could read to every call, so every register a callee might
//! read looks like an argument of the caller too, and every function "returns"
//! rax. `infer` works it out from the callees' signatures, and `program.rs` runs it
//! over the whole binary to a fixpoint. `apply` then rewrites a function to match
//! its signature: calls pass exactly their callee's arguments, registers after a
//! call are resolved, a `void` function returns nothing, and registers that
//! aren't arguments (callee-saved ones, rax, r10, r11) stop being parameters.
//!
//! * **Arguments.** An argument register is a parameter if its entry value is live:
//!   it reaches a store, a branch, the return value, or a call argument the callee
//!   actually takes. Calls to unknown code (indirect calls, imports the libc table
//!   doesn't know) take the argument registers set up in the call's own block,
//!   which is where compilers set them up. Parameters are prefix-closed, like C's:
//!   a function that reads only rsi still takes rdi.
//! * **Return value.** A function returns a value if rax is defined at every
//!   return: not rax's entry value, not `Undef`, not the result of a `void` call,
//!   and not a block parameter that merges one of those in.
//! * **Two-register returns.** It also returns rdx (a 16-byte value, which Rust uses
//!   for slices and pairs) if rdx is defined at every return and some caller reads
//!   rdx after calling it.
//! * **Preserved registers.** A caller-saved register is preserved if its value at
//!   every return is its entry value, possibly through calls that preserve it in
//!   turn. The ABI doesn't promise this, but gcc's interprocedural register
//!   allocation relies on it within a translation unit: a caller keeps a value in
//!   rdi across a call to a function it knows doesn't touch rdi.
//! * **Stack arguments.** Reads at entry-rsp offsets 8, 16, ... (offset 0 is the
//!   return address).
//!
//! Everything is monotone in the callees' signatures, so iterating from "no
//! arguments, returns nothing, preserves nothing" reaches the least fixpoint, also
//! through recursion.
use crate::borrow::{analyze, Off, RSP};
use crate::cfg::Cfg;
use crate::ir::*;
use crate::lift::{CALL_ARGS, CALL_REGS, EXIT_REGS};
use crate::opt::{clean, list_remove, rewrite_uses};
use crate::verify::for_each_operand;

/// x86 register numbers of the System V integer argument registers, in order:
/// rdi, rsi, rdx, rcx, r8, r9.
pub const SYSV_ARGS: [u8; 6] = [7, 6, 2, 1, 8, 9];
pub(crate) const RAX: u8 = 0;
pub(crate) const RDX: u8 = 2;

/// `BlockParam` register number of stack argument `j` (C argument `6 + j`).
pub const STACK_ARG_BASE: u8 = 16;
/// `BlockParam` register number of a promoted stack slot that isn't an argument.
pub const SLOT_PARAM: u8 = u8::MAX;

/// Caller-saved registers whose preservation is tracked, as a mask of x86 numbers.
pub const CALLER_SAVED: u16 = 1 << 0 | 1 << 1 | 1 << 2 | 1 << 6 | 1 << 7 | 1 << 8 | 1 << 9 | 1 << 10 | 1 << 11;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Sig {
    /// Integer arguments in registers: the first `args` of rdi, rsi, rdx, rcx, r8, r9.
    pub args: u8,
    /// 8-byte arguments on the stack, after the six in registers.
    pub stack_args: u8,
    /// Returns a value in rax.
    pub ret: bool,
    /// Also returns a value in rdx: the result is 16 bytes, rax:rdx.
    pub ret2: bool,
    /// Takes more arguments than `args` (printf). Callers pass what they set up.
    pub variadic: bool,
    /// Caller-saved registers (bit = x86 number) the function leaves as it found them.
    pub preserves: u16,
}

impl Sig {
    /// Does the function leave register `reg` (x86 number) as it found it?
    pub fn keeps(self, reg: u8) -> bool {
        self.preserves & (1 << reg) != 0
    }
}

/// A call: a `Call` instruction, or a block that ends in a `TailCall`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Site {
    Call(ValueId),
    Tail(BlockId),
}

impl Site {
    /// The callee value and the argument list.
    pub fn parts(self, f: &Function) -> (ValueId, ListRef) {
        match self {
            Site::Call(id) => match f.insts[id].kind {
                InstKind::Call { callee, args } => (callee, args),
                _ => unreachable!("not a call"),
            },
            Site::Tail(b) => match f.blocks[b].term {
                Terminator::TailCall { callee, args } => (callee, args),
                _ => unreachable!("not a tail call"),
            },
        }
    }

    /// Address of the `call` or `jmp` instruction.
    pub fn ip(self, f: &Function) -> u64 {
        let v = match self {
            Site::Call(id) => id,
            Site::Tail(_) => self.parts(f).0, // created by the jmp
        };
        f.origin[v.index()]
    }

    /// The value register `reg` held at the call, if the lifter recorded it.
    fn before(self, f: &Function, reg: u8) -> Option<ValueId> {
        let args = self.parts(f).1.get(&f.value_pool);
        let k = CALL_REGS.iter().position(|r| r.number() as u8 == reg)?;
        (args.len() == CALL_ARGS).then(|| args[k])
    }
}

/// Every call site, in block order.
pub fn sites(f: &Function) -> Vec<Site> {
    let mut out = Vec::new();
    for (b, blk) in f.blocks.iter() {
        for &id in blk.insts.get(&f.value_pool) {
            if matches!(f.insts[id].kind, InstKind::Call { .. }) {
                out.push(Site::Call(id));
            }
        }
        if matches!(blk.term, Terminator::TailCall { .. }) {
            out.push(Site::Tail(b));
        }
    }
    out
}

/// How many argument registers a call to unknown code passes: up to the last one
/// set in the call's own block (prefix-closed). Valid on lifter output, where a
/// call's arguments are `CALL_ARGS` long.
pub fn guess_args(f: &Function, site: Site) -> u8 {
    let (_, args) = site.parts(f);
    let args = args.get(&f.value_pool);
    if args.len() != CALL_ARGS {
        return args.len().min(6) as u8;
    }
    let block = match site {
        Site::Call(id) => f.blocks.iter().find(|(_, b)| b.insts.get(&f.value_pool).contains(&id)).map(|(b, _)| b),
        Site::Tail(b) => Some(b),
    };
    let Some(block) = block else { return 0 };
    let here = f.blocks[block].insts.get(&f.value_pool);
    let mut n = 0;
    for (k, &a) in args[..6].iter().enumerate() {
        let set_here = here.contains(&a) && !matches!(f.insts[a].kind, InstKind::Undef | InstKind::CallOut { .. });
        if set_here {
            n = k + 1;
        }
    }
    n as u8
}

/// What `infer` found besides the signature: which call sites read rdx after the
/// call (evidence that the callee returns 16 bytes).
pub struct Inferred {
    pub sig: Sig,
    pub rdx_read: Vec<bool>,
}

/// Where a function's control leaves it.
enum Exit {
    /// A return, with rax and the `Exit` registers if the lifter recorded them.
    Return { rax: ValueId, regs: Option<ListRef> },
    /// A tail call: site index.
    Tail(usize),
}

fn exits(f: &Function, cfg: &Cfg, sites: &[Site]) -> Vec<Exit> {
    let mut out = Vec::new();
    for &b in &cfg.rpo {
        let blk = &f.blocks[b];
        match blk.term {
            Terminator::Return(Some(rax)) => {
                let regs = blk.insts.get(&f.value_pool).iter().rev().find_map(|&id| match f.insts[id].kind {
                    InstKind::Exit { regs } => Some(regs),
                    _ => None,
                });
                out.push(Exit::Return { rax, regs });
            }
            Terminator::TailCall { .. } => {
                if let Some(k) = sites.iter().position(|&s| s == Site::Tail(b)) {
                    out.push(Exit::Tail(k));
                }
            }
            _ => {}
        }
    }
    out
}

/// Where a value comes from, as far as register summaries care.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Src {
    /// Not computed yet.
    Top,
    /// Register `r`'s value on entry.
    Entry(u8),
    /// Register `r` after call site `k` (rax: the call's own value).
    After(usize, u8),
    Other,
}

fn sources(f: &Function, cfg: &Cfg, sites: &[Site]) -> Vec<Src> {
    let mut s = vec![Src::Top; f.insts.len()];
    let site_of = |call: ValueId| sites.iter().position(|&x| x == Site::Call(call));
    for (_, blk) in f.blocks.iter() {
        for &id in blk.insts.get(&f.value_pool) {
            s[id.index()] = match f.insts[id].kind {
                InstKind::Call { .. } => site_of(id).map_or(Src::Other, |k| Src::After(k, RAX)),
                InstKind::CallOut { call, reg } => site_of(call).map_or(Src::Other, |k| Src::After(k, reg)),
                _ => Src::Other,
            };
        }
    }
    for &p in f.blocks[f.entry].params.get(&f.value_pool) {
        s[p.index()] = match f.insts[p].kind {
            InstKind::BlockParam(r) => Src::Entry(r),
            _ => Src::Other,
        };
    }
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &cfg.rpo {
            if b == f.entry {
                continue;
            }
            for (k, &p) in f.blocks[b].params.get(&f.value_pool).iter().enumerate() {
                let mut new = Src::Top;
                for &pred in cfg.preds(b) {
                    incoming(f, pred, b, k, |a| {
                        let x = if a == p { Src::Top } else { s[a.index()] };
                        new = match (new, x) {
                            (Src::Top, x) | (x, Src::Top) => x,
                            (x, y) if x == y => x,
                            _ => Src::Other,
                        };
                    });
                }
                if new != s[p.index()] {
                    s[p.index()] = new;
                    changed = true;
                }
            }
        }
    }
    s
}

/// Is `v`, the value of register `reg` at an exit, `reg`'s entry value, possibly
/// through calls that preserve it?
fn traces_to_entry(f: &Function, src: &[Src], sites: &[Site], callee: &dyn Fn(usize) -> Sig, reg: u8, mut v: ValueId) -> bool {
    for _ in 0..64 {
        match src[v.index()] {
            Src::Entry(r) => return r == reg,
            Src::After(k, r) if r == reg && callee(k).keeps(r) => match sites[k].before(f, r) {
                Some(b) => v = b,
                None => return false,
            },
            _ => return false,
        }
    }
    false
}

/// The value of `reg` at each exit (None: unknown, e.g. no `Exit` recorded).
fn exit_value(f: &Function, sites: &[Site], callee: &dyn Fn(usize) -> Sig, e: &Exit, reg: u8) -> Option<ValueId> {
    match *e {
        Exit::Return { rax, .. } if reg == RAX => Some(rax),
        Exit::Return { regs, .. } => {
            let k = EXIT_REGS.iter().position(|r| r.number() as u8 == reg)?;
            Some(regs?.get(&f.value_pool)[k])
        }
        Exit::Tail(k) => {
            // after `jmp g`, reg is what g leaves in it
            if callee(k).keeps(reg) { sites[k].before(f, reg) } else { None }
        }
    }
}

/// The signature of `f` (lifted with `track_exits`, cleaned, not yet `apply`d),
/// given the signature of the callee at each call site (`callee(i)` for
/// `sites[i]`), the function's own signature from the previous round, and whether
/// any caller reads rdx after calling it. `stack_args` is left 0; see `stack_args`.
pub fn infer(f: &Function, sites: &[Site], callee: &dyn Fn(usize) -> Sig, prev: Sig, rdx_wanted: bool) -> Inferred {
    let cfg = Cfg::new(f);
    let ex = exits(f, &cfg, sites);
    let src = sources(f, &cfg, sites);

    // ---- preserved registers ----
    let mut preserves = 0u16;
    for reg in 0..16u8 {
        if CALLER_SAVED & (1 << reg) == 0 {
            continue;
        }
        let ok = ex.iter().all(|e| match e {
            Exit::Tail(k) if !callee(*k).keeps(reg) => false,
            e => exit_value(f, sites, callee, e, reg).is_some_and(|v| traces_to_entry(f, &src, sites, callee, reg, v)),
        });
        if ok && !ex.is_empty() {
            preserves |= 1 << reg;
        }
    }

    // ---- return value: is rax defined at every exit? ----
    let undef = undefined_values(f, &cfg, sites, callee, false);
    let defined_at = |undef: &[bool], reg: u8, e: &Exit| match e {
        Exit::Tail(k) => {
            let c = callee(*k);
            match reg {
                RAX if c.ret => true,
                RDX if c.ret2 => true,
                _ => c.keeps(reg) && sites[*k].before(f, reg).is_some_and(|v| !undef[v.index()]),
            }
        }
        e => exit_value(f, sites, callee, e, reg).is_some_and(|v| !undef[v.index()]),
    };
    let ret = !ex.is_empty() && ex.iter().all(|e| defined_at(&undef, RAX, e));

    // ---- arguments: which entry registers are live ----
    let mut rdx_read = vec![false; sites.len()];
    let live = live_values(f, sites, callee, ret, prev.ret2, &mut rdx_read);
    let mut args = 0;
    for &p in f.blocks[f.entry].params.get(&f.value_pool) {
        if let InstKind::BlockParam(r) = f.insts[p].kind {
            if let Some(k) = SYSV_ARGS.iter().position(|&a| a == r) {
                if live[p.index()] {
                    args = args.max(k as u8 + 1);
                }
            }
        }
    }

    // ---- rdx: returned too? ----
    let ret2 = ret && rdx_wanted && preserves & (1 << RDX) == 0 && {
        let undef = undefined_values(f, &cfg, sites, callee, args < 3);
        ex.iter().all(|e| defined_at(&undef, RDX, e))
    };
    Inferred { sig: Sig { args, stack_args: 0, ret, ret2, variadic: false, preserves }, rdx_read }
}

/// Values that may be undefined: rax on entry (and rdx, if `rdx_undef`: it isn't
/// an argument), `Undef`, registers after a call that the callee neither returns
/// nor preserves, and block parameters that merge any of those in.
fn undefined_values(f: &Function, cfg: &Cfg, sites: &[Site], callee: &dyn Fn(usize) -> Sig, rdx_undef: bool) -> Vec<bool> {
    let mut u = vec![false; f.insts.len()];
    for &p in f.blocks[f.entry].params.get(&f.value_pool) {
        u[p.index()] = match f.insts[p].kind {
            InstKind::BlockParam(RAX) => true,
            InstKind::BlockParam(RDX) => rdx_undef,
            _ => false,
        };
    }
    let site_of = |call: ValueId| sites.iter().position(|&x| x == Site::Call(call));
    // register `reg` after call site `k`
    let after = |u: &[bool], k: Option<usize>, reg: u8| -> bool {
        let Some(k) = k else { return reg != RAX };
        let c = callee(k);
        if (reg == RAX && c.ret) || (reg == RDX && c.ret2) {
            return false;
        }
        !(c.keeps(reg) && sites[k].before(f, reg).is_some_and(|v| !u[v.index()]))
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &cfg.rpo {
            let blk = &f.blocks[b];
            if b != f.entry {
                for (k, &p) in blk.params.get(&f.value_pool).iter().enumerate() {
                    let mut any = false;
                    for &pred in cfg.preds(b) {
                        incoming(f, pred, b, k, |a| any |= u[a.index()]);
                    }
                    if any && !u[p.index()] {
                        u[p.index()] = true;
                        changed = true;
                    }
                }
            }
            for &id in blk.insts.get(&f.value_pool) {
                let now = match f.insts[id].kind {
                    InstKind::Undef => true,
                    InstKind::Call { .. } => after(&u, site_of(id), RAX),
                    InstKind::CallOut { call, reg } => after(&u, site_of(call), reg),
                    _ => false,
                };
                if now != u[id.index()] {
                    u[id.index()] = now;
                    changed = true;
                }
            }
        }
    }
    u
}

/// Edge argument `k` on every edge from `p` into `b`.
pub(crate) fn incoming(f: &Function, p: BlockId, b: BlockId, k: usize, mut cb: impl FnMut(ValueId)) {
    for (s, args) in f.edges(p) {
        if s == b {
            cb(args[k]);
        }
    }
}

/// Liveness from the function's effects. A call only uses the arguments its callee
/// takes; a register after a call that the callee preserves uses its value before
/// the call. The return value counts only if the function `ret`s (rdx at returns,
/// if it `ret2`s). Marks in `rdx_read` the call sites whose rdx is read.
fn live_values(f: &Function, sites: &[Site], callee: &dyn Fn(usize) -> Sig, ret: bool, ret2: bool, rdx_read: &mut [bool]) -> Vec<bool> {
    let n = f.insts.len();
    let mut param_of: Vec<Option<(BlockId, usize)>> = vec![None; n];
    for (b, blk) in f.blocks.iter() {
        for (k, &p) in blk.params.get(&f.value_pool).iter().enumerate() {
            param_of[p.index()] = Some((b, k));
        }
    }
    let site_of = |call: ValueId| sites.iter().position(|&x| x == Site::Call(call));
    let mut live = vec![false; n];
    let mut work = Vec::new();
    fn mark(v: ValueId, live: &mut [bool], work: &mut Vec<ValueId>) {
        if !std::mem::replace(&mut live[v.index()], true) {
            work.push(v);
        }
    }
    let call_uses = |s: Site, live: &mut Vec<bool>, work: &mut Vec<ValueId>| {
        let (c, args) = s.parts(f);
        mark(c, live, work);
        let args = args.get(&f.value_pool);
        let arity = sites.iter().position(|&x| x == s).map_or(0, |k| callee(k).args as usize);
        for &a in &args[..arity.min(args.len()).min(6)] {
            mark(a, live, work);
        }
    };
    for (b, blk) in f.blocks.iter() {
        for &id in blk.insts.get(&f.value_pool) {
            match f.insts[id].kind {
                InstKind::Call { .. } => call_uses(Site::Call(id), &mut live, &mut work),
                InstKind::Store { .. } | InstKind::MemCopy { .. } | InstKind::Opaque { .. }
                | InstKind::Load { volatile: true, .. } | InstKind::Assign { .. } => {
                    for_each_operand(f.insts[id].kind, f, |v| mark(v, &mut live, &mut work));
                }
                InstKind::Exit { regs } if ret2 => mark(regs.get(&f.value_pool)[1], &mut live, &mut work),
                _ => {}
            }
        }
        match blk.term {
            Terminator::Branch { c, .. } => mark(c, &mut live, &mut work),
            Terminator::Switch { v, .. } => mark(v, &mut live, &mut work),
            Terminator::Return(Some(v)) if ret => mark(v, &mut live, &mut work),
            Terminator::TailCall { .. } => call_uses(Site::Tail(b), &mut live, &mut work),
            _ => {}
        }
    }
    let mut ops = Vec::new();
    while let Some(v) = work.pop() {
        ops.clear();
        match f.insts[v].kind {
            // A live call result doesn't make the call's arguments live beyond what
            // the callee takes; those were marked with the call. If the callee
            // returns nothing but keeps rax, the result is rax from before.
            InstKind::Call { .. } => {
                if let Some(k) = site_of(v) {
                    let c = callee(k);
                    if !c.ret && c.keeps(RAX) {
                        ops.extend(sites[k].before(f, RAX));
                    }
                }
            }
            InstKind::CallOut { call, reg } => {
                if let Some(k) = site_of(call) {
                    if reg == RDX {
                        rdx_read[k] = true;
                    }
                    if callee(k).keeps(reg) {
                        ops.extend(sites[k].before(f, reg));
                    }
                }
            }
            k => for_each_operand(k, f, |o| ops.push(o)),
        }
        if let Some((b, k)) = param_of[v.index()] {
            for (p, _) in f.blocks.iter() {
                incoming(f, p, b, k, |a| ops.push(a));
            }
        }
        for &o in &ops {
            mark(o, &mut live, &mut work);
        }
    }
    live
}

/// Stack arguments read: 8-byte words at entry-rsp offsets 8, 16, ... They don't
/// depend on the callees, so `infer` leaves `Sig::stack_args` to this.
pub fn stack_args(f: &Function) -> u8 {
    let entry = f.blocks[f.entry].params.get(&f.value_pool);
    let Some(k) = entry.iter().position(|&p| matches!(f.insts[p].kind, InstKind::BlockParam(RSP))) else { return 0 };
    let a = analyze(f);
    let mut end = 8i64;
    for (_, blk) in f.blocks.iter() {
        for &id in blk.insts.get(&f.value_pool) {
            let (ptr, size) = match f.insts[id].kind {
                InstKind::Load { ptr, .. } => (ptr, bytes(f.insts[id].ty)),
                InstKind::Store { ptr, val, .. } => (ptr, bytes(f.insts[val].ty)),
                _ => continue,
            };
            let o = a.origin[ptr.index()];
            if o.roots == 1 << k {
                if let Off::Known(off) = o.off {
                    if (8..8 + 8 * 64).contains(&off) {
                        end = end.max(off + size as i64);
                    }
                }
            }
        }
    }
    ((end - 8 + 7) / 8) as u8
}

pub(crate) fn bytes(ty: TyId) -> usize {
    match ty {
        TyId::B1 | TyId::BOOL => 1,
        TyId::B2 => 2,
        TyId::B4 => 4,
        TyId::PAIR | TyId::B16 => 16,
        _ => 8,
    }
}

// ---------------------------------------------------------------------------
// Rewriting a function to its signature

/// What a call site calls, as far as `apply` needs to know: the callee's
/// signature, with `args` the register arguments this call passes (a variadic
/// callee gets what the call sets up).
pub type CallShape = Sig;

/// Rewrite `f` (lifted and cleaned) to signature `sig`, with `shape(i)` the callee
/// at `sites[i]`: registers after each call become what the callee leaves there,
/// calls pass exactly their callee's arguments, returns return what `sig` says,
/// non-argument registers on entry become `Undef`, stack slots become values where
/// possible (`frame::promote`), and the result is cleaned again. Returns the call
/// sites, in the same order: a tail call that became a call is a `Site::Call` now.
pub fn apply(f: &mut Function, sig: Sig, sites: &[Site], shape: &dyn Fn(usize) -> CallShape) -> Vec<Site> {
    split_entry(f); // doesn't renumber blocks or values, so `sites` stay valid
    let entry = f.entry;
    let at = f.blocks[entry].insts.get(&f.value_pool).first().and_then(|v| f.origin.get(v.index())).copied().unwrap_or(0);
    let mut undefs = Vec::new(); // go at the top of the entry block, which dominates every use
    let mut map: Vec<(ValueId, ValueId)> = Vec::new();
    let undef = |f: &mut Function, undefs: &mut Vec<ValueId>| {
        let u = new_inst(f, InstKind::Undef, TyId::B8, at);
        undefs.push(u);
        u
    };
    let site_of = |call: ValueId| sites.iter().position(|&x| x == Site::Call(call));

    // 1. Registers after a call: the value from before the call if the callee
    //    preserves the register, the callee's result (rax, and rdx for a 16-byte
    //    result), or Undef.
    for bi in 0..f.blocks.len() {
        let list: Vec<ValueId> = f.blocks[BlockId::new(bi)].insts.get(&f.value_pool).to_vec();
        for id in list {
            let (call, reg) = match f.insts[id].kind {
                InstKind::Call { .. } => (id, RAX),
                InstKind::CallOut { call, reg } => (call, reg),
                _ => continue,
            };
            let Some(k) = site_of(call) else { continue };
            let c = shape(k);
            if (reg == RAX && c.ret) || (reg == RDX && c.ret2) {
                continue;
            }
            let to = match (c.keeps(reg), sites[k].before(f, reg)) {
                (true, Some(v)) => v,
                _ => undef(f, &mut undefs),
            };
            map.push((id, to));
        }
    }

    // 2. Returns: rax, rax:rdx, or nothing. A tail call whose callee returns less
    //    than this function does becomes a call and a return.
    for bi in 0..f.blocks.len() {
        let b = BlockId::new(bi);
        match f.blocks[b].term {
            Terminator::Return(Some(rax)) => {
                let exit = f.blocks[b].insts.get(&f.value_pool).iter().rev().find_map(|&id| match f.insts[id].kind {
                    InstKind::Exit { regs } => Some(regs.get(&f.value_pool)[1]),
                    _ => None,
                });
                let term = if sig.ret2 {
                    let rdx = exit.unwrap_or_else(|| undef(f, &mut undefs));
                    Terminator::Return(Some(pair(f, b, rax, rdx)))
                } else if sig.ret {
                    Terminator::Return(Some(rax))
                } else {
                    Terminator::Return(None)
                };
                f.blocks[b].term = term;
            }
            Terminator::TailCall { callee, args } => {
                let Some(k) = sites.iter().position(|&s| s == Site::Tail(b)) else { continue };
                let c = shape(k);
                if !(sig.ret && !c.ret || sig.ret2 && !c.ret2) {
                    continue;
                }
                // `jmp g` becomes `v = call g; return v (or v:rdx)`
                let call = new_inst(f, InstKind::Call { callee, args }, TyId::B8, f.origin[callee.index()]);
                let mut tail = vec![call];
                let rax = if c.ret {
                    call
                } else {
                    match (c.keeps(RAX), sites[k].before(f, RAX)) {
                        (true, Some(v)) => v,
                        _ => undef(f, &mut undefs),
                    }
                };
                let ret = if sig.ret2 {
                    let rdx = if c.ret2 {
                        let v = new_inst(f, InstKind::CallOut { call, reg: RDX }, TyId::B8, f.origin[callee.index()]);
                        tail.push(v);
                        v
                    } else {
                        match (c.keeps(RDX), sites[k].before(f, RDX)) {
                            (true, Some(v)) => v,
                            _ => undef(f, &mut undefs),
                        }
                    };
                    let p = new_inst(f, InstKind::Aggregate { ty: TyId::PAIR, fields: ListRef::EMPTY }, TyId::PAIR, f.origin[callee.index()]);
                    set_fields(f, p, rax, rdx);
                    tail.push(p);
                    p
                } else {
                    rax
                };
                append(f, b, &tail);
                f.blocks[b].term = Terminator::Return(Some(ret));
            }
            _ => {}
        }
    }
    // The tail calls that became calls are call sites now.
    let mut sites: Vec<Site> = sites.to_vec();
    for s in sites.iter_mut() {
        if let Site::Tail(b) = *s {
            if !matches!(f.blocks[b].term, Terminator::TailCall { .. }) {
                let last = *f.blocks[b].insts.get(&f.value_pool).iter().rev().find(|&&id| matches!(f.insts[id].kind, InstKind::Call { .. })).unwrap();
                *s = Site::Call(last);
            }
        }
    }
    // Exits have served their purpose.
    remove_insts(f, |f, id| matches!(f.insts[id].kind, InstKind::Exit { .. }));

    // 3. Calls pass exactly their callee's arguments; stack arguments are loaded
    //    from the caller's stack at the call.
    let mut inserts: Vec<(ValueId, Vec<ValueId>)> = Vec::new(); // insert before call
    let mut tail_loads: Vec<(BlockId, Vec<ValueId>)> = Vec::new(); // append to block
    for (i, &s) in sites.iter().enumerate() {
        let c = shape(i);
        let (_, args) = s.parts(f);
        let old: Vec<ValueId> = args.get(&f.value_pool).to_vec();
        if old.len() != CALL_ARGS {
            continue; // already applied
        }
        let at = s.ip(f);
        let mut new: Vec<ValueId> = old[..(c.args as usize).min(6)].to_vec();
        let mut loads = Vec::new();
        for j in 0..c.stack_args as i32 {
            // at a call, [rsp] is the first stack argument; at a jmp, [rsp] is our
            // own return address and the arguments follow it
            let disp = 8 * j + if matches!(s, Site::Tail(_)) { 8 } else { 0 };
            let p = new_inst(f, InstKind::PtrOffset { base: old[6], index: None, scale: 1, disp }, TyId::PTR, at);
            let v = new_inst(f, InstKind::Load { ptr: p, align: 1, volatile: false }, TyId::B8, at);
            loads.extend([p, v]);
            new.push(v);
        }
        let start = f.value_pool.len() as u32;
        f.value_pool.extend_from_slice(&new);
        let list = ListRef { start, len: new.len() as u32 };
        match s {
            Site::Call(id) => {
                if let InstKind::Call { args, .. } = &mut f.insts[id].kind {
                    *args = list;
                }
                if !loads.is_empty() {
                    inserts.push((id, loads));
                }
            }
            Site::Tail(b) => {
                if let Terminator::TailCall { args, .. } = &mut f.blocks[b].term {
                    *args = list;
                }
                if !loads.is_empty() {
                    tail_loads.push((b, loads));
                }
            }
        }
    }
    for (b, blk_insts) in rebuild_lists(f, &inserts, &tail_loads) {
        f.blocks[b].insts = blk_insts;
    }

    // 4. Registers that aren't arguments stop being parameters: callee-saved
    //    registers are only saved and restored, and rax, r10, r11 are undefined.
    //    rsp stays for `frame::promote`.
    let mut k = 0;
    while k < f.blocks[entry].params.len as usize {
        let p = f.value_pool[f.blocks[entry].params.start as usize + k];
        let keep = match f.insts[p].kind {
            InstKind::BlockParam(r) => r == RSP || SYSV_ARGS[..sig.args as usize].contains(&r) || r >= STACK_ARG_BASE,
            _ => true,
        };
        if keep {
            k += 1;
            continue;
        }
        let u = undef(f, &mut undefs);
        map.push((p, u));
        let mut params = f.blocks[entry].params;
        let pos = params.start as usize + k;
        list_remove(&mut f.value_pool, &mut params, pos);
        f.blocks[entry].params = params;
    }
    let mut repl: Vec<Option<ValueId>> = vec![None; f.insts.len()];
    for (from, to) in map {
        repl[from.index()] = Some(to);
    }
    prepend(f, entry, &undefs);
    rewrite_uses(f, &repl);
    // Storing an undefined value may as well leave memory as it was. These are
    // mostly pushes of callee-saved registers, and dropping them frees the slot.
    remove_insts(f, |f, id| matches!(f.insts[id].kind, InstKind::Store { val, .. } if matches!(f.insts[val].kind, InstKind::Undef)));

    clean(f);
    crate::frame::promote(f);
    clean(f);
    sites
}

/// Drop the instructions `gone` picks from every block's list.
fn remove_insts(f: &mut Function, gone: impl Fn(&Function, ValueId) -> bool) {
    for bi in 0..f.blocks.len() {
        let b = BlockId::new(bi);
        let l = f.blocks[b].insts;
        let mut w = l.start as usize;
        for r in l.start as usize..(l.start + l.len) as usize {
            let id = f.value_pool[r];
            if !gone(f, id) {
                f.value_pool[w] = id;
                w += 1;
            }
        }
        f.blocks[b].insts.len = (w - l.start as usize) as u32;
    }
}

/// A rax:rdx pair, placed at the end of block `b`.
fn pair(f: &mut Function, b: BlockId, rax: ValueId, rdx: ValueId) -> ValueId {
    let at = f.origin[rax.index()];
    let p = new_inst(f, InstKind::Aggregate { ty: TyId::PAIR, fields: ListRef::EMPTY }, TyId::PAIR, at);
    set_fields(f, p, rax, rdx);
    append(f, b, &[p]);
    p
}

fn set_fields(f: &mut Function, p: ValueId, rax: ValueId, rdx: ValueId) {
    let start = f.value_pool.len() as u32;
    f.value_pool.extend_from_slice(&[rax, rdx]);
    if let InstKind::Aggregate { fields, .. } = &mut f.insts[p].kind {
        *fields = ListRef { start, len: 2 };
    }
}

fn append(f: &mut Function, b: BlockId, vals: &[ValueId]) {
    let old = f.blocks[b].insts;
    let start = f.value_pool.len();
    f.value_pool.extend_from_within(old.start as usize..(old.start + old.len) as usize);
    f.value_pool.extend_from_slice(vals);
    f.blocks[b].insts = ListRef { start: start as u32, len: (f.value_pool.len() - start) as u32 };
}

/// Make sure the entry block has no predecessors (a function whose first
/// instruction is a loop head): add a new entry that jumps to the old one.
fn split_entry(f: &mut Function) {
    let old = f.entry;
    let has_preds = f.blocks.iter().any(|(_, b)| b.term.successors(&f.value_pool).any(|s| s == old));
    if !has_preds {
        return;
    }
    let at = f.blocks[old].insts.get(&f.value_pool).first().and_then(|v| f.origin.get(v.index())).copied().unwrap_or(0);
    let old_params: Vec<ValueId> = f.blocks[old].params.get(&f.value_pool).to_vec();
    let mut params = Vec::new();
    for &p in &old_params {
        let i = f.insts[p];
        params.push(new_inst(f, i.kind, i.ty, at));
    }
    let pstart = f.value_pool.len() as u32;
    f.value_pool.extend_from_slice(&params);
    let astart = f.value_pool.len() as u32;
    f.value_pool.extend_from_slice(&params);
    let n = params.len() as u32;
    let b = f.blocks.push(Block {
        insts: ListRef { start: astart + n, len: 0 },
        params: ListRef { start: pstart, len: n },
        term: Terminator::Jump { to: old, args: ListRef { start: astart, len: n } },
    });
    f.entry = b;
}

/// A new instruction, not yet in any block.
pub(crate) fn new_inst(f: &mut Function, kind: InstKind, ty: TyId, at: u64) -> ValueId {
    let id = f.insts.push(Inst { kind, ty });
    f.origin.push(at);
    id
}

/// Put `vals` at the start of block `b`.
pub(crate) fn prepend(f: &mut Function, b: BlockId, vals: &[ValueId]) {
    if vals.is_empty() {
        return;
    }
    let old = f.blocks[b].insts;
    let start = f.value_pool.len();
    f.value_pool.extend_from_slice(vals);
    f.value_pool.extend_from_within(old.start as usize..(old.start + old.len) as usize);
    f.blocks[b].insts = ListRef { start: start as u32, len: (f.value_pool.len() - start) as u32 };
}

/// New instruction lists for blocks that get values inserted before a call or
/// appended at the end (before a tail call).
fn rebuild_lists(
    f: &mut Function,
    before: &[(ValueId, Vec<ValueId>)],
    at_end: &[(BlockId, Vec<ValueId>)],
) -> Vec<(BlockId, ListRef)> {
    let mut out = Vec::new();
    if before.is_empty() && at_end.is_empty() {
        return out;
    }
    for bi in 0..f.blocks.len() {
        let b = BlockId::new(bi);
        let old = f.blocks[b].insts;
        let list: Vec<ValueId> = old.get(&f.value_pool).to_vec();
        let tail = at_end.iter().find(|(x, _)| *x == b);
        if tail.is_none() && !list.iter().any(|id| before.iter().any(|(a, _)| a == id)) {
            continue;
        }
        let start = f.value_pool.len();
        for id in list {
            if let Some((_, vs)) = before.iter().find(|(a, _)| *a == id) {
                f.value_pool.extend_from_slice(vs);
            }
            f.value_pool.push(id);
        }
        if let Some((_, vs)) = tail {
            f.value_pool.extend_from_slice(vs);
        }
        out.push((b, ListRef { start: start as u32, len: (f.value_pool.len() - start) as u32 }));
    }
    out
}

/// After `apply`: the entry parameter that holds argument `k` (0-5 in registers,
/// 6+ on the stack), if the function uses it.
pub fn arg_param(f: &Function, k: usize) -> Option<ValueId> {
    let reg = if k < 6 { SYSV_ARGS[k] } else { STACK_ARG_BASE + (k - 6) as u8 };
    f.blocks[f.entry]
        .params
        .get(&f.value_pool)
        .iter()
        .copied()
        .find(|&p| matches!(f.insts[p].kind, InstKind::BlockParam(r) if r == reg))
}
