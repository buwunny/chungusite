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
//! * **Float arguments and results.** Float argument `j` is xmm `j`'s low half
//!   (`XMM_PARAM + 2j` on entry), a parameter if live, prefix-closed like the
//!   integer ones. A function returns a float (in xmm0) if xmm0 is defined at
//!   every return and either some caller reads xmm0 after calling it, or at every
//!   return xmm0 holds a float the function computed (or got from a float call,
//!   or a float argument), with a high half that isn't one (packed math is
//!   vector code), that it didn't store (a `void` function that stores a float
//!   it computed leaves it in xmm0 too), and that wins over rax if rax is
//!   defined too: the one computed only to be returned, else the one written
//!   last. Returns of a constant xmm0 don't count either way. `ret` is then set
//!   too; `fret` says the value is in xmm0.
//! * **Preserved xmm registers.** Like the general registers, per 64-bit half:
//!   gcc keeps floats in xmm registers across calls to functions it knows
//!   don't touch them. A function that names no xmm register keeps what all its
//!   callees keep (the lifter doesn't track xmm registers through it).
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
use crate::lift::{CALL_ARGS, CALL_REGS, CALL_XMM, EXIT_REGS, EXIT_XMM0, FLOAT_ARGS, XMM0, XMM_PARAM};
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
    /// Returns a value: in rax, or in xmm0 if `fret`.
    pub ret: bool,
    /// The value it returns is a float, in xmm0.
    pub fret: bool,
    /// Float arguments: the low halves of the first `fargs` of xmm0-7.
    pub fargs: u8,
    /// Also returns a value in rdx: the result is 16 bytes, rax:rdx.
    pub ret2: bool,
    /// Takes more arguments than `args` (printf). Callers pass what they set up.
    pub variadic: bool,
    /// Caller-saved registers (bit = x86 number) the function leaves as it found them.
    pub preserves: u16,
    /// xmm register halves (bit = 2 * xmm + high) it leaves as it found them.
    pub xpreserves: u32,
    /// At one call to a variadic function of the program: how many of the
    /// last integer, float and stack arguments the call doesn't set up,
    /// passed as nothing rather than whatever the registers or the stack
    /// held (`site_sig`).
    pub unset: (u8, u8, u8),
}

impl Sig {
    /// Does the function leave register `reg` (x86 number) as it found it?
    pub fn keeps(self, reg: u8) -> bool {
        match reg {
            0..16 => self.preserves & (1 << reg) != 0,
            _ => reg.checked_sub(XMM_PARAM).is_some_and(|k| k < 32 && self.xpreserves & (1 << k) != 0),
        }
    }

    /// Returns a value in rax.
    pub fn rax(self) -> bool {
        self.ret && !self.fret
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
        let k = match reg.checked_sub(XMM_PARAM) {
            Some(x) if x < 32 => CALL_XMM + x as usize,
            _ => CALL_REGS.iter().position(|r| r.number() as u8 == reg)?,
        };
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
/// the function set up (`set_for`; prefix-closed). Valid on lifter output, where a
/// call's arguments are `CALL_ARGS` long.
///
/// `keeps(call, reg)`: does the direct `call` leave `reg` as it was? A leaf
/// helper gcc's interprocedural register allocation knows keeps a register
/// lets a value set up before it reach the next call (`keeps_alone`).
pub fn guess_args(f: &Function, site: Site, keeps: &dyn Fn(ValueId, u8) -> bool) -> u8 {
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
    let mut n = 0;
    for (k, &a) in args[..6].iter().enumerate() {
        if set_for(f, block, a, Some(CALL_REGS[k].number() as u8), keeps) {
            n = k + 1;
        }
    }
    n as u8
}

/// The argument registers a call to unknown code (`guess_args`) passes on
/// unchanged from the function's own entry, as a bit per register: a wrapper
/// like `malloc(n) { return hooks.malloc(n); }` sets nothing up, so
/// `guess_args` gives it none. Whether these are arguments is up to the
/// function's callers (`Program::build`).
pub fn passed_through(f: &Function, site: Site) -> u8 {
    let (_, args) = site.parts(f);
    let args = args.get(&f.value_pool);
    if args.len() != CALL_ARGS {
        return 0;
    }
    let mut out = 0;
    for (k, &a) in args[..6].iter().enumerate() {
        if from_entry(f, a, CALL_REGS[k].number() as u8, &mut Vec::new()) {
            out |= 1 << k;
        }
    }
    out
}

/// Is `a` the entry's value of `reg` on every path (through block parameters
/// of the same register; a loop's way back adds nothing)?
fn from_entry(f: &Function, a: ValueId, reg: u8, seen: &mut Vec<ValueId>) -> bool {
    let InstKind::BlockParam(r) = f.insts[a].kind else { return false };
    if r != reg {
        return false;
    }
    if f.blocks[f.entry].params.get(&f.value_pool).contains(&a) {
        return true;
    }
    if seen.contains(&a) {
        return true;
    }
    if seen.len() >= 16 {
        return false;
    }
    seen.push(a);
    let Some((b, k)) = f.blocks.iter().find_map(|(b, blk)| {
        blk.params.get(&f.value_pool).iter().position(|&p| p == a).map(|k| (b, k))
    }) else { return false };
    let (mut paths, mut all) = (0, true);
    for (p, blk) in f.blocks.iter() {
        if blk.term.successors(&f.value_pool).any(|s| s == b) {
            incoming(f, p, b, k, |v| {
                paths += 1;
                all = all && from_entry(f, v, reg, seen);
            });
        }
    }
    paths > 0 && all
}

/// Did the function set `a` up for a call in `block`, in register `reg`:
/// computed there, computed before (in a block that dominates it), or moved
/// there from another register, on every path, rather than left in `reg` by
/// the caller or a callee. `free(opaque, p)` through a pointer often loads
/// `p` before a null check and `opaque` after it; `usage(name)` passes `name`
/// on to `fprintf` in r8.
fn set_for(f: &Function, block: BlockId, a: ValueId, reg: Option<u8>, keeps: &dyn Fn(ValueId, u8) -> bool) -> bool {
    match f.insts[a].kind {
        // paths joining, each with what it set up (a float computed on one
        // path, reloaded from the stack after a call on the other)
        InstKind::BlockParam(_) => set_before(f, a, reg, keeps, &mut Vec::new()),
        InstKind::Param(_) => f.blocks[block].insts.get(&f.value_pool).contains(&a),
        _ => set_before(f, a, reg, keeps, &mut Vec::new()),
    }
}

/// `set_for` of a value computed before the call's block. A block parameter
/// (paths joining, each with the arguments it set up, as in
/// `fprintf(stderr, fmt, name, why)` after two error checks) counts if every
/// path into it set its register up; a loop's way back adds nothing.
fn set_before(f: &Function, a: ValueId, reg: Option<u8>, keeps: &dyn Fn(ValueId, u8) -> bool, seen: &mut Vec<ValueId>) -> bool {
    match f.insts[a].kind {
        // another register's value across a call (`mov r8, r10` with r10
        // kept by a callee gcc knows) is set up; the register's own isn't
        InstKind::CallOut { reg: r, .. } if reg.is_some_and(|reg| reg != r) => true,
        // the register's own value across a call that keeps it, set up before:
        // gcc sets r9 for a function pointer, then calls a helper it knows
        // leaves r9 alone
        InstKind::CallOut { call, reg: r } if keeps(call, r) => {
            Site::Call(call).before(f, r).is_some_and(|v| set_before(f, v, reg, keeps, seen))
        }
        InstKind::Undef | InstKind::CallOut { .. } | InstKind::Param(_) => false,
        InstKind::BlockParam(r) if reg.is_some_and(|reg| reg != r) => true,
        InstKind::BlockParam(_) => {
            if seen.contains(&a) {
                return true;
            }
            if seen.len() >= 16 {
                return false;
            }
            seen.push(a);
            let Some((b, k)) = f.blocks.iter().find_map(|(b, blk)| {
                blk.params.get(&f.value_pool).iter().position(|&p| p == a).map(|k| (b, k))
            }) else { return false };
            let (mut paths, mut all) = (0, true);
            for (p, blk) in f.blocks.iter() {
                if blk.term.successors(&f.value_pool).any(|s| s == b) {
                    incoming(f, p, b, k, |v| {
                        paths += 1;
                        all = all && set_before(f, v, reg, keeps, seen);
                    });
                }
            }
            // a block nothing jumps to (padding after a jump, lifted as code)
            // brings nothing
            all && (paths > 0 || b != f.entry)
        }
        _ => true,
    }
}

/// How many 8-byte stack arguments each call that sets up all six argument
/// registers stores just before it, at `[rsp]`, `[rsp+8]`, ... in its own
/// block: the seventh and later arguments of a variadic call (`printf`), or of
/// a function pointer. Registers the function passes on from its own entry
/// count as set up (`passed_through`): liblzma's `block_encode` forwards
/// `out` and `out_pos` in r8 and r9 and pushes the rest.
pub fn guess_stack(f: &Function, sites: &[Site], args: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; sites.len()];
    let six = |k: usize, site: Site| {
        let list = site.parts(f).1.get(&f.value_pool);
        list.len() == CALL_ARGS
            && (args[k] as usize..6).all(|r| from_entry(f, list[r], CALL_REGS[r].number() as u8, &mut Vec::new()))
    };
    if !sites.iter().enumerate().any(|(k, &s)| six(k, s)) {
        return out;
    }
    let a = analyze(f);
    for (k, &site) in sites.iter().enumerate() {
        let Site::Call(id) = site else { continue };
        let list = site.parts(f).1.get(&f.value_pool);
        if !six(k, site) {
            continue;
        }
        let rsp = a.origin[list[6].index()];
        let (Off::Known(base), true) = (rsp.off, rsp.roots != 0) else { continue };
        let Some((b, blk)) = f.blocks.iter().map(|(b, blk)| (b, blk.insts.get(&f.value_pool))).find(|(_, i)| i.contains(&id)) else { continue };
        let words = |insts: &[ValueId], written: &mut [bool; MAX_STACK_ARGS as usize]| {
            for &i in insts {
                if let InstKind::Store { ptr, val, .. } = f.insts[i].kind {
                    // not the prologue saving a callee-saved register (`push rbx`
                    // right before a call in the entry block; rbx, rbp, r12-r15)
                    if [3u8, 5, 12, 13, 14, 15].into_iter().any(|r| from_entry(f, val, r, &mut Vec::new())) {
                        continue;
                    }
                    let o = a.origin[ptr.index()];
                    if let (Off::Known(off), true) = (o.off, o.roots == rsp.roots) {
                        if (0..8 * MAX_STACK_ARGS).contains(&(off - base)) {
                            written[((off - base) / 8) as usize] = true;
                        }
                    }
                }
            }
        };
        let mut written = [false; MAX_STACK_ARGS as usize];
        words(&blk[..blk.iter().position(|&i| i == id).unwrap()], &mut written);
        // and what every block jumping to it stored (gcc shares one `call
        // fprintf` between paths that each push their own arguments)
        let mut joined: Option<[bool; MAX_STACK_ARGS as usize]> = None;
        for (_, p) in f.blocks.iter().filter(|(_, p)| p.term.successors(&f.value_pool).any(|s| s == b)) {
            let mut w = [false; MAX_STACK_ARGS as usize];
            words(p.insts.get(&f.value_pool), &mut w);
            joined = Some(joined.map_or(w, |j| std::array::from_fn(|n| j[n] && w[n])));
        }
        if b != f.entry {
            for (w, j) in written.iter_mut().zip(joined.unwrap_or([false; MAX_STACK_ARGS as usize])) {
                *w |= j;
            }
        }
        out[k] = written.iter().take_while(|&&w| w).count() as u8;
    }
    out
}

/// How many float arguments a call to unknown code passes: up to the last xmm
/// register the function set up, like `guess_args`.
pub fn guess_fargs(f: &Function, site: Site) -> u8 {
    let (_, args) = site.parts(f);
    let args = args.get(&f.value_pool);
    if args.len() != CALL_ARGS {
        return 0;
    }
    let block = match site {
        Site::Call(id) => f.blocks.iter().find(|(_, b)| b.insts.get(&f.value_pool).contains(&id)).map(|(b, _)| b),
        Site::Tail(b) => Some(b),
    };
    let Some(block) = block else { return 0 };
    // a variadic call says in al how many xmm registers it passes
    if let Some(n) = al_constant(f, args[7]).filter(|&n| n <= FLOAT_ARGS as u64) {
        return n as u8;
    }
    let mut n = 0;
    for (j, &a) in args[CALL_XMM..].iter().step_by(2).take(FLOAT_ARGS).enumerate() {
        if set_for(f, block, a, None, &|_, _| false) {
            n = j + 1;
        }
    }
    n as u8
}

/// The call site whose xmm0 a call to unknown code passes on in xmm0, if
/// any: `exp(luaL_checknumber(L, 1))` passes a float argument, if that
/// callee returns one (`Program::build` decides).
pub fn passes_xmm0(f: &Function, sites: &[Site], site: Site) -> Option<usize> {
    let (_, args) = site.parts(f);
    let args = args.get(&f.value_pool);
    let &a = args.get(CALL_XMM)?;
    let InstKind::CallOut { call, reg } = f.insts[a].kind else { return None };
    (reg == XMM0).then(|| sites.iter().position(|&s| s == Site::Call(call))).flatten()
}

/// The low byte of `v` (rax at a call), if the function set it to a constant:
/// `mov eax, 1`, `xor eax, eax`, or `mov al, 2` over whatever rax held.
fn al_constant(f: &Function, v: ValueId) -> Option<u64> {
    let k = |v: ValueId| match f.insts[v].kind {
        InstKind::Const(c) => Some(f.consts[c.index()] as u64 & 0xff),
        InstKind::Cast { v: x, .. } => match f.insts[x].kind {
            InstKind::Const(c) => Some(f.consts[c.index()] as u64 & 0xff),
            _ => None,
        },
        _ => None,
    };
    match f.insts[v].kind {
        InstKind::Bin { op: BinOp::Or, lhs, rhs } => {
            let masked = |x: ValueId| matches!(f.insts[x].kind, InstKind::Bin { op: BinOp::And, rhs: m, .. }
                if matches!(f.insts[m].kind, InstKind::Const(c) if f.consts[c.index()] as u64 & 0xff == 0));
            if masked(lhs) { k(rhs) } else if masked(rhs) { k(lhs) } else { None }
        }
        _ => k(v),
    }
}

/// What `infer` found besides the signature: what each call site reads after
/// the call that the callee may return (`READ_RDX`: evidence that it returns 16
/// bytes, `READ_XMM0`: that it returns a float, `READ_RAX`: that it returns a
/// value at all).
pub struct Inferred {
    pub sig: Sig,
    pub reads: Vec<u8>,
}

pub const READ_RDX: u8 = 1;
pub const READ_XMM0: u8 = 2;
pub const READ_RAX: u8 = 4;
/// Not a read: the function has callers in the program (so `READ_RAX` means something).
pub const CALLED: u8 = 8;
/// Not a read either: some caller tail-calls the function, passing on rax or
/// xmm0, whichever it returns.
pub const TAIL_CALLED: u8 = 16;

/// The call sites that never return: the last call of a block the lifter ended
/// with `Unreachable` (a call to `abort` or a panic function, or one before a trap).
pub fn noreturn_sites(f: &Function, sites: &[Site]) -> Vec<bool> {
    let mut cold = vec![false; sites.len()];
    for (_, blk) in f.blocks.iter() {
        if !matches!(blk.term, Terminator::Unreachable) {
            continue;
        }
        let last = blk.insts.get(&f.value_pool).iter().rev().find(|&&id| matches!(f.insts[id].kind, InstKind::Call { .. }));
        if let Some(k) = last.and_then(|&id| sites.iter().position(|&x| x == Site::Call(id))) {
            cold[k] = true;
        }
    }
    cold
}

/// Where a function's control leaves it.
enum Exit {
    /// A return, with rax and the `Exit` registers (then xmm0) if the lifter recorded them.
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

/// The registers (indexed by number, xmm halves from `XMM_PARAM`) that some
/// exit reached from where a call that never returns would have fallen through
/// (`Function::noreturn_falls`) leaves as they were before that point: defined
/// outside the code that follows. Before such calls ended their blocks, that
/// path reached the exit with the register clobbered by the call, so it wasn't
/// preserved, and rax there wasn't a result unless the code after recomputed it.
fn stale_regs(f: &Function, cfg: &Cfg, sites: &[Site], callee: &dyn Fn(usize) -> Sig) -> Vec<bool> {
    let mut stale = vec![false; 256];
    if f.noreturn_falls.is_empty() {
        return stale;
    }
    let mut def = vec![None; f.insts.len()];
    for (b, blk) in f.blocks.iter() {
        for &v in blk.params.get(&f.value_pool).iter().chain(blk.insts.get(&f.value_pool)) {
            def[v.index()] = Some(b);
        }
    }
    let regs: Vec<u8> = EXIT_REGS.iter().map(|r| r.number() as u8).chain((0..32).map(|x| XMM_PARAM + x)).chain([RAX]).collect();
    for &(_, from) in &f.noreturn_falls {
        let mut region = vec![false; f.blocks.len()];
        let mut work = vec![from];
        while let Some(b) = work.pop() {
            if !std::mem::replace(&mut region[b.index()], true) {
                work.extend(f.blocks[b].term.successors(&f.value_pool));
            }
        }
        let outside = |v: ValueId| def[v.index()].is_none_or(|b| !region[b.index()]);
        // Block parameters that would have merged the clobbered value: those
        // of `from`, and those fed one on an edge inside the region.
        let mut merged = vec![false; f.insts.len()];
        for &p in f.blocks[from].params.get(&f.value_pool) {
            merged[p.index()] = true;
        }
        let mut changed = true;
        while changed {
            changed = false;
            for (b, blk) in f.blocks.iter() {
                if !region[b.index()] || b == from {
                    continue;
                }
                for (k, &p) in blk.params.get(&f.value_pool).iter().enumerate() {
                    if merged[p.index()] {
                        continue;
                    }
                    let mut m = false;
                    for &q in cfg.preds(b).iter().filter(|q| region[q.index()]) {
                        incoming(f, q, b, k, |a| m |= merged[a.index()] || outside(a));
                    }
                    if m {
                        merged[p.index()] = true;
                        changed = true;
                    }
                }
            }
        }
        let outside = |v: ValueId| outside(v) || merged[v.index()];
        for (b, blk) in f.blocks.iter() {
            if !region[b.index()] {
                continue;
            }
            let e = match blk.term {
                Terminator::Return(Some(rax)) => {
                    let regs = blk.insts.get(&f.value_pool).iter().rev().find_map(|&id| match f.insts[id].kind {
                        InstKind::Exit { regs } => Some(regs),
                        _ => None,
                    });
                    Exit::Return { rax, regs }
                }
                Terminator::TailCall { .. } => match sites.iter().position(|&s| s == Site::Tail(b)) {
                    Some(k) => Exit::Tail(k),
                    None => continue,
                },
                _ => continue,
            };
            for &r in &regs {
                // after a tail call rax is the callee's result, recomputed
                let tail_rax = r == RAX && matches!(e, Exit::Tail(_));
                if !tail_rax && exit_value(f, sites, callee, &e, r).is_some_and(outside) {
                    stale[r as usize] = true;
                }
            }
        }
    }
    stale
}

/// Where a value comes from, as far as register summaries care.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Src {
    /// Not computed yet, or in a block nothing reaches.
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
    // (values in blocks nothing reaches stay `Top`: no path brings them)
    for &b in &cfg.rpo {
        for &id in f.blocks[b].insts.get(&f.value_pool) {
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
fn traces_to_entry(f: &Function, src: &[Src], sites: &[Site], callee: &dyn Fn(usize) -> Sig, reg: u8, v: ValueId) -> bool {
    traces(f, src, sites, callee, reg, v, &mut Vec::new())
}

/// `traces_to_entry`, through paths that join with different sources (the
/// entry value on one, a call that keeps it on another; a loop's way back
/// adds nothing).
fn traces(f: &Function, src: &[Src], sites: &[Site], callee: &dyn Fn(usize) -> Sig, reg: u8, mut v: ValueId, seen: &mut Vec<ValueId>) -> bool {
    for _ in 0..64 {
        match src[v.index()] {
            Src::Entry(r) => return r == reg,
            Src::After(k, r) if r == reg && callee(k).keeps(r) => match sites[k].before(f, r) {
                Some(b) => v = b,
                None => return false,
            },
            Src::Other if matches!(f.insts[v].kind, InstKind::BlockParam(r) if r == reg) => {
                if seen.contains(&v) {
                    return true;
                }
                if seen.len() >= 32 {
                    return false;
                }
                seen.push(v);
                let Some((b, k)) = f.blocks.iter().find_map(|(b, blk)| {
                    blk.params.get(&f.value_pool).iter().position(|&p| p == v).map(|k| (b, k))
                }) else { return false };
                let (mut paths, mut all) = (0, true);
                for (p, blk) in f.blocks.iter() {
                    if all && blk.term.successors(&f.value_pool).any(|s| s == b) {
                        incoming(f, p, b, k, |a| {
                            if src[a.index()] == Src::Top {
                                return; // from a block nothing reaches
                            }
                            paths += 1;
                            all = all && traces(f, src, sites, callee, reg, a, seen);
                        });
                    }
                }
                return paths > 0 && all;
            }
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
            let k = match reg.checked_sub(XMM_PARAM) {
                Some(x) if x < 32 => EXIT_XMM0 + x as usize,
                _ => EXIT_REGS.iter().position(|r| r.number() as u8 == reg)?,
            };
            regs?.get(&f.value_pool).get(k).copied()
        }
        Exit::Tail(k) => {
            // after `jmp g`, reg is what g leaves in it
            if callee(k).keeps(reg) { sites[k].before(f, reg) } else { None }
        }
    }
}

/// The caller-saved registers `f` (as for `infer`) keeps without help from
/// any callee, as a mask of x86 numbers: what a helper that calls nothing
/// leaves alone, for `guess_args` before signatures are known.
pub fn keeps_alone(f: &Function) -> u16 {
    let sites = sites(f);
    let callee = |_: usize| Sig::default();
    let cfg = Cfg::new(f);
    let stale = stale_regs(f, &cfg, &sites, &callee);
    let ex = exits(f, &cfg, &sites);
    let src = sources(f, &cfg, &sites);
    let mut keeps = 0u16;
    for reg in 0..16u8 {
        if CALLER_SAVED & (1 << reg) == 0 || stale[reg as usize] || ex.is_empty() {
            continue;
        }
        let ok = ex.iter().all(|e| match e {
            Exit::Tail(_) => false,
            e => exit_value(f, &sites, &callee, e, reg).is_some_and(|v| traces_to_entry(f, &src, &sites, &callee, reg, v)),
        });
        if ok {
            keeps |= 1 << reg;
        }
    }
    keeps
}

/// The signature of `f` (lifted with `track_exits`, cleaned, not yet `apply`d),
/// given the signature of the callee at each call site (`callee(i)` for
/// `sites[i]`), the function's own signature from the previous round, and what
/// its callers read after calling it (`READ_RDX | READ_XMM0 | READ_RAX`, and
/// `CALLED` if it has callers). `set_args` is the most argument registers any
/// caller sets up (6 without callers). `stack_args` is left 0; see `stack_args`.
///
/// A path that ends in a call that never returns is evidence of neither a
/// result nor an argument, so around such calls the callers decide. If rax at
/// a return reached from where the call would have fallen through is still
/// what it was before that point (`stale_regs`; unoptimized code leaves
/// whatever it last computed there), it is a result only if a caller reads it,
/// and such registers aren't preserved, as when the call fell through. A
/// register that is live only because a call that never returns takes it
/// (`assert_failed`'s unused `Option<Arguments>` payload) is an argument only
/// up to what the callers set.
pub fn infer(f: &Function, sites: &[Site], callee: &dyn Fn(usize) -> Sig, prev: Sig, wanted: u8, set_args: u8) -> Inferred {
    let cold = noreturn_sites(f, sites);
    let has_cold = cold.contains(&true);
    let cfg = Cfg::new(f);
    let stale = stale_regs(f, &cfg, sites, callee);
    let ex = exits(f, &cfg, sites);
    let src = sources(f, &cfg, sites);

    // ---- preserved registers ----
    // (a function that never returns keeps every register, vacuously: the
    // `.cold` part of a function that traps is a tail call it makes)
    let mut preserves = 0u16;
    for reg in 0..16u8 {
        if CALLER_SAVED & (1 << reg) == 0 {
            continue;
        }
        let ok = ex.iter().all(|e| match e {
            Exit::Tail(k) if !callee(*k).keeps(reg) => false,
            e => exit_value(f, sites, callee, e, reg).is_some_and(|v| traces_to_entry(f, &src, sites, callee, reg, v)),
        });
        if ok && !stale[reg as usize] {
            preserves |= 1 << reg;
        }
    }
    // the same for xmm registers (gcc keeps floats in them across calls too).
    // A return that lists no xmm registers is in a function that doesn't touch
    // them: it keeps what all its callees keep.
    let mut xpreserves = 0u32;
    for k in 0..32u8 {
        let reg = XMM_PARAM + k;
        let ok = ex.iter().all(|e| match e {
            Exit::Tail(t) if !callee(*t).keeps(reg) => false,
            Exit::Return { regs: Some(l), .. } if l.len as usize == EXIT_XMM0 => (0..sites.len()).all(|t| callee(t).keeps(reg)),
            e => exit_value(f, sites, callee, e, reg).is_some_and(|v| traces_to_entry(f, &src, sites, callee, reg, v)),
        });
        if ok && !stale[reg as usize] {
            xpreserves |= 1 << k;
        }
    }

    // ---- return value: is rax, or xmm0, defined at every exit? ----
    let undef = undefined_values(f, &cfg, sites, callee, false, prev.fargs);
    let defined_at = |undef: &[bool], reg: u8, e: &Exit| match e {
        Exit::Tail(k) => {
            let c = callee(*k);
            match reg {
                RAX if c.rax() => true,
                RDX if c.ret2 => true,
                XMM0 => c.fret,
                _ => c.keeps(reg) && sites[*k].before(f, reg).is_some_and(|v| !undef[v.index()]),
            }
        }
        e => exit_value(f, sites, callee, e, reg).is_some_and(|v| !undef[v.index()]),
    };
    let rax = !ex.is_empty()
        && ex.iter().all(|e| defined_at(&undef, RAX, e))
        && (!stale[RAX as usize] || wanted & CALLED == 0 || wanted & READ_RAX != 0);
    // Callers reading only xmm0 say it's a float, only rax an integer.
    let (read_x, read_r, tail) = (wanted & READ_XMM0 != 0, wanted & READ_RAX != 0, wanted & TAIL_CALLED != 0);
    let fret = !ex.is_empty() && ex.iter().all(|e| defined_at(&undef, XMM0, e)) && (read_x && !read_r || (read_x || !read_r || tail) && {
        // Otherwise (both, a tail call, no callers) each exit votes: a float,
        // an integer, or nothing (xmm0 is a constant, like the 0.0 of an empty sum).
        let fl = floats(f, &cfg, sites, callee, prev.fargs);
        let spill = spills(f);
        let (stored, used) = (stored_values(f, &spill), used_values(f, &spill));
        let votes: Vec<Option<bool>> = ex
            .iter()
            .map(|e| match *e {
                Exit::Tail(k) => Some(callee(k).fret),
                Exit::Return { rax: r, regs } => {
                    let x = exit_value(f, sites, callee, e, XMM0)?;
                    if matches!(f.insts[x].kind, InstKind::Const(_)) {
                        return None;
                    }
                    let hi = regs.and_then(|l| l.get(&f.value_pool).get(EXIT_XMM0 + 1).copied());
                    let packed = hi.is_some_and(|h| fl[h.index()]);
                    let float = fl[x.index()] && !stored[x.index()] && !packed;
                    // of two candidates, the one the function computes only to
                    // return (the other also feeds a compare, an address, ...),
                    // else the one written last, the float on a tie (two loop
                    // carried values)
                    let rax_too = !undef[r.index()];
                    // rax a value (widened) the function also stores, like
                    // `*ok = flag` leaving the flag in eax: a byproduct
                    let (mut s, mut kept) = (r, false);
                    while let InstKind::Cast { kind: CastKind::ZExt | CastKind::SExt | CastKind::Trunc, v } = f.insts[s].kind {
                        s = v;
                        kept |= stored[s.index()];
                    }
                    let used_r = used[r.index()] || kept;
                    Some(float && (!rax_too || match (used[x.index()], used_r) {
                        (false, true) => true,
                        (true, false) => false,
                        _ => f.origin[x.index()] >= f.origin[r.index()],
                    }))
                }
            })
            .collect();
        votes.contains(&Some(true)) && !votes.contains(&Some(false))
    });
    let ret = rax || fret;

    // ---- arguments: which entry registers are live ----
    let mut reads = vec![0u8; sites.len()];
    let live = live_values(f, sites, callee, ret && !fret, prev.ret2, fret, &[], &mut reads);
    // A call whose rdx the function returns as it is, when callers read rdx
    // after calling the function, has its rdx read too: `JS_Eval` returns what
    // `JS_EvalThis2` returns after its stack check, a 16-byte `JSValue`. So
    // a 16-byte result reaches through wrappers, not only through tail calls.
    if rax && wanted & READ_RDX != 0 {
        for e in &ex {
            let Some(v) = exit_value(f, sites, callee, e, RDX) else { continue };
            if let InstKind::CallOut { call, reg: RDX } = f.insts[v].kind {
                if let Some(k) = sites.iter().position(|&s| s == Site::Call(call)) {
                    reads[k] |= READ_RDX;
                }
            }
        }
    }
    let entry_args = |live: &[bool]| {
        let (mut args, mut fargs) = (0, 0);
        for &p in f.blocks[f.entry].params.get(&f.value_pool) {
            if let InstKind::BlockParam(r) = f.insts[p].kind {
                if !live[p.index()] {
                    continue;
                }
                if let Some(k) = SYSV_ARGS.iter().position(|&a| a == r) {
                    args = args.max(k as u8 + 1);
                }
                if let Some(j) = float_arg(r) {
                    fargs = fargs.max(j + 1);
                }
            }
        }
        (args, fargs)
    };
    let (mut args, fargs) = entry_args(&live);
    if has_cold && args > set_args {
        // without what only the calls that don't return read
        let warm = live_values(f, sites, callee, ret && !fret, prev.ret2, fret, &cold, &mut vec![0u8; sites.len()]);
        args = entry_args(&warm).0.max(set_args);
    }

    // ---- rdx: returned too? ----
    let ret2 = rax && !fret && wanted & READ_RDX != 0 && preserves & (1 << RDX) == 0 && {
        let undef = undefined_values(f, &cfg, sites, callee, args < 3, prev.fargs);
        ex.iter().all(|e| defined_at(&undef, RDX, e))
    };
    Inferred { sig: Sig { args, fargs, stack_args: 0, ret, fret, ret2, variadic: f.variadic, preserves, xpreserves, unset: (0, 0, 0) }, reads }
}

/// Values that an instruction other than a call, an `Exit` or a spill to the
/// frame, or a branch, uses: what the function computes with, rather than
/// passes along.
fn used_values(f: &Function, spill: &[bool]) -> Vec<bool> {
    let mut used = vec![false; f.insts.len()];
    for (_, blk) in f.blocks.iter() {
        for &id in blk.insts.get(&f.value_pool) {
            match f.insts[id].kind {
                InstKind::Call { .. } | InstKind::Exit { .. } => {}
                _ if spill[id.index()] => {}
                k => for_each_operand(k, f, |v| used[v.index()] = true),
            }
        }
        match blk.term {
            Terminator::Branch { c, .. } => used[c.index()] = true,
            Terminator::Switch { v, .. } => used[v.index()] = true,
            _ => {}
        }
    }
    used
}

/// Stores to the function's own frame, below the return address: values
/// spilled around a call rather than handed to anyone.
fn spills(f: &Function) -> Vec<bool> {
    let mut spill = vec![false; f.insts.len()];
    let entry = f.blocks[f.entry].params.get(&f.value_pool);
    let Some(k) = entry.iter().position(|&p| matches!(f.insts[p].kind, InstKind::BlockParam(RSP))) else { return spill };
    let a = analyze(f);
    for (_, blk) in f.blocks.iter() {
        for &id in blk.insts.get(&f.value_pool) {
            if let InstKind::Store { ptr, .. } = f.insts[id].kind {
                let o = a.origin[ptr.index()];
                spill[id.index()] = o.roots == 1u128 << k && matches!(o.off, Off::Known(off) if off < 0);
            }
        }
    }
    spill
}

/// Values stored to memory, other than spills.
fn stored_values(f: &Function, spill: &[bool]) -> Vec<bool> {
    let mut stored = vec![false; f.insts.len()];
    for (_, blk) in f.blocks.iter() {
        for &id in blk.insts.get(&f.value_pool) {
            if let InstKind::Store { val, .. } = f.insts[id].kind {
                stored[val.index()] |= !spill[id.index()];
            }
        }
    }
    stored
}

/// Float argument `j` if `reg` is its `BlockParam` number.
fn float_arg(reg: u8) -> Option<u8> {
    let k = reg.checked_sub(XMM_PARAM)?;
    (k % 2 == 0 && ((k / 2) as usize) < FLOAT_ARGS).then_some(k / 2)
}

/// Values that are float bit patterns the function computed: float lane
/// arithmetic and conversions, a float call's result, the first `fargs` float
/// arguments, and bitwise ops, selects and block parameters that pass one on
/// (`andpd` for `fabs`, the merge of a scalar result into its register).
fn floats(f: &Function, cfg: &Cfg, sites: &[Site], callee: &dyn Fn(usize) -> Sig, fargs: u8) -> Vec<bool> {
    let mut fl = vec![false; f.insts.len()];
    for &p in f.blocks[f.entry].params.get(&f.value_pool) {
        if let InstKind::BlockParam(r) = f.insts[p].kind {
            fl[p.index()] = float_arg(r).is_some_and(|j| j < fargs);
        }
    }
    let site_of = |call: ValueId| sites.iter().position(|&x| x == Site::Call(call));
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &cfg.rpo {
            let blk = &f.blocks[b];
            if b != f.entry {
                for (k, &p) in blk.params.get(&f.value_pool).iter().enumerate() {
                    let mut any = false;
                    for &pred in cfg.preds(b) {
                        incoming(f, pred, b, k, |a| any |= fl[a.index()]);
                    }
                    if any && !fl[p.index()] {
                        fl[p.index()] = true;
                        changed = true;
                    }
                }
            }
            for &id in blk.insts.get(&f.value_pool) {
                let now = match f.insts[id].kind {
                    InstKind::Bin { op: BinOp::Lane(op, _), .. } => op.makes_float(),
                    InstKind::Un { op: UnOp::Lane(op, _), .. } => op.makes_float(),
                    InstKind::Bin { op: BinOp::And | BinOp::Or | BinOp::Xor, lhs, rhs } => fl[lhs.index()] || fl[rhs.index()],
                    InstKind::Select { t, f: e, .. } => fl[t.index()] || fl[e.index()],
                    InstKind::CallOut { call, reg: XMM0 } => site_of(call).is_some_and(|k| callee(k).fret),
                    _ => false,
                };
                if now && !fl[id.index()] {
                    fl[id.index()] = true;
                    changed = true;
                }
            }
        }
    }
    fl
}

/// Values that may be undefined: rax on entry (and rdx, if `rdx_undef`: it isn't
/// an argument; and xmm registers but the first `fargs` float arguments),
/// `Undef`, registers after a call that the callee neither returns nor preserves,
/// and block parameters that merge any of those in.
fn undefined_values(f: &Function, cfg: &Cfg, sites: &[Site], callee: &dyn Fn(usize) -> Sig, rdx_undef: bool, fargs: u8) -> Vec<bool> {
    let mut u = vec![false; f.insts.len()];
    for &p in f.blocks[f.entry].params.get(&f.value_pool) {
        u[p.index()] = match f.insts[p].kind {
            InstKind::BlockParam(RAX) => true,
            InstKind::BlockParam(RDX) => rdx_undef,
            InstKind::BlockParam(r) if r >= XMM_PARAM => float_arg(r).is_none_or(|j| j >= fargs),
            _ => false,
        };
    }
    let site_of = |call: ValueId| sites.iter().position(|&x| x == Site::Call(call));
    // register `reg` after call site `k`
    let after = |u: &[bool], k: Option<usize>, reg: u8| -> bool {
        let Some(k) = k else { return reg != RAX };
        let c = callee(k);
        if (reg == RAX && c.rax()) || (reg == RDX && c.ret2) || (reg == XMM0 && c.fret) {
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
    match f.blocks[p].term {
        Terminator::Jump { to, args } if to == b => cb(f.value_pool[args.start as usize + k]),
        Terminator::Branch { t, f: e, args, .. } => {
            if t == b {
                cb(f.value_pool[args.start as usize + k]);
            }
            if e == b {
                cb(f.value_pool[args.start as usize + f.blocks[t].params.len as usize + k]);
            }
        }
        _ => {}
    }
}

/// Liveness from the function's effects. A call only uses the arguments its callee
/// takes; a register after a call that the callee preserves uses its value before
/// the call. The return value counts only if the function returns rax (`ret`;
/// rdx at returns too, if it `ret2`s; xmm0 at returns instead if it `fret`s).
/// Marks in `reads` the call sites whose rax, rdx or xmm0 is read. The arguments
/// of the sites `skip` marks don't count.
#[allow(clippy::too_many_arguments)]
fn live_values(f: &Function, sites: &[Site], callee: &dyn Fn(usize) -> Sig, ret: bool, ret2: bool, fret: bool, skip: &[bool], reads: &mut [u8]) -> Vec<bool> {
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
        let k = sites.iter().position(|&x| x == s);
        if k.is_some_and(|k| skip.get(k) == Some(&true)) {
            return;
        }
        let c = k.map(callee).unwrap_or_default();
        for &a in &args[..(c.args.saturating_sub(c.unset.0) as usize).min(args.len()).min(6)] {
            mark(a, live, work);
        }
        if args.len() == CALL_ARGS {
            for &a in args[CALL_XMM..].iter().step_by(2).take(c.fargs.saturating_sub(c.unset.1) as usize) {
                mark(a, live, work);
            }
        }
    };
    for (b, blk) in f.blocks.iter() {
        for &id in blk.insts.get(&f.value_pool) {
            match f.insts[id].kind {
                InstKind::Call { .. } => call_uses(Site::Call(id), &mut live, &mut work),
                InstKind::Store { .. } | InstKind::MemCopy { .. } | InstKind::MemFill { .. } | InstKind::Opaque { .. }
                | InstKind::Load { volatile: true, .. } | InstKind::Assign { .. } => {
                    for_each_operand(f.insts[id].kind, f, |v| mark(v, &mut live, &mut work));
                }
                InstKind::Exit { regs } if ret2 => mark(regs.get(&f.value_pool)[1], &mut live, &mut work),
                InstKind::Exit { regs } if fret => mark(regs.get(&f.value_pool)[EXIT_XMM0], &mut live, &mut work),
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
                    reads[k] |= READ_RAX;
                    let c = callee(k);
                    if !c.rax() && c.keeps(RAX) {
                        ops.extend(sites[k].before(f, RAX));
                    }
                }
            }
            InstKind::CallOut { call, reg } => {
                if let Some(k) = site_of(call) {
                    if reg == RDX {
                        reads[k] |= READ_RDX;
                    }
                    if reg == XMM0 {
                        reads[k] |= READ_XMM0;
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

/// The most stack arguments a function or call is taken to have (generated
/// programs pass over 80 arrays); 6 register ones more still fit `Sig`'s `u8`s.
pub const MAX_STACK_ARGS: i64 = 192;

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
            if o.roots == 1u128 << k {
                if let Off::Known(off) = o.off {
                    if (8..8 + 8 * MAX_STACK_ARGS).contains(&off) {
                        end = end.max(off + size as i64);
                    }
                }
            }
        }
    }
    ((end - 8 + 7) / 8) as u8
}

/// Does the function store the address of its stack arguments (entry rsp + 8
/// or above) somewhere: a `va_list`'s `overflow_arg_area`, through which
/// `va_arg` reads however many arguments the caller passed?
pub fn stores_stack_area(f: &Function) -> bool {
    let entry = f.blocks[f.entry].params.get(&f.value_pool);
    let Some(k) = entry.iter().position(|&p| matches!(f.insts[p].kind, InstKind::BlockParam(RSP))) else { return false };
    let a = analyze(f);
    f.blocks.iter().any(|(_, blk)| {
        blk.insts.get(&f.value_pool).iter().any(|&id| match f.insts[id].kind {
            // into the function's own frame, where its `va_list` lives
            InstKind::Store { ptr, val, .. } => {
                let (o, p) = (a.origin[val.index()], a.origin[ptr.index()]);
                o.roots == 1u128 << k
                    && matches!(o.off, Off::Known(off) if off >= 8)
                    && p.roots == 1u128 << k
                    && matches!(p.off, Off::Known(off) if off < 0)
            }
            _ => false,
        })
    })
}

pub(crate) fn bytes(ty: TyId) -> usize {
    match ty {
        TyId::B1 | TyId::BOOL => 1,
        TyId::B2 => 2,
        TyId::B4 => 4,
        TyId::PAIR => 16,
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
    //    result), or Undef. A float result (xmm0) becomes the call's own value,
    //    which stands for whatever the callee returns.
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
            if reg == XMM0 && c.fret {
                map.push((id, call));
                continue;
            }
            let to = match (c.keeps(reg), sites[k].before(f, reg)) {
                (true, Some(v)) => v,
                _ => undef(f, &mut undefs),
            };
            map.push((id, to));
        }
    }

    // 2. Returns: rax, rax:rdx, xmm0, or nothing. A tail call whose callee
    //    returns less than this function does (or in another register) becomes a
    //    call and a return.
    for bi in 0..f.blocks.len() {
        let b = BlockId::new(bi);
        match f.blocks[b].term {
            Terminator::Return(Some(rax)) => {
                let exit = |k: usize| f.blocks[b].insts.get(&f.value_pool).iter().rev().find_map(|&id| match f.insts[id].kind {
                    InstKind::Exit { regs } => Some(regs.get(&f.value_pool).get(k).copied()),
                    _ => None,
                }).flatten();
                let (exit, xmm0) = (exit(1), exit(EXIT_XMM0));
                let term = if sig.fret {
                    Terminator::Return(Some(xmm0.unwrap_or_else(|| undef(f, &mut undefs))))
                } else if sig.ret2 {
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
                if !(sig.ret && !c.ret || sig.ret2 && !c.ret2 || sig.ret && sig.fret != c.fret) {
                    continue;
                }
                // `jmp g` becomes `v = call g; return v (or v:rdx)`
                let call = new_inst(f, InstKind::Call { callee, args }, TyId::B8, f.origin[callee.index()]);
                let mut tail = vec![call];
                let rax = if sig.fret {
                    if c.fret { call } else { undef(f, &mut undefs) }
                } else if c.rax() {
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
        let mut floats: Vec<ValueId> = old[CALL_XMM..].iter().step_by(2).take((c.fargs as usize).min(FLOAT_ARGS)).copied().collect();
        // what a call to a variadic function doesn't set up, it doesn't pass
        let set = new.len().saturating_sub(c.unset.0 as usize);
        for a in &mut new[set..] {
            *a = undef(f, &mut undefs);
        }
        let set = floats.len().saturating_sub(c.unset.1 as usize);
        for a in &mut floats[set..] {
            *a = undef(f, &mut undefs);
        }
        let mut loads = Vec::new();
        let set = c.stack_args.saturating_sub(c.unset.2) as i32;
        for j in 0..c.stack_args as i32 {
            if j >= set {
                new.push(undef(f, &mut undefs));
                continue;
            }
            // at a call, [rsp] is the first stack argument; at a jmp, [rsp] is our
            // own return address and the arguments follow it
            let disp = 8 * j + if matches!(s, Site::Tail(_)) { 8 } else { 0 };
            let p = new_inst(f, InstKind::PtrOffset { base: old[6], index: None, scale: 1, disp }, TyId::PTR, at);
            let v = new_inst(f, InstKind::Load { ptr: p, align: 1, volatile: false }, TyId::B8, at);
            loads.extend([p, v]);
            new.push(v);
        }
        new.extend_from_slice(&floats);
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
    //    registers are only saved and restored, and rax, r10, r11 and xmm
    //    registers past the float arguments are undefined. rsp stays for
    //    `frame::promote`.
    let mut k = 0;
    while k < f.blocks[entry].params.len as usize {
        let p = f.value_pool[f.blocks[entry].params.start as usize + k];
        let keep = match f.insts[p].kind {
            InstKind::BlockParam(r) => {
                r == RSP
                    || SYSV_ARGS[..sig.args as usize].contains(&r)
                    || float_arg(r).is_some_and(|j| j < sig.fargs)
                    || (STACK_ARG_BASE..XMM_PARAM).contains(&r)
            }
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
    crate::frame::promote(f, sig.stack_args);
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
