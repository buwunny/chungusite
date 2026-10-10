//! x86_64 -> IR lifter on top of `iced-x86`.
//!
//! The hot path does no heap allocation once a `Lifter` and a `Function` have been
//! used once: every buffer (`leaders`, `state`, and all of `Function`'s arenas) is
//! cleared, not dropped, between functions, so later functions reuse the capacity.
//! `tests/no_alloc.rs` checks this with a counting global allocator.
//!
//! Two linear passes over the bytes:
//! 1. find block leaders (entry, branch targets, fall-throughs) into a sorted `Vec<u64>`;
//! 2. decode again and lift each instruction straight into the arenas.
//!
//! Decoding twice is cheaper than any address->block hash map: iced decodes an
//! instruction in a few nanoseconds, and pass 2 finds block boundaries by comparing
//! against the next leader, which is O(1).
//!
//! SSA uses block parameters. Each block keeps a 16-slot register file (one cache line
//! of `Option<ValueId>`). Reading a register no instruction in the block has written
//! yet creates a `BlockParam`; `finalize` then propagates live-ins to predecessors
//! and fills in the edge arguments.

use crate::ir::*;
use iced_x86::{ConditionCode, Decoder, DecoderOptions, FlowControl, Instruction, Mnemonic, OpKind, Register};

mod asm;
mod sse;
mod x87;

const NGPR: usize = 16;
/// The register file: the 16 GPRs, then xmm0-15 as (low, high) qword pairs,
/// then the upper halves of ymm0-15 the same way, then the x87 stack's eight
/// slots and its control word (`x87::X87`).
const NREG: usize = NGPR + 64 + 9;
/// Where the upper halves of the ymm registers start in the register file.
const UPPER: usize = NGPR + 32;
type RegFile = [Option<ValueId>; NREG];
/// `BlockParam` register number of xmm register half `k` (2 * xmm + high).
/// Float argument `j` is the low half of xmm `j`, `XMM_PARAM + 2 * j`.
pub const XMM_PARAM: u8 = 0xc0;
/// `BlockParam` register number of the upper half `k` of the ymm registers
/// (2 * ymm + high). Never a parameter of the entry block: nothing passes or
/// preserves them.
const YMM_PARAM: u8 = 0xa0;
/// Register number (in `CallOut`, `BlockParam`) of xmm0's low half: the first
/// float argument, and where a float result is returned.
pub const XMM0: u8 = XMM_PARAM;
/// Float arguments in registers: xmm0-7.
pub const FLOAT_ARGS: usize = 8;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LiftError {
    /// Instruction or operand form this lifter does not handle yet.
    Unsupported { ip: u64, mnemonic: Mnemonic },
    /// A Jcc whose flags were set in a different block.
    FlagsNotInBlock { ip: u64 },
    /// A conditional branch (or its fall-through) leaves the code being lifted.
    BranchOutOfRange { ip: u64, target: u64 },
    /// A branch targets the middle of another instruction (overlapping code).
    TargetInsideInstruction { target: u64 },
}

/// A call's arguments as the lifter records them: every register a callee could
/// read. The six argument registers, then rsp (stack arguments), then rax, r10 and
/// r11, whose values `abi` needs when the callee preserves them, then from
/// `CALL_XMM` on the halves of xmm0-15 (low, high): float arguments, and values a
/// callee that preserves the register leaves there.
pub const CALL_REGS: [Register; 10] = [
    Register::RDI, Register::RSI, Register::RDX, Register::RCX, Register::R8, Register::R9,
    Register::RSP, Register::RAX, Register::R10, Register::R11,
];
pub const CALL_XMM: usize = CALL_REGS.len();
pub const CALL_ARGS: usize = CALL_XMM + 32;
type CallRegs = [ValueId; CALL_ARGS];
/// Caller-saved registers besides rax: each is a `CallOut` after a call, and an
/// `Exit` lists them at a return, followed by the halves of xmm0-15 from
/// `EXIT_XMM0` on (xmm0's low half: a float result).
pub const EXIT_REGS: [Register; 8] = [
    Register::RCX, Register::RDX, Register::RSI, Register::RDI, Register::R8, Register::R9, Register::R10, Register::R11,
];
/// Position of xmm0's low half (a float result) in an `Exit`'s list.
pub const EXIT_XMM0: usize = EXIT_REGS.len();
const EXIT_LEN: usize = EXIT_XMM0 + 32;
/// `BlockParam` register number of a condition (a `Bool`) that a block reads from
/// the flags its predecessors left.
pub const FLAG_PARAM: u8 = 0xfe;
/// Condition codes, indexed by `ConditionCode as usize`.
const NCC: usize = 17;
type CondFile = [Option<ValueId>; NCC];

/// Lazy flags: remember what set them, and only build a `Cmp` when a Jcc, CMOVcc or
/// SETcc reads them. `cmp` itself therefore emits nothing, and flags that are never
/// read cost nothing.
#[derive(Copy, Clone)]
enum Flags {
    Unknown,
    /// Nothing in this block has set them yet: they are whatever the predecessors
    /// left, and a condition read from them becomes a block parameter.
    Entry,
    /// cmp / sub / neg: every integer condition is derivable from the operands.
    Sub { lhs: ValueId, rhs: ValueId },
    /// test / and / or / xor: ZF and SF from the result, CF = OF = 0.
    Logic { res: ValueId },
    /// add: ZF, SF, CF (`res < lhs` unsigned) and OF.
    Add { lhs: ValueId, rhs: ValueId, res: ValueId },
    /// inc / dec / shifts: only ZF and SF are modelled (`res == 0`, `res < 0`).
    Res { res: ValueId },
    /// mul / imul: only CF = OF (the product doesn't fit `lo`) is modelled.
    Mul { lhs: ValueId, rhs: ValueId, lo: ValueId, signed: bool },
    /// bt: only CF is defined.
    Carry { cf: ValueId },
    /// adc / sbb: `res = lhs + rhs + cin` (`lhs - rhs - cin` if `sub`), `cin`
    /// the carry in, zero-extended. Every integer condition is derivable.
    AddCarry { lhs: ValueId, rhs: ValueId, cin: ValueId, res: ValueId, sub: bool },
    /// dec: the flags of `lhs - one` except CF, which is left alone.
    Dec { lhs: ValueId, one: ValueId },
    /// inc: the flags of `lhs + one = res` except CF.
    Inc { lhs: ValueId, one: ValueId, res: ValueId },
    /// ucomis / comis of the low `w`-byte floats `a` and `b`: ZF, PF and CF
    /// (all set when unordered), with OF = SF = 0.
    Float { a: ValueId, b: ValueId, w: u8 },
    /// rflags as an instruction kept as inline assembly left them
    /// (`Lifter::lift_asm`); `valid` is the status flags it holds for sure.
    Raw { rflags: ValueId, valid: u16 },
}

/// How many instructions before an indirect `jmp` pass 1 keeps, to recognize the
/// bounds check and table load of a jump table.
const RECENT: usize = 12;
/// Larger tables than this are not believed.
const MAX_CASES: u64 = 4096;

/// A jump table found in pass 1.
#[derive(Copy, Clone)]
struct JumpTable {
    /// The indirect `jmp`.
    jmp: u64,
    /// The instruction that reads the table, where the index register holds the case.
    load: u64,
    index: Register,
    /// The case targets are `cases[start..start + len]`.
    start: u32,
    len: u32,
    /// A bounds check before the jump gave the case count. Otherwise (a Rust
    /// `match` on an enum needs none) the table was read up to the first entry
    /// that leaves the function, and is cut at the first that isn't an instruction.
    bounded: bool,
}

#[derive(Copy, Clone)]
struct BlockState {
    /// Value of each GPR at block exit. Starts as the block's own live-ins.
    out: RegFile,
    /// `BlockParam` created for each live-in register.
    params: RegFile,
    /// `BlockParam` created for each condition read before the block sets the flags.
    cparams: CondFile,
    /// Each condition's value at block exit, filled in `finalize` for the ones a
    /// successor reads.
    cout: CondFile,
    /// The flags at block exit.
    flags: Flags,
    /// Where the block ends, for the instructions `finalize` adds to it.
    exit_ip: u64,
    /// The instruction that first needed a `cparams` entry, for errors.
    flags_read: u64,
}

const EMPTY_STATE: BlockState =
    BlockState { out: [None; NREG], params: [None; NREG], cparams: [None; NCC], cout: [None; NCC], flags: Flags::Unknown, exit_ip: 0, flags_read: 0 };

/// Where an instruction's destination operand lives.
#[derive(Copy, Clone)]
enum Dst {
    Reg(Register),
    Mem(ValueId),
}

pub struct Lifter {
    insn: Instruction,
    leaders: Vec<u64>,
    state: Vec<BlockState>,
    /// Each `Call` and its argument registers plus rsp. The arguments go into
    /// `value_pool` in `finalize`, after every block's instruction list is in place.
    calls: Vec<(ValueId, CallRegs)>,
    /// The last call that doesn't return, while the code after it is skipped.
    fall: Option<ValueId>,
    /// The same for each block that ends in a `TailCall`.
    tails: Vec<(BlockId, CallRegs)>,
    /// Each `Exit` and the registers it lists.
    exits: Vec<(ValueId, [ValueId; EXIT_LEN])>,
    /// Jump tables found in pass 1, in address order.
    tables: Vec<JumpTable>,
    /// Target address of every case of every table, table after table.
    cases: Vec<u64>,
    /// Pass 1: the instructions decoded before the current one, a ring with the
    /// newest at `recent[(nrecent - 1) % RECENT]`.
    recent: [Instruction; RECENT],
    nrecent: usize,
    /// Pass 1: the address each general register was last set to by a
    /// `lea r, [rip + k]` (0 once anything else writes it), in address order.
    /// A loop over a switch sets the table's base once, before the loop,
    /// further back than `recent` reaches.
    rip_lea: [u64; 16],
    /// Pass 1: the address a register holds wherever the function uses it,
    /// when every write to it is the same `lea r, [rip + k]` (besides the `pop`
    /// that restores a callee-saved one): a computed `goto *labels[op]` keeps
    /// its table there through the whole function, past calls and epilogues
    /// that end `rip_lea`'s straight-line view.
    fixed_lea: [u64; 16],
    /// For `fixed_leas`, kept so a warm lifter doesn't allocate.
    info: iced_x86::InstructionInfoFactory,
    /// Pass 1: the address of every instruction, in order.
    starts: Vec<u64>,
    /// The instructions to decode when following the control flow
    /// (`follow_flow`), in address order; empty for a linear sweep.
    walk: Vec<u64>,
    /// Pass 1: some instruction names an xmm (or wider) register. Without one,
    /// calls and returns don't track xmm registers: they can only hold what the
    /// function was entered with, or what its callees left, which `abi` works
    /// out from the callees' signatures. Most functions are like that, and
    /// threading 32 xmm halves through them doubles the lifting time.
    xmm: bool,
    /// Does the function use a ymm register? Then the upper halves are
    /// tracked too: undefined at the entry and after a call, zeroed by a VEX
    /// instruction that writes the xmm part.
    ymm: bool,
    /// Pass 1: the function uses the x87 stack. Then its slots are cleared at
    /// the entry and after calls.
    x87: bool,
    /// The x87 stack's depth in the current block, and at the start of each
    /// block (from the first predecessor lifted).
    fdepth: u8,
    fdepth_in: Vec<Option<u8>>,
    /// A block entered with two different x87 depths, by the branch at this address.
    fdepth_clash: Option<u64>,
    /// Lifting the upper half of a ymm instruction: xmm register `n` means
    /// the upper half of ymm `n`, and memory 16 bytes further on.
    upper: bool,
    /// Pass 2: the case index, read where the current block loads from its table.
    switch_index: Option<ValueId>,
    /// Each block that ends in a `Switch`, and its table (index into `tables`).
    switches: Vec<(BlockId, usize)>,
    /// The address of each conditional branch out of the function (into
    /// another function's `.cold` part), in order. Each gets a block after the
    /// leaders' that tail-calls the target.
    stubs: Vec<u64>,
    /// Switch cases that leave the function (`gcc` sends a `switch`'s unused
    /// cases to the `.cold` part): (the `jmp`, the target), one stub block
    /// each, in `stubs` in the order pass 1 found them.
    case_stubs: Vec<(u64, u64)>,
    /// Stack probe loops (`-fstack-clash-protection`), each (the `sub rsp`,
    /// the probing store, the `jne` back, the register holding the final rsp).
    probes: Vec<(u64, u64, u64, Register)>,
    /// `je`s that skip a variadic function's xmm spills (`spill_skip`),
    /// lifted as falling through.
    spill_skips: Vec<u64>,
    /// The stubs pass 2 has lifted so far.
    nstub: usize,
    /// Record the caller-saved registers at each return in an `Exit` instruction,
    /// for whole-program register summaries (`program.rs` sets this).
    pub track_exits: bool,
    /// What `fs:0` holds, so that thread-locals (`fs:[-k]`) become addresses in
    /// the block `load::tls` lays out (`program.rs` sets this). `None` leaves
    /// them unsupported.
    pub thread_pointer: Option<u64>,
    /// Keep an instruction the lifter has no model of as inline assembly
    /// (`InstKind::Opaque`) instead of failing the function (`lift/asm.rs`).
    pub asm: bool,
    /// Instructions the lifter models, but not in the form found at these
    /// addresses: kept as inline assembly on the next try.
    forced: Vec<u64>,
    /// Each `Opaque` and where its arguments are in `asm_vals`, for `finalize`.
    asm_args: Vec<(ValueId, u32, u32)>,
    asm_vals: Vec<ValueId>,
    cur: usize,
    flags: Flags,
    ip: u64,
}

impl Default for Lifter {
    fn default() -> Self { Self::new() }
}

impl Lifter {
    pub fn new() -> Self {
        Lifter {
            insn: Instruction::default(),
            leaders: Vec::with_capacity(256),
            state: Vec::with_capacity(256),
            calls: Vec::with_capacity(64),
            fall: None,
            tails: Vec::with_capacity(8),
            exits: Vec::with_capacity(8),
            tables: Vec::with_capacity(4),
            cases: Vec::with_capacity(64),
            recent: [Instruction::default(); RECENT],
            nrecent: 0,
            rip_lea: [0; 16],
            fixed_lea: [0; 16],
            info: iced_x86::InstructionInfoFactory::new(),
            starts: Vec::with_capacity(256),
            walk: Vec::new(),
            xmm: false,
            ymm: false,
            x87: false,
            fdepth: 0,
            fdepth_in: Vec::with_capacity(256),
            fdepth_clash: None,
            upper: false,
            switch_index: None,
            switches: Vec::with_capacity(4),
            stubs: Vec::with_capacity(4),
            case_stubs: Vec::new(),
            probes: Vec::new(),
            spill_skips: Vec::new(),
            nstub: 0,
            track_exits: false,
            thread_pointer: None,
            asm: false,
            forced: Vec::new(),
            asm_args: Vec::new(),
            asm_vals: Vec::new(),
            cur: 0,
            flags: Flags::Unknown,
            ip: 0,
        }
    }

    /// Did the last `lift` decode only what the control flow reaches
    /// (`follow_flow`), because a linear sweep of the bytes failed?
    pub fn followed_flow(&self) -> bool {
        !self.walk.is_empty()
    }

    /// Lift one function's bytes, loaded at `ip`, into `f` (which is cleared first).
    /// Jump tables are looked for in `code` itself; see `lift_with_data`.
    pub fn lift(&mut self, code: &[u8], ip: u64, f: &mut Function) -> Result<(), LiftError> {
        self.lift_with_data(code, ip, &[(ip, code)], f)
    }

    /// `lift`, reading jump tables from `data`: the binary's sections as
    /// (load address, bytes), sorted by address.
    pub fn lift_with_data(&mut self, code: &[u8], ip: u64, data: &[(u64, &[u8])], f: &mut Function) -> Result<(), LiftError> {
        self.lift_full(code, ip, data, &|_, _| false, f)
    }

    /// `lift_with_data`, where `noreturn(ip, target)` says whether the call at
    /// `ip` never returns (`abort`, `__stack_chk_fail`): such a call ends its
    /// block. `target` is the callee of a direct call, or the slot of
    /// `call [rip+slot]` (a GOT entry).
    pub fn lift_full(
        &mut self,
        code: &[u8],
        ip: u64,
        data: &[(u64, &[u8])],
        noreturn: &dyn Fn(u64, Option<u64>) -> bool,
        f: &mut Function,
    ) -> Result<(), LiftError> {
        self.forced.clear();
        loop {
            match self.lift_once(code, ip, data, noreturn, f) {
                // an instruction the lifter knows, in a form it doesn't: again,
                // with that one kept as inline assembly
                Err(LiftError::Unsupported { ip: at, .. }) if self.asm && !self.forced.contains(&at) => {
                    let mut i = Instruction::default();
                    let mut dec = Decoder::with_ip(64, code.get(at.wrapping_sub(ip) as usize..).unwrap_or(&[]), at, DecoderOptions::NONE);
                    dec.decode_out(&mut i);
                    if !handled(&i) || !self.asm_fits(&i) {
                        return Err(LiftError::Unsupported { ip: at, mnemonic: i.mnemonic() });
                    }
                    self.forced.push(at);
                }
                r => return r,
            }
        }
    }

    fn lift_once(
        &mut self,
        code: &[u8],
        ip: u64,
        data: &[(u64, &[u8])],
        noreturn: &dyn Fn(u64, Option<u64>) -> bool,
        f: &mut Function,
    ) -> Result<(), LiftError> {
        f.clear();
        // Decode linearly first, the fast path for compiled code. If that
        // fails, the bytes may hold something no path reaches (junk after a
        // `jmp`, data in hand-written code) or instructions that overlap (a
        // jump into the middle of one): decode only what the control flow
        // reaches, from each place it reaches.
        self.walk.clear();
        if let Err(e) = self.find_leaders(code, ip, data) {
            if !matches!(e, LiftError::TargetInsideInstruction { .. } | LiftError::Unsupported { .. }) || !self.follow_flow(code, ip, data, noreturn) {
                return Err(e);
            }
        }
        self.state.clear();
        self.calls.clear();
        self.tails.clear();
        self.exits.clear();
        self.switches.clear();
        self.asm_args.clear();
        self.asm_vals.clear();
        self.nstub = 0;
        self.fdepth_in.clear();
        self.fdepth_in.resize(self.leaders.len() + self.stubs.len(), None);
        self.fdepth_clash = None;
        if self.x87 {
            self.x87_depths(code, ip);
        }
        for _ in 0..self.leaders.len() + self.stubs.len() {
            self.state.push(EMPTY_STATE);
            f.blocks.push(Block { insts: ListRef::EMPTY, params: ListRef::EMPTY, term: Terminator::Unreachable });
        }

        let mut dec = Decoder::with_ip(64, code, ip, DecoderOptions::NONE);
        let mut next_leader = 1;
        let mut open = true;
        // Where the previous instruction falls through to.
        let mut fall_to = ip;
        self.fall = None;
        self.upper = false;
        self.begin_block(0, f);
        if self.ymm {
            self.clear_upper(f, false);
        }
        if self.x87 {
            self.x87_clear(f, true);
        }
        for k in 0..self.starts.len() {
            let at = self.starts[k];
            if at != fall_to {
                // Following the control flow: the previous instruction falls
                // through to code decoded elsewhere (a leader).
                if dec.set_position((at - ip) as usize).is_err() {
                    break;
                }
                dec.set_ip(at);
                if open {
                    let to = self.block_at(fall_to).ok_or(LiftError::TargetInsideInstruction { target: fall_to })?;
                    self.end_block(f, Terminator::Jump { to, args: ListRef::EMPTY });
                    open = false;
                }
                self.fall = None;
            }
            dec.decode_out(&mut self.insn);
            self.ip = self.insn.ip();
            fall_to = self.insn.next_ip();
            if next_leader < self.leaders.len() && self.ip >= self.leaders[next_leader] {
                if self.ip > self.leaders[next_leader] {
                    return Err(LiftError::TargetInsideInstruction { target: self.leaders[next_leader] });
                }
                if open {
                    self.end_block(f, Terminator::Jump { to: BlockId::new(next_leader), args: ListRef::EMPTY });
                }
                if let Some(c) = self.fall.take() {
                    f.noreturn_falls.push((c, BlockId::new(next_leader)));
                }
                self.begin_block(next_leader, f);
                next_leader += 1;
                open = true;
            }
            if open {
                if let Some(t) = self.tables.iter().find(|t| t.load == self.ip) {
                    self.switch_index = Some(self.read_full(f, t.index.number()));
                }
                open = !self.lift_insn(f, noreturn)?;
            } else if let Some(c) = self.fall {
                // Follow the code after a call that doesn't return as if it
                // did, to where it would have gone (`Function::noreturn_falls`).
                let i = self.insn;
                if matches!(i.flow_control(), FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch) && i.op0_kind() == OpKind::NearBranch64 {
                    if let Some(b) = self.block_at(i.near_branch_target()) {
                        f.noreturn_falls.push((c, b));
                    }
                }
                if !matches!(i.flow_control(), FlowControl::Next | FlowControl::Call | FlowControl::IndirectCall | FlowControl::ConditionalBranch) {
                    self.fall = None;
                }
            }
        }
        if open {
            self.end_block(f, Terminator::Unreachable); // ran off the end of the bytes
        }
        if let Some(ip) = self.fdepth_clash {
            return Err(LiftError::Unsupported { ip, mnemonic: Mnemonic::Fld });
        }
        self.finalize(f)
    }

    /// The 64-bit value `reg` holds when block `b` exits, if the block reads or
    /// writes it. Valid after `lift` until the next call.
    pub fn reg_out(&self, b: BlockId, reg: Register) -> Option<ValueId> {
        if !reg.is_gpr() {
            return None;
        }
        self.state.get(b.index())?.out[reg.full_register().number()]
    }

    // ---------- pass 1 ----------

    fn find_leaders(&mut self, code: &[u8], ip: u64, data: &[(u64, &[u8])]) -> Result<(), LiftError> {
        let end = ip + code.len() as u64;
        let in_range = |a: u64| a >= ip && a < end;
        self.leaders.clear();
        self.leaders.push(ip);
        self.tables.clear();
        self.cases.clear();
        self.nrecent = 0;
        self.rip_lea = [0; 16];
        self.fixed_lea = fixed_leas(&mut self.info, code, ip);
        self.starts.clear();
        self.stubs.clear();
        self.case_stubs.clear();
        self.probes.clear();
        self.spill_skips.clear();
        self.xmm = false;
        self.ymm = false;
        self.x87 = false;
        let mut dec = Decoder::with_ip(64, code, ip, DecoderOptions::NONE);
        let mut k = 0;
        loop {
            if self.walk.is_empty() {
                if !dec.can_decode() {
                    break;
                }
            } else {
                // Following the control flow: only the instructions it reaches,
                // in address order, each decoded from its own start.
                let Some(&at) = self.walk.get(k) else { break };
                k += 1;
                if dec.set_position((at - ip) as usize).is_err() {
                    break;
                }
                dec.set_ip(at);
            }
            dec.decode_out(&mut self.insn);
            self.ip = self.insn.ip();
            self.starts.push(self.ip);
            let next = self.insn.next_ip();
            // Falling through to code that isn't decoded next (it overlaps
            // this instruction, or comes after code no path reaches): a leader.
            if !self.walk.is_empty() && self.walk.get(k) != Some(&next) && self.walk.binary_search(&next).is_ok() && falls_through(&self.insn) {
                self.leaders.push(next);
            }
            // Pass 2 lifts every instruction (code after a `ret` starts a block
            // too), so the first one it has no case for fails the function now,
            // before any IR is built.
            if !handled(&self.insn) && !(self.asm && { let i = self.insn; self.asm_fits(&i) }) {
                return Err(self.unsupported());
            }
            let i = &self.insn;
            self.xmm |= (0..i.op_count()).any(|k| i.op_kind(k) == OpKind::Register && i.op_register(k).is_vector_register());
            self.ymm |= (0..i.op_count()).any(|k| i.op_kind(k) == OpKind::Register && i.op_register(k).is_ymm());
            self.x87 |= x87::handled(i.mnemonic());
            // a call (or a jump to another function) may pass on a float the
            // function never touches: `g(x)` with `x` arriving in xmm0, or
            // `print(tonumber(v))`, xmm0 from one call to the next
            self.xmm |= matches!(i.flow_control(), FlowControl::Call | FlowControl::IndirectCall | FlowControl::IndirectBranch)
                || matches!(i.flow_control(), FlowControl::UnconditionalBranch | FlowControl::ConditionalBranch) && !in_range(i.near_branch_target());
            match self.insn.flow_control() {
                FlowControl::ConditionalBranch if self.spill_skip(&code[(next - ip) as usize..], next) => {
                    self.spill_skips.push(self.ip);
                    self.leaders.push(next);
                }
                FlowControl::ConditionalBranch if self.probe_loop().is_some() => {
                    let p = self.probe_loop().unwrap();
                    self.probes.push(p);
                    if in_range(next) { self.leaders.push(next); }
                }
                FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch => {
                    let t = self.insn.near_branch_target();
                    if in_range(t) {
                        self.leaders.push(t);
                    } else if self.insn.flow_control() == FlowControl::ConditionalBranch {
                        self.stubs.push(self.ip);
                    }
                    if in_range(next) && self.walked(next) { self.leaders.push(next); }
                }
                FlowControl::IndirectBranch => {
                    self.find_table(data, ip, end)?;
                    // a stub block for each target out of the function, in case order
                    if let Some(t) = self.tables.last().filter(|t| t.jmp == self.ip).copied() {
                        for k in t.start..t.start + t.len {
                            let c = self.cases[k as usize];
                            if !in_range(c) && !self.case_stubs.contains(&(self.ip, c)) {
                                self.case_stubs.push((self.ip, c));
                                self.stubs.push(self.ip);
                            }
                        }
                    }
                    if in_range(next) && self.walked(next) { self.leaders.push(next); }
                }
                FlowControl::Return if in_range(next) && self.walked(next) => self.leaders.push(next),
                _ => {}
            }
            self.note_rip_lea();
            self.recent[self.nrecent % RECENT] = self.insn;
            self.nrecent += 1;
        }
        // Every case must start an instruction; an unbounded table ends before
        // the first that doesn't.
        for t in &mut self.tables {
            let cases = &self.cases[t.start as usize..(t.start + t.len) as usize];
            let inside = |c: &u64| *c >= ip && *c < end;
            let ok = cases.iter().take_while(|c| self.starts.binary_search(c).is_ok() || (t.bounded && !inside(c))).count();
            if ok == 0 || (t.bounded && ok < cases.len()) {
                return Err(LiftError::Unsupported { ip: t.jmp, mnemonic: Mnemonic::Jmp });
            }
            t.len = ok as u32;
            self.leaders.extend(cases[..ok].iter().filter(|c| inside(c)));
        }
        self.leaders.sort_unstable();
        self.leaders.dedup();
        // Every leader must start an instruction: one that starts inside
        // another needs `follow_flow`.
        if let Some(&target) = self.leaders.iter().find(|l| self.starts.binary_search(l).is_err()) {
            return Err(LiftError::TargetInsideInstruction { target });
        }
        Ok(())
    }

    /// Is `a` decoded? Every instruction is in a linear sweep; following the
    /// control flow, the code after a `jmp` or `ret` may be junk no path reaches.
    fn walked(&self, a: u64) -> bool {
        self.walk.is_empty() || self.walk.binary_search(&a).is_ok()
    }

    /// Retry pass 1 on just the instructions the control flow reaches from the
    /// entry (`walk`), decoding each from where a path reaches it, so the same
    /// bytes can be two different instructions. A jump table found on the way
    /// adds its cases, and the walk repeats. False if pass 1 still fails.
    fn follow_flow(&mut self, code: &[u8], ip: u64, data: &[(u64, &[u8])], noreturn: &dyn Fn(u64, Option<u64>) -> bool) -> bool {
        let end = ip + code.len() as u64;
        let mut dec = Decoder::with_ip(64, code, ip, DecoderOptions::NONE);
        let mut insn = Instruction::default();
        let mut seen = std::collections::HashSet::new();
        let mut todo = vec![ip];
        for _ in 0..8 {
            while let Some(mut at) = todo.pop() {
                while (ip..end).contains(&at) && seen.insert(at) && dec.set_position((at - ip) as usize).is_ok() {
                    dec.set_ip(at);
                    dec.decode_out(&mut insn);
                    self.walk.push(at);
                    if insn.is_invalid() || matches!(insn.mnemonic(), Mnemonic::Ud2 | Mnemonic::Int3 | Mnemonic::Hlt) {
                        break;
                    }
                    let target = match insn.op0_kind() {
                        OpKind::NearBranch64 => Some(insn.near_branch_target()),
                        OpKind::Memory if insn.is_ip_rel_memory_operand() => Some(insn.ip_rel_memory_address()),
                        _ => None,
                    };
                    match insn.flow_control() {
                        FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch => todo.extend(target.filter(|t| (ip..end).contains(t))),
                        FlowControl::Call | FlowControl::IndirectCall if noreturn(at, target) => break,
                        _ => {}
                    }
                    if !falls_through(&insn) {
                        break;
                    }
                    at = insn.next_ip();
                }
            }
            self.walk.sort_unstable();
            match self.find_leaders(code, ip, data) {
                Ok(()) => {
                    // A case or branch target the walk didn't reach yet.
                    todo.extend(self.leaders.iter().copied().filter(|l| !seen.contains(l)));
                    if todo.is_empty() {
                        return true;
                    }
                }
                Err(LiftError::TargetInsideInstruction { target }) if !seen.contains(&target) => todo.push(target),
                Err(_) => break,
            }
        }
        self.walk.clear();
        false
    }

    /// The address `reg` holds where `before(k)` runs, when a `lea reg, [rip + a]`
    /// set it: the last write before it in `recent`, or, if nothing in `recent`
    /// writes it, from `rip_lea`; else `fixed_lea`.
    fn rip_value(&self, reg: Register, k: usize) -> Option<u64> {
        let n = self.nrecent.min(RECENT);
        match (k + 1..n).find_map(|m| self.before(m).filter(|w| writes(w, reg))) {
            Some(w) => (w.mnemonic() == Mnemonic::Lea && w.is_ip_rel_memory_operand()).then(|| w.ip_rel_memory_address()),
            None if reg.is_gpr64() && !(0..=k.min(n.saturating_sub(1))).any(|m| self.before(m).is_some_and(|w| writes(w, reg))) => {
                Some(self.rip_lea[gpr(reg)]).filter(|&a| a != 0)
            }
            None => None,
        }
        .or_else(|| reg.is_gpr64().then(|| self.fixed_lea[gpr(reg)]).filter(|&a| a != 0))
    }

    /// Update `rip_lea` for the current instruction.
    fn note_rip_lea(&mut self) {
        if self.rip_lea.iter().any(|&a| a != 0) {
            let i = self.insn;
            let bits = |rs: &[Register]| rs.iter().fold(0u16, |m, &r| m | 1 << gpr(r));
            use Register::*;
            // What a callee may change, and what instructions change besides
            // their destination (`mul`'s rdx, `rep movsb`'s pointers).
            let mut clobbered = if !handled(&i) {
                u16::MAX // inline assembly: whatever it writes
            } else if i.flow_control() == FlowControl::Call {
                bits(&[RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11])
            } else if i.is_string_instruction() {
                bits(&[RAX, RCX, RSI, RDI])
            } else if (i.mnemonic() == Mnemonic::Imul && i.op_count() == 1)
                || matches!(
                    i.mnemonic(),
                    Mnemonic::Mul | Mnemonic::Div | Mnemonic::Idiv | Mnemonic::Cqo | Mnemonic::Cdq | Mnemonic::Cwd
                        | Mnemonic::Cpuid | Mnemonic::Rdtsc | Mnemonic::Rdtscp | Mnemonic::Syscall | Mnemonic::Cmpxchg
                        | Mnemonic::Cmpxchg8b | Mnemonic::Cmpxchg16b | Mnemonic::Xgetbv
                )
            {
                bits(&[RAX, RCX, RDX, RBX, R11])
            } else {
                0
            };
            // `xchg`'s and `xadd`'s source
            if matches!(i.mnemonic(), Mnemonic::Xchg | Mnemonic::Xadd) && i.op1_kind() == OpKind::Register && i.op1_register().is_gpr() {
                clobbered |= 1 << gpr(i.op1_register().full_register());
            }
            for (n, a) in self.rip_lea.iter_mut().enumerate() {
                if clobbered & 1 << n != 0 || writes(&i, gpr_register(n)) {
                    *a = 0;
                }
            }
        }
        let i = &self.insn;
        if i.mnemonic() == Mnemonic::Lea && i.is_ip_rel_memory_operand() && i.op0_register().is_gpr64() {
            self.rip_lea[gpr(i.op0_register())] = i.ip_rel_memory_address();
        }
    }

    /// Is the `jne` in `self.insn` the end of a stack probe loop, which
    /// `-fstack-clash-protection` (on by default in gcc on most Linux
    /// distributions) emits for a frame larger than a page:
    ///
    /// ```text
    /// lea  r11, [rsp - 0x4000]   ; the final rsp
    /// L: sub rsp, 0x1000
    /// or   qword [rsp], 0        ; touch each page in order (clang: mov qword [rsp], 0)
    /// cmp  rsp, r11
    /// jne  L
    /// ```
    ///
    /// Lifted as one `rsp = r11`, with no loop: the probes only touch the stack.
    fn probe_loop(&self) -> Option<(u64, u64, u64, Register)> {
        let j = &self.insn;
        if j.mnemonic() != Mnemonic::Jne {
            return None;
        }
        let (cmp, store, sub) = (self.before(0)?, self.before(1)?, self.before(2)?);
        let reg = |i: &Instruction, k: u32| (i.op_kind(k) == OpKind::Register).then(|| i.op_register(k));
        let limit = match (cmp.mnemonic(), reg(cmp, 0), reg(cmp, 1)) {
            (Mnemonic::Cmp, Some(Register::RSP), Some(r)) | (Mnemonic::Cmp, Some(r), Some(Register::RSP)) if r.is_gpr64() => r,
            _ => return None,
        };
        let at_rsp = |i: &Instruction| {
            i.op0_kind() == OpKind::Memory && i.memory_base() == Register::RSP && i.memory_index() == Register::None
                && i.memory_displacement64() == 0 && is_imm(i.op1_kind()) && i.immediate(1) == 0
        };
        let ok = matches!(store.mnemonic(), Mnemonic::Or | Mnemonic::Mov) && at_rsp(store)
            && sub.mnemonic() == Mnemonic::Sub && reg(sub, 0) == Some(Register::RSP) && is_imm(sub.op1_kind())
            && sub.ip() == j.near_branch_target();
        ok.then(|| (sub.ip(), store.ip(), j.ip(), limit))
    }

    /// Is the branch in `self.insn` the `test al, al; je` with which a variadic
    /// function skips saving the xmm argument registers when the caller passed
    /// no floats (`al` counts them)? `rest` is the code after it, which must
    /// be nothing but those stores up to the branch target. The output's
    /// callers don't set `al`, so the stores always happen; they only fill the
    /// register save area.
    fn spill_skip(&self, rest: &[u8], next: u64) -> bool {
        let j = &self.insn;
        let Some(t) = self.before(0) else { return false };
        let al = |k: u32| t.op_kind(k) == OpKind::Register && t.op_register(k) == Register::AL;
        if j.mnemonic() != Mnemonic::Je || t.mnemonic() != Mnemonic::Test || !al(0) || !al(1) {
            return false;
        }
        let target = j.near_branch_target();
        if target <= next || target - next > 8 * 16 {
            return false;
        }
        let mut dec = Decoder::with_ip(64, rest, next, DecoderOptions::NONE);
        let mut i = Instruction::default();
        let mut n = 0;
        while dec.can_decode() && dec.ip() < target {
            dec.decode_out(&mut i);
            let spill = matches!(i.mnemonic(), Mnemonic::Movaps | Mnemonic::Movapd | Mnemonic::Movups | Mnemonic::Movupd | Mnemonic::Movdqa | Mnemonic::Movdqu | Mnemonic::Vmovaps | Mnemonic::Vmovapd | Mnemonic::Vmovups | Mnemonic::Vmovupd)
                && i.op0_kind() == OpKind::Memory
                && i.op1_kind() == OpKind::Register
                && i.op_register(1).is_xmm();
            if !spill {
                return false;
            }
            n += 1;
        }
        n > 0 && dec.ip() == target
    }

    /// The `k`-th instruction before the current one in pass 1 (0 is the one
    /// just before).
    fn before(&self, k: usize) -> Option<&Instruction> {
        (k < self.nrecent.min(RECENT)).then(|| &self.recent[(self.nrecent - 1 - k) % RECENT])
    }

    /// Is the indirect `jmp` in `self.insn` a jump through a table of case
    /// targets inside `[lo, hi)`? If so, record it; `find_leaders` checks the
    /// cases and makes them leaders. If it looks like one but the table can't be
    /// read, the jump is unsupported rather than taken for a tail call.
    /// Two shapes, from gcc and clang:
    ///
    /// ```text
    /// cmp  edi, 7              ; or sub, then ja/jae/jbe/jb: the case count
    /// ja   default
    /// jmp  [table + rdi*8]     ; absolute targets (non-PIC)
    ///
    /// cmp  edi, 7
    /// ja   default
    /// lea  rdx, [rip + table]
    /// movsxd rax, [rdx + rdi*4]
    /// add  rax, rdx            ; targets relative to the table (PIC)
    /// jmp  rax
    /// ```
    ///
    /// The `lea` may also come well before, outside a loop around the switch.
    fn find_table(&mut self, data: &[(u64, &[u8])], lo: u64, hi: u64) -> Result<(), LiftError> {
        let j = self.insn;
        // (instructions back to the table load, the index register, table address, entry size)
        // `[table + i*8]`, or `[base + i*8 + disp]` with a `lea base, [rip + table]`
        // before it (a computed `goto *labels[op]` keeps the table in a register)
        // (`m` is the jump itself, or the load `before(k)`)
        let absolute = |l: &Self, m: &Instruction, k: Option<usize>| -> Option<(Register, u64)> {
            if m.memory_index_scale() != 8 || m.memory_index() == Register::None {
                return None;
            }
            let base = match (m.memory_base(), k) {
                (Register::None, _) => 0,
                (b, Some(k)) if b.is_gpr64() => l.rip_value(b, k)?,
                // the jump: the `lea` may be the instruction just before it
                (b, None) if b.is_gpr64() => match l.before(0) {
                    Some(w) if writes(w, b) => (w.mnemonic() == Mnemonic::Lea && w.is_ip_rel_memory_operand()).then(|| w.ip_rel_memory_address())?,
                    _ => l.rip_value(b, 0)?,
                },
                _ => return None,
            };
            Some((m.memory_index(), base.wrapping_add(m.memory_displacement64())))
        };
        let (back, index, table, size) = match j.op0_kind() {
            OpKind::Memory => match absolute(self, &j, None) {
                Some((i, t)) => (0, i, t, 8),
                None => return Ok(()),
            },
            // `mov rax, [table + i*8]; jmp rax`
            OpKind::Register if self.before(0).is_some_and(|m| {
                m.mnemonic() == Mnemonic::Mov && m.op0_kind() == OpKind::Register && m.op0_register() == j.op0_register()
                    && m.op1_kind() == OpKind::Memory && absolute(self, m, Some(0)).is_some()
            }) => {
                let (i, t) = absolute(self, self.before(0).unwrap(), Some(0)).unwrap();
                (1, i, t, 8)
            }
            OpKind::Register => {
                let r = j.op0_register();
                let Some(add) = self.before(0).filter(|a| {
                    a.mnemonic() == Mnemonic::Add && a.op0_kind() == OpKind::Register && a.op0_register() == r
                        && a.op1_kind() == OpKind::Register
                }) else { return Ok(()) };
                let other = add.op1_register();
                // The entry, sign-extended into one of the two: `movsxd`, or
                // `mov eax, [..]; cdqe` (gcc -O0).
                let Some((k, load, dst)) = (1..RECENT).find_map(|k| {
                    let m = self.before(k)?;
                    let dst = m.op0_register();
                    match m.mnemonic() {
                        Mnemonic::Movsxd if m.op1_kind() == OpKind::Memory && (dst == r || dst == other) => Some((k, *m, dst)),
                        Mnemonic::Cdqe if r == Register::RAX || other == Register::RAX => {
                            let l = self.before(k + 1)?;
                            (l.mnemonic() == Mnemonic::Mov && l.op0_register() == Register::EAX && l.op1_kind() == OpKind::Memory)
                                .then_some((k + 1, *l, Register::RAX))
                        }
                        _ => None,
                    }
                }) else { return Ok(()) };
                // the table, added to the entry
                let Some(table) = self.rip_value(if dst == r { other } else { r }, 0) else { return Ok(()) };
                if load.memory_displacement64() != 0 {
                    return Ok(());
                }
                let (b, i) = (load.memory_base(), load.memory_index());
                let at = |reg: Register| reg != Register::None && self.rip_value(reg, k) == Some(table);
                match load.memory_index_scale() {
                    4 if at(b) => (k + 1, i, table, 4),
                    // [table + x] where `lea x, [i*4]` before
                    1 => {
                        let x = if at(b) { i } else if at(i) { b } else { return Ok(()) };
                        let Some(m) = (k + 1..RECENT).find(|&m| self.before(m).is_some_and(|w| writes(w, x))) else { return Ok(()) };
                        let w = self.before(m).unwrap();
                        if w.mnemonic() != Mnemonic::Lea || w.memory_base() != Register::None || w.memory_index_scale() != 4
                            || w.memory_displacement64() != 0
                        {
                            return Ok(());
                        }
                        (m + 1, w.memory_index(), table, 4)
                    }
                    _ => return Ok(()),
                }
            }
            _ => return Ok(()),
        };
        let unknown = self.unsupported();
        if !index.is_gpr64() {
            return Err(unknown);
        }
        // The bounds check: the last `ja`/`jae`/`jbe`/`jb` before the load, right
        // after a `cmp` or `sub` with an immediate, of the index register or of a
        // register moved into it after the check.
        let full = |r: Register| r.full_register();
        let n = (back..RECENT - 1).find_map(|k| {
            let jcc = self.before(k)?;
            let set = self.before(k + 1)?;
            let n = match jcc.mnemonic() {
                Mnemonic::Ja | Mnemonic::Jbe => 1,
                Mnemonic::Jae | Mnemonic::Jb => 0,
                Mnemonic::Jmp | Mnemonic::Ret | Mnemonic::Call => return Some(None), // another block
                _ => return None,
            };
            if !matches!(set.mnemonic(), Mnemonic::Cmp | Mnemonic::Sub) || set.op_count() != 2 || !is_imm(set.op1_kind())
                || set.op0_kind() != OpKind::Register
            {
                return Some(None);
            }
            let checked = full(set.op0_register());
            let moved = (back..k).any(|m| {
                self.before(m).is_some_and(|x| {
                    matches!(x.mnemonic(), Mnemonic::Mov | Mnemonic::Movzx | Mnemonic::Movsxd)
                        && x.op0_kind() == OpKind::Register && full(x.op0_register()) == full(index)
                        && x.op1_kind() == OpKind::Register && full(x.op1_register()) == checked
                })
            });
            Some((checked == full(index) || moved).then(|| set.immediate(1).wrapping_add(n)))
        }).flatten();
        // or a mask of the index (`and eax, 0x7f`) since it was last written
        let n = n.or_else(|| {
            let w = (back..RECENT).find_map(|k| self.before(k).filter(|w| writes(w, index)))?;
            let m = (w.mnemonic() == Mnemonic::And && w.op_count() == 2 && is_imm(w.op1_kind())).then(|| w.immediate(1))?;
            (m != u64::MAX && (m + 1).is_power_of_two()).then_some(m + 1)
        });
        if n.is_some_and(|n| n == 0 || n > MAX_CASES) {
            return Err(unknown);
        }
        let start = self.cases.len();
        for k in 0..n.unwrap_or(MAX_CASES) {
            let Some(e) = read(data, table.wrapping_add(k * size as u64), size) else { break };
            let t = if size == 8 {
                u64::from_le_bytes(e.try_into().unwrap())
            } else {
                table.wrapping_add(i32::from_le_bytes(e.try_into().unwrap()) as i64 as u64)
            };
            // out of the function: a tail call, but only where a bounds check
            // says the entry belongs to the table
            if (t < lo || t >= hi) && n.is_none() {
                break;
            }
            self.cases.push(t);
        }
        let len = self.cases.len() - start;
        if len == 0 || n.is_some_and(|n| len as u64 != n) {
            self.cases.truncate(start);
            return Err(unknown);
        }
        let load = if back == 0 { j.ip() } else { self.before(back - 1).unwrap().ip() };
        self.tables.push(JumpTable { jmp: j.ip(), load, index, start: start as u32, len: len as u32, bounded: n.is_some() });
        Ok(())
    }

    fn block_at(&self, addr: u64) -> Option<BlockId> {
        self.leaders.binary_search(&addr).ok().map(BlockId::new)
    }

    // ---------- pass 2 ----------

    fn begin_block(&mut self, idx: usize, f: &mut Function) {
        self.cur = idx;
        self.fdepth = self.fdepth_in[idx].unwrap_or(0);
        self.flags = Flags::Entry;
        self.switch_index = None;
        f.blocks[BlockId::new(idx)].insts.start = f.value_pool.len() as u32;
    }

    /// The terminator of the current block for a jump to `target`, outside
    /// the function.
    fn tail_call(&mut self, f: &mut Function, target: u64) -> Terminator {
        let a = self.konst(f, target, TyId::B8);
        let callee = self.emit(f, InstKind::IntToPtr(a), TyId::PTR);
        let regs = self.call_regs(f);
        self.tails.push((BlockId::new(self.cur), regs));
        Terminator::TailCall { callee, args: ListRef::EMPTY }
    }

    /// The terminator for the jump `i` out of the function: a tail call, or,
    /// to code that never returns (a `.cold` part that calls `abort`), a call
    /// that doesn't return, so the path is evidence of no result.
    fn jump_out(&mut self, f: &mut Function, i: &Instruction, noreturn: &dyn Fn(u64, Option<u64>) -> bool) -> Result<Terminator, LiftError> {
        let target = i.near_branch_target();
        if noreturn(self.ip, Some(target)) {
            self.call(f, i)?;
            return Ok(Terminator::Unreachable);
        }
        Ok(self.tail_call(f, target))
    }

    /// The address block `b` starts at; a stub's is its branch's.
    fn block_ip(&self, b: usize) -> u64 {
        match self.leaders.get(b) {
            Some(&a) => a,
            None => self.stubs[b - self.leaders.len()],
        }
    }

    fn end_block(&mut self, f: &mut Function, term: Terminator) {
        let len = f.value_pool.len() as u32;
        let b = &mut f.blocks[BlockId::new(self.cur)];
        b.insts.len = len - b.insts.start;
        b.term = term;
        let s = &mut self.state[self.cur];
        s.flags = self.flags;
        s.exit_ip = self.ip;
        // (a block the code never reaches, like padding after a jump, says nothing)
        if self.x87 && self.fdepth_in[self.cur].is_some() {
            let mut succ: Vec<BlockId> = term.successors(&f.value_pool).collect();
            if let Terminator::Switch { .. } = term {
                let t = self.tables[self.table_of(self.cur)];
                succ.extend((0..t.len as usize).filter_map(|k| self.case_block(t, k)));
            }
            for b in succ {
                match self.fdepth_in[b.index()] {
                    None => self.fdepth_in[b.index()] = Some(self.fdepth),
                    Some(d) if d != self.fdepth => self.fdepth_clash = self.fdepth_clash.or(Some(self.ip)),
                    _ => {}
                }
            }
        }
    }

    /// Lift `self.insn`. Returns true if it ended the block.
    fn lift_insn(&mut self, f: &mut Function, noreturn: &dyn Fn(u64, Option<u64>) -> bool) -> Result<bool, LiftError> {
        let i = self.insn; // Instruction is Copy (40 bytes); avoids borrowing self
        // a stack probe loop: rsp goes straight to its final value
        if let Some(&(sub, store, jne, limit)) = self.probes.iter().find(|p| [p.0, p.1, p.2].contains(&self.ip)) {
            if self.ip == sub {
                let v = self.read(f, limit)?;
                self.write(f, Register::RSP, v)?;
            }
            let _ = (store, jne);
            return Ok(false);
        }
        if self.spill_skips.contains(&self.ip) {
            return Ok(false);
        }
        // ud2 / int3 / hlt: traps the compiler puts where control can't continue
        if matches!(i.mnemonic(), Mnemonic::Ud2 | Mnemonic::Int3 | Mnemonic::Hlt) {
            self.end_block(f, Terminator::Unreachable);
            return Ok(true);
        }
        if self.asm && (!handled(&i) || self.forced.contains(&self.ip)) {
            return self.lift_asm(f, &i).map(|_| false);
        }
        match i.flow_control() {
            FlowControl::Next => self.lift_data(f, &i).map(|_| false),
            FlowControl::Call | FlowControl::IndirectCall => {
                self.call(f, &i)?;
                let target = match i.op0_kind() {
                    OpKind::NearBranch64 => Some(i.near_branch_target()),
                    OpKind::Memory if i.is_ip_rel_memory_operand() => Some(i.ip_rel_memory_address()),
                    _ => None,
                };
                if noreturn(self.ip, target) {
                    self.fall = self.calls.last().map(|c| c.0);
                    self.end_block(f, Terminator::Unreachable);
                    return Ok(true);
                }
                Ok(false)
            }
            FlowControl::ConditionalBranch if i.condition_code() != ConditionCode::None => {
                let c = self.condition(f, i.condition_code())?;
                let e = self.target(i.next_ip())?;
                let Some(t) = self.block_at(i.near_branch_target()) else {
                    // Out of the function: a block of its own tail-calls the target.
                    let stub = self.leaders.len() + self.nstub;
                    self.nstub += 1;
                    self.end_block(f, Terminator::Branch { c, t: BlockId::new(stub), f: e, args: ListRef::EMPTY });
                    self.begin_block(stub, f);
                    let term = self.jump_out(f, &i, noreturn)?;
                    self.end_block(f, term);
                    return Ok(true);
                };
                self.end_block(f, Terminator::Branch { c, t, f: e, args: ListRef::EMPTY });
                Ok(true)
            }
            FlowControl::UnconditionalBranch if i.op0_kind() == OpKind::NearBranch64 => {
                let term = match self.block_at(i.near_branch_target()) {
                    Some(to) => Terminator::Jump { to, args: ListRef::EMPTY },
                    // Jump out of this function: a tail call.
                    None => self.jump_out(f, &i, noreturn)?,
                };
                self.end_block(f, term);
                Ok(true)
            }
            // A jump table found in pass 1: switch on the index the table load read.
            FlowControl::IndirectBranch if self.tables.iter().any(|t| t.jmp == self.ip) => {
                let t = self.tables.iter().position(|t| t.jmp == self.ip).unwrap();
                // the load was in another block: the index isn't known here
                let Some(v) = self.switch_index else { return Err(self.unsupported()) };
                self.switches.push((BlockId::new(self.cur), t));
                self.end_block(f, Terminator::Switch { v, table: ListRef::EMPTY, default: BlockId::new(0) });
                for k in 0..self.case_stubs.len() {
                    let (jmp, target) = self.case_stubs[k];
                    if jmp == self.ip {
                        let stub = self.leaders.len() + self.nstub;
                        self.nstub += 1;
                        self.begin_block(stub, f);
                        let term = self.tail_call(f, target);
                        self.end_block(f, term);
                    }
                }
                Ok(true)
            }
            // Any other indirect jump leaves the function: `jmp [rip+x]` is a tail
            // call through the GOT, `jmp rax` or `jmp [rax+8]` one through a pointer.
            FlowControl::IndirectBranch => {
                let callee = self.callee(f, &i)?;
                let regs = self.call_regs(f);
                self.tails.push((BlockId::new(self.cur), regs));
                self.end_block(f, Terminator::TailCall { callee, args: ListRef::EMPTY });
                Ok(true)
            }
            FlowControl::Return => {
                // a long double result (in st0) isn't modelled
                if self.fdepth != 0 {
                    return Err(self.unsupported());
                }
                if self.track_exits {
                    let mut regs = [ValueId::from_u32(0); EXIT_LEN];
                    for (v, r) in regs.iter_mut().zip(EXIT_REGS) {
                        *v = self.read_full(f, r.number());
                    }
                    if self.xmm {
                        for k in 0..32 {
                            regs[EXIT_XMM0 + k] = self.read_full(f, NGPR + k);
                        }
                    }
                    let exit = self.emit(f, InstKind::Exit { regs: ListRef::EMPTY }, TyId::UNIT);
                    self.exits.push((exit, regs));
                }
                let v = self.read(f, Register::RAX)?;
                self.end_block(f, Terminator::Return(Some(v)));
                Ok(true)
            }
            _ => Err(self.unsupported()),
        }
    }

    fn lift_data(&mut self, f: &mut Function, i: &Instruction) -> Result<(), LiftError> {
        match i.mnemonic() {
            // endbr64: CET landing pad; pause: a spin-loop hint. Neither touches data.
            // prefetches are hints with no visible effect
            Mnemonic::Nop | Mnemonic::Endbr64 | Mnemonic::Pause | Mnemonic::Prefetcht0 | Mnemonic::Prefetcht1
            | Mnemonic::Prefetcht2 | Mnemonic::Prefetchnta | Mnemonic::Prefetchw | Mnemonic::Prefetch => {}
            Mnemonic::Mov => match (i.op0_kind(), i.op1_kind()) {
                // mov rax, rcx: no instruction at all, just rename in the register file
                (OpKind::Register, OpKind::Register) => {
                    let v = self.read(f, i.op1_register())?;
                    self.write(f, i.op0_register(), v)?;
                }
                // mov rax, [rdi+8]
                (OpKind::Register, OpKind::Memory) => {
                    let sz = i.op0_register().size();
                    let v = self.operand(f, i, 1, sz)?;
                    self.write(f, i.op0_register(), v)?;
                }
                // mov [rax+8], rcx
                (OpKind::Memory, OpKind::Register) => {
                    let val = self.read(f, i.op1_register())?;
                    let ptr = self.ea(f, i)?;
                    self.emit(f, InstKind::Store { ptr, val, align: 1 }, TyId::UNIT);
                }
                // mov rax, imm
                (OpKind::Register, k) if is_imm(k) => {
                    let ty = TyId::unknown(i.op0_register().size());
                    let c = self.konst(f, imm(i, 1, i.op0_register().size()), ty);
                    self.write(f, i.op0_register(), c)?;
                }
                // mov qword ptr [rax], imm
                (OpKind::Memory, k) if is_imm(k) => {
                    let sz = i.memory_size().size();
                    let val = self.konst(f, imm(i, 1, sz), TyId::unknown(sz));
                    let ptr = self.ea(f, i)?;
                    self.emit(f, InstKind::Store { ptr, val, align: 1 }, TyId::UNIT);
                }
                _ => return Err(self.unsupported()),
            },
            Mnemonic::Lea => {
                let p = self.ea(f, i)?;
                match i.op0_register().size() {
                    8 => self.write(f, i.op0_register(), p)?,
                    // lea eax, [rdi+rsi]: 32-bit arithmetic on the low half
                    4 => {
                        let v = self.emit(f, InstKind::Cast { kind: CastKind::Trunc, v: p }, TyId::B4);
                        self.write(f, i.op0_register(), v)?;
                    }
                    _ => return Err(self.unsupported()),
                }
            }
            Mnemonic::Add | Mnemonic::Sub | Mnemonic::And | Mnemonic::Or | Mnemonic::Xor
            | Mnemonic::Cmp | Mnemonic::Test => self.alu(f, i)?,
            Mnemonic::Inc | Mnemonic::Dec | Mnemonic::Neg | Mnemonic::Not => self.unary(f, i)?,
            Mnemonic::Shl | Mnemonic::Shr | Mnemonic::Sar => self.shift(f, i)?,
            Mnemonic::Mul | Mnemonic::Imul if i.op_count() == 1 => self.mul_wide(f, i)?,
            // two- and three-operand forms; one-operand imul writes rdx:rax
            Mnemonic::Imul => {
                let dst = i.op0_register();
                let sz = dst.size();
                let (lhs, rhs) = if i.op_count() == 3 {
                    (self.operand(f, i, 1, sz)?, self.konst(f, imm(i, 2, sz), TyId::unknown(sz)))
                } else {
                    (self.read(f, dst)?, self.operand(f, i, 1, sz)?)
                };
                let v = self.emit(f, InstKind::Bin { op: BinOp::Mul, lhs, rhs }, TyId::unknown(sz));
                self.write(f, dst, v)?;
                self.flags = Flags::Mul { lhs, rhs, lo: v, signed: true };
            }
            // rep movs: memcpy(rdi, rsi, rcx * size), with the direction flag clear
            // (before `movsd`, which is also the SSE move)
            Mnemonic::Movsb | Mnemonic::Movsw | Mnemonic::Movsd | Mnemonic::Movsq
                if i.has_rep_prefix() && i.op0_kind() == OpKind::MemoryESRDI =>
            {
                let size = i.memory_size().size() as u64;
                let n = self.read(f, Register::RCX)?;
                let len = if size == 1 {
                    n
                } else {
                    let k = self.konst(f, size, TyId::B8);
                    self.emit(f, InstKind::Bin { op: BinOp::Mul, lhs: n, rhs: k }, TyId::B8)
                };
                let dst = self.read(f, Register::RDI)?;
                let src = self.read(f, Register::RSI)?;
                let d = self.emit(f, InstKind::IntToPtr(dst), TyId::PTR);
                let sp = self.emit(f, InstKind::IntToPtr(src), TyId::PTR);
                self.emit(f, InstKind::MemCopy { dst: d, src: sp, len }, TyId::UNIT);
                let d2 = self.emit(f, InstKind::Bin { op: BinOp::Add, lhs: dst, rhs: len }, TyId::B8);
                let s2 = self.emit(f, InstKind::Bin { op: BinOp::Add, lhs: src, rhs: len }, TyId::B8);
                let zero = self.konst(f, 0, TyId::B8);
                self.write(f, Register::RDI, d2)?;
                self.write(f, Register::RSI, s2)?;
                self.write(f, Register::RCX, zero)?;
            }
            // rep stos: rcx copies of al / ax / eax / rax from rdi on
            Mnemonic::Stosb | Mnemonic::Stosw | Mnemonic::Stosd | Mnemonic::Stosq if i.has_rep_prefix() => {
                let size = i.memory_size().size() as u64;
                let val = match size {
                    1 => Register::AL,
                    2 => Register::AX,
                    4 => Register::EAX,
                    _ => Register::RAX,
                };
                let val = self.read(f, val)?;
                let count = self.read(f, Register::RCX)?;
                let dst = self.read(f, Register::RDI)?;
                let d = self.emit(f, InstKind::IntToPtr(dst), TyId::PTR);
                self.emit(f, InstKind::MemFill { dst: d, val, count }, TyId::UNIT);
                let len = if size == 1 {
                    count
                } else {
                    let k = self.konst(f, size, TyId::B8);
                    self.emit(f, InstKind::Bin { op: BinOp::Mul, lhs: count, rhs: k }, TyId::B8)
                };
                let d2 = self.emit(f, InstKind::Bin { op: BinOp::Add, lhs: dst, rhs: len }, TyId::B8);
                let zero = self.konst(f, 0, TyId::B8);
                self.write(f, Register::RDI, d2)?;
                self.write(f, Register::RCX, zero)?;
            }
            // The CPU's answer, read through `simd::cpu`: feature checks pick
            // a code path on the machine the output runs on.
            Mnemonic::Cpuid | Mnemonic::Xgetbv => {
                let cpuid = i.mnemonic() == Mnemonic::Cpuid;
                let c = self.read(f, Register::ECX)?;
                let a = if cpuid { self.read(f, Register::EAX)? } else { c };
                use Register::*;
                let (op, outs) = match cpuid {
                    true => (LaneOp::Cpuid, &[(0, EAX), (1, EBX), (2, ECX), (3, EDX)][..]),
                    false => (LaneOp::Xgetbv, &[(0, EAX), (3, EDX)][..]),
                };
                // All outputs read the inputs before any is written.
                let mut vals = [a; 4];
                for (v, &(k, _)) in vals.iter_mut().zip(outs) {
                    *v = self.emit(f, InstKind::Bin { op: BinOp::Lane(op, k), lhs: a, rhs: c }, TyId::B4);
                }
                for (&(_, r), v) in outs.iter().zip(vals) {
                    self.write(f, r, v)?;
                }
            }
            Mnemonic::Bswap => {
                let r = i.op0_register();
                let v = self.read(f, r)?;
                let v = self.emit(f, InstKind::Un { op: UnOp::Bswap, v }, TyId::unknown(r.size()));
                self.write(f, r, v)?;
            }
            // tzcnt / lzcnt / popcnt: the count of a zero source is the width, like
            // Rust's; bsf / bsr: the index of the lowest / highest set bit, with ZF
            // set (and the result undefined) for a zero source
            Mnemonic::Tzcnt | Mnemonic::Lzcnt | Mnemonic::Popcnt | Mnemonic::Bsf | Mnemonic::Bsr => {
                let dst = i.op0_register();
                let sz = dst.size();
                let ty = TyId::unknown(sz);
                let v = self.operand(f, i, 1, sz)?;
                let op = match i.mnemonic() {
                    Mnemonic::Tzcnt | Mnemonic::Bsf => UnOp::Ctz,
                    Mnemonic::Popcnt => UnOp::Popcnt,
                    _ => UnOp::Clz,
                };
                let mut res = self.emit(f, InstKind::Un { op, v }, ty);
                if i.mnemonic() == Mnemonic::Bsr {
                    let top = self.konst(f, sz as u64 * 8 - 1, ty);
                    res = self.emit(f, InstKind::Bin { op: BinOp::Xor, lhs: res, rhs: top }, ty);
                }
                self.write(f, dst, res)?;
                self.flags = match i.mnemonic() {
                    // ZF: the source was zero
                    Mnemonic::Bsf | Mnemonic::Bsr | Mnemonic::Popcnt => Flags::Logic { res: v },
                    _ => Flags::Res { res },
                };
            }
            Mnemonic::Shld | Mnemonic::Shrd => self.double_shift(f, i)?,
            Mnemonic::Rol | Mnemonic::Ror => {
                let sz = self.op_size(i, 0)?;
                let ty = TyId::unknown(sz);
                let count = match i.op1_kind() {
                    k if is_imm(k) => self.konst(f, i.immediate(1) & if sz == 8 { 63 } else { 31 }, TyId::B1),
                    OpKind::Register if i.op1_register() == Register::CL => self.read(f, Register::CL)?,
                    _ => return Err(self.unsupported()),
                };
                let dst = self.dst(f, i)?;
                let v = self.get(f, dst, sz)?;
                let op = if i.mnemonic() == Mnemonic::Rol { BinOp::RotL } else { BinOp::RotR };
                // Rust rotates by the count modulo the width, as x86 does after masking it
                let res = self.emit(f, InstKind::Bin { op, lhs: v, rhs: count }, ty);
                self.put(f, dst, res)?;
                self.flags = Flags::Unknown; // only CF and OF, which nothing here models
            }
            // adc / sbb: with the carry the previous instruction left
            Mnemonic::Adc | Mnemonic::Sbb => {
                let sz = self.op_size(i, 0)?;
                let ty = TyId::unknown(sz);
                let cf = self.condition(f, ConditionCode::b)?;
                let cf = self.emit(f, InstKind::Cast { kind: CastKind::ZExt, v: cf }, ty);
                let dst = self.dst(f, i)?;
                let a = self.get(f, dst, sz)?;
                let same = i.op0_kind() == OpKind::Register && i.op1_kind() == OpKind::Register && i.op0_register() == i.op1_register();
                let b = if same { a } else { self.operand(f, i, 1, sz)? };
                let op = if i.mnemonic() == Mnemonic::Adc { BinOp::Add } else { BinOp::Sub };
                let t = self.emit(f, InstKind::Bin { op, lhs: a, rhs: b }, ty);
                let res = self.emit(f, InstKind::Bin { op, lhs: t, rhs: cf }, ty);
                self.put(f, dst, res)?;
                self.flags = Flags::AddCarry { lhs: a, rhs: b, cin: cf, res, sub: i.mnemonic() == Mnemonic::Sbb };
            }
            // bt: CF = bit n of the operand; bts / btr / btc then set, clear or flip it
            Mnemonic::Bt | Mnemonic::Bts | Mnemonic::Btr | Mnemonic::Btc => {
                let sz = self.op_size(i, 0)?;
                let ty = TyId::unknown(sz);
                let n = match i.op1_kind() {
                    k if is_imm(k) => self.konst(f, i.immediate(1) & (sz as u64 * 8 - 1), ty),
                    // with a memory operand, a register bit offset can reach past it
                    OpKind::Register if i.op0_kind() == OpKind::Register => {
                        let r = self.read(f, i.op1_register())?;
                        let m = self.konst(f, sz as u64 * 8 - 1, ty);
                        self.emit(f, InstKind::Bin { op: BinOp::And, lhs: r, rhs: m }, ty)
                    }
                    _ => return Err(self.unsupported()),
                };
                let dst = self.dst(f, i)?;
                let v = self.get(f, dst, sz)?;
                let sh = self.emit(f, InstKind::Bin { op: BinOp::LShr, lhs: v, rhs: n }, ty);
                let one = self.konst(f, 1, ty);
                let bit = self.emit(f, InstKind::Bin { op: BinOp::And, lhs: sh, rhs: one }, ty);
                let zero = self.konst(f, 0, ty);
                let cf = self.emit(f, InstKind::Cmp { cc: Cond::Ne, lhs: bit, rhs: zero }, TyId::BOOL);
                if i.mnemonic() != Mnemonic::Bt {
                    let mask = self.emit(f, InstKind::Bin { op: BinOp::Shl, lhs: one, rhs: n }, ty);
                    let (op, mask) = match i.mnemonic() {
                        Mnemonic::Bts => (BinOp::Or, mask),
                        Mnemonic::Btr => (BinOp::And, self.emit(f, InstKind::Un { op: UnOp::Not, v: mask }, ty)),
                        _ => (BinOp::Xor, mask),
                    };
                    let res = self.emit(f, InstKind::Bin { op, lhs: v, rhs: mask }, ty);
                    self.put(f, dst, res)?;
                }
                self.flags = Flags::Carry { cf };
            }
            // xchg: swap (with memory it is atomic; the lifted code is not)
            Mnemonic::Xchg => {
                let sz = self.op_size(i, 0)?;
                let (a, b) = (self.dst(f, i)?, i.op1_register());
                let va = self.get(f, a, sz)?;
                let vb = self.read(f, b)?;
                self.put(f, a, vb)?;
                self.write(f, b, va)?;
            }
            // xadd [m], r: m += r, r = the old m (atomic with `lock`; the lifted code is not)
            Mnemonic::Xadd => {
                let sz = self.op_size(i, 0)?;
                let ty = TyId::unknown(sz);
                let dst = self.dst(f, i)?;
                let old = self.get(f, dst, sz)?;
                let r = self.read(f, i.op1_register())?;
                let res = self.emit(f, InstKind::Bin { op: BinOp::Add, lhs: old, rhs: r }, ty);
                self.put(f, dst, res)?;
                self.write(f, i.op1_register(), old)?;
                self.flags = Flags::Add { lhs: old, rhs: r, res };
            }
            // cmpxchg [m], r: if m == rax { m = r } else { rax = m }; ZF says which.
            // The destination is written either way (atomic with `lock`; the lifted
            // code is not).
            Mnemonic::Cmpxchg => {
                let sz = self.op_size(i, 0)?;
                let ty = TyId::unknown(sz);
                let acc = match sz { 8 => Register::RAX, 4 => Register::EAX, 2 => Register::AX, _ => Register::AL };
                let dst = self.dst(f, i)?;
                let old = self.get(f, dst, sz)?;
                let a = self.read(f, acc)?;
                let r = self.read(f, i.op1_register())?;
                let eq = self.emit(f, InstKind::Cmp { cc: Cond::Eq, lhs: old, rhs: a }, TyId::BOOL);
                let new = self.emit(f, InstKind::Select { c: eq, t: r, f: old }, ty);
                self.put(f, dst, new)?;
                // rax = m, which equals rax when the exchange happened
                self.write(f, acc, old)?;
                self.flags = Flags::Sub { lhs: a, rhs: old };
            }
            Mnemonic::Movzx | Mnemonic::Movsx | Mnemonic::Movsxd => {
                let dst = i.op0_register();
                let src_sz = self.op_size(i, 1)?;
                let v = self.operand(f, i, 1, src_sz)?;
                let v = if src_sz == dst.size() {
                    v // movsxd r32, r/m32 is a plain mov
                } else {
                    let kind = if i.mnemonic() == Mnemonic::Movzx { CastKind::ZExt } else { CastKind::SExt };
                    self.emit(f, InstKind::Cast { kind, v }, TyId::unknown(dst.size()))
                };
                self.write(f, dst, v)?;
            }
            Mnemonic::Div | Mnemonic::Idiv => self.div(f, i)?,
            // cqo / cdq / cwd: rdx (edx, dx) = the sign of rax (eax, ax), ready for idiv
            Mnemonic::Cqo | Mnemonic::Cdq | Mnemonic::Cwd => {
                let (lo, hi, bits) = match i.mnemonic() {
                    Mnemonic::Cqo => (Register::RAX, Register::RDX, 63),
                    Mnemonic::Cdq => (Register::EAX, Register::EDX, 31),
                    _ => (Register::AX, Register::DX, 15),
                };
                let v = self.read(f, lo)?;
                let ty = f.insts[v].ty;
                let s = self.konst(f, bits, TyId::B1);
                let sign = self.emit(f, InstKind::Bin { op: BinOp::AShr, lhs: v, rhs: s }, ty);
                self.write(f, hi, sign)?;
            }
            // cdqe / cwde / cbw: sign-extend the low half of rax (eax, ax) in place
            Mnemonic::Cdqe | Mnemonic::Cwde | Mnemonic::Cbw => {
                let (src, dst) = match i.mnemonic() {
                    Mnemonic::Cdqe => (Register::EAX, Register::RAX),
                    Mnemonic::Cwde => (Register::AX, Register::EAX),
                    _ => (Register::AL, Register::AX),
                };
                let v = self.read(f, src)?;
                let v = self.emit(f, InstKind::Cast { kind: CastKind::SExt, v }, TyId::unknown(dst.size()));
                self.write(f, dst, v)?;
            }
            Mnemonic::Push => {
                let val = match i.op0_kind() {
                    OpKind::Register if i.op0_register().size() == 8 => self.read(f, i.op0_register())?,
                    k if is_imm(k) => self.konst(f, imm(i, 0, 8), TyId::B8),
                    OpKind::Memory if i.memory_size().size() == 8 => {
                        let ptr = self.ea(f, i)?;
                        self.emit(f, InstKind::Load { ptr, align: 1, volatile: false }, TyId::B8)
                    }
                    _ => return Err(self.unsupported()),
                };
                let sp = self.read(f, Register::RSP)?;
                let sp = self.emit(f, InstKind::PtrOffset { base: sp, index: None, scale: 1, disp: -8 }, TyId::PTR);
                self.emit(f, InstKind::Store { ptr: sp, val, align: 1 }, TyId::UNIT);
                self.write(f, Register::RSP, sp)?;
            }
            Mnemonic::Pop => self.pop(f, i)?,
            // leave: mov rsp, rbp; pop rbp
            Mnemonic::Leave => {
                let bp = self.read(f, Register::RBP)?;
                self.write(f, Register::RSP, bp)?;
                self.pop(f, i)?;
            }
            m if cmov_or_setcc(m) == Some(true) => {
                let dst = i.op0_register();
                let sz = dst.size();
                // the old value stays when the condition is false (a 32-bit cmov still
                // zero-extends it), and a memory source is loaded either way
                let old = self.read(f, dst)?;
                let src = self.operand(f, i, 1, sz)?;
                let c = self.condition(f, i.condition_code())?;
                let v = self.emit(f, InstKind::Select { c, t: src, f: old }, TyId::unknown(sz));
                self.write(f, dst, v)?;
            }
            m if cmov_or_setcc(m) == Some(false) => {
                let c = self.condition(f, i.condition_code())?;
                let v = self.emit(f, InstKind::Cast { kind: CastKind::ZExt, v: c }, TyId::B1);
                let dst = self.dst(f, i)?;
                self.put(f, dst, v)?;
            }
            m if sse::handled(m) => self.sse(f, i)?,
            m if x87::handled(m) => self.x87(f, i)?,
            _ => return Err(self.unsupported()),
        }
        Ok(())
    }

    /// ALU ops with register, immediate or memory operands (`add [rdi], eax` loads,
    /// adds and stores).
    fn alu(&mut self, f: &mut Function, i: &Instruction) -> Result<(), LiftError> {
        let m = i.mnemonic();
        let sz = self.op_size(i, 0)?;
        let ty = TyId::unknown(sz);
        let same_reg = i.op0_kind() == OpKind::Register && i.op1_kind() == OpKind::Register
            && i.op1_register() == i.op0_register();

        // xor eax, eax: the zeroing idiom becomes a constant, no data dependency
        if m == Mnemonic::Xor && same_reg {
            let z = self.konst(f, 0, ty);
            self.write(f, i.op0_register(), z)?;
            self.flags = Flags::Logic { res: z };
            return Ok(());
        }

        let dst = self.dst(f, i)?;
        let lhs = self.get(f, dst, sz)?;
        let rhs = if same_reg { lhs } else { self.operand(f, i, 1, sz)? };
        self.flags = match m {
            // cmp emits nothing; the instruction that reads the flags emits the comparison
            Mnemonic::Cmp => Flags::Sub { lhs, rhs },
            // test rax, rax: ZF/SF come straight from rax
            Mnemonic::Test if same_reg => Flags::Logic { res: lhs },
            _ => {
                let op = match m {
                    Mnemonic::Add => BinOp::Add,
                    Mnemonic::Sub => BinOp::Sub,
                    Mnemonic::Or => BinOp::Or,
                    Mnemonic::Xor => BinOp::Xor,
                    _ => BinOp::And, // and, test
                };
                let res = self.emit(f, InstKind::Bin { op, lhs, rhs }, ty);
                if m != Mnemonic::Test {
                    self.put(f, dst, res)?;
                }
                match m {
                    Mnemonic::Sub => Flags::Sub { lhs, rhs },
                    Mnemonic::Add => Flags::Add { lhs, rhs, res },
                    _ => Flags::Logic { res },
                }
            }
        };
        Ok(())
    }

    /// inc, dec, neg, not on a register or memory.
    fn unary(&mut self, f: &mut Function, i: &Instruction) -> Result<(), LiftError> {
        let sz = self.op_size(i, 0)?;
        let ty = TyId::unknown(sz);
        let dst = self.dst(f, i)?;
        let v = self.get(f, dst, sz)?;
        let (kind, flags) = match i.mnemonic() {
            // CF is left as it was; the rest are the flags of the add or sub
            Mnemonic::Inc => {
                let one = self.konst(f, 1, ty);
                let res = self.emit(f, InstKind::Bin { op: BinOp::Add, lhs: v, rhs: one }, ty);
                self.put(f, dst, res)?;
                self.flags = Flags::Inc { lhs: v, one, res };
                return Ok(());
            }
            Mnemonic::Dec => {
                let one = self.konst(f, 1, ty);
                let res = self.emit(f, InstKind::Bin { op: BinOp::Sub, lhs: v, rhs: one }, ty);
                self.put(f, dst, res)?;
                self.flags = Flags::Dec { lhs: v, one };
                return Ok(());
            }
            // neg sets the flags of `0 - v`
            Mnemonic::Neg => {
                let zero = self.konst(f, 0, ty);
                (InstKind::Un { op: UnOp::Neg, v }, Some(Flags::Sub { lhs: zero, rhs: v }))
            }
            _ => (InstKind::Un { op: UnOp::Not, v }, Some(self.flags)), // not leaves the flags alone
        };
        let res = self.emit(f, kind, ty);
        self.put(f, dst, res)?;
        self.flags = flags.unwrap_or(Flags::Res { res });
        Ok(())
    }

    /// shl / shr / sar by an immediate or by cl.
    fn shift(&mut self, f: &mut Function, i: &Instruction) -> Result<(), LiftError> {
        let sz = self.op_size(i, 0)?;
        let ty = TyId::unknown(sz);
        let op = match i.mnemonic() {
            Mnemonic::Shl => BinOp::Shl,
            Mnemonic::Shr => BinOp::LShr,
            _ => BinOp::AShr,
        };
        // x86 masks the count to 5 bits (6 for 64-bit operands)
        let mask = if sz == 8 { 63 } else { 31 };
        let (count, flags_known) = match i.op1_kind() {
            k if is_imm(k) => {
                let n = i.immediate(1) & mask;
                if n == 0 {
                    return Ok(()); // no effect, not even on the flags
                }
                (self.konst(f, n, TyId::B1), n < sz as u64 * 8)
            }
            // a count of 0 leaves the flags alone, so they are unknown after this
            OpKind::Register if i.op1_register() == Register::CL => {
                let cl = self.read(f, Register::CL)?;
                let m = self.konst(f, mask, TyId::B1);
                (self.emit(f, InstKind::Bin { op: BinOp::And, lhs: cl, rhs: m }, TyId::B1), false)
            }
            _ => return Err(self.unsupported()),
        };
        let dst = self.dst(f, i)?;
        let v = self.get(f, dst, sz)?;
        let res = if sz >= 4 {
            self.emit(f, InstKind::Bin { op, lhs: v, rhs: count }, ty)
        } else {
            // 8/16-bit operands can shift everything out (counts up to 31), which
            // Rust's shifts (count modulo the width) don't: shift a 32-bit copy
            let kind = if matches!(op, BinOp::AShr) { CastKind::SExt } else { CastKind::ZExt };
            let w = self.emit(f, InstKind::Cast { kind, v }, TyId::B4);
            let r = self.emit(f, InstKind::Bin { op, lhs: w, rhs: count }, TyId::B4);
            self.emit(f, InstKind::Cast { kind: CastKind::Trunc, v: r }, ty)
        };
        self.put(f, dst, res)?;
        self.flags = if flags_known { Flags::Res { res } } else { Flags::Unknown };
        Ok(())
    }

    /// shld d, s, n: d = d << n | s >> (w - n); shrd d, s, n: d = d >> n | s << (w - n).
    /// A count of 0 leaves d (and the flags) alone.
    fn double_shift(&mut self, f: &mut Function, i: &Instruction) -> Result<(), LiftError> {
        let sz = self.op_size(i, 0)?;
        if sz < 4 {
            return Err(self.unsupported());
        }
        let ty = TyId::unknown(sz);
        let mask = if sz == 8 { 63 } else { 31 };
        let count = match i.op2_kind() {
            k if is_imm(k) => {
                if i.immediate(2) & mask == 0 {
                    return Ok(());
                }
                self.konst(f, i.immediate(2) & mask, TyId::B1)
            }
            OpKind::Register if i.op2_register() == Register::CL => {
                let cl = self.read(f, Register::CL)?;
                let m = self.konst(f, mask, TyId::B1);
                self.emit(f, InstKind::Bin { op: BinOp::And, lhs: cl, rhs: m }, TyId::B1)
            }
            _ => return Err(self.unsupported()),
        };
        let dst = self.dst(f, i)?;
        let d = self.get(f, dst, sz)?;
        let s = self.read(f, i.op1_register())?;
        let w = self.konst(f, sz as u64 * 8, TyId::B1);
        let rest = self.emit(f, InstKind::Bin { op: BinOp::Sub, lhs: w, rhs: count }, TyId::B1);
        let (into_d, from_s) = if i.mnemonic() == Mnemonic::Shld { (BinOp::Shl, BinOp::LShr) } else { (BinOp::LShr, BinOp::Shl) };
        let a = self.emit(f, InstKind::Bin { op: into_d, lhs: d, rhs: count }, ty);
        let b = self.emit(f, InstKind::Bin { op: from_s, lhs: s, rhs: rest }, ty);
        let res = self.emit(f, InstKind::Bin { op: BinOp::Or, lhs: a, rhs: b }, ty);
        // a count of 0 would shift s by the full width, which wraps to no shift
        let zero = self.konst(f, 0, TyId::B1);
        let none = self.emit(f, InstKind::Cmp { cc: Cond::Eq, lhs: count, rhs: zero }, TyId::BOOL);
        let res = self.emit(f, InstKind::Select { c: none, t: d, f: res }, ty);
        self.put(f, dst, res)?;
        self.flags = Flags::Unknown; // a count of 0 leaves them alone
        Ok(())
    }

    /// div / idiv with a 32 or 64-bit divisor. The dividend is rdx:rax. When rdx
    /// only extends rax (zero for div, `xor edx, edx`; the sign of rax for idiv,
    /// `cqo`) it is a plain division of rax. Otherwise a 32-bit division divides
    /// edx:eax as one 64-bit value (`xor edx, edx; idiv` of a value known to be
    /// non-negative, `cdq` in another block, or a real 64-by-32 division); the
    /// 64-bit form would need 128-bit values and stays unsupported.
    fn div(&mut self, f: &mut Function, i: &Instruction) -> Result<(), LiftError> {
        let sz = self.op_size(i, 0)?;
        let (lo, hi) = match sz {
            8 => (Register::RAX, Register::RDX),
            4 => (Register::EAX, Register::EDX),
            _ => return self.div_narrow(f, i, sz),
        };
        let l = self.read(f, lo)?;
        let h = self.read(f, hi)?;
        let signed = i.mnemonic() == Mnemonic::Idiv;
        let extends = if signed {
            matches!(f.insts[h].kind, InstKind::Bin { op: BinOp::AShr, lhs, rhs }
                if lhs == l && const_of(f, rhs) == Some(sz as u64 * 8 - 1))
        } else {
            // `xor edx, edx` leaves rdx = ZExt(0)
            match f.insts[h].kind {
                InstKind::Cast { kind: CastKind::ZExt, v } => const_of(f, v) == Some(0),
                _ => const_of(f, h) == Some(0),
            }
        };
        if !extends && sz == 4 {
            let wide = TyId::unknown(8);
            let n = self.konst(f, 32, TyId::B1);
            let l = self.emit(f, InstKind::Cast { kind: CastKind::ZExt, v: l }, wide);
            let h = self.emit(f, InstKind::Cast { kind: CastKind::ZExt, v: h }, wide);
            let h = self.emit(f, InstKind::Bin { op: BinOp::Shl, lhs: h, rhs: n }, wide);
            let dividend = self.emit(f, InstKind::Bin { op: BinOp::Or, lhs: h, rhs: l }, wide);
            let d = self.operand(f, i, 0, sz)?;
            let ext = if signed { CastKind::SExt } else { CastKind::ZExt };
            let d = self.emit(f, InstKind::Cast { kind: ext, v: d }, wide);
            let (qop, rop) = if signed { (BinOp::SDiv, BinOp::SRem) } else { (BinOp::UDiv, BinOp::URem) };
            let q = self.emit(f, InstKind::Bin { op: qop, lhs: dividend, rhs: d }, wide);
            let r = self.emit(f, InstKind::Bin { op: rop, lhs: dividend, rhs: d }, wide);
            let ty = TyId::unknown(sz);
            let q = self.emit(f, InstKind::Cast { kind: CastKind::Trunc, v: q }, ty);
            let r = self.emit(f, InstKind::Cast { kind: CastKind::Trunc, v: r }, ty);
            self.write(f, lo, q)?;
            self.write(f, hi, r)?;
            self.flags = Flags::Unknown;
            return Ok(());
        }
        if !extends {
            return Err(self.unsupported());
        }
        let d = self.operand(f, i, 0, sz)?;
        let ty = TyId::unknown(sz);
        let (qop, rop) = if signed { (BinOp::SDiv, BinOp::SRem) } else { (BinOp::UDiv, BinOp::URem) };
        let q = self.emit(f, InstKind::Bin { op: qop, lhs: l, rhs: d }, ty);
        let r = self.emit(f, InstKind::Bin { op: rop, lhs: l, rhs: d }, ty);
        self.write(f, lo, q)?;
        self.write(f, hi, r)?;
        self.flags = Flags::Unknown;
        Ok(())
    }

    /// 8-bit div: al, ah = ax / src, ax % src; 16-bit: ax, dx = dx:ax / src, dx:ax % src.
    /// The dividend is twice the operand's width, so divide at that width (a
    /// quotient that doesn't fit traps on x86; here it is truncated).
    fn div_narrow(&mut self, f: &mut Function, i: &Instruction, sz: usize) -> Result<(), LiftError> {
        let signed = i.mnemonic() == Mnemonic::Idiv;
        let ext = if signed { CastKind::SExt } else { CastKind::ZExt };
        let (wide, n) = (TyId::unknown(2 * sz), self.konst(f, 8 * sz as u64, TyId::B1));
        let dividend = if sz == 1 {
            self.read(f, Register::AX)?
        } else {
            let (lo, hi) = (self.read(f, Register::AX)?, self.read(f, Register::DX)?);
            let lo = self.emit(f, InstKind::Cast { kind: CastKind::ZExt, v: lo }, wide);
            let hi = self.emit(f, InstKind::Cast { kind: CastKind::ZExt, v: hi }, wide);
            let hi = self.emit(f, InstKind::Bin { op: BinOp::Shl, lhs: hi, rhs: n }, wide);
            self.emit(f, InstKind::Bin { op: BinOp::Or, lhs: hi, rhs: lo }, wide)
        };
        let d = self.operand(f, i, 0, sz)?;
        let d = self.emit(f, InstKind::Cast { kind: ext, v: d }, wide);
        let (qop, rop) = if signed { (BinOp::SDiv, BinOp::SRem) } else { (BinOp::UDiv, BinOp::URem) };
        let q = self.emit(f, InstKind::Bin { op: qop, lhs: dividend, rhs: d }, wide);
        let r = self.emit(f, InstKind::Bin { op: rop, lhs: dividend, rhs: d }, wide);
        let ty = TyId::unknown(sz);
        let q = self.emit(f, InstKind::Cast { kind: CastKind::Trunc, v: q }, ty);
        let r = self.emit(f, InstKind::Cast { kind: CastKind::Trunc, v: r }, ty);
        let (qr, rr) = if sz == 1 { (Register::AL, Register::AH) } else { (Register::AX, Register::DX) };
        self.write(f, qr, q)?;
        self.write(f, rr, r)?;
        self.flags = Flags::Unknown;
        Ok(())
    }

    /// One-operand mul / imul: rdx:rax = rax * src (edx:eax, dx:ax for narrower
    /// operands, and ax = al * src for bytes).
    fn mul_wide(&mut self, f: &mut Function, i: &Instruction) -> Result<(), LiftError> {
        let sz = self.op_size(i, 0)?;
        let signed = i.mnemonic() == Mnemonic::Imul;
        let (lo_reg, hi_reg) = match sz {
            8 => (Register::RAX, Register::RDX),
            4 => (Register::EAX, Register::EDX),
            2 => (Register::AX, Register::DX),
            _ => {
                // ax = al * src: the whole product in one register
                let a = self.read(f, Register::AL)?;
                let b = self.operand(f, i, 0, 1)?;
                let p = self.wide_product(f, a, b, signed);
                let ax = self.emit(f, InstKind::Cast { kind: CastKind::Trunc, v: p }, TyId::B2);
                let lo = self.emit(f, InstKind::Cast { kind: CastKind::Trunc, v: p }, TyId::B1);
                self.write(f, Register::AX, ax)?;
                self.flags = Flags::Mul { lhs: a, rhs: b, lo, signed };
                return Ok(());
            }
        };
        let a = self.read(f, lo_reg)?;
        let b = self.operand(f, i, 0, sz)?;
        let ty = TyId::unknown(sz);
        let lo = self.emit(f, InstKind::Bin { op: BinOp::Mul, lhs: a, rhs: b }, ty);
        let hi = if sz == 8 {
            let op = if signed { BinOp::SMulHi } else { BinOp::UMulHi };
            self.emit(f, InstKind::Bin { op, lhs: a, rhs: b }, ty)
        } else {
            // the 64-bit product of the extended operands, then its high half
            let p = self.wide_product(f, a, b, signed);
            let k = self.konst(f, 8 * sz as u64, TyId::B1);
            let h = self.emit(f, InstKind::Bin { op: BinOp::LShr, lhs: p, rhs: k }, TyId::B8);
            self.emit(f, InstKind::Cast { kind: CastKind::Trunc, v: h }, ty)
        };
        self.write(f, lo_reg, lo)?;
        self.write(f, hi_reg, hi)?;
        self.flags = Flags::Mul { lhs: a, rhs: b, lo, signed };
        Ok(())
    }

    /// `a * b` of two narrower values, extended to 64 bits first, so it can't overflow.
    fn wide_product(&mut self, f: &mut Function, a: ValueId, b: ValueId, signed: bool) -> ValueId {
        let kind = if signed { CastKind::SExt } else { CastKind::ZExt };
        let a = self.emit(f, InstKind::Cast { kind, v: a }, TyId::B8);
        let b = self.emit(f, InstKind::Cast { kind, v: b }, TyId::B8);
        self.emit(f, InstKind::Bin { op: BinOp::Mul, lhs: a, rhs: b }, TyId::B8)
    }

    /// pop r / pop [mem], and the pop half of `leave` (into rbp).
    fn pop(&mut self, f: &mut Function, i: &Instruction) -> Result<(), LiftError> {
        let sp = self.read(f, Register::RSP)?;
        let v = self.emit(f, InstKind::Load { ptr: sp, align: 1, volatile: false }, TyId::B8);
        let sp = self.emit(f, InstKind::PtrOffset { base: sp, index: None, scale: 1, disp: 8 }, TyId::PTR);
        self.write(f, Register::RSP, sp)?;
        match (i.mnemonic(), i.op0_kind()) {
            (Mnemonic::Leave, _) => self.write(f, Register::RBP, v),
            (_, OpKind::Register) if i.op0_register().size() == 8 => self.write(f, i.op0_register(), v),
            // the address uses rsp after the increment
            (_, OpKind::Memory) if i.memory_size().size() == 8 => {
                let ptr = self.ea(f, i)?;
                self.emit(f, InstKind::Store { ptr, val: v, align: 1 }, TyId::UNIT);
                Ok(())
            }
            _ => Err(self.unsupported()),
        }
    }

    /// A System V call: `rax = callee(rdi, rsi, rdx, rcx, r8, r9)`, and every other
    /// caller-saved register is a `CallOut` of the call; the flags are unknown.
    /// Every register a callee could read is passed (`CALL_REGS`), since which ones
    /// it does isn't known here: `abi::apply` trims the list once the callee's
    /// signature is.
    fn call(&mut self, f: &mut Function, i: &Instruction) -> Result<(), LiftError> {
        let callee = self.callee(f, i)?;
        let args = self.call_regs(f);
        let call = self.emit(f, InstKind::Call { callee, args: ListRef::EMPTY }, TyId::B8);
        self.calls.push((call, args));
        self.state[self.cur].out[Register::RAX.number()] = Some(call);
        for r in EXIT_REGS {
            let v = self.emit(f, InstKind::CallOut { call, reg: r.number() as u8 }, TyId::B8);
            self.state[self.cur].out[r.number()] = Some(v);
        }
        // xmm registers are caller-saved too; xmm0 may hold a float result
        for k in 0..if self.xmm { 32 } else { 0 } {
            let v = self.emit(f, InstKind::CallOut { call, reg: XMM_PARAM + k as u8 }, TyId::B8);
            self.state[self.cur].out[NGPR + k] = Some(v);
        }
        if self.ymm {
            self.clear_upper(f, false);
        }
        if self.x87 {
            // the stack is empty at a call; a long double result isn't modelled
            if self.fdepth != 0 {
                return Err(self.unsupported());
            }
            self.x87_clear(f, false);
        }
        self.flags = Flags::Unknown;
        Ok(())
    }

    /// The registers at a call or tail call, in `CALL_REGS` order.
    fn call_regs(&mut self, f: &mut Function) -> CallRegs {
        let mut regs = [ValueId::from_u32(0); CALL_ARGS];
        for (a, r) in regs.iter_mut().zip(CALL_REGS) {
            *a = self.read_full(f, r.number());
        }
        if self.xmm {
            for k in 0..32 {
                regs[CALL_XMM + k] = self.read_full(f, NGPR + k);
            }
        } else {
            regs[CALL_XMM..].fill(self.emit(f, InstKind::Undef, TyId::B8));
        }
        regs
    }

    /// The target of a call or indirect jump, as a pointer value.
    fn callee(&mut self, f: &mut Function, i: &Instruction) -> Result<ValueId, LiftError> {
        let a = match i.op0_kind() {
            OpKind::NearBranch64 => self.konst(f, i.near_branch_target(), TyId::B8),
            OpKind::Register if i.op0_register().size() == 8 => self.read(f, i.op0_register())?,
            OpKind::Memory if i.memory_size().size() == 8 => {
                let ptr = self.ea(f, i)?;
                self.emit(f, InstKind::Load { ptr, align: 1, volatile: false }, TyId::B8)
            }
            _ => return Err(self.unsupported()),
        };
        Ok(self.emit(f, InstKind::IntToPtr(a), TyId::PTR))
    }

    // ---------- operands ----------

    /// Size in bytes of a register or memory operand; only GPR-sized ones are handled.
    fn op_size(&self, i: &Instruction, op: u32) -> Result<usize, LiftError> {
        let sz = match i.op_kind(op) {
            OpKind::Register => i.op_register(op).size(),
            OpKind::Memory => i.memory_size().size(),
            _ => 0,
        };
        if matches!(sz, 1 | 2 | 4 | 8) { Ok(sz) } else { Err(self.unsupported()) }
    }

    /// Read source operand `op` as a `size`-byte value: a register, an immediate, or a
    /// load from the memory operand.
    fn operand(&mut self, f: &mut Function, i: &Instruction, op: u32, size: usize) -> Result<ValueId, LiftError> {
        match i.op_kind(op) {
            OpKind::Register => self.read(f, i.op_register(op)),
            k if is_imm(k) => Ok(self.konst(f, imm(i, op, size), TyId::unknown(size))),
            OpKind::Memory if is_canary(i) => Ok(self.konst(f, CANARY, TyId::B8)),
            // mov rax, fs:0: the thread pointer, which points at itself
            OpKind::Memory if is_fs_abs(i) && i.memory_displacement64() == 0 && self.thread_pointer.is_some() => {
                if i.memory_size().size() != 8 {
                    return Err(self.unsupported());
                }
                let tp = self.konst(f, self.thread_pointer.unwrap_or_default(), TyId::B8);
                Ok(self.emit(f, InstKind::IntToPtr(tp), TyId::PTR))
            }
            OpKind::Memory => {
                let sz = self.op_size(i, op)?;
                let ptr = self.ea(f, i)?;
                Ok(self.emit(f, InstKind::Load { ptr, align: 1, volatile: false }, TyId::unknown(sz)))
            }
            _ => Err(self.unsupported()),
        }
    }

    /// The destination (operand 0), with its address computed once for read-modify-write.
    fn dst(&mut self, f: &mut Function, i: &Instruction) -> Result<Dst, LiftError> {
        match i.op0_kind() {
            OpKind::Register => Ok(Dst::Reg(i.op0_register())),
            OpKind::Memory => {
                self.op_size(i, 0)?;
                Ok(Dst::Mem(self.ea(f, i)?))
            }
            _ => Err(self.unsupported()),
        }
    }

    fn get(&mut self, f: &mut Function, d: Dst, size: usize) -> Result<ValueId, LiftError> {
        match d {
            Dst::Reg(r) => self.read(f, r),
            Dst::Mem(ptr) => Ok(self.emit(f, InstKind::Load { ptr, align: 1, volatile: false }, TyId::unknown(size))),
        }
    }

    fn put(&mut self, f: &mut Function, d: Dst, val: ValueId) -> Result<(), LiftError> {
        match d {
            Dst::Reg(r) => self.write(f, r, val),
            Dst::Mem(ptr) => {
                self.emit(f, InstKind::Store { ptr, val, align: 1 }, TyId::UNIT);
                Ok(())
            }
        }
    }

    /// Turn the pending flags plus a condition code into a `Bool` value.
    fn condition(&mut self, f: &mut Function, cc: ConditionCode) -> Result<ValueId, LiftError> {
        use ConditionCode as C;
        let (cond, lhs, rhs) = match self.flags {
            Flags::Unknown => return Err(LiftError::FlagsNotInBlock { ip: self.ip }),
            Flags::Raw { rflags, valid } => return self.raw_condition(f, cc, rflags, valid),
            Flags::Entry => return Ok(self.cond_in(f, self.cur, cc, self.ip)),
            // inc / dec: add / sub of 1, without the carry conditions
            Flags::Dec { .. } | Flags::Inc { .. } if matches!(cc, C::b | C::ae | C::be | C::a) => return Err(self.unsupported()),
            Flags::Dec { lhs, one } => {
                self.flags = Flags::Sub { lhs, rhs: one };
                let c = self.condition(f, cc);
                self.flags = Flags::Dec { lhs, one };
                return c;
            }
            Flags::Inc { lhs, one, res } => {
                self.flags = Flags::Add { lhs, rhs: one, res };
                let c = self.condition(f, cc);
                self.flags = Flags::Inc { lhs, one, res };
                return c;
            }
            Flags::Sub { lhs, rhs } => {
                let cond = match cc {
                    C::e => Cond::Eq, C::ne => Cond::Ne,
                    C::b => Cond::Ult, C::ae => Cond::Uge, C::be => Cond::Ule, C::a => Cond::Ugt,
                    C::l => Cond::Slt, C::ge => Cond::Sge, C::le => Cond::Sle, C::g => Cond::Sgt,
                    // sign of the difference
                    C::s | C::ns => {
                        let res = self.emit(f, InstKind::Bin { op: BinOp::Sub, lhs, rhs }, f.insts[lhs].ty);
                        return Ok(self.sign(f, res, cc == C::s));
                    }
                    // signed overflow: the operands' signs differ and the result's
                    // sign differs from lhs, i.e. ((lhs ^ rhs) & (lhs ^ res)) < 0
                    C::o | C::no => {
                        let ty = f.insts[lhs].ty;
                        let res = self.emit(f, InstKind::Bin { op: BinOp::Sub, lhs, rhs }, ty);
                        let a = self.emit(f, InstKind::Bin { op: BinOp::Xor, lhs, rhs }, ty);
                        let b = self.emit(f, InstKind::Bin { op: BinOp::Xor, lhs, rhs: res }, ty);
                        let x = self.emit(f, InstKind::Bin { op: BinOp::And, lhs: a, rhs: b }, ty);
                        return Ok(self.sign(f, x, cc == C::o));
                    }
                    _ => return Err(self.unsupported()),
                };
                (cond, lhs, rhs)
            }
            Flags::Logic { res } => {
                // CF = OF = 0, so the unsigned conditions reduce to ZF and the signed
                // ones to a signed compare of the result with zero
                let cond = match cc {
                    C::e | C::be => Cond::Eq, C::ne | C::a => Cond::Ne,
                    C::s | C::l => Cond::Slt, C::ns | C::ge => Cond::Sge,
                    C::le => Cond::Sle, C::g => Cond::Sgt,
                    C::b | C::o => return Ok(self.konst(f, 0, TyId::BOOL)),
                    C::ae | C::no => return Ok(self.konst(f, 1, TyId::BOOL)),
                    _ => return Err(self.unsupported()),
                };
                (cond, res, self.konst(f, 0, f.insts[res].ty))
            }
            Flags::Add { lhs, rhs, res } => {
                let cond = match cc {
                    C::e => Cond::Eq, C::ne => Cond::Ne,
                    C::s | C::ns => return Ok(self.sign(f, res, cc == C::s)),
                    // signed order of the exact sum, when rhs is a constant k:
                    // lhs + k < 0 is lhs < -k (-k fits unless k is the minimum)
                    C::l | C::ge | C::le | C::g => {
                        let ty = f.insts[res].ty;
                        let mask = width_mask(ty);
                        let Some(k) = const_of(f, rhs).map(|k| k & mask) else { return Err(self.unsupported()) };
                        if k == 0 || k == mask / 2 + 1 {
                            return Err(self.unsupported());
                        }
                        let neg = self.konst(f, k.wrapping_neg() & mask, ty);
                        let cond = match cc { C::l => Cond::Slt, C::ge => Cond::Sge, C::le => Cond::Sle, _ => Cond::Sgt };
                        return Ok(self.emit(f, InstKind::Cmp { cc: cond, lhs, rhs: neg }, TyId::BOOL));
                    }
                    // carry: the sum wrapped
                    C::b => return Ok(self.emit(f, InstKind::Cmp { cc: Cond::Ult, lhs: res, rhs: lhs }, TyId::BOOL)),
                    C::ae => return Ok(self.emit(f, InstKind::Cmp { cc: Cond::Uge, lhs: res, rhs: lhs }, TyId::BOOL)),
                    // signed overflow: both operands' signs differ from the result's,
                    // i.e. ((lhs ^ res) & (rhs ^ res)) < 0
                    C::o | C::no => {
                        let ty = f.insts[res].ty;
                        let a = self.emit(f, InstKind::Bin { op: BinOp::Xor, lhs, rhs: res }, ty);
                        let b = self.emit(f, InstKind::Bin { op: BinOp::Xor, lhs: rhs, rhs: res }, ty);
                        let x = self.emit(f, InstKind::Bin { op: BinOp::And, lhs: a, rhs: b }, ty);
                        return Ok(self.sign(f, x, cc == C::o));
                    }
                    _ => return Err(self.unsupported()),
                };
                (cond, res, self.konst(f, 0, f.insts[res].ty))
            }
            // CF = OF = the full product doesn't fit in `lo`
            Flags::Mul { lhs, rhs, lo, signed } => {
                let cond = match cc {
                    C::o | C::b => Cond::Ne,
                    C::no | C::ae => Cond::Eq,
                    _ => return Err(self.unsupported()),
                };
                let ty = f.insts[lo].ty;
                let (full, fits) = if ty == TyId::B8 {
                    let op = if signed { BinOp::SMulHi } else { BinOp::UMulHi };
                    let hi = self.emit(f, InstKind::Bin { op, lhs, rhs }, TyId::B8);
                    let fits = if signed {
                        let k = self.konst(f, 63, TyId::B1);
                        self.emit(f, InstKind::Bin { op: BinOp::AShr, lhs: lo, rhs: k }, TyId::B8)
                    } else {
                        self.konst(f, 0, TyId::B8)
                    };
                    (hi, fits)
                } else {
                    let p = self.wide_product(f, lhs, rhs, signed);
                    let kind = if signed { CastKind::SExt } else { CastKind::ZExt };
                    (p, self.emit(f, InstKind::Cast { kind, v: lo }, TyId::B8))
                };
                (cond, full, fits)
            }
            Flags::Carry { cf } => {
                return match cc {
                    C::b => Ok(cf),
                    C::ae => {
                        let no = self.konst(f, 0, TyId::BOOL);
                        Ok(self.emit(f, InstKind::Cmp { cc: Cond::Eq, lhs: cf, rhs: no }, TyId::BOOL))
                    }
                    _ => Err(self.unsupported()),
                };
            }
            Flags::AddCarry { lhs, rhs, cin, res, sub } => return self.carry_condition(f, cc, lhs, rhs, cin, res, sub),
            Flags::Res { res } => {
                let cond = match cc {
                    C::e => Cond::Eq, C::ne => Cond::Ne, C::s => Cond::Slt, C::ns => Cond::Sge,
                    _ => return Err(self.unsupported()),
                };
                (cond, res, self.konst(f, 0, f.insts[res].ty))
            }            Flags::Float { a, b, w } => {
                // each condition is one lane compare, true or inverted
                let (op, set) = match cc {
                    C::a => (LaneOp::FCmpGt, true),
                    C::ae => (LaneOp::FCmpGe, true),
                    C::b => (LaneOp::FCmpGe, false),
                    C::be => (LaneOp::FCmpGt, false),
                    C::e | C::le => (LaneOp::FCmpLtGt, false),
                    C::ne | C::g => (LaneOp::FCmpLtGt, true),
                    C::p => (LaneOp::FCmpUnord, true),
                    C::np => (LaneOp::FCmpUnord, false),
                    C::l | C::s | C::o => return Ok(self.konst(f, 0, TyId::BOOL)),
                    C::ge | C::ns | C::no => return Ok(self.konst(f, 1, TyId::BOOL)),
                    _ => return Err(self.unsupported()),
                };
                let mut m = self.emit(f, InstKind::Bin { op: BinOp::Lane(op, w), lhs: a, rhs: b }, TyId::B8);
                if w == 4 {
                    let low = self.konst(f, 0xffff_ffff, TyId::B8);
                    m = self.emit(f, InstKind::Bin { op: BinOp::And, lhs: m, rhs: low }, TyId::B8);
                }
                (if set { Cond::Ne } else { Cond::Eq }, m, self.konst(f, 0, TyId::B8))
            }
        };
        Ok(self.emit(f, InstKind::Cmp { cc: cond, lhs, rhs }, TyId::BOOL))
    }

    /// A condition after adc / sbb (`Flags::AddCarry`). CF is the carry (borrow)
    /// out of either step, OF the signs of the operands against the result's.
    #[allow(clippy::too_many_arguments)]
    fn carry_condition(
        &mut self, f: &mut Function, cc: ConditionCode, lhs: ValueId, rhs: ValueId, cin: ValueId, res: ValueId, sub: bool,
    ) -> Result<ValueId, LiftError> {
        use ConditionCode as C;
        let ty = f.insts[res].ty;
        let zero = self.konst(f, 0, ty);
        let not = |l: &mut Self, f: &mut Function, v: ValueId| {
            let no = l.konst(f, 0, TyId::BOOL);
            l.emit(f, InstKind::Cmp { cc: Cond::Eq, lhs: v, rhs: no }, TyId::BOOL)
        };
        let or = |l: &mut Self, f: &mut Function, a: ValueId, b: ValueId| {
            let a = l.emit(f, InstKind::Cast { kind: CastKind::ZExt, v: a }, TyId::B1);
            let b = l.emit(f, InstKind::Cast { kind: CastKind::ZExt, v: b }, TyId::B1);
            let v = l.emit(f, InstKind::Bin { op: BinOp::Or, lhs: a, rhs: b }, TyId::B1);
            let z = l.konst(f, 0, TyId::B1);
            l.emit(f, InstKind::Cmp { cc: Cond::Ne, lhs: v, rhs: z }, TyId::BOOL)
        };
        let carry = |l: &mut Self, f: &mut Function| {
            let op = if sub { BinOp::Sub } else { BinOp::Add };
            let t = l.emit(f, InstKind::Bin { op, lhs, rhs }, ty);
            // sub: lhs < rhs, or the difference < cin; add: either sum wrapped
            let (a, b) = if sub {
                (l.emit(f, InstKind::Cmp { cc: Cond::Ult, lhs, rhs }, TyId::BOOL), l.emit(f, InstKind::Cmp { cc: Cond::Ult, lhs: t, rhs: cin }, TyId::BOOL))
            } else {
                (l.emit(f, InstKind::Cmp { cc: Cond::Ult, lhs: t, rhs: lhs }, TyId::BOOL), l.emit(f, InstKind::Cmp { cc: Cond::Ult, lhs: res, rhs: t }, TyId::BOOL))
            };
            or(l, f, a, b)
        };
        // the sign bit of OF (and of SF ^ OF, for the signed orders)
        let overflow = |l: &mut Self, f: &mut Function| {
            let (a, b) = if sub { (rhs, lhs) } else { (res, rhs) };
            let a = l.emit(f, InstKind::Bin { op: BinOp::Xor, lhs, rhs: a }, ty);
            let b = l.emit(f, InstKind::Bin { op: BinOp::Xor, lhs: b, rhs: res }, ty);
            l.emit(f, InstKind::Bin { op: BinOp::And, lhs: a, rhs: b }, ty)
        };
        let less = |l: &mut Self, f: &mut Function| {
            let o = overflow(l, f);
            let x = l.emit(f, InstKind::Bin { op: BinOp::Xor, lhs: res, rhs: o }, ty);
            l.sign(f, x, true)
        };
        let is_zero = |l: &mut Self, f: &mut Function| l.emit(f, InstKind::Cmp { cc: Cond::Eq, lhs: res, rhs: zero }, TyId::BOOL);
        Ok(match cc {
            C::e => is_zero(self, f),
            C::ne => self.emit(f, InstKind::Cmp { cc: Cond::Ne, lhs: res, rhs: zero }, TyId::BOOL),
            C::s | C::ns => self.sign(f, res, cc == C::s),
            C::b => carry(self, f),
            C::ae => {
                let c = carry(self, f);
                not(self, f, c)
            }
            C::be | C::a => {
                let (c, z) = (carry(self, f), is_zero(self, f));
                let v = or(self, f, c, z);
                if cc == C::be { v } else { not(self, f, v) }
            }
            C::o | C::no => {
                let o = overflow(self, f);
                self.sign(f, o, cc == C::o)
            }
            C::l => less(self, f),
            C::ge => {
                let v = less(self, f);
                not(self, f, v)
            }
            C::le | C::g => {
                let (lt, z) = (less(self, f), is_zero(self, f));
                let v = or(self, f, lt, z);
                if cc == C::le { v } else { not(self, f, v) }
            }
            _ => return Err(self.unsupported()),
        })
    }

    /// `v < 0` (signed) if `neg`, else `v >= 0`.
    fn sign(&mut self, f: &mut Function, v: ValueId, neg: bool) -> ValueId {
        let zero = self.konst(f, 0, f.insts[v].ty);
        let cc = if neg { Cond::Slt } else { Cond::Sge };
        self.emit(f, InstKind::Cmp { cc, lhs: v, rhs: zero }, TyId::BOOL)
    }

    /// Effective address of the memory operand as one `PtrOffset` node.
    fn ea(&mut self, f: &mut Function, i: &Instruction) -> Result<ValueId, LiftError> {
        // a thread-local at a constant offset from the thread pointer
        if let (true, Some(tp)) = (is_fs_abs(i), self.thread_pointer) {
            let a = self.konst(f, tp.wrapping_add(i.memory_displacement64()), TyId::B8);
            return Ok(self.emit(f, InstKind::IntToPtr(a), TyId::PTR));
        }
        if matches!(i.memory_segment(), Register::FS | Register::GS) {
            return Err(self.unsupported()); // TLS through a register (initial-exec), or gs:
        }
        if i.is_ip_rel_memory_operand() {
            let a = self.konst(f, i.ip_rel_memory_address(), TyId::B8);
            return Ok(self.emit(f, InstKind::IntToPtr(a), TyId::PTR));
        }
        let (base, index) = (i.memory_base(), i.memory_index());
        let disp = i.memory_displacement64();
        if base == Register::None && index == Register::None {
            let a = self.konst(f, disp, TyId::B8);
            return Ok(self.emit(f, InstKind::IntToPtr(a), TyId::PTR));
        }
        let Ok(disp) = i32::try_from(disp as i64) else { return Err(self.unsupported()) };
        let base = match base {
            Register::None => self.konst(f, 0, TyId::B8),
            r if r.is_gpr64() => self.read(f, r)?,
            _ => return Err(self.unsupported()),
        };
        let index = match index {
            Register::None => None,
            r if r.is_gpr64() => Some(self.read(f, r)?),
            _ => return Err(self.unsupported()),
        };
        let scale = i.memory_index_scale() as u8;
        Ok(self.emit(f, InstKind::PtrOffset { base, index, scale, disp }, TyId::PTR))
    }

    // ---------- register file ----------

    fn read(&mut self, f: &mut Function, reg: Register) -> Result<ValueId, LiftError> {
        if !reg.is_gpr() {
            return Err(self.unsupported());
        }
        let n = reg.full_register().number();
        let full = self.read_full(f, n);
        let sz = reg.size();
        if sz == 8 {
            return Ok(full);
        }
        if is_high_byte(reg) {
            let eight = self.konst(f, 8, TyId::B1);
            let v = self.emit(f, InstKind::Bin { op: BinOp::LShr, lhs: full, rhs: eight }, TyId::B8);
            return Ok(self.emit(f, InstKind::Cast { kind: CastKind::Trunc, v }, TyId::B1));
        }
        match f.insts[full].kind {
            // eax after `mov eax, x` is ZExt(x); read x back instead of Trunc(ZExt(x)).
            InstKind::Cast { kind: CastKind::ZExt, v } if f.insts[v].ty == TyId::unknown(sz) => return Ok(v),
            // al after `mov al, x` is (old & !0xff) | ZExt(x); read x back.
            InstKind::Bin { op: BinOp::Or, lhs, rhs } => {
                if let (InstKind::Bin { op: BinOp::And, rhs: m, .. }, InstKind::Cast { kind: CastKind::ZExt, v }) =
                    (f.insts[lhs].kind, f.insts[rhs].kind)
                {
                    if f.insts[v].ty == TyId::unknown(sz) && const_of(f, m) == Some(!low_mask(sz)) {
                        return Ok(v);
                    }
                }
            }
            _ => {}
        }
        Ok(self.emit(f, InstKind::Cast { kind: CastKind::Trunc, v: full }, TyId::unknown(sz)))
    }

    /// The 64-bit value of GPR `n`, creating a live-in if the block hasn't seen it yet.
    /// Doesn't check for clobbering.
    fn read_full(&mut self, f: &mut Function, n: usize) -> ValueId {
        match self.state[self.cur].out[n] {
            Some(v) => v,
            None => self.live_in(f, self.cur, n),
        }
    }

    fn write(&mut self, f: &mut Function, reg: Register, v: ValueId) -> Result<(), LiftError> {
        if !reg.is_gpr() {
            return Err(self.unsupported());
        }
        let n = reg.full_register().number();
        let v = match reg.size() {
            8 => v,
            // writing a 32-bit register zero-extends into the 64-bit one
            4 => self.emit(f, InstKind::Cast { kind: CastKind::ZExt, v }, TyId::B8),
            // 8/16-bit writes merge into the old value: (old & !mask) | (ZExt(v) << shift).
            sz => {
                let old = self.read_full(f, n);
                let (mask, shift) = if is_high_byte(reg) { (0xff00, 8) } else { (low_mask(sz), 0) };
                let mut wide = self.emit(f, InstKind::Cast { kind: CastKind::ZExt, v }, TyId::B8);
                if shift != 0 {
                    let s = self.konst(f, shift, TyId::B1);
                    wide = self.emit(f, InstKind::Bin { op: BinOp::Shl, lhs: wide, rhs: s }, TyId::B8);
                }
                let m = self.konst(f, !mask, TyId::B8);
                let kept = self.emit(f, InstKind::Bin { op: BinOp::And, lhs: old, rhs: m }, TyId::B8);
                self.emit(f, InstKind::Bin { op: BinOp::Or, lhs: kept, rhs: wide }, TyId::B8)
            }
        };
        self.state[self.cur].out[n] = Some(v);
        Ok(())
    }

    /// Create a block parameter for GPR `n` in block `b`. Not part of the block's
    /// instruction list; `finalize` lists it in `Block::params`.
    fn live_in(&mut self, f: &mut Function, b: usize, n: usize) -> ValueId {
        let reg = match n {
            _ if n < NGPR => n as u8,
            _ if n < UPPER => XMM_PARAM + (n - NGPR) as u8,
            _ if n < x87::X87 => YMM_PARAM + (n - UPPER) as u8,
            _ => x87::X87_PARAM + (n - x87::X87) as u8,
        };
        let id = f.insts.push(Inst { kind: InstKind::BlockParam(reg), ty: TyId::B8 });
        f.origin.push(self.block_ip(b));
        self.state[b].params[n] = Some(id);
        self.state[b].out[n] = Some(id);
        id
    }

    /// Condition `cc` of the flags block `b` was entered with: a `Bool` block
    /// parameter, which `finalize` has each predecessor compute.
    /// `read` is the instruction that needs it.
    fn cond_in(&mut self, f: &mut Function, b: usize, cc: ConditionCode, read: u64) -> ValueId {
        let st = &mut self.state[b];
        if let Some(v) = st.cparams[cc as usize] {
            return v;
        }
        if st.cparams.iter().all(Option::is_none) {
            st.flags_read = read;
        }
        let id = f.insts.push(Inst { kind: InstKind::BlockParam(FLAG_PARAM), ty: TyId::BOOL });
        f.origin.push(self.block_ip(b));
        self.state[b].cparams[cc as usize] = Some(id);
        id
    }

    /// Condition `cc` of the flags block `b` exits with, computed at its end.
    /// Called from `finalize`, after every block is lifted: `b`'s instructions are
    /// moved to the end of the pool first if another block's follow them. `read`
    /// is the instruction in a successor that needs it, for the error when `b`
    /// leaves flags that don't give `cc`.
    fn cond_out(&mut self, f: &mut Function, b: usize, cc: ConditionCode, read: u64) -> Result<ValueId, LiftError> {
        let st = self.state[b];
        if let Some(v) = st.cout[cc as usize] {
            return Ok(v);
        }
        let v = match st.flags {
            Flags::Entry => self.cond_in(f, b, cc, read),
            Flags::Unknown => return Err(LiftError::FlagsNotInBlock { ip: read }),
            flags => {
                let id = BlockId::new(b);
                let l = f.blocks[id].insts;
                let (start, end) = (l.start as usize, (l.start + l.len) as usize);
                if end != f.value_pool.len() {
                    let at = f.value_pool.len() as u32;
                    f.value_pool.extend_from_within(start..end);
                    f.blocks[id].insts.start = at;
                }
                (self.cur, self.flags, self.ip) = (b, flags, st.exit_ip);
                let before = f.value_pool.len();
                let v = self.condition(f, cc).map_err(|_| LiftError::FlagsNotInBlock { ip: read })?;
                f.blocks[id].insts.len += (f.value_pool.len() - before) as u32;
                v
            }
        };
        self.state[b].cout[cc as usize] = Some(v);
        Ok(v)
    }

    // ---------- emission ----------

    #[inline]
    fn emit(&mut self, f: &mut Function, kind: InstKind, ty: TyId) -> ValueId {
        let id = f.insts.push(Inst { kind, ty });
        f.origin.push(self.ip);
        f.value_pool.push(id);
        id
    }

    #[inline]
    fn konst(&mut self, f: &mut Function, val: u64, ty: TyId) -> ValueId {
        let c = ConstId::new(f.consts.len());
        f.consts.push(val as u128);
        self.emit(f, InstKind::Const(c), ty)
    }

    fn target(&self, addr: u64) -> Result<BlockId, LiftError> {
        self.block_at(addr).ok_or(LiftError::BranchOutOfRange { ip: self.ip, target: addr })
    }

    fn unsupported(&self) -> LiftError {
        LiftError::Unsupported { ip: self.ip, mnemonic: self.insn.mnemonic() }
    }

    // ---------- SSA wiring ----------

    /// How many successors block `b` has for live-in propagation: a switch's
    /// are its cases (repeats included), before `finalize` routes them.
    fn succ_count(&self, f: &Function, b: usize) -> usize {
        match f.blocks[BlockId::new(b)].term {
            Terminator::Switch { .. } => self.tables[self.table_of(b)].len as usize,
            t => succs(t).iter().flatten().count(),
        }
    }

    fn succ_at(&self, f: &Function, b: usize, k: usize) -> usize {
        match f.blocks[BlockId::new(b)].term {
            Terminator::Switch { .. } => {
                let t = self.tables[self.table_of(b)];
                self.case_block(t, k).expect("case targets are leaders or stubs").index()
            }
            t => succs(t)[k].expect("successors come first").index(),
        }
    }

    /// The block case `k` of table `t` goes to: a leader, or the stub that
    /// tail-calls a target out of the function.
    fn case_block(&self, t: JumpTable, k: usize) -> Option<BlockId> {
        let c = self.cases[t.start as usize + k];
        if let Some(b) = self.block_at(c) {
            return Some(b);
        }
        // `stubs` is in the order pass 2 lifts them: the j-th stub at this jmp
        // is its j-th target out of the function
        let j = self.case_stubs.iter().filter(|s| s.0 == t.jmp).position(|s| s.1 == c)?;
        let first = self.stubs.iter().position(|&a| a == t.jmp)?;
        Some(BlockId::new(self.leaders.len() + first + j))
    }

    fn table_of(&self, b: usize) -> usize {
        self.switches.iter().find(|s| s.0.index() == b).expect("a switch block").1
    }

    fn finalize(&mut self, f: &mut Function) -> Result<(), LiftError> {
        f.variadic = !self.spill_skips.is_empty();
        let n = self.state.len();
        // 0. A condition read from flags a predecessor set is a live-in like a
        //    register (step 1): each predecessor computes it at its end from the
        //    flags it leaves, or passes on a parameter of its own if it doesn't
        //    touch them. Nothing sets the flags before the entry.
        loop {
            let mut changed = false;
            for b in 0..n {
                for k in 0..self.succ_count(f, b) {
                    let s = self.succ_at(f, b, k);
                    for (c, &cc) in CC.iter().enumerate() {
                        if self.state[s].cparams[c].is_some() && self.state[b].cout[c].is_none() {
                            let new_param = matches!(self.state[b].flags, Flags::Entry) && self.state[b].cparams[c].is_none();
                            self.cond_out(f, b, cc, self.state[s].flags_read)?;
                            changed |= new_param;
                        }
                    }
                }
            }
            if !changed { break; }
        }
        if self.state[0].cparams.iter().any(Option::is_some) {
            return Err(LiftError::FlagsNotInBlock { ip: self.state[0].flags_read });
        }
        // 1. A successor's live-in must be defined at the end of each predecessor;
        //    if the predecessor never touched that register it becomes a live-in there too.
        //    A switch's successors here are its case targets.
        loop {
            let mut changed = false;
            for b in 0..n {
                for k in 0..self.succ_count(f, b) {
                    let s = self.succ_at(f, b, k);
                    for r in 0..NREG {
                        if self.state[s].params[r].is_none() {
                            continue;
                        }
                        if self.state[b].out[r].is_none() {
                            self.live_in(f, b, r);
                            changed = true;
                        }
                    }
                }
            }
            if !changed { break; }
        }
        // 1b. Switch edges carry no arguments: each case goes through a new block
        //     without parameters (one per target) that jumps on with them.
        for si in 0..self.switches.len() {
            let (b, t) = self.switches[si];
            let (start, len) = (self.tables[t].start as usize, self.tables[t].len as usize);
            let table = f.value_pool.len();
            for k in 0..len {
                let target = self.cases[start + k];
                let landing = match (0..k).find(|&j| self.cases[start + j] == target) {
                    Some(j) => f.value_pool[table + j],
                    None => {
                        let to = self.case_block(self.tables[t], k).expect("case targets are leaders or stubs");
                        let BlockState { out, cout, .. } = self.state[b.index()];
                        self.state.push(BlockState { out, cout, ..EMPTY_STATE });
                        let term = Terminator::Jump { to, args: ListRef::EMPTY };
                        f.blocks.push(Block { insts: ListRef::EMPTY, params: ListRef::EMPTY, term }).as_value()
                    }
                };
                f.value_pool.push(landing);
            }
            // the default is the target with the most cases
            let cases = &f.value_pool[table..table + len];
            let default = *cases.iter().max_by_key(|&&c| cases.iter().filter(|&&d| d == c).count()).expect("tables aren't empty");
            if let Terminator::Switch { table: l, default: d, .. } = &mut f.blocks[b].term {
                *l = ListRef { start: table as u32, len: len as u32 };
                *d = BlockId::from_value(default);
            }
        }
        let n = self.state.len();
        // 2. Write each call's and tail call's argument list.
        for &(call, args) in &self.calls {
            let start = f.value_pool.len() as u32;
            f.value_pool.extend_from_slice(&args);
            if let InstKind::Call { args, .. } = &mut f.insts[call].kind {
                *args = ListRef { start, len: CALL_ARGS as u32 };
            }
        }
        for &(exit, regs) in &self.exits {
            let start = f.value_pool.len() as u32;
            f.value_pool.extend_from_slice(&regs);
            if let InstKind::Exit { regs } = &mut f.insts[exit].kind {
                *regs = ListRef { start, len: if self.xmm { EXIT_LEN } else { EXIT_XMM0 } as u32 };
            }
        }
        for &(op, start, len) in &self.asm_args {
            let at = f.value_pool.len() as u32;
            f.value_pool.extend_from_slice(&self.asm_vals[start as usize..(start + len) as usize]);
            if let InstKind::Opaque { args, .. } = &mut f.insts[op].kind {
                *args = ListRef { start: at, len };
            }
        }
        for &(b, args) in &self.tails {
            let start = f.value_pool.len() as u32;
            f.value_pool.extend_from_slice(&args);
            if let Terminator::TailCall { args, .. } = &mut f.blocks[b].term {
                *args = ListRef { start, len: CALL_ARGS as u32 };
            }
        }
        // 3. Write each block's parameter list, in register order.
        for b in 0..n {
            let start = f.value_pool.len();
            f.value_pool.extend(self.state[b].params.iter().flatten());
            f.value_pool.extend(self.state[b].cparams.iter().flatten());
            f.blocks[BlockId::new(b)].params = ListRef { start: start as u32, len: (f.value_pool.len() - start) as u32 };
        }
        // 4. Write edge arguments in the same order (Branch: true args, then false args).
        for b in 0..n {
            let start = f.value_pool.len();
            for s in succs(f.blocks[BlockId::new(b)].term).into_iter().flatten() {
                for r in 0..NREG {
                    if self.state[s.index()].params[r].is_some() {
                        f.value_pool.push(self.state[b].out[r].expect("filled by step 1"));
                    }
                }
                for c in 0..NCC {
                    if self.state[s.index()].cparams[c].is_some() {
                        f.value_pool.push(self.state[b].cout[c].expect("filled by step 0"));
                    }
                }
            }
            let list = ListRef { start: start as u32, len: (f.value_pool.len() - start) as u32 };
            if let Terminator::Jump { args, .. } | Terminator::Branch { args, .. } = &mut f.blocks[BlockId::new(b)].term {
                *args = list;
            }
        }
        // Trivial parameters (same value on every edge) are left in; a later
        // cleanup pass removes them (Braun et al., "trivial phi" removal).
        Ok(())
    }
}

/// `ConditionCode` by its number.
const CC: [ConditionCode; NCC] = {
    use ConditionCode as C;
    [C::None, C::o, C::no, C::b, C::ae, C::e, C::ne, C::be, C::a, C::s, C::ns, C::p, C::np, C::l, C::ge, C::le, C::g]
};
const _: () = {
    let mut c = 0;
    while c < NCC {
        assert!(CC[c] as usize == c);
        c += 1;
    }
};

/// Successors of a `Jump` or `Branch`, in edge-argument order.
fn succs(t: Terminator) -> [Option<BlockId>; 2] {
    match t {
        Terminator::Jump { to, .. } => [Some(to), None],
        Terminator::Branch { t, f, .. } => [Some(t), Some(f)],
        _ => [None, None],
    }
}

/// Does execution go on to the next instruction (always or sometimes)?
fn falls_through(i: &Instruction) -> bool {
    !matches!(i.mnemonic(), Mnemonic::Ud2 | Mnemonic::Int3 | Mnemonic::Hlt)
        && !matches!(i.flow_control(), FlowControl::UnconditionalBranch | FlowControl::IndirectBranch | FlowControl::Return | FlowControl::Exception)
}

/// Could `Lifter::lift_insn` lift this instruction? False only when it can't
/// whatever the operands are, so pass 1 can fail early; pass 2 checks the rest.
fn handled(i: &Instruction) -> bool {
    use Mnemonic::*;
    let m = i.mnemonic();
    if matches!(m, Ud2 | Int3 | Hlt) {
        return true;
    }
    match i.flow_control() {
        FlowControl::Next => {
            matches!(
                m,
                Nop | Endbr64 | Mov | Lea | Add | Sub | And | Or | Xor | Cmp | Test | Inc | Dec | Neg | Not | Shl | Shr
                    | Sar | Mul | Imul | Movzx | Movsx | Movsxd | Div | Idiv | Cqo | Cdq | Cwd | Cdqe | Cwde | Cbw | Push | Pop
                    | Leave | Movsb | Movsw | Movsd | Movsq | Stosb | Stosw | Stosd | Stosq | Bswap | Cpuid | Xgetbv | Tzcnt | Lzcnt | Popcnt | Bsf | Bsr
                    | Shld | Shrd | Rol | Ror | Adc | Sbb | Bt | Bts | Btr | Btc | Xchg | Xadd | Cmpxchg | Pause
                    | Prefetcht0 | Prefetcht1 | Prefetcht2 | Prefetchnta | Prefetchw | Prefetch
            ) || cmov_or_setcc(m).is_some()
                || sse::handled(m)
                || x87::handled(m)
        }
        FlowControl::ConditionalBranch => i.condition_code() != ConditionCode::None,
        FlowControl::UnconditionalBranch | FlowControl::IndirectBranch | FlowControl::Return | FlowControl::Call
        | FlowControl::IndirectCall => true,
        _ => false,
    }
}

/// The stack protector's canary, `fs:[0x28]` on Linux. The lifted code doesn't
/// model thread-local storage; any constant works for the canary, since all the
/// function does with it is check that the copy on its stack is unchanged.
const CANARY: u64 = 0x2f8a_61c3_9d0e_7b00;

fn is_canary(i: &Instruction) -> bool {
    is_fs_abs(i) && i.memory_displacement64() == 0x28 && i.memory_size().size() == 8
}

/// `fs:[disp]`, with no base or index register.
fn is_fs_abs(i: &Instruction) -> bool {
    i.memory_segment() == Register::FS && i.memory_base() == Register::None && i.memory_index() == Register::None
}

/// `len` bytes at `addr` in one of `data`'s (address, bytes) sections.
fn read<'a>(data: &[(u64, &'a [u8])], addr: u64, len: usize) -> Option<&'a [u8]> {
    data.iter().find_map(|&(at, bytes)| {
        let off = usize::try_from(addr.checked_sub(at)?).ok()?;
        bytes.get(off..off.checked_add(len)?)
    })
}

#[inline]
fn is_imm(k: OpKind) -> bool {
    matches!(
        k,
        OpKind::Immediate8 | OpKind::Immediate16 | OpKind::Immediate32 | OpKind::Immediate64
            | OpKind::Immediate8to16 | OpKind::Immediate8to32 | OpKind::Immediate8to64 | OpKind::Immediate32to64
    )
}

/// Immediate operand, truncated to the operation size (iced sign-extends to 64 bits).
#[inline]
fn imm(i: &Instruction, op: u32, size: usize) -> u64 {
    let v = i.immediate(op);
    if size >= 8 { v } else { v & ((1u64 << (size * 8)) - 1) }
}

fn is_high_byte(r: Register) -> bool {
    matches!(r, Register::AH | Register::CH | Register::DH | Register::BH)
}

/// The bits an 8/16/32-bit register write replaces.
fn low_mask(size: usize) -> u64 {
    (1u64 << (size * 8)) - 1
}

/// All ones in the width of the integer type `ty`.
fn width_mask(ty: TyId) -> u64 {
    match ty {
        TyId::B1 => 0xff,
        TyId::B2 => 0xffff,
        TyId::B4 => 0xffff_ffff,
        _ => u64::MAX,
    }
}

fn const_of(f: &Function, v: ValueId) -> Option<u64> {
    match f.insts[v].kind {
        InstKind::Const(c) => Some(f.consts[c.index()] as u64),
        _ => None,
    }
}

/// `Some(true)` for CMOVcc, `Some(false)` for SETcc.
fn cmov_or_setcc(m: Mnemonic) -> Option<bool> {
    use Mnemonic::*;
    match m {
        Cmovo | Cmovno | Cmovb | Cmovae | Cmove | Cmovne | Cmovbe | Cmova | Cmovs | Cmovns | Cmovp | Cmovnp
        | Cmovl | Cmovge | Cmovle | Cmovg => Some(true),
        Seto | Setno | Setb | Setae | Sete | Setne | Setbe | Seta | Sets | Setns | Setp | Setnp | Setl | Setge
        | Setle | Setg => Some(false),
        _ => None,
    }
}

/// Does `i` (probably) write `reg`, as its destination register or as `cdqe`?
fn writes(i: &Instruction, reg: Register) -> bool {
    let full = reg.full_register();
    (i.op_count() > 0 && i.op0_kind() == OpKind::Register && i.op0_register().full_register() == full
        && !matches!(i.mnemonic(), Mnemonic::Cmp | Mnemonic::Test))
        || (matches!(i.mnemonic(), Mnemonic::Cdqe | Mnemonic::Cwde | Mnemonic::Cbw | Mnemonic::Cqo) && full == Register::RAX)
}

/// `Lifter::fixed_lea` for the function `code` at `ip`. Calls don't count as
/// writes: where code after one uses the register, the callee keeps it (gcc
/// knows which caller-saved registers its own functions leave alone).
fn fixed_leas(info: &mut iced_x86::InstructionInfoFactory, code: &[u8], ip: u64) -> [u64; 16] {
    use iced_x86::OpAccess;
    const SAVED: u16 = 1 << 3 | 1 << 5 | 0xf000; // rbx, rbp, r12-r15
    let mut out = [0u64; 16];
    let mut dec = Decoder::with_ip(64, code, ip, DecoderOptions::NONE);
    let mut i = Instruction::default();
    while dec.can_decode() {
        dec.decode_out(&mut i);
        if matches!(i.flow_control(), FlowControl::Call | FlowControl::IndirectCall) {
            continue;
        }
        let lea = (i.mnemonic() == Mnemonic::Lea && i.is_ip_rel_memory_operand()).then(|| i.ip_rel_memory_address());
        for u in info.info(&i).used_registers() {
            let r = u.register();
            if !r.is_gpr() || !matches!(u.access(), OpAccess::Write | OpAccess::CondWrite | OpAccess::ReadWrite | OpAccess::ReadCondWrite) {
                continue;
            }
            let n = gpr(r.full_register());
            // the epilogue restoring a callee-saved register
            if i.mnemonic() == Mnemonic::Pop && SAVED & 1 << n != 0 {
                continue;
            }
            out[n] = match lea {
                Some(a) if out[n] == 0 || out[n] == a => a,
                _ => u64::MAX,
            };
        }
    }
    out.map(|a| if a == u64::MAX { 0 } else { a })
}

/// Index of a 64-bit general register (rax = 0 .. r15 = 15).
fn gpr(r: Register) -> usize {
    r as usize - Register::RAX as usize
}

/// The 64-bit general register with index `n` (`gpr`'s inverse).
fn gpr_register(n: usize) -> Register {
    const ALL: [Register; 16] = [
        Register::RAX, Register::RCX, Register::RDX, Register::RBX, Register::RSP, Register::RBP, Register::RSI, Register::RDI,
        Register::R8, Register::R9, Register::R10, Register::R11, Register::R12, Register::R13, Register::R14, Register::R15,
    ];
    ALL[n]
}
