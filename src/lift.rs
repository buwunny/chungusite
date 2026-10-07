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

const NGPR: usize = 16;
type RegFile = [Option<ValueId>; NGPR];

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
/// r11, whose values `abi` needs when the callee preserves them.
pub const CALL_REGS: [Register; 10] = [
    Register::RDI, Register::RSI, Register::RDX, Register::RCX, Register::R8, Register::R9,
    Register::RSP, Register::RAX, Register::R10, Register::R11,
];
pub const CALL_ARGS: usize = CALL_REGS.len();
type CallRegs = [ValueId; CALL_ARGS];
/// Caller-saved registers besides rax: each is a `CallOut` after a call, and an
/// `Exit` lists them at a return.
pub const EXIT_REGS: [Register; 8] = [
    Register::RCX, Register::RDX, Register::RSI, Register::RDI, Register::R8, Register::R9, Register::R10, Register::R11,
];

/// Lazy flags: remember what set them, and only build a `Cmp` when a Jcc, CMOVcc or
/// SETcc reads them. `cmp` itself therefore emits nothing, and flags that are never
/// read cost nothing.
#[derive(Copy, Clone)]
enum Flags {
    Unknown,
    /// cmp / sub / neg: every integer condition is derivable from the operands.
    Sub { lhs: ValueId, rhs: ValueId },
    /// test / and / or / xor: ZF and SF from the result, CF = OF = 0.
    Logic { res: ValueId },
    /// add: ZF, SF, CF (`res < lhs` unsigned) and OF.
    Add { lhs: ValueId, rhs: ValueId, res: ValueId },
    /// inc / dec / shifts: only ZF and SF are modelled (`res == 0`, `res < 0`).
    Res { res: ValueId },
}

#[derive(Copy, Clone)]
struct BlockState {
    /// Value of each GPR at block exit. Starts as the block's own live-ins.
    out: RegFile,
    /// `BlockParam` created for each live-in GPR.
    params: RegFile,
}

const EMPTY_STATE: BlockState = BlockState { out: [None; NGPR], params: [None; NGPR] };

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
    /// The same for each block that ends in a `TailCall`.
    tails: Vec<(BlockId, CallRegs)>,
    /// Each `Exit` and the registers it lists.
    exits: Vec<(ValueId, [ValueId; 8])>,
    /// Record the caller-saved registers at each return in an `Exit` instruction,
    /// for whole-program register summaries (`program.rs` sets this).
    pub track_exits: bool,
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
            tails: Vec::with_capacity(8),
            exits: Vec::with_capacity(8),
            track_exits: false,
            cur: 0,
            flags: Flags::Unknown,
            ip: 0,
        }
    }

    /// Lift one function's bytes, loaded at `ip`, into `f` (which is cleared first).
    pub fn lift(&mut self, code: &[u8], ip: u64, f: &mut Function) -> Result<(), LiftError> {
        f.clear();
        self.find_leaders(code, ip);
        self.state.clear();
        self.calls.clear();
        self.tails.clear();
        self.exits.clear();
        for _ in 0..self.leaders.len() {
            self.state.push(EMPTY_STATE);
            f.blocks.push(Block { insts: ListRef::EMPTY, params: ListRef::EMPTY, term: Terminator::Unreachable });
        }

        let mut dec = Decoder::with_ip(64, code, ip, DecoderOptions::NONE);
        let mut next_leader = 1;
        let mut open = true;
        self.begin_block(0, f);
        while dec.can_decode() {
            dec.decode_out(&mut self.insn);
            self.ip = self.insn.ip();
            if next_leader < self.leaders.len() && self.ip >= self.leaders[next_leader] {
                if self.ip > self.leaders[next_leader] {
                    return Err(LiftError::TargetInsideInstruction { target: self.leaders[next_leader] });
                }
                if open {
                    self.end_block(f, Terminator::Jump { to: BlockId::new(next_leader), args: ListRef::EMPTY });
                }
                self.begin_block(next_leader, f);
                next_leader += 1;
                open = true;
            }
            if open {
                open = !self.lift_insn(f)?;
            }
        }
        if open {
            self.end_block(f, Terminator::Unreachable); // ran off the end of the bytes
        }
        self.finalize(f);
        Ok(())
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

    fn find_leaders(&mut self, code: &[u8], ip: u64) {
        let end = ip + code.len() as u64;
        let in_range = |a: u64| a >= ip && a < end;
        self.leaders.clear();
        self.leaders.push(ip);
        let mut dec = Decoder::with_ip(64, code, ip, DecoderOptions::NONE);
        while dec.can_decode() {
            dec.decode_out(&mut self.insn);
            match self.insn.flow_control() {
                FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch => {
                    let t = self.insn.near_branch_target();
                    if in_range(t) { self.leaders.push(t); }
                    if in_range(self.insn.next_ip()) { self.leaders.push(self.insn.next_ip()); }
                }
                FlowControl::Return | FlowControl::IndirectBranch if in_range(self.insn.next_ip()) => {
                    self.leaders.push(self.insn.next_ip());
                }
                _ => {}
            }
        }
        self.leaders.sort_unstable();
        self.leaders.dedup();
    }

    fn block_at(&self, addr: u64) -> Option<BlockId> {
        self.leaders.binary_search(&addr).ok().map(BlockId::new)
    }

    // ---------- pass 2 ----------

    fn begin_block(&mut self, idx: usize, f: &mut Function) {
        self.cur = idx;
        self.flags = Flags::Unknown;
        f.blocks[BlockId::new(idx)].insts.start = f.value_pool.len() as u32;
    }

    fn end_block(&mut self, f: &mut Function, term: Terminator) {
        let len = f.value_pool.len() as u32;
        let b = &mut f.blocks[BlockId::new(self.cur)];
        b.insts.len = len - b.insts.start;
        b.term = term;
    }

    /// Lift `self.insn`. Returns true if it ended the block.
    fn lift_insn(&mut self, f: &mut Function) -> Result<bool, LiftError> {
        let i = self.insn; // Instruction is Copy (40 bytes); avoids borrowing self
        // ud2 / int3 / hlt: traps the compiler puts where control can't continue
        if matches!(i.mnemonic(), Mnemonic::Ud2 | Mnemonic::Int3 | Mnemonic::Hlt) {
            self.end_block(f, Terminator::Unreachable);
            return Ok(true);
        }
        match i.flow_control() {
            FlowControl::Next => self.lift_data(f, &i).map(|_| false),
            FlowControl::Call | FlowControl::IndirectCall => self.call(f, &i).map(|_| false),
            FlowControl::ConditionalBranch if i.condition_code() != ConditionCode::None => {
                let c = self.condition(f, i.condition_code())?;
                let t = self.target(i.near_branch_target())?;
                let e = self.target(i.next_ip())?;
                self.end_block(f, Terminator::Branch { c, t, f: e, args: ListRef::EMPTY });
                Ok(true)
            }
            FlowControl::UnconditionalBranch if i.op0_kind() == OpKind::NearBranch64 => {
                let term = match self.block_at(i.near_branch_target()) {
                    Some(to) => Terminator::Jump { to, args: ListRef::EMPTY },
                    None => {
                        // Jump out of this function: a tail call.
                        let a = self.konst(f, i.near_branch_target(), TyId::B8);
                        let callee = self.emit(f, InstKind::IntToPtr(a), TyId::PTR);
                        let regs = self.call_regs(f);
                        self.tails.push((BlockId::new(self.cur), regs));
                        Terminator::TailCall { callee, args: ListRef::EMPTY }
                    }
                };
                self.end_block(f, term);
                Ok(true)
            }
            // jmp [rip+x]: a tail call through the GOT. Other indirect jumps are
            // usually jump tables, which need recovering into a `Switch`.
            FlowControl::IndirectBranch if i.op0_kind() == OpKind::Memory && i.is_ip_rel_memory_operand() => {
                let callee = self.callee(f, &i)?;
                let regs = self.call_regs(f);
                self.tails.push((BlockId::new(self.cur), regs));
                self.end_block(f, Terminator::TailCall { callee, args: ListRef::EMPTY });
                Ok(true)
            }
            FlowControl::Return => {
                if self.track_exits {
                    let mut regs = [ValueId::from_u32(0); 8];
                    for (v, r) in regs.iter_mut().zip(EXIT_REGS) {
                        *v = self.read_full(f, r.number());
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
            Mnemonic::Nop | Mnemonic::Endbr64 => {} // endbr64: CET landing pad, no effect on data
            Mnemonic::Mov => match (i.op0_kind(), i.op1_kind()) {
                // mov rax, rcx: no instruction at all, just rename in the register file
                (OpKind::Register, OpKind::Register) => {
                    let v = self.read(f, i.op1_register())?;
                    self.write(f, i.op0_register(), v)?;
                }
                // mov rax, [rdi+8]
                (OpKind::Register, OpKind::Memory) => {
                    let ptr = self.ea(f, i)?;
                    let ty = TyId::unknown(i.op0_register().size());
                    let v = self.emit(f, InstKind::Load { ptr, align: 1, volatile: false }, ty);
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
            // two- and three-operand forms; one-operand imul writes rdx:rax
            Mnemonic::Imul if i.op_count() >= 2 => {
                let dst = i.op0_register();
                let sz = dst.size();
                let (lhs, rhs) = if i.op_count() == 3 {
                    (self.operand(f, i, 1, sz)?, self.konst(f, imm(i, 2, sz), TyId::unknown(sz)))
                } else {
                    (self.read(f, dst)?, self.operand(f, i, 1, sz)?)
                };
                let v = self.emit(f, InstKind::Bin { op: BinOp::Mul, lhs, rhs }, TyId::unknown(sz));
                self.write(f, dst, v)?;
                self.flags = Flags::Unknown; // only CF/OF (overflow) are defined
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
            // cqo / cdq: rdx (edx) = the sign of rax (eax), ready for idiv
            Mnemonic::Cqo | Mnemonic::Cdq => {
                let (lo, hi, bits) =
                    if i.mnemonic() == Mnemonic::Cqo { (Register::RAX, Register::RDX, 63) } else { (Register::EAX, Register::EDX, 31) };
                let v = self.read(f, lo)?;
                let ty = f.insts[v].ty;
                let s = self.konst(f, bits, TyId::B1);
                let sign = self.emit(f, InstKind::Bin { op: BinOp::AShr, lhs: v, rhs: s }, ty);
                self.write(f, hi, sign)?;
            }
            // cdqe / cwde: sign-extend the low half of rax (eax) in place
            Mnemonic::Cdqe | Mnemonic::Cwde => {
                let (src, dst) = if i.mnemonic() == Mnemonic::Cdqe { (Register::EAX, Register::RAX) } else { (Register::AX, Register::EAX) };
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
            Mnemonic::Inc | Mnemonic::Dec => {
                let one = self.konst(f, 1, ty);
                let op = if i.mnemonic() == Mnemonic::Inc { BinOp::Add } else { BinOp::Sub };
                (InstKind::Bin { op, lhs: v, rhs: one }, None) // CF is left as it was: model ZF/SF only
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
                if n >= sz as u64 * 8 {
                    // 8/16-bit operands can shift everything out, which Rust's
                    // wrapping shifts (count mod width) don't model
                    return Err(self.unsupported());
                }
                (self.konst(f, n, TyId::B1), true)
            }
            // a count of 0 leaves the flags alone, so they are unknown after this
            OpKind::Register if i.op1_register() == Register::CL && sz >= 4 => {
                let cl = self.read(f, Register::CL)?;
                let m = self.konst(f, mask, TyId::B1);
                (self.emit(f, InstKind::Bin { op: BinOp::And, lhs: cl, rhs: m }, TyId::B1), false)
            }
            _ => return Err(self.unsupported()),
        };
        let dst = self.dst(f, i)?;
        let v = self.get(f, dst, sz)?;
        let res = self.emit(f, InstKind::Bin { op, lhs: v, rhs: count }, ty);
        self.put(f, dst, res)?;
        self.flags = if flags_known { Flags::Res { res } } else { Flags::Unknown };
        Ok(())
    }

    /// div / idiv with a 32 or 64-bit divisor. The dividend is rdx:rax, which this
    /// handles when rdx only extends rax: zero for div (`xor edx, edx`), the sign of
    /// rax for idiv (`cqo`). Then it is a plain division of rax.
    fn div(&mut self, f: &mut Function, i: &Instruction) -> Result<(), LiftError> {
        let sz = self.op_size(i, 0)?;
        let (lo, hi) = match sz {
            8 => (Register::RAX, Register::RDX),
            4 => (Register::EAX, Register::EDX),
            _ => return Err(self.unsupported()), // 8/16-bit divide ax / dx:ax
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
        self.flags = Flags::Unknown;
        Ok(())
    }

    /// The registers at a call or tail call, in `CALL_REGS` order.
    fn call_regs(&mut self, f: &mut Function) -> CallRegs {
        let mut regs = [ValueId::from_u32(0); CALL_ARGS];
        for (a, r) in regs.iter_mut().zip(CALL_REGS) {
            *a = self.read_full(f, r.number());
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
            Flags::Res { res } => {
                let cond = match cc {
                    C::e => Cond::Eq, C::ne => Cond::Ne, C::s => Cond::Slt, C::ns => Cond::Sge,
                    _ => return Err(self.unsupported()),
                };
                (cond, res, self.konst(f, 0, f.insts[res].ty))
            }
        };
        Ok(self.emit(f, InstKind::Cmp { cc: cond, lhs, rhs }, TyId::BOOL))
    }

    /// `v < 0` (signed) if `neg`, else `v >= 0`.
    fn sign(&mut self, f: &mut Function, v: ValueId, neg: bool) -> ValueId {
        let zero = self.konst(f, 0, f.insts[v].ty);
        let cc = if neg { Cond::Slt } else { Cond::Sge };
        self.emit(f, InstKind::Cmp { cc, lhs: v, rhs: zero }, TyId::BOOL)
    }

    /// Effective address of the memory operand as one `PtrOffset` node.
    fn ea(&mut self, f: &mut Function, i: &Instruction) -> Result<ValueId, LiftError> {
        if matches!(i.memory_segment(), Register::FS | Register::GS) {
            return Err(self.unsupported()); // TLS
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
        let id = f.insts.push(Inst { kind: InstKind::BlockParam(n as u8), ty: TyId::B8 });
        f.origin.push(self.leaders[b]);
        self.state[b].params[n] = Some(id);
        self.state[b].out[n] = Some(id);
        id
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

    fn finalize(&mut self, f: &mut Function) {
        let n = self.state.len();
        // 1. A successor's live-in must be defined at the end of each predecessor;
        //    if the predecessor never touched that register it becomes a live-in there too.
        loop {
            let mut changed = false;
            for b in 0..n {
                for s in succs(f.blocks[BlockId::new(b)].term).into_iter().flatten() {
                    for r in 0..NGPR {
                        if self.state[s.index()].params[r].is_some() && self.state[b].out[r].is_none() {
                            self.live_in(f, b, r);
                            changed = true;
                        }
                    }
                }
            }
            if !changed { break; }
        }
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
                *regs = ListRef { start, len: 8 };
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
            f.blocks[BlockId::new(b)].params = ListRef { start: start as u32, len: (f.value_pool.len() - start) as u32 };
        }
        // 4. Write edge arguments in the same order (Branch: true args, then false args).
        for b in 0..n {
            let start = f.value_pool.len();
            for s in succs(f.blocks[BlockId::new(b)].term).into_iter().flatten() {
                for r in 0..NGPR {
                    if self.state[s.index()].params[r].is_some() {
                        f.value_pool.push(self.state[b].out[r].expect("filled by step 1"));
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
    }
}

fn succs(t: Terminator) -> [Option<BlockId>; 2] {
    t.successors()
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
