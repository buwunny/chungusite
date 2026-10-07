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

/// Lazy flags: remember what set them, and only build a `Cmp` when a Jcc reads them.
/// `cmp` itself therefore emits nothing, and flags that are never read cost nothing.
#[derive(Copy, Clone)]
enum Flags {
    Unknown,
    /// cmp / sub: every integer condition is derivable from the operands.
    Sub { lhs: ValueId, rhs: ValueId },
    /// test / and / or / xor / add: only ZF and SF are modelled (`res == 0`, `res < 0`).
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

pub struct Lifter {
    insn: Instruction,
    leaders: Vec<u64>,
    state: Vec<BlockState>,
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
        match i.flow_control() {
            FlowControl::Next => self.lift_data(f, &i).map(|_| false),
            FlowControl::ConditionalBranch if i.condition_code() != ConditionCode::None => {
                let (cc, lhs, rhs) = self.condition(f, i.condition_code())?;
                let c = self.emit(f, InstKind::Cmp { cc, lhs, rhs }, TyId::BOOL);
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
                        Terminator::TailCall { callee, args: ListRef::EMPTY }
                    }
                };
                self.end_block(f, term);
                Ok(true)
            }
            FlowControl::Return => {
                let v = self.read(f, Register::RAX)?;
                self.end_block(f, Terminator::Return(Some(v)));
                Ok(true)
            }
            _ => Err(self.unsupported()),
        }
    }

    fn lift_data(&mut self, f: &mut Function, i: &Instruction) -> Result<(), LiftError> {
        match i.mnemonic() {
            Mnemonic::Nop => {}
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
            Mnemonic::Lea if i.op0_register().size() == 8 => {
                let p = self.ea(f, i)?;
                self.write(f, i.op0_register(), p)?;
            }
            Mnemonic::Add | Mnemonic::Sub | Mnemonic::And | Mnemonic::Or | Mnemonic::Xor
            | Mnemonic::Cmp | Mnemonic::Test => self.alu(f, i)?,
            _ => return Err(self.unsupported()),
        }
        Ok(())
    }

    /// Register/register or register/immediate ALU ops.
    fn alu(&mut self, f: &mut Function, i: &Instruction) -> Result<(), LiftError> {
        if i.op0_kind() != OpKind::Register {
            return Err(self.unsupported());
        }
        let m = i.mnemonic();
        let dst = i.op0_register();
        let sz = dst.size();
        let ty = TyId::unknown(sz);
        let same_reg = i.op1_kind() == OpKind::Register && i.op1_register() == dst;

        // xor eax, eax: the zeroing idiom becomes a constant, no data dependency
        if m == Mnemonic::Xor && same_reg {
            let z = self.konst(f, 0, ty);
            self.write(f, dst, z)?;
            self.flags = Flags::Res { res: z };
            return Ok(());
        }

        let lhs = self.read(f, dst)?;
        let rhs = match i.op1_kind() {
            OpKind::Register => self.read(f, i.op1_register())?,
            k if is_imm(k) => self.konst(f, imm(i, 1, sz), ty),
            _ => return Err(self.unsupported()),
        };
        self.flags = match m {
            // cmp emits nothing; the Jcc that reads the flags emits the comparison
            Mnemonic::Cmp => Flags::Sub { lhs, rhs },
            // test rax, rax: ZF/SF come straight from rax
            Mnemonic::Test if same_reg => Flags::Res { res: lhs },
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
                    self.write(f, dst, res)?;
                }
                if m == Mnemonic::Sub { Flags::Sub { lhs, rhs } } else { Flags::Res { res } }
            }
        };
        Ok(())
    }

    /// Turn the pending flags plus a condition code into an IR comparison.
    fn condition(&mut self, f: &mut Function, cc: ConditionCode) -> Result<(Cond, ValueId, ValueId), LiftError> {
        use ConditionCode as C;
        match self.flags {
            Flags::Unknown => Err(LiftError::FlagsNotInBlock { ip: self.ip }),
            Flags::Sub { lhs, rhs } => {
                let cond = match cc {
                    C::e => Cond::Eq, C::ne => Cond::Ne,
                    C::b => Cond::Ult, C::ae => Cond::Uge, C::be => Cond::Ule, C::a => Cond::Ugt,
                    C::l => Cond::Slt, C::ge => Cond::Sge, C::le => Cond::Sle, C::g => Cond::Sgt,
                    _ => return Err(self.unsupported()),
                };
                Ok((cond, lhs, rhs))
            }
            Flags::Res { res } => {
                let cond = match cc {
                    C::e => Cond::Eq, C::ne => Cond::Ne, C::s => Cond::Slt, C::ns => Cond::Sge,
                    _ => return Err(self.unsupported()),
                };
                let zero = self.konst(f, 0, f.insts[res].ty);
                Ok((cond, res, zero))
            }
        }
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
        if !reg.is_gpr() || matches!(reg, Register::AH | Register::CH | Register::DH | Register::BH) {
            return Err(self.unsupported());
        }
        let n = reg.full_register().number();
        let full = match self.state[self.cur].out[n] {
            Some(v) => v,
            None => self.live_in(f, self.cur, n),
        };
        let sz = reg.size();
        if sz == 8 {
            return Ok(full);
        }
        // eax after `mov eax, x` is ZExt(x); read x back instead of Trunc(ZExt(x)).
        if let InstKind::Cast { kind: CastKind::ZExt, v } = f.insts[full].kind {
            if f.insts[v].ty == TyId::unknown(sz) {
                return Ok(v);
            }
        }
        Ok(self.emit(f, InstKind::Cast { kind: CastKind::Trunc, v: full }, TyId::unknown(sz)))
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
            // 8/16-bit writes merge into the old value; not handled in this sketch
            _ => return Err(self.unsupported()),
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
        // 2. Write each block's parameter list, in register order.
        for b in 0..n {
            let start = f.value_pool.len();
            f.value_pool.extend(self.state[b].params.iter().flatten());
            f.blocks[BlockId::new(b)].params = ListRef { start: start as u32, len: (f.value_pool.len() - start) as u32 };
        }
        // 3. Write edge arguments in the same order (Branch: true args, then false args).
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
