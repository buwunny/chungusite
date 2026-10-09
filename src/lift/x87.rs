//! x87: the floating-point register stack, as eight slots in the register file.
//!
//! Compiled code keeps the stack's depth the same at each instruction whichever
//! way control got there (empty at calls and returns), so the lifter tracks the
//! depth as a number per block, not a value: `st(i)` is slot `depth - 1 - i`, a
//! push writes slot `depth`. Values are f64 bit patterns. That loses the 80-bit
//! format's extra precision, so the lifted code can differ from the original in
//! the last bits of a `long double`; values in memory (`fld tbyte`, `fstp tbyte`)
//! are converted to and from the 80-bit format. The control word is one more
//! slot, read by `fist` for its rounding mode: `fnstcw`, set the rounding bits,
//! `fldcw`, `fistp` is how C truncates a `long double` to an integer.
use super::*;
use iced_x86::MemorySize;

/// Where the x87 slots start in the register file; the control word follows them.
pub(super) const X87: usize = NGPR + 64;
pub(super) const FPUCW: usize = X87 + 8;
/// `BlockParam` register number of x87 slot `k`, and of the control word after them.
pub(super) const X87_PARAM: u8 = 0xe0;
/// The control word at the start: all exceptions masked, 64-bit precision,
/// round to nearest.
const DEFAULT_CW: u64 = 0x037f;

/// The mnemonics `Lifter::x87` lifts (some only in particular operand forms).
pub(super) fn handled(m: Mnemonic) -> bool {
    use Mnemonic::*;
    matches!(
        m,
        Fld | Fild | Fld1 | Fldz | Fst | Fstp | Fist | Fistp | Fisttp | Fxch | Fadd | Faddp | Fiadd | Fsub | Fsubp
            | Fisub | Fsubr | Fsubrp | Fisubr | Fmul | Fmulp | Fimul | Fdiv | Fdivp | Fidiv | Fdivr | Fdivrp | Fidivr
            | Fchs | Fabs | Fcomi | Fcomip | Fucomi | Fucomip | Fnstcw | Fldcw
    )
}

impl Lifter {
    /// At the entry, and after a call: an empty stack (slots `Undef`). The
    /// control word is the default at the entry, and callees leave it alone.
    pub(super) fn x87_clear(&mut self, f: &mut Function, entry: bool) {
        for k in 0..8 {
            let u = self.emit(f, InstKind::Undef, TyId::B8);
            self.state[self.cur].out[X87 + k] = Some(u);
        }
        if entry {
            let cw = self.konst(f, DEFAULT_CW, TyId::B8);
            self.state[self.cur].out[FPUCW] = Some(cw);
        }
    }

    /// Slot of `st(i)`.
    fn st_slot(&self, i: usize) -> Result<usize, LiftError> {
        match (self.fdepth as usize).checked_sub(i + 1) {
            Some(k) => Ok(X87 + k),
            // a long double returned by a callee, or code that isn't compiled C
            None => Err(self.unsupported()),
        }
    }

    fn st(&mut self, f: &mut Function, i: usize) -> Result<ValueId, LiftError> {
        let n = self.st_slot(i)?;
        Ok(self.read_full(f, n))
    }

    fn set_st(&mut self, i: usize, v: ValueId) -> Result<(), LiftError> {
        let n = self.st_slot(i)?;
        self.state[self.cur].out[n] = Some(v);
        Ok(())
    }

    fn fpush(&mut self, v: ValueId) -> Result<(), LiftError> {
        if self.fdepth == 8 {
            return Err(self.unsupported());
        }
        self.state[self.cur].out[X87 + self.fdepth as usize] = Some(v);
        self.fdepth += 1;
        Ok(())
    }

    fn fpop(&mut self) -> Result<(), LiftError> {
        self.st_slot(0)?;
        self.fdepth -= 1;
        Ok(())
    }

    /// `st(i)` of register operand `k`.
    fn st_of(&self, i: &Instruction, k: u32) -> Result<usize, LiftError> {
        match i.op_register(k) {
            r if (Register::ST0..=Register::ST7).contains(&r) => Ok(r as usize - Register::ST0 as usize),
            _ => Err(self.unsupported()),
        }
    }

    /// A float memory operand as an f64 bit pattern.
    fn load_float(&mut self, f: &mut Function, i: &Instruction) -> Result<ValueId, LiftError> {
        let p = self.ea(f, i)?;
        let load = |l: &mut Self, f: &mut Function, p, ty| l.emit(f, InstKind::Load { ptr: p, align: 1, volatile: false }, ty);
        Ok(match i.memory_size() {
            MemorySize::Float32 => {
                let v = load(self, f, p, TyId::B4);
                let v = self.emit(f, InstKind::Cast { kind: CastKind::ZExt, v }, TyId::B8);
                self.emit(f, InstKind::Un { op: UnOp::Lane(LaneUn::F32ToF64, 8), v }, TyId::B8)
            }
            MemorySize::Float64 => load(self, f, p, TyId::B8),
            MemorySize::Float80 => {
                let lo = load(self, f, p, TyId::B8);
                let at = self.emit(f, InstKind::PtrOffset { base: p, index: None, scale: 1, disp: 8 }, TyId::PTR);
                let hi = load(self, f, at, TyId::B2);
                let hi = self.emit(f, InstKind::Cast { kind: CastKind::ZExt, v: hi }, TyId::B8);
                self.emit(f, InstKind::Bin { op: BinOp::Lane(LaneOp::F80ToF64, 8), lhs: lo, rhs: hi }, TyId::B8)
            }
            _ => return Err(self.unsupported()),
        })
    }

    /// An integer memory operand (`fild`, `fiadd`) as an f64 bit pattern.
    fn load_int(&mut self, f: &mut Function, i: &Instruction) -> Result<ValueId, LiftError> {
        let w = i.memory_size().size();
        if !matches!(w, 2 | 4 | 8) {
            return Err(self.unsupported());
        }
        let p = self.ea(f, i)?;
        let v = self.emit(f, InstKind::Load { ptr: p, align: 1, volatile: false }, TyId::unknown(w));
        let v = if w == 8 { v } else { self.emit(f, InstKind::Cast { kind: CastKind::ZExt, v }, TyId::B8) };
        Ok(self.emit(f, InstKind::Un { op: UnOp::Lane(LaneUn::IntToF64, w as u8), v }, TyId::B8))
    }

    /// Store f64 bit pattern `v` to the float memory operand.
    fn store_float(&mut self, f: &mut Function, i: &Instruction, v: ValueId) -> Result<(), LiftError> {
        let p = self.ea(f, i)?;
        let store = |l: &mut Self, f: &mut Function, p, val| l.emit(f, InstKind::Store { ptr: p, val, align: 1 }, TyId::UNIT);
        match i.memory_size() {
            MemorySize::Float32 => {
                let s = self.emit(f, InstKind::Un { op: UnOp::Lane(LaneUn::F64ToF32, 8), v }, TyId::B8);
                let s = self.emit(f, InstKind::Cast { kind: CastKind::Trunc, v: s }, TyId::B4);
                store(self, f, p, s);
            }
            MemorySize::Float64 => {
                store(self, f, p, v);
            }
            MemorySize::Float80 => {
                let lo = self.emit(f, InstKind::Un { op: UnOp::Lane(LaneUn::F64ToF80Lo, 8), v }, TyId::B8);
                let hi = self.emit(f, InstKind::Un { op: UnOp::Lane(LaneUn::F64ToF80Hi, 8), v }, TyId::B8);
                let hi = self.emit(f, InstKind::Cast { kind: CastKind::Trunc, v: hi }, TyId::B2);
                store(self, f, p, lo);
                let at = self.emit(f, InstKind::PtrOffset { base: p, index: None, scale: 1, disp: 8 }, TyId::PTR);
                store(self, f, at, hi);
            }
            _ => return Err(self.unsupported()),
        }
        Ok(())
    }

    fn fop(&mut self, f: &mut Function, op: LaneOp, a: ValueId, b: ValueId) -> ValueId {
        self.emit(f, InstKind::Bin { op: BinOp::Lane(op, 8), lhs: a, rhs: b }, TyId::B8)
    }

    pub(super) fn x87(&mut self, f: &mut Function, i: &Instruction) -> Result<(), LiftError> {
        use Mnemonic::*;
        let m = i.mnemonic();
        match m {
            Fld if i.op0_kind() == OpKind::Register => {
                let v = self.st(f, self.st_of(i, 0)?)?;
                self.fpush(v)?;
            }
            Fld => {
                let v = self.load_float(f, i)?;
                self.fpush(v)?;
            }
            Fild => {
                let v = self.load_int(f, i)?;
                self.fpush(v)?;
            }
            Fld1 | Fldz => {
                let v = self.konst(f, if m == Fld1 { 1f64.to_bits() } else { 0 }, TyId::B8);
                self.fpush(v)?;
            }
            Fst | Fstp => {
                let v = self.st(f, 0)?;
                match i.op0_kind() {
                    OpKind::Register => {
                        let k = self.st_of(i, 0)?;
                        self.set_st(k, v)?;
                    }
                    _ => self.store_float(f, i, v)?,
                }
                if m == Fstp {
                    self.fpop()?;
                }
            }
            Fist | Fistp | Fisttp => {
                let w = i.memory_size().size();
                if !matches!(w, 2 | 4 | 8) {
                    return Err(self.unsupported());
                }
                let v = self.st(f, 0)?;
                // fisttp truncates whatever the control word says
                let cw = match m {
                    Fisttp => self.konst(f, 0xc00, TyId::B8),
                    _ => self.read_full(f, FPUCW),
                };
                let n = self.emit(f, InstKind::Bin { op: BinOp::Lane(LaneOp::Fist, w as u8), lhs: v, rhs: cw }, TyId::B8);
                let n = if w == 8 { n } else { self.emit(f, InstKind::Cast { kind: CastKind::Trunc, v: n }, TyId::unknown(w)) };
                let p = self.ea(f, i)?;
                self.emit(f, InstKind::Store { ptr: p, val: n, align: 1 }, TyId::UNIT);
                if m != Fist {
                    self.fpop()?;
                }
            }
            Fxch => {
                // `fxch st(i)`, sometimes written with st(0) first
                let k = (0..i.op_count()).map(|k| self.st_of(i, k)).find(|r| r.as_ref().is_ok_and(|&r| r != 0)).unwrap_or(Ok(1))?;
                let (a, b) = (self.st(f, 0)?, self.st(f, k)?);
                self.set_st(0, b)?;
                self.set_st(k, a)?;
            }
            Fadd | Faddp | Fiadd | Fsub | Fsubp | Fisub | Fsubr | Fsubrp | Fisubr | Fmul | Fmulp | Fimul | Fdiv
            | Fdivp | Fidiv | Fdivr | Fdivrp | Fidivr => {
                let (op, rev) = match m {
                    Fadd | Faddp | Fiadd => (LaneOp::FAdd, false),
                    Fsub | Fsubp | Fisub => (LaneOp::FSub, false),
                    Fsubr | Fsubrp | Fisubr => (LaneOp::FSub, true),
                    Fmul | Fmulp | Fimul => (LaneOp::FMul, false),
                    Fdiv | Fdivp | Fidiv => (LaneOp::FDiv, false),
                    _ => (LaneOp::FDiv, true),
                };
                // destination st(d) = st(d) op source (reversed: source op st(d))
                let (d, src) = if i.op_count() == 1 && i.op0_kind() == OpKind::Memory {
                    let src = match m {
                        Fiadd | Fisub | Fisubr | Fimul | Fidiv | Fidivr => self.load_int(f, i)?,
                        _ => self.load_float(f, i)?,
                    };
                    (0, src)
                } else if i.op_count() == 2 {
                    let (d, s) = (self.st_of(i, 0)?, self.st_of(i, 1)?);
                    (d, self.st(f, s)?)
                } else {
                    return Err(self.unsupported());
                };
                let a = self.st(f, d)?;
                let v = if rev { self.fop(f, op, src, a) } else { self.fop(f, op, a, src) };
                self.set_st(d, v)?;
                if matches!(m, Faddp | Fsubp | Fsubrp | Fmulp | Fdivp | Fdivrp) {
                    self.fpop()?;
                }
            }
            Fchs | Fabs => {
                let v = self.st(f, 0)?;
                let (op, mask) = if m == Fchs { (BinOp::Xor, 1 << 63) } else { (BinOp::And, !(1u64 << 63)) };
                let mask = self.konst(f, mask, TyId::B8);
                let v = self.emit(f, InstKind::Bin { op, lhs: v, rhs: mask }, TyId::B8);
                self.set_st(0, v)?;
            }
            // ZF, PF and CF as `ucomisd` sets them
            Fcomi | Fcomip | Fucomi | Fucomip => {
                let k = self.st_of(i, 1)?;
                let (a, b) = (self.st(f, 0)?, self.st(f, k)?);
                self.flags = Flags::Float { a, b, w: 8 };
                if matches!(m, Fcomip | Fucomip) {
                    self.fpop()?;
                }
            }
            Fnstcw => {
                let cw = self.read_full(f, FPUCW);
                let v = self.emit(f, InstKind::Cast { kind: CastKind::Trunc, v: cw }, TyId::B2);
                let p = self.ea(f, i)?;
                self.emit(f, InstKind::Store { ptr: p, val: v, align: 1 }, TyId::UNIT);
            }
            Fldcw => {
                let p = self.ea(f, i)?;
                let v = self.emit(f, InstKind::Load { ptr: p, align: 1, volatile: false }, TyId::B2);
                let v = self.emit(f, InstKind::Cast { kind: CastKind::ZExt, v }, TyId::B8);
                self.state[self.cur].out[FPUCW] = Some(v);
            }
            _ => return Err(self.unsupported()),
        }
        Ok(())
    }
}

/// How an instruction changes the depth of the x87 stack.
fn depth_change(m: Mnemonic) -> i8 {
    use Mnemonic::*;
    match m {
        Fld | Fild | Fld1 | Fldz => 1,
        Fstp | Fistp | Fisttp | Faddp | Fsubp | Fsubrp | Fmulp | Fdivp | Fdivrp | Fcomip | Fucomip => -1,
        _ => 0,
    }
}

impl Lifter {
    /// The x87 depth each block starts with, into `fdepth_in`, following the
    /// code from the entry (empty there and after calls). Pass 2 lifts blocks in
    /// address order, so a loop's body can come before the branch that enters it.
    pub(super) fn x87_depths(&mut self, code: &[u8], ip: u64) {
        let end = ip + code.len() as u64;
        let mut dec = Decoder::with_ip(64, code, ip, DecoderOptions::NONE);
        let mut insn = Instruction::default();
        let mut todo = vec![(ip, 0i8)];
        while let Some((mut at, mut depth)) = todo.pop() {
            loop {
                if let Some(b) = self.block_at(at) {
                    if self.fdepth_in[b.index()].is_some() {
                        break;
                    }
                    self.fdepth_in[b.index()] = Some(depth.clamp(0, 8) as u8);
                }
                if !(ip..end).contains(&at) || dec.set_position((at - ip) as usize).is_err() {
                    break;
                }
                dec.set_ip(at);
                dec.decode_out(&mut insn);
                if insn.is_invalid() {
                    break;
                }
                // (random bytes can push or pop any number of times)
                depth = (depth + depth_change(insn.mnemonic())).clamp(-1, 9);
                at = insn.next_ip();
                match insn.flow_control() {
                    FlowControl::Call | FlowControl::IndirectCall => depth = 0,
                    FlowControl::ConditionalBranch => todo.push((insn.near_branch_target(), depth)),
                    FlowControl::UnconditionalBranch if insn.op0_kind() == OpKind::NearBranch64 => {
                        todo.push((insn.near_branch_target(), depth));
                        break;
                    }
                    FlowControl::IndirectBranch => {
                        for t in self.tables.iter().filter(|t| t.jmp == insn.ip()) {
                            todo.extend(self.cases[t.start as usize..(t.start + t.len) as usize].iter().map(|&c| (c, depth)));
                        }
                        break;
                    }
                    FlowControl::Return | FlowControl::UnconditionalBranch | FlowControl::Interrupt => break,
                    _ => {}
                }
            }
        }
    }
}
