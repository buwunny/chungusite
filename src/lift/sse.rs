//! SSE: xmm registers as (low, high) qword pairs in the register file.
//!
//! A vector instruction works on each 64-bit half separately: integer and float
//! lane ops become `BinOp::Lane` / `UnOp::Lane` (see `crate::simd`), shuffles and
//! inserts are built from shifts and masks on the halves, and scalar float ops
//! (`addsd`, `addss`) change only the low lane. A VEX (AVX) encoding of the same
//! instruction on xmm registers is lifted the same way, with its separate first
//! source; ymm registers are unsupported.
use super::*;
use iced_x86::EncodingKind;

/// The mnemonics `Lifter::sse` lifts (some only in particular operand forms).
pub(super) fn handled(m: Mnemonic) -> bool {
    use Mnemonic::*;
    lane_int(m).is_some()
        || float_op(m).is_some()
        || matches!(
            m,
            Movups | Movaps | Movdqu | Movdqa | Movupd | Movapd | Lddqu | Vmovups | Vmovaps | Vmovdqu | Vmovdqa
                | Vmovupd | Vmovapd | Movq | Movd | Movsd | Vmovq | Vmovd | Movss | Movhps | Movlps | Movhpd | Movlpd
                | Vzeroupper | Pand | Andps | Andpd | Vpand | Vandps | Vandpd | Por | Orps | Orpd | Vpor | Vorps | Vorpd
                | Pxor | Xorps | Xorpd | Vpxor | Vxorps | Vxorpd | Pandn | Andnps | Andnpd | Vpandn | Pcmpeqb | Pcmpeqw
                | Pcmpeqd | Pcmpeqq | Vpcmpeqb | Vpcmpeqw | Vpcmpeqd | Pmuludq | Punpcklbw | Punpcklwd | Punpckldq
                | Punpckhbw | Punpckhwd | Punpckhdq | Punpcklqdq | Punpckhqdq | Movlhps | Movhlps | Unpcklpd | Unpckhpd
                | Unpcklps | Unpckhps | Psllw | Pslld | Psllq | Psrlw | Psrld | Psrlq | Psraw | Psrad | Pslldq | Psrldq
                | Pmovmskb | Vpmovmskb | Movmskps | Movmskpd | Pshufd | Pshuflw | Pshufhw | Shufps | Shufpd | Pinsrw
                | Pextrw | Cmpsd | Cmpss | Cmppd | Cmpps | Ucomisd | Comisd | Ucomiss | Comiss | Cvtsi2sd | Cvtsi2ss
                | Cvttsd2si | Cvtsd2si | Cvttss2si | Cvtss2si | Cvtss2sd | Cvtsd2ss | Sqrtsd | Sqrtss | Sqrtpd | Sqrtps
        )
}

/// Integer lane ops: the op and the lane width in bytes.
fn lane_int(m: Mnemonic) -> Option<(LaneOp, u8)> {
    use LaneOp as L;
    use Mnemonic::*;
    Some(match m {
        Paddb | Vpaddb => (L::Add, 1),
        Paddw | Vpaddw => (L::Add, 2),
        Paddd | Vpaddd => (L::Add, 4),
        Paddq | Vpaddq => (L::Add, 8),
        Psubb | Vpsubb => (L::Sub, 1),
        Psubw | Vpsubw => (L::Sub, 2),
        Psubd | Vpsubd => (L::Sub, 4),
        Psubq | Vpsubq => (L::Sub, 8),
        Paddusb => (L::AddSatU, 1),
        Paddusw => (L::AddSatU, 2),
        Psubusb | Vpsubusb => (L::SubSatU, 1),
        Psubusw => (L::SubSatU, 2),
        Paddsb => (L::AddSatS, 1),
        Paddsw => (L::AddSatS, 2),
        Psubsb => (L::SubSatS, 1),
        Psubsw => (L::SubSatS, 2),
        Pminub | Vpminub => (L::MinU, 1),
        Pmaxub | Vpmaxub => (L::MaxU, 1),
        Pminsw => (L::MinS, 2),
        Pmaxsw => (L::MaxS, 2),
        Pavgb => (L::AvgU, 1),
        Pavgw => (L::AvgU, 2),
        Pmullw => (L::MulLo, 2),
        Pcmpgtb | Vpcmpgtb => (L::CmpGtS, 1),
        Pcmpgtw | Vpcmpgtw => (L::CmpGtS, 2),
        Pcmpgtd | Vpcmpgtd => (L::CmpGtS, 4),
        Psadbw => (L::SumAbsDiff, 1),
        _ => return None,
    })
}

/// Float arithmetic: the op, the lane width, and whether it is scalar (only the
/// low lane changes).
fn float_op(m: Mnemonic) -> Option<(LaneOp, u8, bool)> {
    use LaneOp as L;
    use Mnemonic::*;
    let (op, w, scalar) = match m {
        Addsd | Vaddsd => (L::FAdd, 8, true),
        Subsd | Vsubsd => (L::FSub, 8, true),
        Mulsd | Vmulsd => (L::FMul, 8, true),
        Divsd | Vdivsd => (L::FDiv, 8, true),
        Minsd => (L::FMin, 8, true),
        Maxsd => (L::FMax, 8, true),
        Addss | Vaddss => (L::FAdd, 4, true),
        Subss | Vsubss => (L::FSub, 4, true),
        Mulss | Vmulss => (L::FMul, 4, true),
        Divss | Vdivss => (L::FDiv, 4, true),
        Minss => (L::FMin, 4, true),
        Maxss => (L::FMax, 4, true),
        Addpd => (L::FAdd, 8, false),
        Subpd => (L::FSub, 8, false),
        Mulpd => (L::FMul, 8, false),
        Divpd => (L::FDiv, 8, false),
        Minpd => (L::FMin, 8, false),
        Maxpd => (L::FMax, 8, false),
        Addps => (L::FAdd, 4, false),
        Subps => (L::FSub, 4, false),
        Mulps => (L::FMul, 4, false),
        Divps => (L::FDiv, 4, false),
        Minps => (L::FMin, 4, false),
        Maxps => (L::FMax, 4, false),
        _ => return None,
    };
    Some((op, w, scalar))
}

type Pair = (ValueId, ValueId);

impl Lifter {
    /// Lift an SSE instruction (`handled`).
    pub(super) fn sse(&mut self, f: &mut Function, i: &Instruction) -> Result<(), LiftError> {
        use Mnemonic::*;
        let m = i.mnemonic();
        // `movsd` and `cmpsd` are also string instructions (without a rep prefix
        // here), whose operands are [rsi] / [rdi]
        if (0..i.op_count()).any(|k| matches!(i.op_kind(k), OpKind::MemorySegRSI | OpKind::MemorySegESI | OpKind::MemorySegSI
            | OpKind::MemoryESRDI | OpKind::MemoryESEDI | OpKind::MemoryESDI))
        {
            return Err(self.unsupported());
        }
        if let Some((op, w)) = lane_int(m) {
            return self.lanes(f, i, |l, f, a, b| l.emit(f, InstKind::Bin { op: BinOp::Lane(op, w), lhs: a, rhs: b }, TyId::B8));
        }
        if let Some((op, w, scalar)) = float_op(m) {
            let (d, a, b) = self.srcs(f, i)?;
            let lo = self.lane(f, op, w, a.0, b.0);
            let v = match scalar {
                true => (self.low_lane(f, w, lo, a.0), a.1),
                false => (lo, self.lane(f, op, w, a.1, b.1)),
            };
            self.xmm_set(d, v);
            return Ok(());
        }
        match m {
            Movups | Movaps | Movdqu | Movdqa | Movupd | Movapd | Lddqu | Vmovups | Vmovaps | Vmovdqu | Vmovdqa
            | Vmovupd | Vmovapd => self.mov128(f, i)?,
            Movq | Movd | Movsd | Vmovq | Vmovd => self.movq(f, i)?,
            Movss => self.movss(f, i)?,
            Movhps | Movhpd | Movlps | Movlpd => self.mov_half(f, i)?,
            // only clears the upper halves of the ymm registers
            Vzeroupper => {}
            Pand | Andps | Andpd | Vpand | Vandps | Vandpd => self.lanes(f, i, |l, f, a, b| l.bin(f, BinOp::And, a, b))?,
            Por | Orps | Orpd | Vpor | Vorps | Vorpd => self.lanes(f, i, |l, f, a, b| l.bin(f, BinOp::Or, a, b))?,
            // xor of a register with itself: the zeroing idiom
            Pxor | Xorps | Xorpd | Vpxor | Vxorps | Vxorpd if self.same_srcs(i) => {
                let z = self.konst(f, 0, TyId::B8);
                self.xmm_set(self.xmm_of(i.op0_register())?, (z, z));
            }
            Pxor | Xorps | Xorpd | Vpxor | Vxorps | Vxorpd => self.lanes(f, i, |l, f, a, b| l.bin(f, BinOp::Xor, a, b))?,
            Pandn | Andnps | Andnpd | Vpandn => self.lanes(f, i, |l, f, a, b| {
                let na = l.emit(f, InstKind::Un { op: UnOp::Not, v: a }, TyId::B8);
                l.bin(f, BinOp::And, na, b)
            })?,
            // compare with itself: all ones
            Pcmpeqb | Pcmpeqw | Pcmpeqd | Pcmpeqq | Vpcmpeqb | Vpcmpeqw | Vpcmpeqd if self.same_srcs(i) => {
                let ones = self.konst(f, u64::MAX, TyId::B8);
                self.xmm_set(self.xmm_of(i.op0_register())?, (ones, ones));
            }
            Pcmpeqb | Pcmpeqw | Pcmpeqd | Pcmpeqq | Vpcmpeqb | Vpcmpeqw | Vpcmpeqd => {
                let w = match m {
                    Pcmpeqb | Vpcmpeqb => 1,
                    Pcmpeqw | Vpcmpeqw => 2,
                    Pcmpeqd | Vpcmpeqd => 4,
                    _ => 8,
                };
                self.lanes(f, i, |l, f, a, b| l.lane(f, LaneOp::CmpEq, w, a, b))?;
            }
            // the low dwords of each qword, multiplied into the whole qword
            Pmuludq => self.lanes(f, i, |l, f, a, b| {
                let m = l.konst(f, 0xffff_ffff, TyId::B8);
                let a = l.bin(f, BinOp::And, a, m);
                let b = l.bin(f, BinOp::And, b, m);
                l.bin(f, BinOp::Mul, a, b)
            })?,
            Punpcklbw | Punpcklwd | Punpckldq | Unpcklps | Punpckhbw | Punpckhwd | Punpckhdq | Unpckhps => {
                let w = match m {
                    Punpcklbw | Punpckhbw => 1,
                    Punpcklwd | Punpckhwd => 2,
                    _ => 4,
                };
                let high = matches!(m, Punpckhbw | Punpckhwd | Punpckhdq | Unpckhps);
                let (d, a, b) = self.srcs(f, i)?;
                let (x, y) = if high { (a.1, b.1) } else { (a.0, b.0) };
                let lo = self.lane(f, LaneOp::UnpackLo, w, x, y);
                let hi = self.lane(f, LaneOp::UnpackHi, w, x, y);
                self.xmm_set(d, (lo, hi));
            }
            Punpcklqdq | Movlhps | Unpcklpd => {
                let (d, a, b) = self.srcs(f, i)?;
                self.xmm_set(d, (a.0, b.0));
            }
            Punpckhqdq | Unpckhpd => {
                let (d, a, b) = self.srcs(f, i)?;
                self.xmm_set(d, (a.1, b.1));
            }
            Movhlps => {
                let (d, a, b) = self.srcs(f, i)?;
                self.xmm_set(d, (b.1, a.1));
            }
            Psllw | Pslld | Psllq | Psrlw | Psrld | Psrlq | Psraw | Psrad => {
                let w = match m {
                    Psllw | Psrlw | Psraw => 2,
                    Pslld | Psrld | Psrad => 4,
                    _ => 8,
                };
                let op = match m {
                    Psllw | Pslld | Psllq => LaneOp::Shl,
                    Psrlw | Psrld | Psrlq => LaneOp::LShr,
                    _ => LaneOp::AShr,
                };
                let d = self.xmm_of(i.op0_register())?;
                let a = self.xmm_get(f, d)?;
                // the count: an immediate, or the low qword of an xmm register or memory
                let n = match i.op1_kind() {
                    k if is_imm(k) => self.konst(f, i.immediate(1) & 0xff, TyId::B8),
                    _ => self.xmm_operand(f, i)?.0,
                };
                let lo = self.lane(f, op, w, a.0, n);
                let hi = self.lane(f, op, w, a.1, n);
                self.xmm_set(d, (lo, hi));
            }
            // the whole register shifted by bytes
            Pslldq | Psrldq => {
                let d = self.xmm_of(i.op0_register())?;
                let a = self.xmm_get(f, d)?;
                let n = (i.immediate(1) & 0xff).min(16) as u32;
                let v = self.byte_shift(f, a, n, m == Pslldq);
                self.xmm_set(d, v);
            }
            Pmovmskb | Vpmovmskb | Movmskps | Movmskpd => {
                let w = match m {
                    Movmskps => 4,
                    Movmskpd => 8,
                    _ => 1,
                };
                let n = self.xmm_of(i.op1_register())?;
                let (lo, hi) = self.xmm_get(f, n)?;
                let lo = self.emit(f, InstKind::Un { op: UnOp::Lane(LaneUn::MoveMask, w), v: lo }, TyId::B8);
                let hi = self.emit(f, InstKind::Un { op: UnOp::Lane(LaneUn::MoveMask, w), v: hi }, TyId::B8);
                let k = self.konst(f, 8 / w as u64, TyId::B1);
                let hi = self.bin(f, BinOp::Shl, hi, k);
                let v = self.bin(f, BinOp::Or, lo, hi);
                let r = i.op0_register();
                let v = if r.size() == 8 { v } else { self.emit(f, InstKind::Cast { kind: CastKind::Trunc, v }, TyId::B4) };
                self.write(f, r, v)?;
            }
            Pshufd => {
                let d = self.xmm_of(i.op0_register())?;
                let s = self.xmm_operand(f, i)?;
                let k = i.immediate(2) as u32;
                let dw = [self.dword(f, s.0, 0), self.dword(f, s.0, 1), self.dword(f, s.1, 0), self.dword(f, s.1, 1)];
                let pick = |j: u32| dw[(k >> (2 * j) & 3) as usize];
                let lo = self.join(f, &[pick(0), pick(1)], 32);
                let hi = self.join(f, &[pick(2), pick(3)], 32);
                self.xmm_set(d, (lo, hi));
            }
            // the words of one half shuffled, the other half copied
            Pshuflw | Pshufhw => {
                let d = self.xmm_of(i.op0_register())?;
                let s = self.xmm_operand(f, i)?;
                let k = i.immediate(2) as u32;
                let half = if m == Pshuflw { s.0 } else { s.1 };
                let words: Vec<ValueId> = (0..4).map(|j| self.field(f, half, 16 * j, 16)).collect();
                let picked: Vec<ValueId> = (0..4).map(|j| words[(k >> (2 * j) & 3) as usize]).collect();
                let shuffled = self.join(f, &picked, 16);
                self.xmm_set(d, if m == Pshuflw { (shuffled, s.1) } else { (s.0, shuffled) });
            }
            // dwords 0 and 1 from the destination, 2 and 3 from the source
            Shufps => {
                let d = self.xmm_of(i.op0_register())?;
                let a = self.xmm_get(f, d)?;
                let b = self.xmm_operand(f, i)?;
                let k = i.immediate(2) as u32;
                let da = [self.dword(f, a.0, 0), self.dword(f, a.0, 1), self.dword(f, a.1, 0), self.dword(f, a.1, 1)];
                let db = [self.dword(f, b.0, 0), self.dword(f, b.0, 1), self.dword(f, b.1, 0), self.dword(f, b.1, 1)];
                let lo = self.join(f, &[da[(k & 3) as usize], da[(k >> 2 & 3) as usize]], 32);
                let hi = self.join(f, &[db[(k >> 4 & 3) as usize], db[(k >> 6 & 3) as usize]], 32);
                self.xmm_set(d, (lo, hi));
            }
            Shufpd => {
                let d = self.xmm_of(i.op0_register())?;
                let a = self.xmm_get(f, d)?;
                let b = self.xmm_operand(f, i)?;
                let k = i.immediate(2);
                let lo = if k & 1 == 0 { a.0 } else { a.1 };
                let hi = if k & 2 == 0 { b.0 } else { b.1 };
                self.xmm_set(d, (lo, hi));
            }
            Pinsrw => {
                let d = self.xmm_of(i.op0_register())?;
                let a = self.xmm_get(f, d)?;
                let k = (i.immediate(2) & 7) as u32;
                let w = match i.op1_kind() {
                    OpKind::Register => self.read(f, i.op1_register())?,
                    _ => self.operand(f, i, 1, 2)?,
                };
                let w = self.emit(f, InstKind::Cast { kind: CastKind::ZExt, v: w }, TyId::B8);
                let w = self.field(f, w, 0, 16);
                let half = if k < 4 { a.0 } else { a.1 };
                let at = 16 * (k % 4);
                let keep = self.konst(f, !(0xffff << at), TyId::B8);
                let kept = self.bin(f, BinOp::And, half, keep);
                let s = self.konst(f, at as u64, TyId::B1);
                let moved = self.bin(f, BinOp::Shl, w, s);
                let half = self.bin(f, BinOp::Or, kept, moved);
                self.xmm_set(d, if k < 4 { (half, a.1) } else { (a.0, half) });
            }
            Pextrw => {
                let n = self.xmm_of(i.op1_register())?;
                let a = self.xmm_get(f, n)?;
                let k = (i.immediate(2) & 7) as u32;
                let v = self.field(f, if k < 4 { a.0 } else { a.1 }, 16 * (k % 4), 16);
                let r = i.op0_register();
                let v = if r.size() == 8 { v } else { self.emit(f, InstKind::Cast { kind: CastKind::Trunc, v }, TyId::B4) };
                self.write(f, r, v)?;
            }
            Cmpsd | Cmpss | Cmppd | Cmpps => {
                let op = match i.immediate(2) & 7 {
                    0 => LaneOp::FCmpEq,
                    1 => LaneOp::FCmpLt,
                    2 => LaneOp::FCmpLe,
                    3 => LaneOp::FCmpUnord,
                    4 => LaneOp::FCmpNeq,
                    5 => LaneOp::FCmpNlt,
                    6 => LaneOp::FCmpNle,
                    _ => LaneOp::FCmpOrd,
                };
                let w = if matches!(m, Cmpsd | Cmppd) { 8 } else { 4 };
                let d = self.xmm_of(i.op0_register())?;
                let a = self.xmm_get(f, d)?;
                let b = self.xmm_operand(f, i)?;
                let lo = self.lane(f, op, w, a.0, b.0);
                let v = match m {
                    Cmpsd | Cmpss => (self.low_lane(f, w, lo, a.0), a.1),
                    _ => (lo, self.lane(f, op, w, a.1, b.1)),
                };
                self.xmm_set(d, v);
            }
            Ucomisd | Comisd | Ucomiss | Comiss => {
                let a = self.xmm_of(i.op0_register())?;
                let a = self.xmm_get(f, a)?.0;
                let b = self.xmm_operand(f, i)?.0;
                let w = if matches!(m, Ucomisd | Comisd) { 8 } else { 4 };
                self.flags = Flags::Float { a, b, w };
            }
            Cvtsi2sd | Cvtsi2ss => {
                let d = self.xmm_of(i.op0_register())?;
                let a = self.xmm_kept(f, d);
                let sz = self.op_size(i, 1)?;
                let v = self.operand(f, i, 1, sz)?;
                let v = if sz == 8 { v } else { self.emit(f, InstKind::Cast { kind: CastKind::ZExt, v }, TyId::B8) };
                let (op, w) = if m == Cvtsi2sd { (LaneUn::IntToF64, 8) } else { (LaneUn::IntToF32, 4) };
                let r = self.emit(f, InstKind::Un { op: UnOp::Lane(op, sz as u8), v }, TyId::B8);
                let lo = self.low_lane(f, w, r, a.0);
                self.xmm_set(d, (lo, a.1));
            }
            Cvttsd2si | Cvtsd2si | Cvttss2si | Cvtss2si => {
                let r = i.op0_register();
                let v = self.xmm_operand(f, i)?.0;
                let op = match m {
                    Cvttsd2si => LaneUn::F64ToIntTrunc,
                    Cvtsd2si => LaneUn::F64ToInt,
                    Cvttss2si => LaneUn::F32ToIntTrunc,
                    _ => LaneUn::F32ToInt,
                };
                let v = self.emit(f, InstKind::Un { op: UnOp::Lane(op, r.size() as u8), v }, TyId::B8);
                let v = if r.size() == 8 { v } else { self.emit(f, InstKind::Cast { kind: CastKind::Trunc, v }, TyId::B4) };
                self.write(f, r, v)?;
            }
            Cvtss2sd | Cvtsd2ss => {
                let d = self.xmm_of(i.op0_register())?;
                let a = self.xmm_kept(f, d);
                let b = self.xmm_operand(f, i)?.0;
                let (op, w) = if m == Cvtss2sd { (LaneUn::F32ToF64, 8) } else { (LaneUn::F64ToF32, 4) };
                let r = self.emit(f, InstKind::Un { op: UnOp::Lane(op, 8), v: b }, TyId::B8);
                let lo = self.low_lane(f, w, r, a.0);
                self.xmm_set(d, (lo, a.1));
            }
            Sqrtsd | Sqrtss | Sqrtpd | Sqrtps => {
                let w = if matches!(m, Sqrtsd | Sqrtpd) { 8 } else { 4 };
                let d = self.xmm_of(i.op0_register())?;
                let b = self.xmm_operand(f, i)?;
                let sqrt = |l: &mut Self, f: &mut Function, v| l.emit(f, InstKind::Un { op: UnOp::Lane(LaneUn::FSqrt, w), v }, TyId::B8);
                let lo = sqrt(self, f, b.0);
                let v = match m {
                    Sqrtsd | Sqrtss => {
                        let a = self.xmm_kept(f, d);
                        (self.low_lane(f, w, lo, a.0), a.1)
                    }
                    _ => (lo, sqrt(self, f, b.1)),
                };
                self.xmm_set(d, v);
            }
            _ => return Err(self.unsupported()),
        }
        Ok(())
    }

    /// The bits of xmm register `n` that a scalar conversion or square root
    /// leaves alone. Compilers treat them as garbage (that's why they write
    /// `xorps` first when the false dependency matters), so a half this block
    /// hasn't set is `Undef` rather than a live-in, which would fail at entry,
    /// where the register holds an argument or nothing.
    fn xmm_kept(&mut self, f: &mut Function, n: usize) -> Pair {
        let st = self.state[self.cur];
        let half = |l: &mut Self, f: &mut Function, r: usize| match st.out[r] {
            Some(v) if st.clobbered >> r & 1 == 0 => v,
            _ => l.emit(f, InstKind::Undef, TyId::B8),
        };
        (half(self, f, NGPR + 2 * n), half(self, f, NGPR + 2 * n + 1))
    }

    /// Is this a VEX (AVX) encoding, with its first source separate from the destination?
    fn vex(i: &Instruction) -> bool {
        i.encoding() == EncodingKind::VEX && i.op_count() == 3
    }

    /// Are both sources the same register (`pxor xmm0, xmm0`)?
    fn same_srcs(&self, i: &Instruction) -> bool {
        let (a, b) = if Self::vex(i) { (1, 2) } else { (0, 1) };
        i.op_kind(b) == OpKind::Register && i.op_register(a) == i.op_register(b)
    }

    /// The destination register and both sources of a two-source instruction:
    /// `op0 = op(op0, op1)`, or `op0 = op(op1, op2)` for a VEX encoding.
    fn srcs(&mut self, f: &mut Function, i: &Instruction) -> Result<(usize, Pair, Pair), LiftError> {
        let d = self.xmm_of(i.op0_register())?;
        let (a, b) = if Self::vex(i) { (1, 2) } else { (0, 1) };
        let a = self.xmm_of(i.op_register(a))?;
        let a = self.xmm_get(f, a)?;
        let b = self.xmm_op(f, i, b)?;
        Ok((d, a, b))
    }

    /// `op` on both halves of the sources, into the destination.
    fn lanes(
        &mut self,
        f: &mut Function,
        i: &Instruction,
        op: impl Fn(&mut Self, &mut Function, ValueId, ValueId) -> ValueId,
    ) -> Result<(), LiftError> {
        let (d, a, b) = self.srcs(f, i)?;
        let lo = op(self, f, a.0, b.0);
        let hi = op(self, f, a.1, b.1);
        self.xmm_set(d, (lo, hi));
        Ok(())
    }

    fn lane(&mut self, f: &mut Function, op: LaneOp, w: u8, a: ValueId, b: ValueId) -> ValueId {
        self.emit(f, InstKind::Bin { op: BinOp::Lane(op, w), lhs: a, rhs: b }, TyId::B8)
    }

    fn bin(&mut self, f: &mut Function, op: BinOp, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.emit(f, InstKind::Bin { op, lhs, rhs }, TyId::B8)
    }

    /// A scalar result `v` in the low `w`-byte lane of `old`: all of `v` for an
    /// f64, its low dword over `old`'s high dword for an f32.
    fn low_lane(&mut self, f: &mut Function, w: u8, v: ValueId, old: ValueId) -> ValueId {
        if w == 8 {
            return v;
        }
        let lo = self.konst(f, 0xffff_ffff, TyId::B8);
        let hi = self.konst(f, !0xffff_ffff, TyId::B8);
        let v = self.bin(f, BinOp::And, v, lo);
        let old = self.bin(f, BinOp::And, old, hi);
        self.bin(f, BinOp::Or, old, v)
    }

    /// Bits `at..at + len` of `v`, at the bottom.
    fn field(&mut self, f: &mut Function, v: ValueId, at: u32, len: u32) -> ValueId {
        let v = if at == 0 {
            v
        } else {
            let s = self.konst(f, at as u64, TyId::B1);
            self.bin(f, BinOp::LShr, v, s)
        };
        if at + len >= 64 {
            return v;
        }
        let m = self.konst(f, (1 << len) - 1, TyId::B8);
        self.bin(f, BinOp::And, v, m)
    }

    /// Dword `k` (0 or 1) of the qword `v`.
    fn dword(&mut self, f: &mut Function, v: ValueId, k: u32) -> ValueId {
        self.field(f, v, 32 * k, 32)
    }

    /// `parts`, each `bits` wide and at the bottom of its value, packed from the bottom up.
    fn join(&mut self, f: &mut Function, parts: &[ValueId], bits: u32) -> ValueId {
        let mut v = parts[0];
        for (k, &p) in parts.iter().enumerate().skip(1) {
            let s = self.konst(f, (bits * k as u32) as u64, TyId::B1);
            let p = self.bin(f, BinOp::Shl, p, s);
            v = self.bin(f, BinOp::Or, v, p);
        }
        v
    }

    /// The 128-bit value `(lo, hi)` shifted left (towards higher bytes) or right by `n` bytes.
    fn byte_shift(&mut self, f: &mut Function, (lo, hi): Pair, n: u32, left: bool) -> Pair {
        let zero = self.konst(f, 0, TyId::B8);
        if n == 0 {
            return (lo, hi);
        }
        if n >= 16 {
            return (zero, zero);
        }
        let (op, back) = if left { (BinOp::Shl, BinOp::LShr) } else { (BinOp::LShr, BinOp::Shl) };
        // the half the bytes move out of, and the one they move into
        let (from, into) = if left { (lo, hi) } else { (hi, lo) };
        let (new_from, new_into) = if n >= 8 {
            let s = self.konst(f, 8 * (n as u64 - 8), TyId::B1);
            (zero, self.bin(f, op, from, s))
        } else {
            let s = self.konst(f, 8 * n as u64, TyId::B1);
            let r = self.konst(f, 64 - 8 * n as u64, TyId::B1);
            let moved = self.bin(f, op, from, s);
            let shifted = self.bin(f, op, into, s);
            let carried = self.bin(f, back, from, r);
            (moved, self.bin(f, BinOp::Or, shifted, carried))
        };
        if left { (new_from, new_into) } else { (new_into, new_from) }
    }

    /// Operand `k` as a (low, high) pair: an xmm register, or memory of the
    /// operand's size (16 bytes, or 8 or 4 for a scalar, zero-extended).
    fn xmm_op(&mut self, f: &mut Function, i: &Instruction, k: u32) -> Result<Pair, LiftError> {
        match i.op_kind(k) {
            OpKind::Register => {
                let n = self.xmm_of(i.op_register(k))?;
                self.xmm_get(f, n)
            }
            OpKind::Memory => {
                let p = self.ea(f, i)?;
                match i.memory_size().size() {
                    16 => {
                        let lo = self.emit(f, InstKind::Load { ptr: p, align: 1, volatile: false }, TyId::B8);
                        let p8 = self.emit(f, InstKind::PtrOffset { base: p, index: None, scale: 1, disp: 8 }, TyId::PTR);
                        let hi = self.emit(f, InstKind::Load { ptr: p8, align: 1, volatile: false }, TyId::B8);
                        Ok((lo, hi))
                    }
                    8 => {
                        let v = self.emit(f, InstKind::Load { ptr: p, align: 1, volatile: false }, TyId::B8);
                        Ok((v, self.konst(f, 0, TyId::B8)))
                    }
                    4 => {
                        let v = self.emit(f, InstKind::Load { ptr: p, align: 1, volatile: false }, TyId::B4);
                        let v = self.emit(f, InstKind::Cast { kind: CastKind::ZExt, v }, TyId::B8);
                        Ok((v, self.konst(f, 0, TyId::B8)))
                    }
                    _ => Err(self.unsupported()),
                }
            }
            _ => Err(self.unsupported()),
        }
    }

    /// movss: a load zeroes the rest of the register, a register move only
    /// replaces the low dword, a store writes the low dword.
    fn movss(&mut self, f: &mut Function, i: &Instruction) -> Result<(), LiftError> {
        match (i.op0_kind(), i.op1_kind()) {
            (OpKind::Register, OpKind::Register) => {
                let d = self.xmm_of(i.op0_register())?;
                let a = self.xmm_get(f, d)?;
                let b = self.xmm_operand(f, i)?;
                let lo = self.low_lane(f, 4, b.0, a.0);
                self.xmm_set(d, (lo, a.1));
            }
            (OpKind::Register, OpKind::Memory) => {
                let d = self.xmm_of(i.op0_register())?;
                let v = self.xmm_op(f, i, 1)?;
                self.xmm_set(d, v);
            }
            (OpKind::Memory, OpKind::Register) => {
                let n = self.xmm_of(i.op1_register())?;
                let lo = self.xmm_get(f, n)?.0;
                let v = self.emit(f, InstKind::Cast { kind: CastKind::Trunc, v: lo }, TyId::B4);
                let p = self.ea(f, i)?;
                self.emit(f, InstKind::Store { ptr: p, val: v, align: 1 }, TyId::UNIT);
            }
            _ => return Err(self.unsupported()),
        }
        Ok(())
    }

    /// movhps / movhpd / movlps / movlpd: one half of an xmm register to or from memory.
    fn mov_half(&mut self, f: &mut Function, i: &Instruction) -> Result<(), LiftError> {
        let high = matches!(i.mnemonic(), Mnemonic::Movhps | Mnemonic::Movhpd);
        match (i.op0_kind(), i.op1_kind()) {
            (OpKind::Register, OpKind::Memory) => {
                let d = self.xmm_of(i.op0_register())?;
                let a = self.xmm_get(f, d)?;
                let p = self.ea(f, i)?;
                let v = self.emit(f, InstKind::Load { ptr: p, align: 1, volatile: false }, TyId::B8);
                self.xmm_set(d, if high { (a.0, v) } else { (v, a.1) });
            }
            (OpKind::Memory, OpKind::Register) => {
                let n = self.xmm_of(i.op1_register())?;
                let a = self.xmm_get(f, n)?;
                let p = self.ea(f, i)?;
                self.emit(f, InstKind::Store { ptr: p, val: if high { a.1 } else { a.0 }, align: 1 }, TyId::UNIT);
            }
            _ => return Err(self.unsupported()),
        }
        Ok(())
    }

    /// movups / movaps / movdqu / movdqa between xmm registers and memory: two
    /// qword loads or stores.
    pub(super) fn mov128(&mut self, f: &mut Function, i: &Instruction) -> Result<(), LiftError> {
        match (i.op0_kind(), i.op1_kind()) {
            (OpKind::Register, _) => {
                let d = self.xmm_of(i.op0_register())?;
                let v = self.xmm_operand(f, i)?;
                self.xmm_set(d, v);
            }
            (OpKind::Memory, OpKind::Register) => {
                let (lo, hi) = { let n = self.xmm_of(i.op1_register())?; self.xmm_get(f, n) }?;
                let p = self.ea(f, i)?;
                let p8 = self.emit(f, InstKind::PtrOffset { base: p, index: None, scale: 1, disp: 8 }, TyId::PTR);
                self.emit(f, InstKind::Store { ptr: p, val: lo, align: 1 }, TyId::UNIT);
                self.emit(f, InstKind::Store { ptr: p8, val: hi, align: 1 }, TyId::UNIT);
            }
            _ => return Err(self.unsupported()),
        }
        Ok(())
    }

    /// movq / movd / movsd between an xmm register and a general register or
    /// memory. Writing the xmm register zeroes the rest of it, except for
    /// `movsd xmm, xmm`, which only replaces the low qword.
    pub(super) fn movq(&mut self, f: &mut Function, i: &Instruction) -> Result<(), LiftError> {
        let sz = match i.mnemonic() {
            Mnemonic::Movd => 4,
            _ => 8,
        };
        let xmm_dst = i.op0_kind() == OpKind::Register && i.op0_register().is_xmm();
        let xmm_src = i.op1_kind() == OpKind::Register && i.op1_register().is_xmm();
        if i.mnemonic() == Mnemonic::Movsd && !(xmm_dst || xmm_src) {
            return Err(self.unsupported()); // the string instruction
        }
        match (xmm_dst, xmm_src) {
            (true, true) => {
                let d = self.xmm_of(i.op0_register())?;
                let (lo, _) = { let n = self.xmm_of(i.op1_register())?; self.xmm_get(f, n) }?;
                let hi = if i.mnemonic() == Mnemonic::Movsd {
                    self.xmm_get(f, d)?.1
                } else {
                    self.konst(f, 0, TyId::B8)
                };
                self.xmm_set(d, (lo, hi));
            }
            (true, false) => {
                let d = self.xmm_of(i.op0_register())?;
                let mut lo = self.operand(f, i, 1, sz)?;
                if sz == 4 {
                    lo = self.emit(f, InstKind::Cast { kind: CastKind::ZExt, v: lo }, TyId::B8);
                }
                let hi = self.konst(f, 0, TyId::B8);
                self.xmm_set(d, (lo, hi));
            }
            (false, true) => {
                let (mut lo, _) = { let n = self.xmm_of(i.op1_register())?; self.xmm_get(f, n) }?;
                if sz == 4 {
                    lo = self.emit(f, InstKind::Cast { kind: CastKind::Trunc, v: lo }, TyId::B4);
                }
                if i.op0_kind() == OpKind::Register && i.op0_register().size() != sz {
                    return Err(self.unsupported());
                }
                let dst = self.dst(f, i)?;
                self.put(f, dst, lo)?;
            }
            (false, false) => return Err(self.unsupported()), // mmx
        }
        Ok(())
    }

    /// Source operand 1 as a 16-byte (low, high) pair: an xmm register or two loads.
    /// The source operand (operand 1): an xmm register, or 16, 8 or 4 bytes of
    /// memory zero-extended.
    pub(super) fn xmm_operand(&mut self, f: &mut Function, i: &Instruction) -> Result<(ValueId, ValueId), LiftError> {
        self.xmm_op(f, i, 1)
    }

    pub(super) fn xmm_of(&self, r: Register) -> Result<usize, LiftError> {
        let n = (r as usize).wrapping_sub(Register::XMM0 as usize);
        if n < 16 { Ok(n) } else { Err(self.unsupported()) }
    }

    /// The value of xmm register `n`: a live-in pair if this block hasn't set it.
    pub(super) fn xmm_get(&mut self, f: &mut Function, n: usize) -> Result<(ValueId, ValueId), LiftError> {
        let (lo, hi) = (NGPR + 2 * n, NGPR + 2 * n + 1);
        if self.state[self.cur].clobbered >> lo & 1 != 0 {
            return Err(LiftError::XmmNotSet { ip: self.ip });
        }
        let st = &mut self.state[self.cur];
        if st.out[lo].is_none() && st.xmm_read == 0 {
            st.xmm_read = self.ip;
        }
        let lo = match self.state[self.cur].out[lo] {
            Some(v) => v,
            None => self.live_in(f, self.cur, lo),
        };
        let hi = match self.state[self.cur].out[hi] {
            Some(v) => v,
            None => self.live_in(f, self.cur, hi),
        };
        Ok((lo, hi))
    }

    pub(super) fn xmm_set(&mut self, n: usize, (lo, hi): (ValueId, ValueId)) {
        let st = &mut self.state[self.cur];
        st.out[NGPR + 2 * n] = Some(lo);
        st.out[NGPR + 2 * n + 1] = Some(hi);
        st.clobbered &= !(3 << (NGPR + 2 * n));
    }

}
