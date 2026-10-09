//! Instructions the lifter has no model of, kept as they are (`Lifter::asm`):
//! an `Opaque` that runs the instruction as inline assembly on the values of
//! the registers it reads, and an `AsmOut` for each register it writes.
//! Hand-written code uses instructions no compiler emits for plain C or Rust
//! (`rdtsc`, `crc32`, `pshufb`, `aesenc`, `rcl`), and before this one of them
//! failed its whole function.
//!
//! An instruction is kept if all it touches is general registers (rsp only in
//! an address), xmm registers, the status flags, and memory through its one
//! memory operand. The operand's address is computed outside the `asm!` the
//! way the lifter computes any other, so a stack slot or a static is where the
//! decompiled code keeps it. Flags it reads are built from the pending flags
//! and loaded into rflags first (`push; popfq`); flags it writes are read back
//! after it (`pushfq; pop`) and become `Flags::Raw`.

use super::*;
use iced_x86::{EncodingKind, Formatter, IntelFormatter, MemorySizeOptions, OpAccess, RflagsBits};

/// rflags bits.
pub(super) const CF: u16 = 1 << 0;
pub(super) const PF: u16 = 1 << 2;
pub(super) const AF: u16 = 1 << 4;
pub(super) const ZF: u16 = 1 << 6;
pub(super) const SF: u16 = 1 << 7;
pub(super) const OF: u16 = 1 << 11;
const STATUS: u16 = CF | PF | AF | ZF | SF | OF;

/// The status flags as iced numbers them, with their rflags bit and the
/// condition that reads each alone (none for AF).
const FLAGS: [(u32, u16, Option<ConditionCode>); 6] = [
    (RflagsBits::CF, CF, Some(ConditionCode::b)),
    (RflagsBits::PF, PF, Some(ConditionCode::p)),
    (RflagsBits::AF, AF, None),
    (RflagsBits::ZF, ZF, Some(ConditionCode::e)),
    (RflagsBits::SF, SF, Some(ConditionCode::s)),
    (RflagsBits::OF, OF, Some(ConditionCode::o)),
];

/// The flags `i` reads or writes (`STATUS` bits). The direction flag is
/// always clear, as the ABI wants it; any other flag isn't modelled.
fn flags_of(bits: u32) -> Option<u16> {
    let known = FLAGS.iter().fold(RflagsBits::DF, |m, f| m | f.0);
    (bits & !known == 0).then(|| FLAGS.iter().filter(|f| bits & f.0 != 0).fold(0, |m, f| m | f.1))
}

/// Memory operand sizes in LLVM's Intel syntax.
const SIZES: [&str; 8] = ["byte ptr", "word ptr", "dword ptr", "fword ptr", "qword ptr", "tbyte ptr", "xmmword ptr", "ymmword ptr"];

const GPR_NAMES: [&str; 16] = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"];
const XMM_NAMES: [&str; 16] = [
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",
];

/// A register the instruction uses: a full GPR or an xmm register.
struct Bind {
    reg: Register,
    read: bool,
    write: bool,
    /// An 8- or 16-bit part of it is written, which keeps the rest.
    partial: bool,
    /// Used as ah, bh, ch or dh somewhere.
    high: bool,
}

/// Note that the instruction uses `reg` (any part of it) with `access`.
fn add(binds: &mut Vec<Bind>, reg: Register, access: OpAccess) {
    let full = if reg.is_gpr() { reg.full_register() } else { reg };
    let k = match binds.iter().position(|b| b.reg == full) {
        Some(k) => k,
        None => {
            binds.push(Bind { reg: full, read: false, write: false, partial: false, high: false });
            binds.len() - 1
        }
    };
    let b = &mut binds[k];
    b.read |= reads(access);
    b.write |= writes(access);
    b.partial |= writes(access) && reg.is_gpr() && reg.size() <= 2;
    b.high |= is_high_byte(reg);
}

fn reads(a: OpAccess) -> bool {
    matches!(a, OpAccess::Read | OpAccess::CondRead | OpAccess::ReadWrite | OpAccess::ReadCondWrite)
}

fn writes(a: OpAccess) -> bool {
    matches!(a, OpAccess::Write | OpAccess::CondWrite | OpAccess::ReadWrite | OpAccess::ReadCondWrite)
}

/// `rbx` and `rbp` can't be named in an `asm!` (LLVM keeps them for itself);
/// as operands they get a register of the compiler's choice instead.
fn unnamed(full: Register) -> bool {
    matches!(full, Register::RBX | Register::RBP)
}

impl Lifter {
    /// Can `lift_asm` keep `i` as inline assembly? Everything but the flags
    /// it reads, which depend on where it is, is checked here.
    pub(super) fn asm_fits(&mut self, i: &Instruction) -> bool {
        if i.is_invalid() || i.flow_control() != FlowControl::Next || !matches!(i.encoding(), EncodingKind::Legacy | EncodingKind::VEX) {
            return false;
        }
        if flags_of(i.rflags_read()).is_none() {
            return false;
        }
        let mut mem = false;
        for k in 0..i.op_count() {
            match i.op_kind(k) {
                OpKind::Register => {
                    let r = i.op_register(k);
                    if !(r.is_gpr() && r.full_register() != Register::RSP || r.is_xmm()) {
                        return false;
                    }
                }
                OpKind::Memory if !mem && !i.is_vsib() && i.memory_segment() != Register::GS => mem = true,
                k if is_imm(k) => {}
                _ => return false,
            }
        }
        let explicit = |full: Register| (0..i.op_count()).any(|k| i.op_kind(k) == OpKind::Register && i.op_register(k).full_register() == full);
        let addr = |full: Register| mem && (i.memory_base().full_register() == full || i.memory_index().full_register() == full);
        let info = self.info.info(i);
        // memory only through the operand
        if info.used_memory().iter().any(|m| !mem || m.base() != i.memory_base() || m.index() != i.memory_index()) {
            return false;
        }
        info.used_registers().iter().all(|u| {
            let r = u.register();
            if r.is_segment_register() {
                return r != Register::GS;
            }
            if r.is_ip() {
                return mem && i.is_ip_rel_memory_operand();
            }
            if r.is_xmm() {
                return true;
            }
            if !r.is_gpr() {
                return false;
            }
            let full = r.full_register();
            let in_addr = addr(full) && u.access() == OpAccess::Read;
            match full {
                Register::RSP => in_addr,
                _ if unnamed(full) => in_addr || explicit(full),
                _ => true,
            }
        })
    }

    /// Lift `i` as inline assembly (see the module comment).
    pub(super) fn lift_asm(&mut self, f: &mut Function, i: &Instruction) -> Result<(), LiftError> {
        if !self.asm_fits(i) {
            return Err(self.unsupported());
        }
        let mem = (0..i.op_count()).find(|&k| i.op_kind(k) == OpKind::Memory);
        // the registers, operands first
        let mut binds: Vec<Bind> = Vec::new();
        let info = self.info.info(i);
        for k in 0..i.op_count() {
            if i.op_kind(k) == OpKind::Register {
                let r = i.op_register(k);
                add(&mut binds, r, info.op_access(k));
            }
        }
        let addr = |full: Register| mem.is_some() && (i.memory_base().full_register() == full || i.memory_index().full_register() == full);
        for u in info.used_registers() {
            let r = u.register();
            if !(r.is_gpr() || r.is_xmm()) || r.full_register() == Register::RSP {
                continue;
            }
            // a register only used in the address, which is computed outside
            if r.is_gpr() && addr(r.full_register()) && u.access() == OpAccess::Read && !binds.iter().any(|b| b.reg == r.full_register()) {
                continue;
            }
            add(&mut binds, r, u.access());
        }
        let (read, written) = (flags_of(i.rflags_read()).unwrap_or(STATUS), flags_of(i.rflags_modified()).unwrap_or(STATUS));

        // The flags to load into rflags: those `i` reads, and if it writes some,
        // the ones it leaves alone, so that reading them back gets them right.
        let mut flags_in = None;
        let mut valid = 0;
        if read != 0 || written != 0 {
            match self.flags {
                Flags::Raw { rflags, valid: v } => {
                    if read & !v != 0 {
                        return Err(self.unsupported());
                    }
                    (flags_in, valid) = (Some(rflags), v);
                }
                // a block's flags on entry are worked out by its predecessors,
                // which may not all give every one: ask only for what `i` reads
                Flags::Entry | Flags::Unknown if written != 0 && read == 0 => {}
                flags => {
                    let want = if matches!(flags, Flags::Entry | Flags::Unknown) { read } else { read | STATUS & !written };
                    let mut acc = None;
                    for &(_, bit, cc) in &FLAGS {
                        if want & bit == 0 {
                            continue;
                        }
                        let c = match cc.map(|cc| self.condition(f, cc)) {
                            Some(Ok(c)) => c,
                            _ if read & bit != 0 => return Err(self.unsupported()),
                            _ => continue,
                        };
                        let wide = self.emit(f, InstKind::Cast { kind: CastKind::ZExt, v: c }, TyId::B8);
                        let s = self.konst(f, bit.trailing_zeros() as u64, TyId::B1);
                        let v = self.emit(f, InstKind::Bin { op: BinOp::Shl, lhs: wide, rhs: s }, TyId::B8);
                        acc = Some(match acc {
                            Some(a) => self.emit(f, InstKind::Bin { op: BinOp::Or, lhs: a, rhs: v }, TyId::B8),
                            None => v,
                        });
                        valid |= bit;
                    }
                    flags_in = acc;
                }
            }
        }
        let read_flags = read != 0 || flags_in.is_some() && written != 0 && written & STATUS != STATUS;
        let flags_in = flags_in.filter(|_| read_flags);

        // Operands: `{k}` ones (the address, the flags, rbx and rbp) first,
        // then the named registers.
        let mut ops: Vec<AsmOp> = Vec::new();
        let mut args: Vec<ValueId> = Vec::new();
        let mut outs = 0u8;
        let mut out = |n: u8| {
            let k = outs;
            outs += n;
            Some(k)
        };
        let mem_op = match mem {
            Some(_) => {
                let p = self.ea(f, i)?;
                args.push(p);
                ops.push(AsmOp { reg: "reg", named: false, xmm: false, input: Some(0), output: None });
                Some(ops.len() - 1)
            }
            None => None,
        };
        let flags_op = (read_flags || written != 0).then(|| {
            let input = flags_in.map(|v| {
                args.push(v);
                (args.len() - 1) as u8
            });
            ops.push(AsmOp { reg: "reg", named: false, xmm: false, input, output: if written != 0 { out(1) } else { None } });
            ops.len() - 1
        });
        let mut slot = Vec::with_capacity(binds.len());
        for b in binds.iter().filter(|b| unnamed(b.reg)).chain(binds.iter().filter(|b| !unnamed(b.reg))) {
            // a write of 8 or 16 bits keeps the rest of the register
            let input = match (b.read || b.partial, b.reg.is_xmm()) {
                (false, _) => None,
                (true, true) => {
                    let (lo, hi) = self.xmm_get(f, b.reg.number())?;
                    args.extend([lo, hi]);
                    Some((args.len() - 2) as u8)
                }
                (true, false) => {
                    args.push(self.read_full(f, b.reg.number()));
                    Some((args.len() - 1) as u8)
                }
            };
            let (reg, named) = match b.reg {
                r if unnamed(r) && b.high => ("reg_abcd", false),
                r if unnamed(r) => ("reg", false),
                r if r.is_xmm() => (XMM_NAMES[r.number()], true),
                r => (GPR_NAMES[r.number()], true),
            };
            let output = if b.write { out(if b.reg.is_xmm() { 2 } else { 1 }) } else { None };
            ops.push(AsmOp { reg, named, xmm: b.reg.is_xmm(), input, output });
            slot.push((b.reg, ops.len() - 1));
        }

        // The template.
        let mut fmt = IntelFormatter::new();
        let o = fmt.options_mut();
        o.set_hex_prefix("0x");
        o.set_hex_suffix("");
        o.set_uppercase_hex(false);
        o.set_signed_immediate_operands(true);
        o.set_memory_size_options(MemorySizeOptions::Default);
        o.set_space_after_operand_separator(true);
        let mut text = String::new();
        if let (true, Some(k)) = (read_flags, flags_op) {
            text.push_str(&format!("push {{{k}}}\npopfq\n"));
        }
        fmt.format_mnemonic(i, &mut text);
        for op in 0..fmt.operand_count(i) {
            text.push_str(if op == 0 { " " } else { ", " });
            let mut s = String::new();
            fmt.format_operand(i, &mut s, op).map_err(|_| self.unsupported())?;
            match fmt.get_instruction_operand(i, op).ok().flatten() {
                Some(k) if i.op_kind(k) == OpKind::Memory => {
                    // `qword ptr fs:[rax + 8]` is `qword ptr [{0}]`
                    let size = s.split('[').next().unwrap_or("");
                    let size = size.rsplit_once(' ').map_or("", |(sz, _)| sz);
                    // iced spells out a size only where the operands don't
                    // give it; one LLVM doesn't know (`fxsave`'s) goes
                    let size = if SIZES.contains(&size) { size } else { "" };
                    let size = if size.is_empty() { String::new() } else { format!("{size} ") };
                    text.push_str(&format!("{size}[{{{}}}]", mem_op.unwrap()));
                    continue;
                }
                Some(k) if i.op_kind(k) == OpKind::Register && unnamed(i.op_register(k).full_register()) => {
                    let r = i.op_register(k);
                    let at = slot.iter().find(|s| s.0 == r.full_register()).unwrap().1;
                    let m = match r.size() {
                        8 => "",
                        4 => ":e",
                        2 => ":x",
                        _ if is_high_byte(r) => ":h",
                        _ => ":l",
                    };
                    text.push_str(&format!("{{{at}{m}}}"));
                    continue;
                }
                _ => {}
            }
            if s.contains(['{', '}']) {
                return Err(self.unsupported());
            }
            text.push_str(&s);
        }
        if let (true, Some(k)) = (written != 0, flags_op) {
            text.push_str(&format!("\npushfq\npop {{{k}}}"));
        }
        let stack = flags_op.is_some();

        // The IR: the `Opaque`, then each register's new value.
        let a = f.asm.len() as u32;
        f.asm.push(Asm { text, ops: ops.clone(), stack });
        let op = self.emit(f, InstKind::Opaque { asm: a, args: ListRef::EMPTY }, TyId::UNIT);
        self.asm_args.push((op, self.asm_vals.len() as u32, args.len() as u32));
        self.asm_vals.extend_from_slice(&args);
        let res = |l: &mut Self, f: &mut Function, k: u8| l.emit(f, InstKind::AsmOut { asm: op, k }, TyId::B8);
        if let Some(k) = flags_op.and_then(|k| ops[k].output) {
            let rflags = res(self, f, k);
            self.flags = Flags::Raw { rflags, valid: valid & !written | written };
        }
        for &(reg, at) in &slot {
            let Some(k) = ops[at].output else { continue };
            if reg.is_xmm() {
                let (lo, hi) = (res(self, f, k), res(self, f, k + 1));
                self.xmm_set(reg.number(), (lo, hi));
                // a VEX instruction clears the rest of the ymm register
                if self.ymm && i.encoding() == EncodingKind::VEX {
                    let z = self.konst(f, 0, TyId::B8);
                    self.upper = true;
                    self.xmm_set(reg.number(), (z, z));
                    self.upper = false;
                }
            } else {
                let v = res(self, f, k);
                self.state[self.cur].out[reg.number()] = Some(v);
            }
        }
        Ok(())
    }

    /// Condition `cc` of flags read back from rflags (`Flags::Raw`); `valid`
    /// says which of them are known.
    pub(super) fn raw_condition(&mut self, f: &mut Function, cc: ConditionCode, rflags: ValueId, valid: u16) -> Result<ValueId, LiftError> {
        use ConditionCode as C;
        // the bits to test (a set bit means the condition holds), and SF != OF
        let (bits, sf_ne_of, holds) = match cc {
            C::e | C::ne => (ZF, false, cc == C::e),
            C::b | C::ae => (CF, false, cc == C::b),
            C::be | C::a => (CF | ZF, false, cc == C::be),
            C::s | C::ns => (SF, false, cc == C::s),
            C::o | C::no => (OF, false, cc == C::o),
            C::p | C::np => (PF, false, cc == C::p),
            C::l | C::ge => (0, true, cc == C::l),
            C::le | C::g => (ZF, true, cc == C::le),
            C::None => return Err(self.unsupported()),
        };
        let need = bits | if sf_ne_of { SF | OF } else { 0 };
        if need & !valid != 0 {
            return Err(self.unsupported());
        }
        let mut x = rflags;
        if sf_ne_of {
            // SF (bit 7) ^ OF (bit 11), at bit 7
            let four = self.konst(f, 4, TyId::B1);
            let of = self.emit(f, InstKind::Bin { op: BinOp::LShr, lhs: rflags, rhs: four }, TyId::B8);
            x = self.emit(f, InstKind::Bin { op: BinOp::Xor, lhs: rflags, rhs: of }, TyId::B8);
            if bits != 0 {
                // with ZF from rflags itself
                let m = self.konst(f, SF as u64, TyId::B8);
                let sf = self.emit(f, InstKind::Bin { op: BinOp::And, lhs: x, rhs: m }, TyId::B8);
                let zf_m = self.konst(f, ZF as u64, TyId::B8);
                let zf = self.emit(f, InstKind::Bin { op: BinOp::And, lhs: rflags, rhs: zf_m }, TyId::B8);
                x = self.emit(f, InstKind::Bin { op: BinOp::Or, lhs: sf, rhs: zf }, TyId::B8);
            }
        }
        let mask = self.konst(f, (bits | if sf_ne_of { SF } else { 0 }) as u64, TyId::B8);
        let set = self.emit(f, InstKind::Bin { op: BinOp::And, lhs: x, rhs: mask }, TyId::B8);
        let zero = self.konst(f, 0, TyId::B8);
        let cond = if holds { Cond::Ne } else { Cond::Eq };
        Ok(self.emit(f, InstKind::Cmp { cc: cond, lhs: set, rhs: zero }, TyId::BOOL))
    }
}
