//! SSA spike: `mov eax, 1; add eax, 2` must lift to one Add of two 32-bit constants,
//! zero-extended into RAX, with no value lost between the two instructions.
use chungusite::{dump::dump, ir::*, lift::Lifter, verify::verify};
use iced_x86::Register;

const SPIKE: &[u8] = &[0xB8, 0x01, 0x00, 0x00, 0x00, 0x83, 0xC0, 0x02];

#[test]
fn mov_add_spike() {
    let mut lifter = Lifter::new();
    let mut f = Function::with_capacity(16, 2);
    lifter.lift(SPIKE, 0x1000, &mut f).unwrap();
    verify(&f).unwrap();
    println!("{:#?}", f.blocks);

    assert_eq!(
        dump(&f),
        "\
bb0():
  v0 = const 0x1
  v1 = ZExt v0
  v2 = const 0x2
  v3 = Add v0, v2
  v4 = ZExt v3
  Unreachable
"
    );

    // RAX holds ZExt(Add(1, 2)), all 32-bit until the final zero-extension.
    let rax = lifter.reg_out(BlockId::from_u32(0), Register::RAX).unwrap();
    let InstKind::Cast { kind: CastKind::ZExt, v: sum } = f.insts[rax].kind else { panic!("{:?}", f.insts[rax]) };
    assert_eq!(f.insts[rax].ty, TyId::B8);
    let InstKind::Bin { op: BinOp::Add, lhs, rhs } = f.insts[sum].kind else { panic!("{:?}", f.insts[sum]) };
    assert_eq!(f.insts[sum].ty, TyId::B4);
    let konst = |v: ValueId| match f.insts[v].kind {
        InstKind::Const(c) => (f.consts[c.index()], f.insts[v].ty),
        k => panic!("{k:?}"),
    };
    assert_eq!(konst(lhs), (1, TyId::B4));
    assert_eq!(konst(rhs), (2, TyId::B4));

    // EAX and AX/AL are views of the same physical register.
    for r in [Register::EAX, Register::AX, Register::AL] {
        assert_eq!(lifter.reg_out(BlockId::from_u32(0), r), Some(rax));
    }
    // Registers the code never touched have no value, rather than a stale one.
    assert_eq!(lifter.reg_out(BlockId::from_u32(0), Register::RCX), None);
    assert_eq!(lifter.reg_out(BlockId::from_u32(0), Register::XMM0), None);
}

#[test]
fn every_gpr_maps_to_its_own_slot() {
    use iced_x86::code_asm::*;
    // mov r, imm for all 16 GPRs, each with a distinct value; then check each slot.
    let regs = [rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi, r8, r9, r10, r11, r12, r13, r14, r15];
    let mut a = CodeAssembler::new(64).unwrap();
    for (i, &r) in regs.iter().enumerate() {
        a.mov(r, 0x100 + i as u64).unwrap();
    }
    let code = a.assemble(0).unwrap();
    let mut lifter = Lifter::new();
    let mut f = Function::with_capacity(32, 2);
    lifter.lift(&code, 0, &mut f).unwrap();
    verify(&f).unwrap();

    let iced = [
        Register::RAX, Register::RCX, Register::RDX, Register::RBX, Register::RSP, Register::RBP,
        Register::RSI, Register::RDI, Register::R8, Register::R9, Register::R10, Register::R11,
        Register::R12, Register::R13, Register::R14, Register::R15,
    ];
    for (i, &r) in iced.iter().enumerate() {
        let v = lifter.reg_out(BlockId::from_u32(0), r).unwrap();
        let InstKind::Const(c) = f.insts[v].kind else { panic!("{r:?}: {:?}", f.insts[v]) };
        assert_eq!(f.consts[c.index()], 0x100 + i as u128, "{r:?}");
    }
}
