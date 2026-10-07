mod common;
use chungusite::{dump::dump, ir::{BlockId, Function}, lift::{LiftError, Lifter}};
use iced_x86::code_asm::*;

#[test]
fn lifts_loop_with_block_params() {
    let code = common::sum_loop();
    let mut f = Function::with_capacity(64, 8);
    Lifter::new().lift(&code, common::BASE, &mut f).unwrap();
    // Registers by x86 number: rax=0, rcx=1, rdx=2, rsi=6, rdi=7. Block params and
    // edge args are listed in that order. v3 is the store, which the dump prints unnamed.
    let expected = "\
bb0(v1, v18, v0):
  v2 = ptr v0 + 8
  store v2 <- v1
  v4 = const 0x0
  v5 = ZExt v4
  jump bb1(v0, v5, v18, v0)
bb1(v19, v6, v7, v20):
  v8 = cmp.Uge v6, v7
  br v8 bb3(v19) bb2(v19, v6, v7, v20)
bb2(v13, v10, v21, v9):
  v11 = ptr v9 + v10*8 + 16
  v12 = load v11
  v14 = Add v13, v12
  v15 = const 0x1
  v16 = Add v10, v15
  jump bb1(v14, v16, v21, v9)
bb3(v17):
  ret v17
";
    assert_eq!(dump(&f), expected);
}

#[test]
fn reports_unsupported_instead_of_guessing() {
    let mut a = CodeAssembler::new(64).unwrap();
    a.mov(al, 1).unwrap(); // 8-bit write needs a merge, not handled yet
    a.ret().unwrap();
    let code = a.assemble(0).unwrap();
    let mut f = Function::with_capacity(8, 2);
    let err = Lifter::new().lift(&code, 0, &mut f).unwrap_err();
    assert!(matches!(err, LiftError::Unsupported { ip: 0, .. }), "{err:?}");
}

#[test]
fn endbr64_is_a_nop() {
    // endbr64 ; mov eax, 1 ; ret (what gcc emits by default with CET on)
    let code = [0xF3, 0x0F, 0x1E, 0xFA, 0xB8, 0x01, 0x00, 0x00, 0x00, 0xC3];
    let mut f = Function::with_capacity(16, 2);
    Lifter::new().lift(&code, 0x1000, &mut f).unwrap();
    assert_eq!(f.blocks[BlockId::from_u32(0)].insts.len, 2, "const + zext, nothing for endbr64");
}
