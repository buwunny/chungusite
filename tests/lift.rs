mod common;
use chungusite::{dump::dump, ir::{BlockId, Function}, lift::{LiftError, Lifter}, verify::verify};
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
    a.cpuid().unwrap(); // no data-flow model of it at all
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

/// Lift `build`'s code at 0x1000, check it verifies, and return the IR dump.
fn ir(build: impl FnOnce(&mut CodeAssembler)) -> String {
    let mut a = CodeAssembler::new(64).unwrap();
    build(&mut a);
    let code = a.assemble(0x1000).unwrap();
    let mut f = Function::with_capacity(64, 8);
    Lifter::new().lift(&code, 0x1000, &mut f).unwrap();
    verify(&f).unwrap();
    dump(&f)
}

#[test]
fn call_passes_registers_and_leaves_placeholders_for_the_rest() {
    let out = ir(|a| {
        a.mov(rdi, rsi).unwrap();
        a.call(0x2000).unwrap();
        a.add(rax, rdx).unwrap(); // rdx: the high half of a 16-byte result, or preserved
        a.add(rax, rcx).unwrap(); // rcx: clobbered, unless the callee preserves it
        a.ret().unwrap();
    });
    // The call lists rdi (= rsi), rsi, rdx, rcx, r8, r9, rsp, rax, r10, r11; every
    // caller-saved register afterwards is a `callout` that `abi::apply` resolves
    // once the callee's signature is known.
    let expected = "\
bb0(v8, v4, v3, v7, v0, v5, v6, v9, v10):
  v1 = const 0x2000
  v2 = inttoptr v1
  v11 = call v2(v0, v0, v3, v4, v5, v6, v7, v8, v9, v10)
  v12 = callout v11 r1
  v13 = callout v11 r2
  v14 = callout v11 r6
  v15 = callout v11 r7
  v16 = callout v11 r8
  v17 = callout v11 r9
  v18 = callout v11 r10
  v19 = callout v11 r11
  v20 = Add v11, v13
  v21 = Add v20, v12
  ret v21
";
    assert_eq!(out, expected);
}

#[test]
fn push_and_pop_move_rsp() {
    let out = ir(|a| {
        a.push(rbx).unwrap();
        a.mov(rbx, rdi).unwrap();
        a.pop(rbx).unwrap();
        a.ret().unwrap();
    });
    let expected = "\
bb0(v7, v0, v1, v4):
  v2 = ptr v1 + -8
  store v2 <- v0
  v5 = load v2
  v6 = ptr v2 + 8
  ret v7
";
    assert_eq!(out, expected);
}

#[test]
fn setcc_and_cmov_read_the_flags() {
    let out = ir(|a| {
        a.cmp(rdi, rsi).unwrap();
        a.setl(al).unwrap();
        a.cmovb(rdi, rsi).unwrap();
        a.ret().unwrap();
    });
    // setl merges into rax: (rax & !0xff) | ZExt(cond)
    let expected = "\
bb0(v4, v1, v0):
  v2 = cmp.Slt v0, v1
  v3 = ZExt v2
  v5 = ZExt v3
  v6 = const 0xffffffffffffff00
  v7 = And v4, v6
  v8 = Or v7, v5
  v9 = cmp.Ult v0, v1
  v10 = select v9, v1, v0
  ret v8
";
    assert_eq!(out, expected);
}

#[test]
fn memory_operands_load_and_store() {
    let out = ir(|a| {
        a.add(dword_ptr(rdi + 4), esi).unwrap();
        a.cmp(byte_ptr(rdi), 0).unwrap();
        a.movzx(eax, byte_ptr(rdi + 1)).unwrap();
        a.ret().unwrap();
    });
    let expected = "\
bb0(v3, v0):
  v1 = ptr v0 + 4
  v2 = load v1
  v4 = Trunc v3
  v5 = Add v2, v4
  store v1 <- v5
  v7 = ptr v0 + 0
  v8 = load v7
  v9 = const 0x0
  v10 = ptr v0 + 1
  v11 = load v10
  v12 = ZExt v11
  v13 = ZExt v12
  ret v13
";
    assert_eq!(out, expected);
}

#[test]
fn shifts_divides_and_sign_extension() {
    let out = ir(|a| {
        a.mov(rax, rdi).unwrap();
        a.cqo().unwrap();
        a.idiv(rsi).unwrap();
        a.shl(rax, 3).unwrap();
        a.sar(rdx, cl).unwrap();
        a.movsxd(rcx, edx).unwrap();
        a.add(rax, rcx).unwrap();
        a.ret().unwrap();
    });
    assert!(out.contains("AShr v0, v1") && out.contains("SDiv v0, v3") && out.contains("SRem v0, v3"), "{out}");
    assert!(out.contains("Shl v4, v6"), "{out}");
    assert!(out.contains("SExt"), "{out}");
}

#[test]
fn byte_register_write_reads_back_without_a_mask() {
    let out = ir(|a| {
        a.mov(al, 1).unwrap();
        a.mov(cl, al).unwrap(); // reads the byte straight back
        a.mov(ah, cl).unwrap();
        a.ret().unwrap();
    });
    assert!(!out.contains("Trunc"), "al and cl read back as the written byte:\n{out}");
    assert!(out.contains("const 0xffffffffffff00ff"), "ah merges into bits 8..16:\n{out}");
}

#[test]
fn traps_end_the_block() {
    let out = ir(|a| {
        a.ud2().unwrap();
    });
    assert_eq!(out, "bb0():\n  Unreachable\n");
}

#[test]
fn one_operand_mul_writes_both_halves() {
    let out = ir(|a| {
        a.mov(rax, rdi).unwrap();
        a.mul(rsi).unwrap();
        a.seto(cl).unwrap(); // CF = OF = the high half isn't zero
        a.add(rax, rdx).unwrap();
        a.ret().unwrap();
    });
    assert!(out.contains("v2 = Mul v0, v1\n  v3 = UMulHi v0, v1\n"), "{out}");
    assert!(out.contains("cmp.Ne"), "{out}");
    assert!(out.contains("Add v2, v3"), "{out}");
}

#[test]
fn sixteen_byte_copies_and_zeroing() {
    let out = ir(|a| {
        a.movups(xmm0, xmmword_ptr(rsi)).unwrap();
        a.movups(xmmword_ptr(rdi), xmm0).unwrap();
        a.xorps(xmm1, xmm1).unwrap();
        a.movaps(xmmword_ptr(rdi + 16), xmm1).unwrap();
        a.ret().unwrap();
    });
    let expected = "\
bb0(v15, v0, v5):
  v1 = ptr v0 + 0
  v2 = load v1
  v3 = ptr v1 + 8
  v4 = load v3
  v6 = ptr v5 + 0
  v7 = ptr v6 + 8
  store v6 <- v2
  store v7 <- v4
  v10 = const 0x0
  v11 = ptr v5 + 16
  v12 = ptr v11 + 8
  store v11 <- v10
  store v12 <- v10
  ret v15
";
    assert_eq!(out, expected);
}

#[test]
fn jump_tables_become_switches() {
    let (code, table) = common::jump_table(0x3000);
    let mut f = Function::with_capacity(64, 8);
    Lifter::new().lift_with_data(&code, common::BASE, &[(0x3000, &table)], &mut f).unwrap();
    verify(&f).unwrap();
    let out = dump(&f);
    // one parameterless block per case target (bb7..bb9) passes the registers on
    assert!(out.contains("switch v6 [bb7, bb8, bb9] default bb9"), "{out}");
    assert!(out.contains("bb7():\n  jump bb2(v4)"), "{out}");

    // Without the table the jump can't be followed: unsupported, not a tail call.
    let err = Lifter::new().lift(&code, common::BASE, &mut f).unwrap_err();
    assert!(matches!(err, LiftError::Unsupported { mnemonic: iced_x86::Mnemonic::Jmp, .. }), "{err:?}");
}

#[test]
fn indirect_jump_without_a_table_is_a_tail_call() {
    let out = ir(|a| {
        a.mov(rax, qword_ptr(rdi + 0x18)).unwrap();
        a.jmp(rax).unwrap();
    });
    assert!(out.contains("tailcall"), "{out}");
}
