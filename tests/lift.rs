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
    a.rdtsc().unwrap(); // no data-flow model of it at all
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
    // The call lists rdi (= rsi), rsi, rdx, rcx, r8, r9, rsp, rax, r10, r11, then
    // the xmm halves: the function never touches an xmm register, but a call
    // may take floats it was passed, so they are the entry's (v11..v42); every
    // caller-saved register afterwards, xmm halves too, is a `callout` that
    // `abi::apply` resolves once the callee's signature is known.
    let params: Vec<String> = (11..43).map(|k| format!("v{k}")).collect();
    let xmm = params.join(", ");
    let outs: String = (0..32).map(|k| format!("  v{} = callout v43 r{}\n", 52 + k, 192 + k)).collect();
    let expected = format!("\
bb0(v8, v4, v3, v7, v0, v5, v6, v9, v10, {xmm}):
  v1 = const 0x2000
  v2 = inttoptr v1
  v43 = call v2(v0, v0, v3, v4, v5, v6, v7, v8, v9, v10, {xmm})
  v44 = callout v43 r1
  v45 = callout v43 r2
  v46 = callout v43 r6
  v47 = callout v43 r7
  v48 = callout v43 r8
  v49 = callout v43 r9
  v50 = callout v43 r10
  v51 = callout v43 r11
{outs}  v84 = Add v43, v45
  v85 = Add v84, v44
  ret v85
");
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
fn narrow_sign_extensions_and_prefetch() {
    // gcc -O0 widens a signed char with cbw and a short dividend with cwd
    let out = ir(|a| {
        a.mov(eax, edi).unwrap();
        a.prefetcht0(byte_ptr(rsi)).unwrap(); // a hint: nothing to lift
        a.cbw().unwrap(); // ax = sext(al)
        a.cwd().unwrap(); // dx = the sign of ax
        a.idiv(cx).unwrap();
        a.ret().unwrap();
    });
    assert!(out.contains("SExt"), "{out}");
    assert!(out.contains("AShr") && out.contains("const 0xf"), "{out}");
    assert!(out.contains("SDiv") && out.contains("SRem"), "{out}");
    assert!(!out.contains("Load"), "prefetch reads nothing:\n{out}");
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
fn relative_jump_tables_at_o0() {
    // gcc -O0 computes the offset into the table first, adds it to the table's
    // address, and sign-extends the entry with `cdqe`:
    //   cmp edi, 2 ; ja default ; mov eax, edi ; lea rdx, [rax*4] ; lea rax, [rip+table]
    //   mov eax, [rdx+rax] ; cdqe ; lea rdx, [rip+table] ; add rax, rdx ; jmp rax
    use iced_x86::{BlockEncoderOptions, Code, Instruction, MemoryOperand, Register};
    const TABLE: u64 = 0x3000;
    let lea_table = |r| Instruction::with2(Code::Lea_r64_m, r, MemoryOperand::with_base_displ(Register::RIP, TABLE as i64)).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    let mut cases = [a.create_label(), a.create_label(), a.create_label()];
    let mut default = a.create_label();
    let mut done = a.create_label();
    a.cmp(edi, 2).unwrap();
    a.ja(default).unwrap();
    a.mov(eax, edi).unwrap();
    a.lea(rdx, ptr(rax * 4)).unwrap();
    a.add_instruction(lea_table(Register::RAX)).unwrap();
    a.mov(eax, dword_ptr(rdx + rax)).unwrap();
    a.cdqe().unwrap();
    a.add_instruction(lea_table(Register::RDX)).unwrap();
    a.add(rax, rdx).unwrap();
    a.jmp(rax).unwrap();
    for (k, l) in cases.iter_mut().enumerate() {
        a.set_label(l).unwrap();
        a.lea(eax, dword_ptr(rdi + 10 * (k as i32 + 1))).unwrap();
        a.jmp(done).unwrap();
    }
    a.set_label(&mut default).unwrap();
    a.xor(eax, eax).unwrap();
    a.set_label(&mut done).unwrap();
    a.ret().unwrap();
    let r = a.assemble_options(common::BASE, BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS).unwrap();
    let table: Vec<u8> = cases.iter().flat_map(|l| (r.label_ip(l).unwrap().wrapping_sub(TABLE) as i32).to_le_bytes()).collect();
    let mut f = Function::with_capacity(64, 8);
    Lifter::new().lift_with_data(&r.inner.code_buffer, common::BASE, &[(TABLE, &table)], &mut f).unwrap();
    verify(&f).unwrap();
    let out = dump(&f);
    assert!(out.contains("switch") && !out.contains("tailcall"), "{out}");
}

#[test]
fn relative_jump_table_with_a_store_before_the_jump() {
    // gcc -O2 may schedule a store between the add and the jump (zstd's
    // ZSTD_decodeLiteralsBlock):
    //   lea rsi, [rip+table] ; movsxd rcx, [rsi+rax*4] ; movzx edx, byte [rdx]
    //   add rcx, rsi ; mov [rsp+8], rdx ; jmp rcx
    use iced_x86::{BlockEncoderOptions, Code, Instruction, MemoryOperand, Register};
    const TABLE: u64 = 0x3000;
    let mut a = CodeAssembler::new(64).unwrap();
    let mut cases = [a.create_label(), a.create_label(), a.create_label()];
    let mut default = a.create_label();
    let mut done = a.create_label();
    a.cmp(edi, 2).unwrap();
    a.ja(default).unwrap();
    a.mov(eax, edi).unwrap();
    a.add_instruction(Instruction::with2(Code::Lea_r64_m, Register::RSI, MemoryOperand::with_base_displ(Register::RIP, TABLE as i64)).unwrap()).unwrap();
    a.movsxd(rcx, dword_ptr(rsi + rax * 4)).unwrap();
    a.movzx(edx, byte_ptr(rdx)).unwrap();
    a.add(rcx, rsi).unwrap();
    a.mov(qword_ptr(rsp + 8), rdx).unwrap();
    a.jmp(rcx).unwrap();
    for (k, l) in cases.iter_mut().enumerate() {
        a.set_label(l).unwrap();
        a.lea(eax, dword_ptr(rdi + 10 * (k as i32 + 1))).unwrap();
        a.jmp(done).unwrap();
    }
    a.set_label(&mut default).unwrap();
    a.xor(eax, eax).unwrap();
    a.set_label(&mut done).unwrap();
    a.ret().unwrap();
    let r = a.assemble_options(common::BASE, BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS).unwrap();
    let table: Vec<u8> = cases.iter().flat_map(|l| (r.label_ip(l).unwrap().wrapping_sub(TABLE) as i32).to_le_bytes()).collect();
    let mut f = Function::with_capacity(64, 8);
    Lifter::new().lift_with_data(&r.inner.code_buffer, common::BASE, &[(TABLE, &table)], &mut f).unwrap();
    verify(&f).unwrap();
    let out = dump(&f);
    assert!(out.contains("switch") && !out.contains("tailcall"), "{out}");
}

#[test]
fn indirect_jump_without_a_table_is_a_tail_call() {
    let out = ir(|a| {
        a.mov(rax, qword_ptr(rdi + 0x18)).unwrap();
        a.jmp(rax).unwrap();
    });
    assert!(out.contains("tailcall"), "{out}");
}

#[test]
fn flags_set_in_another_block_become_a_parameter() {
    // cmp ; je ; jl in the fall-through block: one predecessor computes `<` for it
    let out = ir(|a| {
        let mut eq = a.create_label();
        let mut lt = a.create_label();
        a.cmp(rdi, rsi).unwrap();
        a.je(eq).unwrap();
        a.jl(lt).unwrap();
        a.mov(eax, 1).unwrap();
        a.ret().unwrap();
        a.set_label(&mut eq).unwrap();
        a.xor(eax, eax).unwrap();
        a.ret().unwrap();
        a.set_label(&mut lt).unwrap();
        a.mov(rax, -1i64).unwrap();
        a.ret().unwrap();
    });
    assert!(out.starts_with("bb0(v1, v0):\n  v2 = cmp.Eq v0, v1\n  v9 = cmp.Slt v0, v1\n  br v2 bb3() bb1(v9)\nbb1(v3):\n  br v3 bb4() bb2()\n"), "{out}");
    // a loop whose head reads flags from before the loop and from itself
    let out = ir(|a| {
        let mut l = a.create_label();
        a.cmp(rdi, rsi).unwrap();
        a.jmp(l).unwrap();
        a.set_label(&mut l).unwrap();
        a.je(l).unwrap();
        a.ret().unwrap();
    });
    assert!(out.contains("  v4 = cmp.Eq v0, v1\n  jump bb1(v6, v4)\nbb1(v5, v2):\n  br v2 bb1(v5, v2) bb2(v5)\n"), "{out}");
}

#[test]
fn inc_and_dec_give_the_signed_conditions() {
    let out = ir(|a| {
        a.dec(rdi).unwrap();
        a.setg(al).unwrap();
        a.inc(rsi).unwrap();
        a.setl(cl).unwrap();
        a.ret().unwrap();
    });
    // dec: rdi - 1 > 0 is rdi > 1; inc: rsi + 1 < 0 is rsi < -1
    assert!(out.contains("cmp.Sgt v0, v1"), "{out}");
    assert!(out.contains("const 0xffffffffffffffff\n") && out.contains("cmp.Slt"), "{out}");
    // the carry conditions are left alone by inc and dec
    let mut f = Function::with_capacity(16, 2);
    let code = [0x48, 0xFF, 0xC7, 0x0F, 0x92, 0xC0, 0xC3]; // inc rdi ; setb al ; ret
    assert!(Lifter::new().lift(&code, 0x1000, &mut f).is_err());
}

#[test]
fn narrow_mul_div_and_bit_set() {
    let out = ir(|a| {
        a.mov(eax, edi).unwrap();
        a.div(sil).unwrap(); // al, ah = ax / sil, ax % sil
        a.mul(cx).unwrap(); // dx:ax = ax * cx
        a.bts(r8, r9).unwrap();
        a.pause().unwrap();
        a.ret().unwrap();
    });
    assert!(out.contains("UDiv") && out.contains("URem"), "{out}");
    assert!(out.contains("Mul"), "{out}");
    assert!(out.contains("Shl") && out.contains("Or"), "{out}");
}

#[test]
fn thread_locals_are_addresses_below_the_thread_pointer() {
    // mov rax, fs:[0] ; mov ecx, fs:[-8] ; add eax, ecx ; ret
    let code = [0x64, 0x48, 0x8B, 0x04, 0x25, 0, 0, 0, 0, 0x64, 0x8B, 0x0C, 0x25, 0xF8, 0xFF, 0xFF, 0xFF, 0x01, 0xC8, 0xC3];
    let mut f = Function::with_capacity(16, 2);
    // without a thread-local block, unsupported
    assert!(Lifter::new().lift(&code, 0x1000, &mut f).is_err());
    let mut l = Lifter::new();
    l.thread_pointer = Some(0x8000);
    l.lift(&code, 0x1000, &mut f).unwrap();
    verify(&f).unwrap();
    let out = dump(&f);
    assert!(out.contains("const 0x8000\n") && out.contains("const 0x7ff8\n"), "{out}");
}

#[test]
fn xmm_values_cross_blocks_as_block_params() {
    let out = ir(|a| {
        let mut skip = a.create_label();
        a.movdqu(xmm0, xmmword_ptr(rsi)).unwrap();
        a.test(edx, edx).unwrap();
        a.je(skip).unwrap();
        a.paddd(xmm0, xmm0).unwrap();
        a.set_label(&mut skip).unwrap();
        a.movdqu(xmmword_ptr(rdi), xmm0).unwrap();
        a.ret().unwrap();
    });
    // the join block takes both halves of xmm0: the loads, or the sums
    assert!(out.contains("Lane(Add, 4)"), "{out}");
    assert!(out.contains("jump bb2(v23, v24, v11, v12)"), "{out}");
    assert!(out.contains("bb2(v21, v22, v2, v4) bb1("), "{out}");
}

#[test]
fn xmm_arguments_and_call_results_are_values() {
    // an argument in xmm0 is an entry param; a call defines all xmm registers
    let out = ir(|a| {
        a.movq(rax, xmm0).unwrap();
        a.call(0x2000).unwrap();
        a.movq(rdx, xmm0).unwrap();
        a.add(rax, rdx).unwrap();
        a.ret().unwrap();
    });
    // xmm0's halves are entry params v0, v1, passed on to the call ...
    assert!(out.contains("v12, v0, v1, v13"), "{out}");
    // ... and its result is read back through a callout
    assert!(out.contains("v52 = callout v43 r192"), "{out}");
    assert!(out.contains("Add v43, v52"), "{out}");
}

#[test]
fn scalar_conversion_leaves_the_rest_undefined_not_an_argument() {
    // cvtsi2sd keeps xmm0's upper half, which is garbage at entry
    let out = ir(|a| {
        a.cvtsi2sd(xmm0, edi).unwrap();
        a.cvttsd2si(rax, xmm0).unwrap();
        a.ret().unwrap();
    });
    assert!(out.contains("undef"), "{out}");
}

#[test]
fn ucomisd_flags_become_float_compares() {
    let out = ir(|a| {
        a.movq(xmm0, rdi).unwrap();
        a.movq(xmm1, rsi).unwrap();
        a.ucomisd(xmm0, xmm1).unwrap();
        a.seta(al).unwrap();
        a.setp(cl).unwrap();
        a.or(al, cl).unwrap();
        a.ret().unwrap();
    });
    assert!(out.contains("Lane(FCmpGt, 8)") && out.contains("Lane(FCmpUnord, 8)"), "{out}");
}

#[test]
fn rep_stos_is_a_fill() {
    let out = ir(|a| {
        a.xor(eax, eax).unwrap();
        a.mov(ecx, 6).unwrap();
        a.rep().stosq().unwrap();
        a.mov(rax, rdi).unwrap();
        a.ret().unwrap();
    });
    assert!(out.contains("MemFill") || out.contains("fill"), "{out}");
}

#[test]
fn string_cmpsd_and_movsd_are_not_the_sse_ones() {
    for code in [[0xA7u8, 0xC3], [0xA5, 0xC3]] {
        // cmpsd / movsd dword [rsi], [rdi] ; ret
        let mut f = Function::with_capacity(8, 2);
        let err = Lifter::new().lift(&code, 0, &mut f).unwrap_err();
        assert!(matches!(err, LiftError::Unsupported { ip: 0, .. }), "{err:?}");
    }
}

#[test]
fn cpuid_and_xgetbv_ask_the_cpu() {
    let out = ir(|a| {
        a.cpuid().unwrap();
        a.add(eax, ebx).unwrap();
        a.add(ecx, edx).unwrap();
        a.xgetbv().unwrap();
        a.ret().unwrap();
    });
    for op in ["Lane(Cpuid, 0)", "Lane(Cpuid, 1)", "Lane(Cpuid, 2)", "Lane(Cpuid, 3)", "Lane(Xgetbv, 0)", "Lane(Xgetbv, 3)"] {
        assert!(out.contains(op), "{op} missing:\n{out}");
    }
}

#[test]
fn conditional_branch_out_of_the_function_is_a_tail_call() {
    // test rdi, rdi ; jne <f.cold at 0x5000> ; mov eax, 1 ; ret
    let out = ir(|a| {
        a.test(rdi, rdi).unwrap();
        a.jne(0x5000).unwrap();
        a.mov(eax, 1).unwrap();
        a.ret().unwrap();
    });
    let tail = out.lines().position(|l| l.contains("tailcall")).unwrap_or_else(|| panic!("no tail call:\n{out}"));
    let lines: Vec<&str> = out.lines().collect();
    assert!(lines[..tail].iter().any(|l| l.contains("const 0x5000")), "{out}");
    assert!(out.contains("ret"), "{out}");
}

#[test]
fn unknown_instructions_become_inline_assembly() {
    let lift = |code: &[u8]| {
        let mut f = Function::with_capacity(16, 2);
        let mut l = Lifter::new();
        l.asm = true;
        l.lift(code, 0, &mut f).map(|()| f)
    };
    // rdtsc ; ret: an asm! that writes rax and rdx, and no argument
    let f = lift(&[0x0F, 0x31, 0xC3]).unwrap();
    assert_eq!(f.asm.len(), 1);
    assert_eq!(f.asm[0].text, "rdtsc");
    let named: Vec<_> = f.asm[0].ops.iter().map(|o| (o.reg, o.input.is_some(), o.output.is_some())).collect();
    assert_eq!(named, [("rax", false, true), ("rdx", false, true)]);
    // mov rax, rsi ; mov rcx, rdx ; mov rdx, rdi ; div rcx ; ret: a 128-bit
    // dividend, which the lifter's own `div` doesn't do, kept as it is
    let f = lift(&[0x48, 0x89, 0xF0, 0x48, 0x89, 0xD1, 0x48, 0x89, 0xFA, 0x48, 0xF7, 0xF1, 0xC3]).unwrap();
    assert_eq!(f.asm.len(), 1);
    assert!(f.asm[0].text.starts_with("div rcx\n"), "{}", f.asm[0].text);
    // enter moves rsp: still unsupported
    let err = lift(&[0xC8, 0x10, 0x00, 0x00, 0xC3]).unwrap_err();
    assert!(matches!(err, LiftError::Unsupported { ip: 0, .. }), "{err:?}");
    // without `asm`, as before
    let mut f = Function::with_capacity(8, 2);
    assert!(matches!(Lifter::new().lift(&[0x0F, 0x31, 0xC3], 0, &mut f), Err(LiftError::Unsupported { ip: 0, .. })));
}

#[test]
fn flags_from_inline_assembly() {
    // cmp rdi, rsi ; rcl rax, 1 ; jb L ; ret ; L: ret: rcl reads CF from the
    // cmp, and the jb reads the CF rcl leaves, from rflags
    let code = [0x48, 0x39, 0xF7, 0x48, 0xD1, 0xD0, 0x72, 0x01, 0xC3, 0xC3];
    let mut f = Function::with_capacity(32, 4);
    let mut l = Lifter::new();
    l.asm = true;
    l.lift(&code, 0, &mut f).unwrap();
    assert_eq!(f.asm[0].text, "push {0}\npopfq\nrcl rax, 1\npushfq\npop {0}");
    let ir = dump(&f);
    assert!(ir.contains("And v") && ir.contains("const 0x1\n"), "{ir}");
    // jb right after the entry reads flags nothing set
    let code = [0xF9, 0x72, 0x01, 0xC3, 0xC3]; // stc ; jb L ; ret ; L: ret
    l.lift(&code, 0, &mut f).unwrap();
    assert_eq!(f.asm[0].text, "stc\npushfq\npop {0}");
}
