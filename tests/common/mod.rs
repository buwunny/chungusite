use iced_x86::code_asm::*;

pub const BASE: u64 = 0x1000;

/// sum(p, n): writes p[1] = rcx, then sums n qwords starting at p+16.
///
///     mov  rax, rdi
///     mov  [rax+8], rcx
///     xor  edx, edx
///   top:
///     cmp  rdx, rsi
///     jae  done
///     mov  r8, [rdi+rdx*8+16]
///     add  rax, r8
///     add  rdx, 1
///     jmp  top
///   done:
///     ret
pub fn sum_loop() -> Vec<u8> {
    let mut a = CodeAssembler::new(64).unwrap();
    let mut top = a.create_label();
    let mut done = a.create_label();
    a.mov(rax, rdi).unwrap();
    a.mov(qword_ptr(rax + 8), rcx).unwrap();
    a.xor(edx, edx).unwrap();
    a.set_label(&mut top).unwrap();
    a.cmp(rdx, rsi).unwrap();
    a.jae(done).unwrap();
    a.mov(r8, qword_ptr(rdi + rdx * 8 + 16)).unwrap();
    a.add(rax, r8).unwrap();
    a.add(rdx, 1).unwrap();
    a.jmp(top).unwrap();
    a.set_label(&mut done).unwrap();
    a.ret().unwrap();
    a.assemble(BASE).unwrap()
}
