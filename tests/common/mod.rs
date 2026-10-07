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
#[allow(dead_code)]
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

/// Random programs built only from instructions the lifter supports, with Jcc/JMP to
/// random instruction boundaries. Deterministic for a given seed.
#[allow(dead_code)]
pub fn random_programs(count: usize, mut seed: u64) -> Vec<Vec<u8>> {
    let mut rnd = |n: u64| {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        seed.wrapping_mul(0x2545_F491_4F6C_DD1D) % n
    };
    let gprs = [rax, rcx, rdx, rbx, rsi, rdi, r8, r9];
    let gpr32 = [eax, ecx, edx, ebx, esi, edi, r8d, r9d];
    (0..count)
        .map(|_| {
            let n = 2 + rnd(24) as usize;
            let mut a = CodeAssembler::new(64).unwrap();
            let mut labels: Vec<CodeLabel> = (0..n).map(|_| a.create_label()).collect();
            for i in 0..n {
                a.set_label(&mut labels[i]).unwrap();
                let (r, s) = (gprs[rnd(8) as usize], gprs[rnd(8) as usize]);
                let to = labels[rnd(n as u64) as usize];
                match rnd(12) {
                    0 => a.mov(r, s).unwrap(),
                    1 => a.mov(r, rnd(10_000)).unwrap(),
                    2 => a.mov(qword_ptr(r + 8), s).unwrap(),
                    3 => a.mov(r, qword_ptr(s + r * 8 - 16)).unwrap(),
                    4 => a.add(r, s).unwrap(),
                    5 => a.sub(r, 1).unwrap(),
                    6 => a.xor(gpr32[rnd(8) as usize], gpr32[rnd(8) as usize]).unwrap(),
                    7 => a.lea(r, qword_ptr(s + 0x20)).unwrap(),
                    8 => { a.cmp(r, s).unwrap(); a.jb(to).unwrap() }
                    9 => { a.test(r, r).unwrap(); a.jne(to).unwrap() }
                    10 => a.jmp(to).unwrap(),
                    _ => a.ret().unwrap(),
                }
            }
            a.ret().unwrap();
            a.assemble(BASE).unwrap()
        })
        .collect()
}
