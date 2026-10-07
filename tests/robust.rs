//! Branch targets and arbitrary bytes must never panic the lifter, and anything it
//! accepts must pass the verifier (no dangling ids, no edge dropping a value).
use chungusite::{borrow::analyze, ir::*, lift::{LiftError, Lifter}, opt::clean, verify::verify};
use iced_x86::code_asm::*;

fn lift(code: &[u8]) -> (Result<(), LiftError>, Function) {
    let mut f = Function::with_capacity(64, 8);
    let r = Lifter::new().lift(code, 0x1000, &mut f);
    if r.is_ok() {
        verify(&f).unwrap_or_else(|e| panic!("{e:?} for {code:02x?}"));
    }
    (r, f)
}

#[test]
fn branch_into_middle_of_instruction_is_an_error() {
    // 1000: jmp 1003 ; 1002: mov eax, 1 (1002..1007) ; 1007: ret
    let code = [0xEB, 0x01, 0xB8, 0x01, 0x00, 0x00, 0x00, 0xC3];
    assert_eq!(lift(&code).0, Err(LiftError::TargetInsideInstruction { target: 0x1003 }));
}

#[test]
fn conditional_branch_out_of_range_is_an_error() {
    // test rdi, rdi ; je +0x7f (past the end)
    let code = [0x48, 0x85, 0xFF, 0x74, 0x7F, 0xC3];
    assert_eq!(lift(&code).0, Err(LiftError::BranchOutOfRange { ip: 0x1003, target: 0x1084 }));
}

#[test]
fn fallthrough_past_end_is_an_error() {
    // test rdi, rdi ; je 1000 ; (no more bytes)
    let code = [0x48, 0x85, 0xFF, 0x74, 0xFB];
    assert_eq!(lift(&code).0, Err(LiftError::BranchOutOfRange { ip: 0x1003, target: 0x1005 }));
}

#[test]
fn unconditional_jump_out_of_range_is_a_tail_call() {
    let code = [0xE9, 0x00, 0x10, 0x00, 0x00]; // jmp 0x2005
    let (r, f) = lift(&code);
    r.unwrap();
    assert!(matches!(f.blocks[BlockId::from_u32(0)].term, Terminator::TailCall { .. }));
}

#[test]
fn flags_from_another_block_are_an_error() {
    // cmp rdi, rsi ; jmp L ; L: je L ; ret
    let code = [0x48, 0x39, 0xF7, 0xEB, 0x00, 0x74, 0xFE, 0xC3];
    assert_eq!(lift(&code).0, Err(LiftError::FlagsNotInBlock { ip: 0x1005 }));
}

#[test]
fn truncated_instruction_does_not_panic() {
    let code = [0x48, 0x8B]; // first two bytes of mov r64, r/m64
    assert!(matches!(lift(&code).0, Err(LiftError::Unsupported { .. })));
}

/// xorshift64*, so the test needs no extra crate and is reproducible.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 { self.next() % n }
}

#[test]
fn random_bytes_never_panic() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut buf = [0u8; 64];
    for _ in 0..20_000 {
        let len = 1 + rng.below(64) as usize;
        for b in &mut buf[..len] {
            *b = rng.next() as u8;
        }
        let _ = lift(&buf[..len]);
    }
}

/// Random programs built only from instructions the lifter supports, with Jcc/JMP to
/// random instruction boundaries, so most of them lift and exercise the SSA wiring.
#[test]
fn random_supported_programs_verify() {
    let gprs = [rax, rcx, rdx, rbx, rsi, rdi, r8, r9];
    let gpr32 = [eax, ecx, edx, ebx, esi, edi, r8d, r9d];
    let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
    let mut ok = 0;
    for _ in 0..2_000 {
        let n = 2 + rng.below(20) as usize;
        let mut a = CodeAssembler::new(64).unwrap();
        let mut labels: Vec<CodeLabel> = (0..n).map(|_| a.create_label()).collect();
        for i in 0..n {
            a.set_label(&mut labels[i]).unwrap();
            let r = gprs[rng.below(8) as usize];
            let s = gprs[rng.below(8) as usize];
            let to = labels[rng.below(n as u64) as usize];
            let (r32, s32) = (gpr32[rng.below(8) as usize], gpr32[rng.below(8) as usize]);
            match rng.below(26) {
                12 => a.call(0x3000).unwrap(),
                13 => a.push(r).unwrap(),
                14 => a.pop(r).unwrap(),
                15 => { a.cmp(r, s).unwrap(); a.cmovae(r, qword_ptr(s + 8)).unwrap() }
                16 => { a.test(r32, s32).unwrap(); a.setg(byte_ptr(r)).unwrap() }
                17 => a.sub(dword_ptr(r + 4), s32).unwrap(),
                18 => { a.add(r, s).unwrap(); a.jo(to).unwrap() }
                19 => a.movsx(r32, byte_ptr(s)).unwrap(),
                20 => a.sar(r, cl).unwrap(),
                21 => a.neg(qword_ptr(s)).unwrap(),
                22 => a.mov(ah, bl).unwrap(),
                23 => { a.xor(edx, edx).unwrap(); a.div(r).unwrap() }
                24 => a.leave().unwrap(),
                25 => a.lea(r32, qword_ptr(s + r * 4)).unwrap(),
                0 => a.mov(r, s).unwrap(),
                1 => a.mov(r, rng.next()).unwrap(),
                2 => a.mov(qword_ptr(r + 8), s).unwrap(),
                3 => a.mov(r, qword_ptr(s + r * 8 + 16)).unwrap(),
                4 => a.add(r, s).unwrap(),
                5 => a.sub(r, 1).unwrap(),
                6 => a.xor(gpr32[rng.below(8) as usize], gpr32[rng.below(8) as usize]).unwrap(),
                7 => a.lea(r, qword_ptr(s + 0x20)).unwrap(),
                8 => { a.cmp(r, s).unwrap(); a.jb(to).unwrap() }
                9 => { a.test(r, r).unwrap(); a.jne(to).unwrap() }
                10 => a.jmp(to).unwrap(),
                _ => a.ret().unwrap(),
            }
        }
        a.ret().unwrap();
        let code = a.assemble(0x1000).unwrap();
        let (r, mut f) = lift(&code);
        if r.is_ok() {
            ok += 1;
            // Cleanup must keep the function well formed, and analysis must not panic.
            clean(&mut f);
            verify(&f).unwrap_or_else(|e| panic!("after clean: {e:?} for {code:02x?}"));
            analyze(&f);
        }
    }
    // Make sure the generator really exercises the success path.
    println!("{ok} of 2000 programs lifted");
    assert!(ok > 800, "only {ok} of 2000 programs lifted");
}
