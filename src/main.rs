//! Driver: decode a few mock functions, lift them, and print the IR.
//!
//!     cargo run                         # built-in samples
//!     cargo run -- "b8 01 00 00 00 c3"  # your own bytes, hex
use chungusite::{dump::dump, ir::*, lift::Lifter, verify::verify};
use iced_x86::{Decoder, DecoderOptions, Instruction, Register};

const BASE: u64 = 0x1000;

/// The SSA spike: `mov eax, 1; add eax, 2`.
const SPIKE: &[u8] = &[
    0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1
    0x83, 0xC0, 0x02,             // add eax, 2
];

/// A small CFG: `if p == null { 0 } else { p[1] = v; p[2] }`.
const CFG: &[u8] = &[
    0x48, 0x85, 0xFF,       // 1000: test rdi, rdi
    0x74, 0x09,             // 1003: je   100e
    0x48, 0x89, 0x77, 0x08, // 1005: mov  [rdi+8], rsi
    0x48, 0x8B, 0x47, 0x10, // 1009: mov  rax, [rdi+16]
    0xC3,                   // 100d: ret
    0x31, 0xC0,             // 100e: xor  eax, eax
    0xC3,                   // 1010: ret
];

fn main() {
    let arg = std::env::args().nth(1);
    let user_bytes = arg.as_deref().map(parse_hex);
    let samples: Vec<(&str, &[u8])> = match &user_bytes {
        Some(b) => vec![("input", b.as_slice())],
        None => vec![("spike: mov eax, 1; add eax, 2", SPIKE), ("cfg: null check", CFG)],
    };

    // Reused across samples: after the first, lifting allocates nothing.
    let mut lifter = Lifter::new();
    let mut func = Function::with_capacity(256, 16);

    for (name, bytes) in samples {
        println!("==== {name} ====");

        // The raw decode, so you can see what the lifter sees.
        let mut dec = Decoder::with_ip(64, bytes, BASE, DecoderOptions::NONE);
        let mut insn = Instruction::default();
        while dec.can_decode() {
            dec.decode_out(&mut insn);
            println!("  {:#x}  {:?}", insn.ip(), insn.code());
        }

        match lifter.lift(bytes, BASE, &mut func) {
            Ok(()) => {
                if let Err(e) = verify(&func) {
                    println!("IR failed verification: {e:?}");
                }
                println!("\n{}", dump(&func));
                let ir_blocks = &func.blocks;
                println!("{ir_blocks:#?}");
                println!("register file at each block exit:");
                for (b, _) in func.blocks.iter() {
                    let regs: Vec<String> = GPRS
                        .iter()
                        .filter_map(|&r| lifter.reg_out(b, r).map(|v| format!("{r:?}=v{}", v.index())))
                        .collect();
                    println!("  bb{}: {}", b.index(), regs.join(" "));
                }
            }
            Err(e) => println!("lift failed: {e:?}"),
        }
        println!();
    }
}

const GPRS: [Register; 16] = [
    Register::RAX, Register::RCX, Register::RDX, Register::RBX, Register::RSP, Register::RBP,
    Register::RSI, Register::RDI, Register::R8, Register::R9, Register::R10, Register::R11,
    Register::R12, Register::R13, Register::R14, Register::R15,
];

fn parse_hex(s: &str) -> Vec<u8> {
    let digits: Vec<u8> = s.bytes().filter(u8::is_ascii_hexdigit).collect();
    digits
        .chunks(2)
        .map(|c| u8::from_str_radix(std::str::from_utf8(c).unwrap(), 16).expect("hex byte"))
        .collect()
}
