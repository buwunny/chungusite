//! The Rust emitter and the CLI, checked by compiling (and running) what they print.
//! These tests shell out to `rustc`, the same one cargo uses.
mod common;
use chungusite::{
    emit::{emit_function, Mode},
    ir::*,
    lift::Lifter,
    opt::clean,
    verify::verify,
};
use iced_x86::code_asm::*;
use std::path::{Path, PathBuf};
use std::process::Command;

fn emit(code: &[u8], name: &str, mode: Mode) -> String {
    let mut f = Function::with_capacity(64, 8);
    Lifter::new().lift(code, common::BASE, &mut f).unwrap();
    clean(&mut f);
    verify(&f).unwrap();
    let mut out = String::new();
    emit_function(&f, name, mode, &|_| None, &mut out);
    out
}

const PRELUDE: &str = "#![allow(unused_mut, unused_variables, unused_assignments, unreachable_code, non_snake_case, unused_parens, unused_unsafe)]\n";

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("chungusite-test-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Compile `src` with rustc; panics with the compiler's errors on failure.
fn rustc(dir: &Path, file: &str, src: &str, args: &[&str]) {
    let path = dir.join(file);
    std::fs::write(&path, src).unwrap();
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let out = Command::new(rustc)
        .args(["--edition", "2021", "-A", "warnings", "--out-dir"])
        .arg(dir)
        .args(args)
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success(), "rustc failed on {}:\n{}\n--- source ---\n{src}", path.display(), String::from_utf8_lossy(&out.stderr));
}

#[test]
fn sum_loop_runs_the_same_in_both_modes() {
    let code = common::sum_loop();
    let fast = emit(&code, "sum", Mode::Fast);
    let safe = emit(&code, "sum", Mode::Safe);
    // Safe mode proved rdi is written through, so it arrives as &mut [u8] and
    // every access is checked; fast mode keeps raw pointers.
    assert!(safe.starts_with("pub fn sum(rdi_ref: &mut [u8], mut rsi: u64, mut rcx: u64) -> u64 {"), "{safe}");
    assert!(!safe.contains("unsafe"), "{safe}");
    assert!(fast.starts_with("pub unsafe fn sum(mut rdi: u64, mut rsi: u64, mut rcx: u64) -> u64 {"), "{fast}");

    let main = r#"
mod fast { include!("fast.rs"); }
mod safe { include!("safe.rs"); }
fn main() {
    // p[1] = rcx, then rax = p + p[2] + p[3] + p[4]
    let mut buf = [0u8; 40];
    for (i, v) in [5u64, 6, 7].iter().enumerate() {
        buf[16 + 8 * i..24 + 8 * i].copy_from_slice(&v.to_le_bytes());
    }
    let base = buf.as_ptr() as u64;
    let r = safe::sum(&mut buf, 3, 99);
    assert_eq!(r.wrapping_sub(base), 18);
    assert_eq!(buf[8], 99);
    let r = unsafe { fast::sum(buf.as_mut_ptr() as u64, 3, 77) };
    assert_eq!(r.wrapping_sub(base), 18);
    assert_eq!(buf[8], 77);
    // Too short for the loop: safe mode panics instead of reading past the end.
    let mut short = [0u8; 24];
    assert!(std::panic::catch_unwind(move || safe::sum(&mut short, 3, 0)).is_err());
}
"#;
    let dir = scratch("sum");
    std::fs::write(dir.join("fast.rs"), format!("{PRELUDE}{fast}").replace("#![allow", "#[allow")).unwrap();
    std::fs::write(dir.join("safe.rs"), format!("{PRELUDE}{safe}").replace("#![allow", "#[allow")).unwrap();
    rustc(&dir, "main.rs", main, &["-o", dir.join("run").to_str().unwrap()]);
    let out = Command::new(dir.join("run")).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
}

/// xorshift64*, as in tests/robust.rs.
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

/// Whatever the lifter accepts, the emitter must turn into Rust that type-checks.
#[test]
fn random_programs_emit_rust_that_compiles() {
    let gprs = [rax, rcx, rdx, rbx, rsi, rdi, r8, r9];
    let gpr32 = [eax, ecx, edx, ebx, esi, edi, r8d, r9d];
    let mut rng = Rng(0x0123_4567_89AB_CDEF);
    let mut src = String::from(PRELUDE);
    let mut n_ok = 0;
    for p in 0..400 {
        let n = 2 + rng.below(16) as usize;
        let mut a = CodeAssembler::new(64).unwrap();
        let mut labels: Vec<CodeLabel> = (0..n).map(|_| a.create_label()).collect();
        for i in 0..n {
            a.set_label(&mut labels[i]).unwrap();
            let r = gprs[rng.below(8) as usize];
            let s = gprs[rng.below(8) as usize];
            let (r32, s32) = (gpr32[rng.below(8) as usize], gpr32[rng.below(8) as usize]);
            let to = labels[rng.below(n as u64) as usize];
            match rng.below(28) {
                16 => a.call(0x2000).unwrap(),
                17 => a.push(r).unwrap(),
                18 => a.pop(r).unwrap(),
                19 => { a.cmp(r, s).unwrap(); a.cmovl(r, s).unwrap() }
                20 => { a.test(r32, r32).unwrap(); a.setne(al).unwrap() }
                21 => a.add(qword_ptr(r + 8), s).unwrap(),
                22 => { a.cmp(dword_ptr(s), 3).unwrap(); a.jbe(to).unwrap() }
                23 => a.movzx(r32, byte_ptr(s + 1)).unwrap(),
                24 => a.movsxd(r, r32).unwrap(),
                25 => a.shr(r, 3).unwrap(),
                26 => a.imul_3(r32, s32, 10).unwrap(),
                27 => a.mov(cl, dl).unwrap(),
                0 => a.mov(r, s).unwrap(),
                1 => a.mov(r, rng.next()).unwrap(),
                2 => a.mov(qword_ptr(r + 8), s).unwrap(),
                3 => a.mov(r, qword_ptr(s + r * 8 + 16)).unwrap(),
                4 => a.add(r, s).unwrap(),
                5 => a.sub(r, 1).unwrap(),
                6 => a.xor(r32, s32).unwrap(),
                7 => a.lea(r, qword_ptr(s - 0x20)).unwrap(),
                8 => { a.cmp(r, s).unwrap(); a.jl(to).unwrap() }
                9 => { a.test(r, r).unwrap(); a.jne(to).unwrap() }
                10 => a.jmp(to).unwrap(),
                11 => a.mov(dword_ptr(r), s32).unwrap(),
                12 => a.mov(r32, dword_ptr(s + 4)).unwrap(),
                13 => { a.and(r32, s32).unwrap(); a.js(to).unwrap() }
                14 => a.mov(dword_ptr(r + 12), 7).unwrap(),
                _ => a.ret().unwrap(),
            }
        }
        a.ret().unwrap();
        let code = a.assemble(common::BASE).unwrap();
        let mut f = Function::with_capacity(64, 8);
        if Lifter::new().lift(&code, common::BASE, &mut f).is_err() {
            continue;
        }
        clean(&mut f);
        for (mode, tag) in [(Mode::Fast, "fast"), (Mode::Safe, "safe")] {
            src.push_str(&format!("// {code:02x?}\n"));
            emit_function(&f, &format!("p{p}_{tag}"), mode, &|_| None, &mut src);
        }
        n_ok += 1;
    }
    assert!(n_ok > 120, "only {n_ok} programs lifted");
    let dir = scratch("random");
    rustc(&dir, "random.rs", &src, &["--crate-type", "lib", "--emit", "metadata"]);
}

/// Build a small ELF with two functions, then drive the real `chungusite` binary.
#[test]
fn cli_decompiles_an_elf() {
    use object::write::{Object as Obj, StandardSection, Symbol, SymbolSection};
    use object::{Architecture, BinaryFormat, Endianness, SymbolFlags, SymbolKind, SymbolScope};

    let get_count = [0x48, 0x8B, 0x47, 0x08, 0xC3]; // mov rax, [rdi+8]; ret
    let mul = [0x48, 0xF7, 0xE1, 0xC3]; // mul rcx; ret (rdx:rax result, not liftable yet)
    let mut obj = Obj::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);
    let text = obj.section_id(StandardSection::Text);
    for (name, code) in [("get_count", &get_count[..]), ("muller", &mul[..])] {
        let off = obj.append_section_data(text, code, 16);
        obj.add_symbol(Symbol {
            name: name.as_bytes().to_vec(),
            value: off,
            size: code.len() as u64,
            kind: SymbolKind::Text,
            scope: SymbolScope::Linkage,
            weak: false,
            section: SymbolSection::Section(text),
            flags: SymbolFlags::None,
        });
    }
    let dir = scratch("cli");
    let elf = dir.join("two.o");
    std::fs::write(&elf, obj.write().unwrap()).unwrap();
    let bin = env!("CARGO_BIN_EXE_chungusite");

    let list = Command::new(bin).arg(&elf).arg("--list").output().unwrap();
    let text = String::from_utf8(list.stdout).unwrap();
    assert!(text.contains("ok    get_count @ 0x0, 5 bytes"), "{text}");
    assert!(text.contains("FAIL  muller @ 0x10, 4 bytes: unsupported instruction Mul at 0x10"), "{text}");
    assert_eq!(list.status.code(), Some(1), "not everything lifted");

    for mode in ["fast", "safe"] {
        let out = Command::new(bin).arg(&elf).args(["--mode", mode]).output().unwrap();
        let src = String::from_utf8(out.stdout).unwrap();
        assert!(src.contains("pub fn muller() -> u64 {\n    todo!(\"not lifted: unsupported instruction Mul at 0x10\")"), "{src}");
        if mode == "safe" {
            assert!(src.contains("pub fn get_count(rdi_ref: &[u8]) -> u64 {"), "{src}");
        }
        rustc(&dir, &format!("{mode}.rs"), &src, &["--crate-type", "lib", "--emit", "metadata"]);
    }

    let one = Command::new(bin).arg(&elf).args(["-f", "get_count", "--emit", "ir"]).output().unwrap();
    assert!(one.status.success());
    let ir = String::from_utf8(one.stdout).unwrap();
    assert!(ir.contains("load v1") && !ir.contains("muller"), "{ir}");

    let missing = Command::new(bin).arg(&elf).args(["-f", "nope"]).output().unwrap();
    assert_eq!(missing.status.code(), Some(2));
}
