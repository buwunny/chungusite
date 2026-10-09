//! Code written to throw a disassembler off, as hand-written assembly: a junk
//! byte after a `jmp`, a jump into the middle of its own instruction, a table
//! of data after a function's `ret`, and a branch that always goes one way to
//! garbage. The symbols carry no type (`.globl f` without `.type`), and the
//! local labels between them aren't functions. The program is decompiled into
//! a Cargo project in both modes, which must print what the original does.
//! Also: instructions the lifter has no model of run as inline assembly, and
//! a binary whose code looks compressed gets a warning.
use std::path::PathBuf;
use std::process::Command;

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok_and(|o| o.status.success())
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("chungusite-test-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

const ASM: &str = r#"
.intel_syntax noprefix
.text
.globl junk_after_jmp, overlap, data_in_code, opaque
junk_after_jmp:
    lea eax, [rdi + 1]
    jmp over
    .byte 0xe8                  # starts a 5-byte call over the real code
over:
    add eax, esi
    ret
overlap:
    mov eax, edi
    .byte 0xeb, 0xff, 0xc0      # jmp -1, into its own 0xff: inc eax
    add eax, 2
    ret
data_in_code:
    and edi, 3
    lea rax, [rip + tab]
    mov eax, [rax + rdi * 4]
    ret
tab:
    .long 0x0fffffff, 0xc3c3c3c3, 0xffffffff, 0x12345678
opaque:
    mov eax, edi
    xor ecx, ecx
    test ecx, ecx
    jnz junk                    # never taken
    imul eax, eax, 3
    ret
junk:
    .byte 0x0f, 0x0b, 0xff, 0xff, 0x62
.section .note.GNU-stack, "", @progbits
"#;

const MAIN: &str = r#"#include <stdio.h>
int junk_after_jmp(int, int); int overlap(int); int data_in_code(int); int opaque(int);
int main(void) {
    for (int i = 0; i < 4; i++) printf("%d %d %d %d\n", junk_after_jmp(i, 5), overlap(i), data_in_code(i), opaque(i));
    return 0;
}
"#;

#[test]
fn anti_disassembly_runs_like_the_original() {
    if !have("cc") || !have("cargo") {
        eprintln!("hostile: no C compiler or cargo, skipping");
        return;
    }
    let dir = scratch("hostile");
    std::fs::write(dir.join("anti.s"), ASM).unwrap();
    std::fs::write(dir.join("main.c"), MAIN).unwrap();
    let cc = Command::new("cc").args(["-O1", "-o"]).arg(dir.join("prog")).arg(dir.join("main.c")).arg(dir.join("anti.s")).status().unwrap();
    assert!(cc.success());
    let want = Command::new(dir.join("prog")).output().unwrap();
    assert!(want.status.success());

    let bin = env!("CARGO_BIN_EXE_chungusite");
    let list = Command::new(bin).arg(dir.join("prog")).arg("--list").output().unwrap();
    let list = String::from_utf8_lossy(&list.stdout);
    for (f, size) in [("junk_after_jmp", 9), ("overlap", 9), ("data_in_code", 30), ("opaque", 17)] {
        assert!(list.lines().any(|l| l.starts_with("ok ") && l.contains(&format!(" {f} @ ")) && l.ends_with(&format!(", {size} bytes"))), "{f}:\n{list}");
    }
    for local in ["over", "tab", "junk"] {
        assert!(!list.contains(&format!(" {local} @ ")), "{local} is a label, not a function:\n{list}");
    }
    let opaque = Command::new(bin).arg(dir.join("prog")).args(["-f", "opaque"]).output().unwrap();
    let opaque = String::from_utf8_lossy(&opaque.stdout);
    assert!(!opaque.contains("if "), "the branch to the garbage is gone:\n{opaque}");

    for mode in ["fast", "safe"] {
        let project = dir.join(mode);
        let out = Command::new(bin).arg(dir.join("prog")).args(["--mode", mode, "--cargo"]).arg(&project).output().unwrap();
        assert!(out.status.success(), "{mode}: {}", String::from_utf8_lossy(&out.stderr));
        let target = dir.join("target");
        let build = Command::new("cargo")
            .args(["build", "--quiet", "--offline", "--manifest-path"])
            .arg(project.join("Cargo.toml"))
            .env("CARGO_TARGET_DIR", &target)
            .output()
            .unwrap();
        assert!(build.status.success(), "{mode}: cargo build failed:\n{}", String::from_utf8_lossy(&build.stderr));
        let got = Command::new(target.join("debug/prog")).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&got.stdout), String::from_utf8_lossy(&want.stdout), "{mode}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Instructions the lifter has no model of (SSE4.2's `crc32`, SSSE3's
/// `pshufb`, rotates through the carry, `rdtsc`), with a memory operand on the
/// heap and on the stack, flags in and out, and rbx as an operand.
const UNKNOWN: &str = r#"
.intel_syntax noprefix
.text
.globl crc, shuffle, mem_crc, stack_crc, rotate_carry, carry_out, rbx_crc, tsc_ok
crc:
    mov eax, edi
    crc32 rax, rsi
    ret
shuffle:                        # the low 8 bytes reversed
    movq xmm0, rdi
    lea rax, [rip + rev]
    movdqu xmm1, [rax]
    pshufb xmm0, xmm1
    movq rax, xmm0
    ret
mem_crc:                        # crc32 of rsi qwords at rdi
    xor eax, eax
    test rsi, rsi
    je 2f
1:  crc32 rax, qword ptr [rdi]
    add rdi, 8
    dec rsi
    jne 1b
2:  ret
stack_crc:
    sub rsp, 24
    mov [rsp + 8], rdi
    mov eax, 7
    crc32 rax, qword ptr [rsp + 8]
    add rsp, 24
    ret
rotate_carry:                   # (rdx << 1) | (rdi < rsi)
    cmp rdi, rsi
    mov rax, rdx
    rcl rax, 1
    ret
carry_out:                      # the bit rcr shifts out picks the result
    mov rax, rdi
    clc
    rcr rax, 1
    jc 1f
    ret
1:  not rax
    ret
rbx_crc:
    push rbx
    mov rbx, rdi
    mov eax, esi
    crc32 eax, ebx
    crc32 eax, bh
    pop rbx
    ret
tsc_ok:
    lfence
    rdtsc
    shl rdx, 32
    or rax, rdx
    setne al
    movzx eax, al
    ret
.section .rodata
rev: .byte 7, 6, 5, 4, 3, 2, 1, 0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
.section .note.GNU-stack, "", @progbits
"#;

const UNKNOWN_MAIN: &str = r#"#include <stdio.h>
typedef unsigned long long u64;
u64 crc(u64, u64); u64 shuffle(u64); u64 mem_crc(const u64 *, u64); u64 stack_crc(u64);
u64 rotate_carry(u64, u64, u64); u64 carry_out(u64); u64 rbx_crc(u64, u64); int tsc_ok(void);
int main(void) {
    u64 buf[4] = {1, 2, 0x123456789abcdef0ull, ~0ull};
    for (u64 i = 0; i < 4; i++)
        printf("%llx %llx %llx %llx %llx %llx %llx\n", crc(i, buf[i]), shuffle(buf[i] + i), mem_crc(buf, i), stack_crc(buf[i]),
               rotate_carry(i, 2, buf[i]), carry_out(buf[i] + i), rbx_crc(buf[i] << 8, i));
    printf("%d\n", tsc_ok());
    return 0;
}
"#;

#[test]
fn unknown_instructions_run_as_inline_assembly() {
    if !have("cc") || !have("cargo") {
        eprintln!("hostile: no C compiler or cargo, skipping");
        return;
    }
    let dir = scratch("asm");
    std::fs::write(dir.join("unknown.s"), UNKNOWN).unwrap();
    std::fs::write(dir.join("main.c"), UNKNOWN_MAIN).unwrap();
    let cc = Command::new("cc").args(["-O1", "-o"]).arg(dir.join("prog")).arg(dir.join("main.c")).arg(dir.join("unknown.s")).status().unwrap();
    assert!(cc.success());
    let want = Command::new(dir.join("prog")).output().unwrap();
    assert!(want.status.success());

    let bin = env!("CARGO_BIN_EXE_chungusite");
    let fns = ["crc", "shuffle", "mem_crc", "stack_crc", "rotate_carry", "carry_out", "rbx_crc", "tsc_ok"];
    let list = Command::new(bin).arg(dir.join("prog")).arg("--list").output().unwrap();
    let list = String::from_utf8_lossy(&list.stdout);
    for f in fns {
        assert!(list.lines().any(|l| l.starts_with("ok ") && l.contains(&format!(" {f} @ "))), "{f}:\n{list}");
    }
    let src = Command::new(bin).arg(dir.join("prog")).args(["-f", "rbx_crc"]).output().unwrap();
    let src = String::from_utf8_lossy(&src.stdout);
    assert!(src.contains(r#"core::arch::asm!("crc32 eax, {0:h}", in(reg_abcd) "#), "{src}");
    // without the fallback, each of them fails
    let list = Command::new(bin).arg(dir.join("prog")).args(["--list", "--no-asm"]).output().unwrap();
    let list = String::from_utf8_lossy(&list.stdout);
    for f in fns {
        assert!(list.lines().any(|l| l.starts_with("FAIL ") && l.contains(&format!(" {f} @ "))), "{f}:\n{list}");
    }

    for mode in ["fast", "safe"] {
        let project = dir.join(mode);
        let out = Command::new(bin).arg(dir.join("prog")).args(["--mode", mode, "--cargo"]).arg(&project).output().unwrap();
        assert!(out.status.success(), "{mode}: {}", String::from_utf8_lossy(&out.stderr));
        assert!(String::from_utf8_lossy(&out.stderr).contains("11 instructions the lifter has no model of"), "{mode}: {}", String::from_utf8_lossy(&out.stderr));
        let target = dir.join("target");
        let build = Command::new("cargo")
            .args(["build", "--quiet", "--offline", "--manifest-path"])
            .arg(project.join("Cargo.toml"))
            .env("CARGO_TARGET_DIR", &target)
            .output()
            .unwrap();
        assert!(build.status.success(), "{mode}: cargo build failed:\n{}", String::from_utf8_lossy(&build.stderr));
        let got = Command::new(target.join("debug/prog")).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&got.stdout), String::from_utf8_lossy(&want.stdout), "{mode}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compressed_code_gets_a_warning() {
    if !have("cc") {
        eprintln!("hostile: no C compiler, skipping");
        return;
    }
    let dir = scratch("packed");
    // 64 KiB of random bytes in the code segment, as a packer's compressed payload
    let mut x = 0x2545f4914f6cdd1du64;
    let mut asm = String::from(".text\n.globl payload\npayload:\n");
    for _ in 0..65536 / 16 {
        let bytes: Vec<String> = (0..16)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x & 0xff).to_string()
            })
            .collect();
        asm += &format!(".byte {}\n", bytes.join(","));
    }
    asm += ".section .note.GNU-stack, \"\", @progbits\n";
    std::fs::write(dir.join("payload.s"), asm).unwrap();
    std::fs::write(dir.join("main.c"), "int main(void) { return 0; }\n").unwrap();
    let cc = Command::new("cc").arg("-o").arg(dir.join("packed")).arg(dir.join("main.c")).arg(dir.join("payload.s")).status().unwrap();
    assert!(cc.success());
    let bin = env!("CARGO_BIN_EXE_chungusite");
    let out = Command::new(bin).arg(dir.join("packed")).arg("--list").output().unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("looks compressed or encrypted"), "{err}");
    // an ordinary program doesn't
    let cc = Command::new("cc").arg("-o").arg(dir.join("plain")).arg(dir.join("main.c")).status().unwrap();
    assert!(cc.success());
    let out = Command::new(bin).arg(dir.join("plain")).arg("--list").output().unwrap();
    assert!(!String::from_utf8_lossy(&out.stderr).contains("warning"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Malformed files fail with a message, never a crash: a header that claims a
/// thread-local block of petabytes, then bytes changed at random.
#[test]
fn corrupt_binaries_never_crash() {
    if !have("cc") {
        eprintln!("hostile: no C compiler, skipping");
        return;
    }
    let dir = scratch("corrupt");
    std::fs::write(dir.join("t.c"), "__thread int counter;\nint bump(int x) { counter += x; return counter; }\nint main(void) { return bump(2); }\n").unwrap();
    assert!(Command::new("cc").args(["-O1", "-o"]).arg(dir.join("t")).arg(dir.join("t.c")).status().unwrap().success());
    let good = std::fs::read(dir.join("t")).unwrap();
    let bin = env!("CARGO_BIN_EXE_chungusite");
    let check = |bytes: &[u8], what: &str| {
        let p = dir.join("bad");
        std::fs::write(&p, bytes).unwrap();
        let out = Command::new(bin).arg(&p).args(["-o", "/dev/null", "-j", "1"]).output().unwrap();
        let code = out.status.code();
        assert!(matches!(code, Some(0..=2)), "{what}: exit {code:?}\n{}", String::from_utf8_lossy(&out.stderr));
    };

    // .tbss's sh_size (ELF64: section headers at e_shoff, 64 bytes each; a
    // NOBITS section with SHF_TLS; sh_size at +32)
    let u64_at = |b: &[u8], at: usize| u64::from_le_bytes(b[at..at + 8].try_into().unwrap());
    let (shoff, shnum) = (u64_at(&good, 0x28) as usize, u16::from_le_bytes([good[0x3c], good[0x3d]]) as usize);
    let tbss = (0..shnum)
        .map(|k| shoff + 64 * k)
        .find(|&h| u32::from_le_bytes(good[h + 4..h + 8].try_into().unwrap()) == 8 && u64_at(&good, h + 8) & 0x400 != 0)
        .expect(".tbss");
    let mut bad = good.clone();
    bad[tbss + 32..tbss + 40].copy_from_slice(&0x0030_0000_0000_0000u64.to_le_bytes());
    check(&bad, "huge thread-local block");

    let mut x = 0x9e3779b97f4a7c15u64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for round in 0..60 {
        let mut bad = good.clone();
        for _ in 0..1 + next() % 16 {
            // half the time in the headers, where the tables are
            let at = if next() % 2 == 0 { next() as usize % 4096.min(bad.len()) } else { next() as usize % bad.len() };
            bad[at] = next() as u8;
        }
        check(&bad, &format!("round {round}"));
    }
    let _ = std::fs::remove_dir_all(&dir);
}
