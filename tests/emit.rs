//! The Rust emitter and the CLI, checked by compiling (and running) what they print.
//! These tests shell out to `rustc`, the same one cargo uses.
mod common;
use chungusite::{
    emit::{emit_function, emit_function_with, Mode},
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
    // The loop is structured: a `while` with the exit test as its condition,
    // the load inlined into the sum, and the return after it.
    assert!(fast.contains("    while v6 < rsi {\n") && !fast.contains("match bb"), "{fast}");
    assert!(fast.contains("(v19, v6) = (v19.wrapping_add(unsafe { (rdi.wrapping_add(v6.wrapping_mul(8)).wrapping_add(0x10) as *const u64).read_unaligned() }), v6.wrapping_add(1));"), "{fast}");
    assert!(fast.ends_with("    }\n    return v19;\n}\n"), "{fast}");

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

/// `a || b` folds into one condition, and an irreducible cycle becomes a
/// `match bb` over just its own blocks; both compile and run like the original.
#[test]
fn short_circuits_and_irreducible_regions() {
    // or2(a, b) = if a == 0 || b > 5 { 1 } else { 2 }
    let mut a = CodeAssembler::new(64).unwrap();
    let mut yes = a.create_label();
    a.test(rdi, rdi).unwrap();
    a.je(yes).unwrap();
    a.cmp(rsi, 5).unwrap();
    a.ja(yes).unwrap();
    a.mov(eax, 2).unwrap();
    a.ret().unwrap();
    a.set_label(&mut yes).unwrap();
    a.mov(eax, 1).unwrap();
    a.ret().unwrap();
    let or2 = emit(&a.assemble(common::BASE).unwrap(), "or2", Mode::Fast);
    assert!(or2.contains("    if rdi == 0 || rsi > 5 {\n") && !or2.contains("'b"), "{or2}");

    // tangle(a, n, k): rax = k * 2; if a == 0 jump into the middle of the cycle;
    // the cycle adds 3 and subtracts 1 from n until it is zero; then rax += a.
    let mut a = CodeAssembler::new(64).unwrap();
    let (mut top, mut mid, mut out) = (a.create_label(), a.create_label(), a.create_label());
    a.lea(rax, qword_ptr(rdx + rdx)).unwrap();
    a.test(rdi, rdi).unwrap();
    a.je(mid).unwrap();
    a.set_label(&mut top).unwrap();
    a.add(rax, 3).unwrap();
    a.set_label(&mut mid).unwrap();
    a.test(rsi, rsi).unwrap();
    a.je(out).unwrap();
    a.sub(rsi, 1).unwrap();
    a.jmp(top).unwrap();
    a.set_label(&mut out).unwrap();
    a.add(rax, rdi).unwrap();
    a.ret().unwrap();
    let tangle = emit(&a.assemble(common::BASE).unwrap(), "tangle", Mode::Fast);
    // The code before the cycle stays outside the `match`, which has an arm per
    // block of the cycle only (the exit, with one predecessor, is inlined there).
    let (before, rest) = tangle.split_once("match bb {").expect(&tangle);
    assert!(before.contains("rdx.wrapping_add(rdx)") && before.contains("if rdi == 0 {"), "{tangle}");
    assert_eq!(rest.matches(" => {").count(), 3, "{tangle}");

    let main = r#"
mod code { include!("code.rs"); }
fn main() {
    for (a, b) in [(0, 0), (0, 9), (1, 5), (1, 6), (7, 2)] {
        assert_eq!(unsafe { code::or2(a, b) }, if a == 0 || b > 5 { 1 } else { 2 });
    }
    for (a, n, k) in [(0u64, 0u64, 5u64), (0, 4, 1), (2, 0, 1), (2, 3, 10)] {
        let adds = if a == 0 { n } else { n + 1 };
        assert_eq!(unsafe { code::tangle(a, n, k) }, k * 2 + 3 * adds + a);
    }
}
"#;
    let dir = scratch("tangle");
    std::fs::write(dir.join("code.rs"), format!("{PRELUDE}{or2}{tangle}").replace("#![allow", "#[allow")).unwrap();
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

/// Structured output must compute the same thing as the state machine, which is a
/// direct transcription of the CFG. Random register-only programs (so any input is
/// valid), both forms run on the same random inputs. Loops get fuel so a program
/// that doesn't terminate is skipped instead of hanging the test; the state
/// machine counts every block, so it gets far more than the structured form.
#[test]
fn structured_control_flow_matches_the_state_machine() {
    let gprs = [rax, rcx, rdx, rbx, rsi, rdi, r8, r9];
    let gpr32 = [eax, ecx, edx, ebx, esi, edi, r8d, r9d];
    let mut rng = Rng(0x5EED_0FC0_FFEE);
    let mut lib = String::new();
    let mut calls = String::new();
    let (mut structured, mut machines) = (0, 0);
    for p in 0..300 {
        let n = 3 + rng.below(20) as usize;
        let mut a = CodeAssembler::new(64).unwrap();
        let mut labels: Vec<CodeLabel> = (0..n).map(|_| a.create_label()).collect();
        for i in 0..n {
            a.set_label(&mut labels[i]).unwrap();
            let (r, s) = (gprs[rng.below(8) as usize], gprs[rng.below(8) as usize]);
            let to = labels[rng.below(n as u64) as usize];
            match rng.below(13) {
                0 => a.mov(r, s).unwrap(),
                1 => a.mov(r, rng.below(100)).unwrap(),
                2 | 3 => a.sub(r, 1).unwrap(),
                4 => a.add(r, s).unwrap(),
                5 => a.xor(gpr32[rng.below(8) as usize], gpr32[rng.below(8) as usize]).unwrap(),
                6 => a.lea(r, qword_ptr(s + r * 2 + 3)).unwrap(),
                7 => { a.cmp(r, s).unwrap(); a.jl(to).unwrap() }
                8 => { a.test(r, r).unwrap(); a.jne(to).unwrap() }
                9 => { a.cmp(r, 7).unwrap(); a.jae(to).unwrap() }
                10 => a.jmp(to).unwrap(),
                11 => { a.test(r, s).unwrap(); a.je(to).unwrap() }
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
        let (mut st, mut sm) = (String::new(), String::new());
        let stats = emit_function_with(&f, &format!("st{p}"), Mode::Fast, true, &|_| None, &|_| None, &mut st);
        emit_function_with(&f, &format!("sm{p}"), Mode::Fast, false, &|_| None, &|_| None, &mut sm);
        if stats.state_machines > 0 {
            machines += 1;
        } else if st.contains("loop {") || st.contains("while ") || st.contains("if ") {
            structured += 1;
        }
        // every loop, `while` loops included, burns fuel each time round
        let fuel = |src: String, n: u32| {
            let is_loop = |l: &str| {
                let l = l.trim_start();
                let l = if l.starts_with('\'') { l.split_once(": ").map_or(l, |x| x.1) } else { l };
                l.ends_with(" {") && (l == "loop {" || l.starts_with("while "))
            };
            src.lines().map(|l| if is_loop(l) { format!("{l} fuel!({n});\n") } else { format!("{l}\n") }).collect::<String>()
        };
        lib.push_str(&format!("// {code:02x?}\n{}{}", fuel(st, 1_000), fuel(sm, 1_000_000)));
        let arity = sm_arity(&lib, p);
        let args = (0..arity).map(|k| format!("x[{k}]")).collect::<Vec<_>>().join(", ");
        calls.push_str(&format!(
            "    for x in &inputs {{ FUEL.set(0); let a = catch(|| unsafe {{ st{p}({args}) }}); \
             if a.is_none() {{ continue; }} FUEL.set(0); let b = catch(|| unsafe {{ sm{p}({args}) }}); \
             assert_eq!(a, b, \"program {p} on {{x:?}}\"); compared += 1; }}\n"
        ));
    }
    assert!(structured > 100, "only {structured} programs with control flow were structured");
    assert!(machines > 0, "no irreducible program exercised the fallback");

    let main = format!(
        r#"{PRELUDE}
use std::cell::Cell;
thread_local! {{ static FUEL: Cell<u32> = Cell::new(0); }}
macro_rules! fuel {{ ($n:expr) => {{ FUEL.set(FUEL.get() + 1); if FUEL.get() > $n {{ panic!("out of fuel"); }} }} }}
fn catch(f: impl FnOnce() -> u64 + std::panic::UnwindSafe) -> Option<u64> {{ std::panic::catch_unwind(f).ok() }}
{lib}
fn main() {{
    std::panic::set_hook(Box::new(|_| {{}}));
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {{ s ^= s >> 12; s ^= s << 25; s ^= s >> 27; s.wrapping_mul(0x2545_F491_4F6C_DD1D) % 20 }};
    let inputs: Vec<[u64; 16]> = (0..16).map(|_| std::array::from_fn(|_| next())).collect();
    let mut compared = 0;
{calls}    assert!(compared > 1500, "only {{compared}} runs terminated");
    println!("{{compared}}");
}}
"#
    );
    let dir = scratch("structured");
    rustc(&dir, "main.rs", &main, &["-O", "-o", dir.join("run").to_str().unwrap()]);
    let out = Command::new(dir.join("run")).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    eprintln!("{structured} structured, {machines} state machines, {} runs compared", String::from_utf8_lossy(&out.stdout).trim());
}

/// Number of parameters of `sm{p}` in `src`.
fn sm_arity(src: &str, p: usize) -> usize {
    let head = format!("fn sm{p}(");
    let start = src.find(&head).unwrap() + head.len();
    let params = &src[start..start + src[start..].find(')').unwrap()];
    if params.trim().is_empty() { 0 } else { params.split(", ").count() }
}
