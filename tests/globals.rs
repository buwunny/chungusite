//! Globals and demangling on a real linked binary: rustc builds a cdylib whose
//! functions read and write statics, chungusite decompiles it, and the decompiled
//! functions must see the same data when they run.
use std::path::{Path, PathBuf};
use std::process::Command;

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("chungusite-test-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

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

const ORIGINAL: &str = r#"
#[no_mangle]
pub static TABLE: [u32; 4] = [10, 20, 30, 40];
#[no_mangle]
pub static mut COUNTER: u32 = 7;

#[no_mangle]
pub extern "C" fn third() -> u32 {
    unsafe { core::ptr::read_volatile(&TABLE[2]) }
}
#[no_mangle]
pub extern "C" fn set_counter(v: u32) {
    unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!(COUNTER), v) }
}
#[no_mangle]
pub extern "C" fn counter() -> u32 {
    unsafe { core::ptr::read_volatile(core::ptr::addr_of!(COUNTER)) }
}
// Mangled, so the decompiler has to demangle it.
#[inline(never)]
pub fn first() -> u32 {
    unsafe { core::ptr::read_volatile(&TABLE[0]) }
}
#[no_mangle]
pub extern "C" fn keep_first() -> extern "C" fn() -> u32 {
    extern "C" fn f() -> u32 { first() }
    f
}
"#;

const MAIN: &str = r#"
fn main() {
    unsafe {
        assert_eq!(dec::third(), 30);
        assert_eq!(dec::counter(), 7);
        dec::set_counter(99);
        assert_eq!(dec::counter(), 99);
        assert_eq!(dec::orig__first(), 10);
    }
}
"#;

#[test]
fn statics_and_demangled_names_from_a_cdylib() {
    let dir = scratch("globals");
    rustc(&dir, "orig.rs", ORIGINAL, &["--crate-type", "cdylib", "-C", "opt-level=2", "--crate-name", "orig"]);
    let so = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "so" || e == "dylib" || e == "dll"))
        .expect("cdylib");
    let bin = env!("CARGO_BIN_EXE_chungusite");

    let list = Command::new(bin).arg(&so).arg("--list").output().unwrap();
    let list = String::from_utf8(list.stdout).unwrap();
    assert!(list.contains("ok    orig::first @ "), "{list}");

    for mode in ["fast", "safe"] {
        let out = Command::new(bin)
            .arg(&so)
            .args(["--mode", mode, "-f", "third", "-f", "counter", "-f", "set_counter", "-f", "orig::first"])
            .output()
            .unwrap();
        let src = String::from_utf8(out.stdout).unwrap();
        let err = String::from_utf8(out.stderr).unwrap();
        assert!(out.status.success(), "{err}\n{src}");
        assert!(src.contains("pub static TABLE: Bytes<16> = Bytes { b: *b\"\\n\\0\\0\\0\\x14\\0\\0\\0\\x1e\\0\\0\\0(\\0\\0\\0\" };"), "{src}");
        assert!(src.contains("pub static mut COUNTER: Bytes<4>"), "{src}");
        assert!(src.contains("fn orig__first("), "{src}");

        let d = dir.join(mode);
        std::fs::create_dir_all(&d).unwrap();
        rustc(&d, "dec.rs", &src, &["--crate-type", "rlib", "--crate-name", "dec"]);
        let rlib = d.join("libdec.rlib");
        rustc(&d, "main.rs", MAIN, &["--extern", &format!("dec={}", rlib.display()), "-o", d.join("run").to_str().unwrap()]);
        let run = Command::new(d.join("run")).output().unwrap();
        assert!(run.status.success(), "{mode}: {}", String::from_utf8_lossy(&run.stderr));
    }
}

const SWITCHES: &str = r#"
#[no_mangle]
pub extern "C" fn dispatch(op: u32, a: u64, b: u64) -> u64 {
    match op {
        0 => a ^ b,
        1 => a & b,
        2 => a | b,
        3 => !a,
        4 => a,
        5 => b,
        6 => a ^ 0x55,
        7 => b & 0xff,
        _ => 7,
    }
}
#[repr(u8)]
#[derive(Clone, Copy)]
pub enum Shape { Dot, Line, Tri, Square, Penta, Hexa }
// Unoptimized, a `match` on an enum is a jump table with no bounds check.
#[no_mangle]
pub extern "C" fn corners(s: Shape, x: u64) -> u64 {
    match s {
        Shape::Dot => x,
        Shape::Line => x ^ 2,
        Shape::Tri => x | 3,
        Shape::Square => x & 4,
        Shape::Penta => !x,
        Shape::Hexa => 6,
    }
}
"#;

// Arguments and results `as _`: type recovery narrows them (`op: u32`).
const SWITCHES_MAIN: &str = r#"
fn main() {
    for op in 0..12u64 {
        for (a, b) in [(0u64, 0u64), (0x1234, 0xff00), (u64::MAX, 3)] {
            let want = match op { 0 => a ^ b, 1 => a & b, 2 => a | b, 3 => !a, 4 => a, 5 => b, 6 => a ^ 0x55, 7 => b & 0xff, _ => 7 };
            assert_eq!(unsafe { dec::dispatch(op as _, a as _, b as _) } as u64, want, "dispatch({op}, {a}, {b})");
        }
    }
    for s in 0..6u64 {
        let x = 0x5a5a;
        let want = [x, x ^ 2, x | 3, x & 4, !x, 6][s as usize];
        assert_eq!(unsafe { dec::corners(s as _, x as _) } as u64, want, "corners({s})");
    }
}
"#;

/// Jump tables as rustc lays them out, bounds-checked (`-O2`) and not (`-O0`),
/// lift into switches that compute the same thing.
#[test]
fn jump_tables_from_a_cdylib() {
    for opt in ["0", "2"] {
        let dir = scratch(&format!("switches{opt}"));
        rustc(&dir, "orig.rs", SWITCHES, &["--crate-type", "cdylib", "-C", &format!("opt-level={opt}"), "--crate-name", "orig"]);
        let so = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| p.extension().is_some_and(|e| e == "so" || e == "dylib" || e == "dll"))
            .expect("cdylib");
        let bin = env!("CARGO_BIN_EXE_chungusite");
        for mode in ["fast", "safe"] {
            let out = Command::new(bin).arg(&so).args(["--mode", mode, "-f", "dispatch", "-f", "corners"]).output().unwrap();
            let src = String::from_utf8(out.stdout).unwrap();
            assert!(out.status.success(), "{}\n{src}", String::from_utf8_lossy(&out.stderr));
            assert!(!src.contains("todo!"), "-O{opt}:\n{src}");
            let d = dir.join(mode);
            std::fs::create_dir_all(&d).unwrap();
            rustc(&d, "dec.rs", &src, &["--crate-type", "rlib", "--crate-name", "dec"]);
            let rlib = d.join("libdec.rlib");
            rustc(&d, "main.rs", SWITCHES_MAIN, &["--extern", &format!("dec={}", rlib.display()), "-o", d.join("run").to_str().unwrap()]);
            let run = Command::new(d.join("run")).output().unwrap();
            assert!(run.status.success(), "-O{opt} {mode}: {}\n{src}", String::from_utf8_lossy(&run.stderr));
        }
    }
}

/// The same output however many threads do the work.
#[test]
fn parallel_output_is_deterministic() {
    let bin = env!("CARGO_BIN_EXE_chungusite");
    let run = |jobs: &str| {
        let out = Command::new(bin).arg(bin).args(["-j", jobs, "--mode", "safe"]).output().unwrap();
        // exit 1 only says some functions weren't lifted; anything else is a crash
        assert!(out.status.code().is_some_and(|c| c <= 1), "-j {jobs}: {}\n{}", out.status, String::from_utf8_lossy(&out.stderr));
        out.stdout
    };
    let one = run("1");
    assert!(one.len() > 100_000);
    assert_eq!(one, run("4"));
}

const STDERR_C: &str = r#"
#include <stdio.h>
int report(int x) {
    fprintf(stderr, "x=%d\n", x);
    return fileno(stderr) * 100 + x;
}
int main(int argc, char **argv) { return report(argc); }
"#;

const STDERR_MAIN: &str = r#"
fn main() {
    assert_eq!(unsafe { dec::report(5 as _) } as i32, 205);
}
"#;

/// The C library's own data that a program reads through a copy relocation
/// (`stderr` in a PIE) is the library's, not a copy of the bytes in the file.
#[test]
fn copy_relocated_library_data() {
    let dir = scratch("copyreloc");
    let c = dir.join("orig.c");
    std::fs::write(&c, STDERR_C).unwrap();
    let exe = dir.join("orig");
    let Ok(out) = Command::new("cc").args(["-O2", "-o"]).arg(&exe).arg(&c).output() else {
        assert!(std::env::var_os("CI").is_none(), "no C compiler");
        return;
    };
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let bin = env!("CARGO_BIN_EXE_chungusite");
    for mode in ["fast", "safe"] {
        let out = Command::new(bin).arg(&exe).args(["--mode", mode, "-f", "report"]).output().unwrap();
        let src = String::from_utf8(out.stdout).unwrap();
        assert!(out.status.success(), "{}\n{src}", String::from_utf8_lossy(&out.stderr));
        assert!(src.contains("#[link_name = \"stderr\"]"), "{mode}:\n{src}");
        let d = dir.join(mode);
        std::fs::create_dir_all(&d).unwrap();
        rustc(&d, "dec.rs", &src, &["--crate-type", "rlib", "--crate-name", "dec"]);
        let rlib = d.join("libdec.rlib");
        rustc(&d, "main.rs", STDERR_MAIN, &["--extern", &format!("dec={}", rlib.display()), "-o", d.join("run").to_str().unwrap()]);
        let run = Command::new(d.join("run")).output().unwrap();
        assert!(run.status.success(), "{mode}: {}\n{src}", String::from_utf8_lossy(&run.stderr));
        assert_eq!(String::from_utf8_lossy(&run.stderr), "x=5\n", "{mode}");
    }
}
