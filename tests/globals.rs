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

/// The same output however many threads do the work.
#[test]
fn parallel_output_is_deterministic() {
    let bin = env!("CARGO_BIN_EXE_chungusite");
    let run = |jobs: &str| Command::new(bin).arg(bin).args(["-j", jobs, "--mode", "safe"]).output().unwrap().stdout;
    let one = run("1");
    assert!(one.len() > 100_000);
    assert_eq!(one, run("4"));
}
