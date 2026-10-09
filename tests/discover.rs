//! Function discovery on stripped binaries: `tests/differential/corpus.c` is
//! linked into a program with each C compiler found, a copy is stripped, and
//! the functions `--list` finds in the stripped copy must be the ones the
//! symbol table gives for the original. The stripped copy's Rust must type-check.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("chungusite-test-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn chungusite(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_chungusite")).args(args).output().unwrap();
    (out.status.code().unwrap_or(-1), String::from_utf8_lossy(&out.stdout).into(), String::from_utf8_lossy(&out.stderr).into())
}

/// `--list`: address -> (size, name).
fn list(bin: &Path) -> BTreeMap<u64, (u64, String)> {
    let (code, out, err) = chungusite(&[bin.to_str().unwrap(), "--list"]);
    assert!(code == 0 || code == 1, "--list failed on {}: {err}", bin.display());
    let mut m = BTreeMap::new();
    for line in out.lines() {
        let rest = line.trim_start_matches("ok").trim_start_matches("FAIL").trim_start();
        let Some((name, tail)) = rest.split_once(" @ ") else { continue };
        let (addr, tail) = tail.split_once(", ").unwrap();
        let size = tail.split(' ').next().unwrap().parse().unwrap();
        m.insert(u64::from_str_radix(addr.trim_start_matches("0x"), 16).unwrap(), (size, name.to_string()));
    }
    m
}

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok_and(|o| o.status.success())
}

#[test]
fn stripped_binaries_have_the_same_functions() {
    let ccs: Vec<&str> = ["cc", "clang"].into_iter().filter(|c| have(c)).collect();
    if ccs.is_empty() || !have("strip") {
        eprintln!("discover: no C compiler or strip, skipping");
        return;
    }
    let dir = scratch("discover");
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/differential/corpus.c");
    let main_c = dir.join("main.c");
    std::fs::write(&main_c, "int main(void) { return 0; }\n").unwrap();
    // With unwind tables (the default) every boundary comes from `.eh_frame`;
    // without them, from calls and the code left between functions.
    // gcc -O2 without unwind tables ends some functions in a call to
    // `__stack_chk_fail` and uses jump tables; clang -Os packs functions with
    // no padding between them.
    let variants: [(&str, &[&str]); 5] = [
        ("eh", &["-O2"]),
        ("noeh", &["-O0", "-fno-asynchronous-unwind-tables"]),
        ("noeh-o2", &["-O2", "-fno-asynchronous-unwind-tables"]),
        ("noeh-os", &["-Os", "-fno-asynchronous-unwind-tables"]),
        ("nopie", &["-O1", "-fno-pie", "-no-pie"]),
    ];
    for cc in &ccs {
        for (tag, flags) in variants {
            let full = dir.join(format!("{cc}-{tag}"));
            let stripped = dir.join(format!("{cc}-{tag}-stripped"));
            let out = Command::new(cc).args(flags).arg("-o").arg(&full).arg(&corpus).arg(&main_c).arg("-lm").output().unwrap();
            assert!(out.status.success(), "{cc} {flags:?}: {}", String::from_utf8_lossy(&out.stderr));
            let out = Command::new("strip").arg("-o").arg(&stripped).arg(&full).output().unwrap();
            assert!(out.status.success(), "strip: {}", String::from_utf8_lossy(&out.stderr));

            let want = list(&full);
            let got = list(&stripped);
            let label = format!("{cc} {flags:?}");
            assert!(want.len() > 80, "{label}: only {} functions with symbols", want.len());
            let missing: Vec<_> = want.iter().filter(|(a, _)| !got.contains_key(a)).map(|(_, (_, n))| n).collect();
            let extra: Vec<_> = got.keys().filter(|a| !want.contains_key(a)).map(|a| format!("{a:#x}")).collect();
            assert!(missing.is_empty() && extra.is_empty(), "{label}: missed {missing:?}, invented {extra:?}");
            // Sizes: symbol sizes of the hand-written crt functions include their
            // trailing padding; everything compiled matches exactly.
            let exact = want.iter().filter(|(a, (s, _))| got[a].0 == *s).count();
            assert!(exact + 4 >= want.len(), "{label}: only {exact} of {} sizes match", want.len());
            let main = want.iter().find(|(_, (_, n))| n == "main").map(|(a, _)| *a).unwrap();
            assert_eq!(got[&main].1, "main", "{label}: main not named");
            eprintln!("{label}: {} functions, {exact} exact", want.len());
        }
    }

    // The stripped program decompiles end to end, and the output type-checks.
    let stripped = dir.join(format!("{}-eh-stripped", ccs[0]));
    for mode in ["fast", "safe"] {
        let rs = dir.join(format!("stripped_{mode}.rs"));
        let (code, _, err) = chungusite(&[stripped.to_str().unwrap(), "--mode", mode, "-o", rs.to_str().unwrap()]);
        assert!(code == 0 || code == 1, "{err}");
        assert!(err.contains("no symbol table"), "{err}");
        let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
        let out = Command::new(rustc)
            .args(["--edition", "2021", "-A", "warnings", "--crate-type", "lib", "--emit", "metadata", "--crate-name", "stripped", "--out-dir"])
            .arg(&dir)
            .arg(&rs)
            .output()
            .unwrap();
        assert!(out.status.success(), "{mode} output of the stripped binary doesn't type-check:\n{}", String::from_utf8_lossy(&out.stderr));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Functions that never return, without unwind tables: `die` is one because
/// every path through it ends in `exit` or `abort`, so `checked` and `more` end
/// at their call to it, and `unused`, which nothing calls, isn't swallowed by
/// `checked`. `via_slot` calls `die` through a pointer the loader fills in (as
/// Rust calls `handle_alloc_error` through the GOT) and `via_got` calls `abort`
/// through the GOT; both branch on flags set before the call at the address
/// after it, which only lifts if the call is known not to return.
const NORETURN_C: &str = r#"
#include <stdio.h>
#include <stdlib.h>

__attribute__((noinline, noreturn)) void die(const char *m, int code) {
    fputs(m, stderr);
    if (code)
        exit(code);
    abort();
}

__attribute__((noinline)) int checked(int x) {
    if (x < 0)
        die("negative\n", 2);
    return x * 3;
}

__attribute__((noinline)) int unused(int x) { return x ^ 0x55; }

__attribute__((noinline)) int more(int x, int y) {
    if (x == y)
        die("equal\n", 0);
    return x - y;
}

__asm__(
    ".text\n"
    ".globl via_slot\n"
    ".type via_slot, @function\n"
    "via_slot:\n"
    "  test %rdi, %rdi\n"
    "  jns 1f\n"
    "  xor %esi, %esi\n"
    "  call *die_slot(%rip)\n"
    "1: je 2f\n"
    "  mov $1, %eax\n"
    "  ret\n"
    "2: xor %eax, %eax\n"
    "  ret\n"
    ".size via_slot, .-via_slot\n"
    ".globl via_got\n"
    ".type via_got, @function\n"
    "via_got:\n"
    "  test %rdi, %rdi\n"
    "  jns 1f\n"
    "  call *abort@GOTPCREL(%rip)\n"
    "1: je 2f\n"
    "  mov $1, %eax\n"
    "  ret\n"
    "2: xor %eax, %eax\n"
    "  ret\n"
    ".size via_got, .-via_got\n"
    ".section .data.rel.ro, \"aw\"\n"
    ".p2align 3\n"
    "die_slot: .quad die\n"
    ".text\n");

int via_slot(long), via_got(long);

int main(int argc, char **argv) { return checked(argc) + more(argc, 3) + via_slot(argc) + via_got(argc); }
"#;

#[test]
fn calls_that_dont_return_end_functions() {
    let ccs: Vec<&str> = ["cc", "clang"].into_iter().filter(|c| have(c)).collect();
    if ccs.is_empty() || !have("strip") {
        eprintln!("discover: no C compiler or strip, skipping");
        return;
    }
    let dir = scratch("noreturn");
    let src = dir.join("noreturn.c");
    std::fs::write(&src, NORETURN_C).unwrap();
    let ours = ["die", "checked", "unused", "more", "via_slot", "via_got"];
    for cc in &ccs {
        for opt in ["-O0", "-O1", "-O2", "-Os"] {
            let label = format!("{cc} {opt}");
            let full = dir.join(format!("{cc}{opt}"));
            let stripped = dir.join(format!("{cc}{opt}-stripped"));
            let flags = [opt, "-fno-asynchronous-unwind-tables", "-fno-unwind-tables"];
            let out = Command::new(cc).args(flags).arg("-o").arg(&full).arg(&src).output().unwrap();
            assert!(out.status.success(), "{label}: {}", String::from_utf8_lossy(&out.stderr));
            let out = Command::new("strip").arg("-o").arg(&stripped).arg(&full).output().unwrap();
            assert!(out.status.success(), "strip: {}", String::from_utf8_lossy(&out.stderr));

            // The calls through a slot end their blocks, so both lift.
            let (_, out, _) = chungusite(&[full.to_str().unwrap(), "--list"]);
            for f in ["via_slot", "via_got"] {
                let line = out.lines().find(|l| l.contains(&format!(" {f} @ "))).unwrap_or_else(|| panic!("{label}: no {f}"));
                assert!(line.starts_with("ok"), "{label}: {line}");
            }

            // Without symbols or unwind tables, every function is found with its exact size.
            let want = list(&full);
            let got = list(&stripped);
            for (a, (size, name)) in want.iter().filter(|(_, (_, n))| ours.contains(&n.as_str())) {
                assert_eq!(got.get(a).map(|g| g.0), Some(*size), "{label}: {name} at {a:#x}");
            }
            let extra: Vec<_> = got.keys().filter(|a| !want.contains_key(a)).map(|a| format!("{a:#x}")).collect();
            assert!(extra.is_empty(), "{label}: invented {extra:?}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

const COLD_C: &str = r#"
#include <stdlib.h>
__attribute__((noinline)) long pick(long x) {
    if (__builtin_expect(x > 100, 0)) abort();
    return x * 3 + 1;
}
int main(int argc, char **argv) { return pick(argc); }
"#;

/// gcc moves the call to `abort` into `pick.cold` and jumps there: a jump to
/// code that never returns says nothing about the result, so `pick` still
/// returns one, with and without symbols.
#[test]
fn jumps_to_cold_parts_that_dont_return() {
    if !have("cc") || !have("strip") {
        eprintln!("discover: no C compiler or strip, skipping");
        return;
    }
    let dir = scratch("cold");
    let src = dir.join("cold.c");
    std::fs::write(&src, COLD_C).unwrap();
    let full = dir.join("cold");
    let stripped = dir.join("cold-stripped");
    let out = Command::new("cc").args(["-O2", "-o"]).arg(&full).arg(&src).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let out = Command::new("strip").arg("-o").arg(&stripped).arg(&full).output().unwrap();
    assert!(out.status.success(), "strip: {}", String::from_utf8_lossy(&out.stderr));
    let addr = list(&full).into_iter().find(|(_, (_, n))| n == "pick").map(|(a, _)| a).expect("no pick");
    for (bin, name) in [(&full, "pick".to_string()), (&stripped, format!("sub_{addr:x}"))] {
        let (code, out, err) = chungusite(&[bin.to_str().unwrap()]);
        assert!(code == 0 || code == 1, "{err}");
        let sig = out.lines().find(|l| l.contains(&format!("fn {name}("))).unwrap_or_else(|| panic!("no {name}:\n{out}"));
        assert!(sig.contains(") -> "), "{}: {sig}", bin.display());
    }
    let _ = std::fs::remove_dir_all(&dir);
}
