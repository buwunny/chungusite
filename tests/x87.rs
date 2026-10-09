//! `long double` (x87) code and stack arguments in a frame that can't be
//! followed: a C program is decompiled into a Cargo project in both modes,
//! which must print what the original does.
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

/// `human` divides a `long double` in a loop, truncates it to an integer
/// (`fistp` under a changed control word) and passes it to `snprintf` in
/// memory; `order` takes two on the stack and compares them (`fcomi`);
/// `many`'s `alloca` leaves a frame whose offsets can't be followed, and it
/// reads three stack arguments.
const SOURCE: &str = r#"#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <alloca.h>

__attribute__((noinline)) void human(unsigned long n, char *buf, size_t len) {
    static const char units[] = " KMGTPEZY";
    long double x = n;
    int p = 0;
    while (x >= 1024 && p < 8) { x /= 1024; p++; }
    long t = (long) (x * 10);
    snprintf(buf, len, "%ld.%ld%c %.3Lf", t / 10, t % 10, units[p], x);
}

__attribute__((noinline)) int order(long double a, long double b) { return a < b ? -1 : a > b; }

__attribute__((noinline)) int many(int a, int b, int c, int d, int e, int f, int g, int h, int n) {
    char *p = alloca(n + 1);
    memset(p, 'x', n);
    p[n] = 0;
    return a + b + c + d + e + f + g * 100 + h * 1000 + (int) strlen(p);
}

int main(int argc, char **argv) {
    char buf[64];
    unsigned long sizes[] = {0, 1023, 1024, 1536, 999999999, 123456789012345UL, (unsigned long) argc << 40};
    for (int i = 0; i < 7; i++) {
        human(sizes[i], buf, sizeof buf);
        printf("%s\n", buf);
    }
    printf("%d\n", order(argc, 2.5L));
    printf("%d\n", order(3.5L, argc));
    printf("%d\n", order(argc, argc));
    printf("%d\n", many(1, 2, 3, 4, 5, 6, 7, 8, argc + 4));
    return 0;
}
"#;

#[test]
fn long_double_and_stack_arguments() {
    if !have("cc") || !have("cargo") {
        eprintln!("x87: no C compiler or cargo, skipping");
        return;
    }
    let dir = scratch("x87");
    std::fs::write(dir.join("t.c"), SOURCE).unwrap();
    assert!(Command::new("cc").args(["-O2", "-o"]).arg(dir.join("t")).arg(dir.join("t.c")).status().unwrap().success());
    let args = ["a", "b"];
    let want = Command::new(dir.join("t")).args(args).output().unwrap();
    assert!(want.status.success());
    let bin = env!("CARGO_BIN_EXE_chungusite");
    for mode in ["fast", "safe"] {
        let project = dir.join(mode);
        let out = Command::new(bin).arg(dir.join("t")).args(["--mode", mode, "--cargo"]).arg(&project).output().unwrap();
        assert!(out.status.success(), "chungusite failed:\n{}", String::from_utf8_lossy(&out.stderr));
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(!err.contains("unsupported"), "{mode}:\n{err}");
        let target = dir.join("target");
        let build = Command::new("cargo")
            .args(["build", "--quiet", "--offline", "--manifest-path"])
            .arg(project.join("Cargo.toml"))
            .env("CARGO_TARGET_DIR", &target)
            .output()
            .unwrap();
        assert!(build.status.success(), "{mode}: cargo build failed:\n{}", String::from_utf8_lossy(&build.stderr));
        let got = Command::new(target.join("debug/t")).args(args).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&got.stdout), String::from_utf8_lossy(&want.stdout), "{mode}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
