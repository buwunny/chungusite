//! `--cargo DIR`: a C program in three files, compiled with debug info, is
//! decompiled into a Cargo project in both modes; the project must build, have
//! a module per source file, and print and return what the original does.
use std::path::{Path, PathBuf};
use std::process::Command;

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok_and(|o| o.status.success())
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("chungusite-test-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

const SOURCES: &[(&str, &str)] = &[
    (
        "list.c",
        r#"#include <stdlib.h>
struct node { int v; struct node *next; };
struct node *push(struct node *h, int v) { struct node *n = malloc(sizeof *n); n->v = v; n->next = h; return n; }
int total(struct node *h) { int s = 0; for (; h; h = h->next) s += h->v; return s; }
void freeall(struct node *h) { while (h) { struct node *n = h->next; free(h); h = n; } }
"#,
    ),
    (
        "util.c",
        r#"int count_char(const char *s, char c) { int n = 0; for (; *s; s++) if (*s == c) n++; return n; }
unsigned hash(const char *s) { unsigned h = 5381; while (*s) h = h * 33 + (unsigned char)*s++; return h; }
"#,
    ),
    (
        "main.c",
        r#"#include <stdio.h>
struct node; struct node *push(struct node *, int); int total(struct node *); void freeall(struct node *);
int count_char(const char *, char); unsigned hash(const char *);
static const int weights[4] = {3, 1, 4, 1};
int main(int argc, char **argv) {
    struct node *h = 0;
    for (int i = 1; i < argc; i++) h = push(h, count_char(argv[i], 'a') * weights[i & 3]);
    printf("total=%d hash=%u\n", total(h), argc > 1 ? hash(argv[1]) : 0);
    freeall(h);
    return argc - 1;
}
"#,
    ),
];

fn run(cmd: &mut Command) -> (String, i32) {
    let out = cmd.output().unwrap();
    (String::from_utf8_lossy(&out.stdout).into_owned(), out.status.code().unwrap_or(-1))
}

#[test]
fn cargo_project_runs_like_the_original() {
    if !have("cc") || !have("cargo") {
        eprintln!("cargo: no C compiler or cargo, skipping");
        return;
    }
    let dir = scratch("cargo");
    let mut cc = Command::new("cc");
    cc.args(["-O2", "-g", "-o"]).arg(dir.join("prog"));
    for (name, src) in SOURCES {
        std::fs::write(dir.join(name), src).unwrap();
        cc.arg(dir.join(name));
    }
    assert!(cc.status().unwrap().success());
    let args = ["banana", "apple", "aardvark"];
    let want = run(Command::new(dir.join("prog")).args(args));
    assert_eq!(want.1, 3);

    let bin = env!("CARGO_BIN_EXE_chungusite");
    for mode in ["fast", "safe"] {
        let project = dir.join(mode);
        let out = Command::new(bin).arg(dir.join("prog")).args(["--mode", mode, "--cargo"]).arg(&project).output().unwrap();
        assert!(out.status.success(), "chungusite failed:\n{}", String::from_utf8_lossy(&out.stderr));
        let modules = project.join("src/decompiled");
        for m in ["mod.rs", "list_c.rs", "util_c.rs", "main_c.rs", "crt.rs", "ffi.rs", "data.rs"] {
            assert!(modules.join(m).exists(), "{mode}: no {m}");
        }
        let main_c = std::fs::read_to_string(modules.join("main_c.rs")).unwrap();
        assert!(main_c.contains("fn main"), "{mode}:\n{main_c}");
        assert!(std::fs::read_to_string(modules.join("list_c.rs")).unwrap().contains("fn total"));

        let target = dir.join("target");
        let build = Command::new("cargo")
            .args(["build", "--quiet", "--offline", "--manifest-path"])
            .arg(project.join("Cargo.toml"))
            .env("CARGO_TARGET_DIR", &target)
            .output()
            .unwrap();
        assert!(build.status.success(), "{mode}: cargo build failed:\n{}", String::from_utf8_lossy(&build.stderr));
        assert!(build.stderr.is_empty(), "{mode}: warnings:\n{}", String::from_utf8_lossy(&build.stderr));
        let got = run(Command::new(target.join("debug/prog")).args(args));
        assert_eq!(got, want, "{mode}");
    }

    // Running it again replaces the project; a Cargo.toml it didn't write stays.
    let out = Command::new(bin).arg(dir.join("prog")).arg("--cargo").arg(dir.join("fast")).output().unwrap();
    assert!(out.status.success());
    let other = dir.join("other");
    std::fs::create_dir_all(&other).unwrap();
    std::fs::write(other.join("Cargo.toml"), "[package]\nname = \"mine\"\n").unwrap();
    let out = Command::new(bin).arg(dir.join("prog")).arg("--cargo").arg(&other).output().unwrap();
    assert!(!out.status.success());
    assert!(std::fs::read_to_string(other.join("Cargo.toml")).unwrap().contains("mine"));
    let _ = std::fs::remove_dir_all(Path::new(&dir));
}

/// `-mcmodel=medium`: arrays over 64 KiB go in `.lbss`/`.ldata`, which the code
/// reaches as the GOT's address plus a 64-bit offset; one is over 1 GB, so the
/// project needs the medium code model too. `sum` takes 80 stack arguments.
const LARGE: &str = r#"#include <stdio.h>
#include <stdint.h>
static uint32_t big[40000];
uint32_t init[20000] = {1, 2, 3, 4, 5};
static uint8_t huge[1200u << 20];
uint64_t small = 7;
#define A8(p) long p##0, long p##1, long p##2, long p##3, long p##4, long p##5, long p##6, long p##7
#define S8(p) p##0 + p##1 + p##2 + p##3 + p##4 + p##5 + p##6 + p##7
#define V8(n) n, n + 1, n + 2, n + 3, n + 4, n + 5, n + 6, n + 7
__attribute__((noinline)) long sum(long a, long b, long c, long d, long e, long f,
    A8(g), A8(h), A8(i), A8(j), A8(k), A8(l), A8(m), A8(n), A8(o), A8(p)) {
    return a + b + c + d + e + f + S8(g) + S8(h) + S8(i) + S8(j) + S8(k) + S8(l) + S8(m) + S8(n) + S8(o) + S8(p) * 3;
}
__attribute__((noinline)) void fill(int n) { for (int i = 0; i < n; i++) big[i] = i * 3 + init[i % 5]; }
int main(int argc, char **argv) {
    fill(40000);
    huge[sizeof huge - argc] = 9;
    uint64_t s = small;
    for (int i = 0; i < 40000; i++) s += big[i];
    long t = sum(1, 2, 3, 4, 5, 6, V8(10), V8(20), V8(30), V8(40), V8(50), V8(60), V8(70), V8(80), V8(90), V8(100));
    printf("%lu %u %d %ld\n", (unsigned long)s, init[4], huge[sizeof huge - 1], t);
    return 0;
}
"#;

#[test]
fn medium_code_model_data() {
    if !have("cc") || !have("cargo") {
        eprintln!("cargo: no C compiler or cargo, skipping");
        return;
    }
    let dir = scratch("medium");
    std::fs::write(dir.join("large.c"), LARGE).unwrap();
    let bin = env!("CARGO_BIN_EXE_chungusite");
    for (opt, name) in [("-O0", "large0"), ("-O2", "large2")] {
        let prog = dir.join(name);
        let cc = Command::new("cc").args([opt, "-s", "-mcmodel=medium", "-o"]).arg(&prog).arg(dir.join("large.c")).status().unwrap();
        assert!(cc.success());
        let want = run(&mut Command::new(&prog));
        assert_eq!(want, ("2400060007 5 9 4915\n".to_string(), 0), "{opt}: the original");

        let project = dir.join(format!("p{name}"));
        let out = Command::new(bin).arg(&prog).arg("--cargo").arg(&project).output().unwrap();
        assert!(out.status.success(), "chungusite failed:\n{}", String::from_utf8_lossy(&out.stderr));
        assert!(std::fs::read_to_string(project.join(".cargo/config.toml")).unwrap().contains("code-model=medium"));
        let target = dir.join("target");
        let build = Command::new("cargo")
            .args(["build", "--quiet", "--offline", "--manifest-path"])
            .arg(project.join("Cargo.toml"))
            .env("CARGO_TARGET_DIR", &target)
            .output()
            .unwrap();
        assert!(build.status.success(), "{opt}: cargo build failed:\n{}", String::from_utf8_lossy(&build.stderr));
        let got = run(&mut Command::new(target.join("debug").join(name)));
        assert_eq!(got, want, "{opt}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A constructor picks the function a pointer calls (xz picks its CRCs so),
/// and the call through it passes three stack arguments, two of the registers
/// forwarded from the caller's own entry; the callee never reads r9.
const CTOR: &str = r#"#include <stdio.h>
typedef long (*fn9)(void *, long, long, long *, long, long *, long *, long, int);
struct coder { void *c; fn9 code; };
__attribute__((noinline)) long impl(void *c, long a, long b, long *p, long d, long *o, long *op, long os, int act) {
    *op += os + act + a + b + d;
    return *p + (long)c;
}
static struct coder co;
__attribute__((constructor)) static void pick(void) { co.c = (void *)3; co.code = impl; }
__attribute__((noinline)) long block(struct coder *k, long a, long b, long *p, long d, long *o, long *op, long os, int act) {
    long r = k->code(k->c, a, b, p, d, o, op, os, act);
    return r + *op;
}
int main(int argc, char **argv) {
    long p = 5, o = 0, op = 1;
    long r = block(&co, argc, 2, &p, 4, &o, &op, 100, argc + 6);
    printf("%ld %ld\n", r, op);
    return 0;
}
"#;

#[test]
fn constructors_and_stack_arguments_through_a_pointer() {
    if !have("cc") || !have("cargo") {
        eprintln!("cargo: no C compiler or cargo, skipping");
        return;
    }
    let dir = scratch("ctor");
    std::fs::write(dir.join("ctor.c"), CTOR).unwrap();
    let bin = env!("CARGO_BIN_EXE_chungusite");
    for (opt, name) in [("-O0", "ctor0"), ("-O2", "ctor2")] {
        let prog = dir.join(name);
        assert!(Command::new("cc").args([opt, "-s", "-o"]).arg(&prog).arg(dir.join("ctor.c")).status().unwrap().success());
        let want = run(&mut Command::new(&prog));
        assert_eq!(want, ("123 115\n".to_string(), 0), "{opt}: the original");
        for mode in ["fast", "safe"] {
            let project = dir.join(format!("p{name}{mode}"));
            let out = Command::new(bin).arg(&prog).args(["--mode", mode, "--cargo"]).arg(&project).output().unwrap();
            assert!(out.status.success(), "chungusite failed:\n{}", String::from_utf8_lossy(&out.stderr));
            let target = dir.join("target");
            let build = Command::new("cargo")
                .args(["build", "--quiet", "--offline", "--manifest-path"])
                .arg(project.join("Cargo.toml"))
                .env("CARGO_TARGET_DIR", &target)
                .output()
                .unwrap();
            assert!(build.status.success(), "{opt} {mode}: cargo build failed:\n{}", String::from_utf8_lossy(&build.stderr));
            let got = run(&mut Command::new(target.join("debug").join(name)));
            assert_eq!(got, want, "{opt} {mode}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
