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
