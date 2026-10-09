//! `--emit dataset`: a C library compiled with debug info gives one JSON line per
//! argument and return value, labeled with the declared C type and named after
//! the source, with the text `--refine` would show the model.
use std::process::Command;

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok_and(|o| o.status.success())
}

#[test]
fn dataset_labels_come_from_the_prototypes() {
    if !have("cc") {
        eprintln!("dataset: no C compiler, skipping");
        return;
    }
    let dir = std::env::temp_dir().join(format!("chungusite-test-{}-dataset", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("u.c"),
        "int count_char(const char *s, char c) { int n = 0; for (; *s; s++) if (*s == c) n++; return n; }\n\
         unsigned hash(const char *s) { unsigned h = 5381; while (*s) h = h * 33 + (unsigned char)*s++; return h; }\n",
    )
    .unwrap();
    let lib = dir.join("u.so");
    let cc = Command::new("cc").args(["-O1", "-g", "-shared", "-fPIC", "-o"]).arg(&lib).arg(dir.join("u.c")).status().unwrap();
    assert!(cc.success());

    let bin = env!("CARGO_BIN_EXE_chungusite");
    let out = Command::new(bin).arg(&lib).args(["--emit", "dataset", "-f", "count_char", "-f", "hash"]).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    let rows: Vec<&str> = text.lines().collect();
    let row = |func: &str, var: &str| {
        let key = format!("\"func\": \"{func}\", ");
        let var = format!("\"var\": \"{var}\", ");
        *rows.iter().find(|r| r.contains(&key) && r.contains(&var)).unwrap_or_else(|| panic!("no {func} {var} in\n{text}"))
    };
    assert_eq!(rows.len(), 5, "{text}");
    for (func, var, label, name) in [
        ("count_char", "arg0", "char *", "\"s\""),
        ("count_char", "arg1", "char", "\"c\""),
        ("count_char", "ret", "int", "null"),
        ("hash", "arg0", "char *", "\"s\""),
        ("hash", "ret", "unsigned int", "null"),
    ] {
        let r = row(func, var);
        assert!(r.contains(&format!("\"label\": \"{label}\", \"name\": {name}, ")), "{r}");
        // the row ends with the function's text and the question about one value
        let value = r.split("\"value\": \"").nth(1).unwrap().split('"').next().unwrap();
        assert!(r.ends_with(&format!("\\nvar {value}\"}}")), "{r}");
    }
    assert!(String::from_utf8_lossy(&out.stderr).contains("5 examples from 2 functions"));

    let no_dwarf = Command::new(bin).arg(&lib).args(["--emit", "dataset", "--no-dwarf"]).output().unwrap();
    assert_eq!(no_dwarf.status.code(), Some(2));
    std::fs::remove_dir_all(&dir).ok();
}
