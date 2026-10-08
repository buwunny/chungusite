//! The last safe-mode stage (docs/ownership.md, stage 7): compile the output with
//! rustc and report the functions it rejects, so the caller can emit them again
//! in fast mode, which always compiles.
//!
//! One crate with every function takes rustc minutes on a large binary, so the
//! functions are checked in batches, in parallel. A batch holds the shared items
//! (the `ffi` declarations and the statics), the batch's functions in full, and a
//! stub (`{ loop {} }` body) for every other function, so calls across batches
//! still resolve. Each error is mapped back, through its primary span's line, to
//! the function it is in.
use rayon::prelude::*;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Functions per rustc invocation.
pub const BATCH: usize = 400;

/// What rustc said about the output.
#[derive(Debug, Default)]
pub struct Report {
    /// Indices (into the `funcs` given to `check`) of functions with errors.
    pub failing: Vec<usize>,
    /// The first error in each failing function, as rendered by rustc.
    pub messages: Vec<(usize, String)>,
    /// Errors outside any function (in the shared items), as rendered by rustc.
    pub other: Vec<String>,
}

/// Check `funcs` (each a complete `pub fn`), given the shared items `shared`
/// that they may refer to (`mod ffi`, statics). Scratch files go in `dir`.
pub fn check(shared: &str, funcs: &[&str], dir: &Path) -> std::io::Result<Report> {
    std::fs::create_dir_all(dir)?;
    let stubs: Vec<String> = funcs.iter().map(|f| stub(f)).collect();
    let batches: Vec<(usize, usize)> = (0..funcs.len()).step_by(BATCH).map(|s| (s, (s + BATCH).min(funcs.len()))).collect();
    let results: Vec<std::io::Result<Report>> = batches
        .par_iter()
        .enumerate()
        .map(|(b, &(lo, hi))| {
            let mut src = String::from(HEADER);
            src.push_str(shared);
            for (i, s) in stubs.iter().enumerate() {
                if i < lo || i >= hi {
                    src.push_str(s);
                }
            }
            // line ranges of the functions checked in full
            let mut ranges = Vec::new();
            for (i, f) in funcs.iter().enumerate().take(hi).skip(lo) {
                let first = src.lines().count() + 1;
                src.push_str(f);
                if !f.ends_with('\n') {
                    src.push('\n');
                }
                ranges.push((first, src.lines().count(), i));
            }
            let path = dir.join(format!("batch{b}.rs"));
            std::fs::write(&path, &src)?;
            let out = rustc(&path, dir)?;
            let mut r = Report::default();
            for (line, rendered) in errors(&out) {
                match ranges.iter().find(|&&(a, z, _)| line >= a && line <= z) {
                    Some(&(_, _, i)) => {
                        if !r.failing.contains(&i) {
                            r.messages.push((i, rendered));
                        }
                        r.failing.push(i);
                    }
                    None => r.other.push(rendered),
                }
            }
            Ok(r)
        })
        .collect();
    let mut report = Report::default();
    for r in results {
        let r = r?;
        report.failing.extend(r.failing);
        report.messages.extend(r.messages);
        report.other.extend(r.other);
    }
    report.failing.sort_unstable();
    report.failing.dedup();
    report.other.sort();
    report.other.dedup();
    Ok(report)
}

const HEADER: &str = "#![allow(unused_mut, unused_variables, unused_assignments, unreachable_code, non_snake_case, \
non_upper_case_globals, unused_parens, unused_unsafe, dead_code, clippy::all)]\n";

/// Every function's signature (a function's source may hold its raw twin too),
/// with a body that type-checks as anything.
fn stub(f: &str) -> String {
    let mut out = String::new();
    for sig in f.lines().filter(|l| l.starts_with("pub ")) {
        let _ = writeln!(out, "{sig} loop {{}} }}");
    }
    out
}

/// rustc's JSON diagnostics for one file.
fn rustc(path: &Path, dir: &Path) -> std::io::Result<String> {
    let out: PathBuf = path.with_extension("rmeta");
    let o = Command::new(std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into()))
        .args(["--edition", "2021", "--crate-type", "lib", "--emit=metadata", "--error-format=json", "--cap-lints", "allow"])
        .arg("-o")
        .arg(&out)
        .arg(path)
        .current_dir(dir)
        .output()?;
    Ok(String::from_utf8_lossy(&o.stderr).into_owned())
}

/// `(line of the primary span, rendered message)` for every error.
fn errors(json: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    for l in json.lines() {
        if !l.contains("\"level\":\"error\"") || !l.contains("\"$message_type\":\"diagnostic\"") {
            continue;
        }
        // the summary line at the end
        if l.contains("\"message\":\"aborting due to") {
            continue;
        }
        let spans = l.split("\"spans\":[").nth(1).and_then(|s| s.split("],\"children\"").next()).unwrap_or("");
        let rendered = l.split("\"rendered\":\"").nth(1).map_or(String::new(), |r| {
            r.split("\"}").next().unwrap_or(r).replace("\\n", "\n").replace("\\\"", "\"")
        });
        let mut line = None;
        for span in spans.split("{\"file_name\"").skip(1) {
            if span.contains("\"is_primary\":true") {
                line = number_after(span, "\"line_start\":");
                break;
            }
        }
        match line {
            Some(n) => out.push((n, rendered)),
            // no span: an error about the crate as a whole
            None => out.push((0, rendered)),
        }
    }
    out
}

fn number_after(s: &str, key: &str) -> Option<usize> {
    let rest = &s[s.find(key)? + key.len()..];
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// The first line of a few errors, for the summary.
pub fn describe(errors: &[String]) -> String {
    let mut s = String::new();
    for e in errors.iter().take(5) {
        let first = e.lines().next().unwrap_or("");
        let _ = writeln!(s, "    {first}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_map_to_the_function_they_are_in() {
        let ok = "pub fn ok(x: &mut [u8]) -> u64 {\n    x[0] = 1;\n    0\n}\n";
        let bad = "pub fn bad(x: &mut [u8]) -> u64 {\n    let a = &mut x[0..];\n    let b = &mut x[1..];\n    a[0] = b[0];\n    0\n}\n";
        let calls = "pub fn calls(x: &mut [u8]) -> u64 {\n    ok(x) + bad(x) + twin(0)\n}\n\npub fn twin(x: u64) -> u64 {\n    x\n}\n";
        let dir = std::env::temp_dir().join(format!("chungusite-check-test-{}", std::process::id()));
        let r = check("", &[ok, bad, calls], &dir).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(r.failing, vec![1], "{r:?}");
        assert!(r.other.is_empty(), "{r:?}");
    }
}
