//! Differential testing (docs/roadmap.md, item 9): does the decompiled Rust compute
//! the same thing as the machine code it came from?
//!
//! `tests/differential/corpus.c` is compiled with every C compiler found (`cc`,
//! `clang`) at -O1, -O2 and -Os, each once without and once with debug info (`+g`,
//! where types come from DWARF instead of inference). The object file is also
//! linked into a program (`-nostartfiles`), so that its data and jump tables sit at
//! real addresses. Each function in that program is lifted, cleaned and emitted in
//! both modes, exactly as the CLI does, with the data it points into as statics. A
//! generated Rust program links the original object file next to the decompiled
//! source and calls both on the same random inputs, comparing return values and
//! every byte of every buffer argument afterwards. Each (function, mode) runs in
//! its own process, so a crash or an infinite loop in decompiled code is reported
//! against that function alone.
//!
//! A function that doesn't lift yet is counted, not failed: the pass rate printed at
//! the end is the share of (compiler, -O level, function) combinations that lift and
//! agree with the original in both modes. A function that lifts but computes a
//! different result, panics, crashes or hangs fails the test, unless `KNOWN_BAD`
//! lists it with the reason.
//!
//! Environment knobs:
//! * `CHUNGUSITE_DIFF_CC="gcc clang-18"`: compilers to use (default: `cc` and `clang`, whichever exist).
//! * `CHUNGUSITE_DIFF_TRIALS=N`: random inputs per function and mode (default 500).
//! * `CHUNGUSITE_DIFF_SEED=N`: change the inputs (default fixed, so failures reproduce).
//! * `CHUNGUSITE_DIFF_ONLY=name`: only functions whose name contains this.
//!
//! The full table is written to `$CARGO_TARGET_TMPDIR/differential/report.txt`.
use chungusite::{
    emit::Mode,
    globals::{Globals, PRELUDE},
    load::Binary,
    program::{BuildOptions, Input, Options, Program},
};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Decompiled code that is known to compute the wrong thing, as
/// `("compiler/-Olevel/function/mode" substring, reason)`. Each entry is a bug to fix;
/// remove it once the case passes (the test prints a note when it does).
const KNOWN_BAD: &[(&str, &str)] = &[];

const OPT_LEVELS: [&str; 3] = ["-O1", "-O2", "-Os"];
const TIMEOUT: Duration = Duration::from_secs(30);

// ---------------------------------------------------------------------------
// The corpus: `// @diff name: ret(arg, ...)` lines in corpus.c

#[derive(Clone, Copy, Debug, PartialEq)]
enum Int {
    I8, U8, I16, U16, I32, U32, I64, U64,
}

impl Int {
    fn parse(s: &str) -> Option<Int> {
        Some(match s {
            "i8" => Int::I8, "u8" => Int::U8, "i16" => Int::I16, "u16" => Int::U16,
            "i32" => Int::I32, "u32" => Int::U32, "i64" => Int::I64, "u64" => Int::U64,
            _ => return None,
        })
    }
    fn bits(self) -> u32 {
        match self {
            Int::I8 | Int::U8 => 8,
            Int::I16 | Int::U16 => 16,
            Int::I32 | Int::U32 => 32,
            Int::I64 | Int::U64 => 64,
        }
    }
    fn signed(self) -> bool {
        matches!(self, Int::I8 | Int::I16 | Int::I32 | Int::I64)
    }
    fn rust(self) -> &'static str {
        match self {
            Int::I8 => "i8", Int::U8 => "u8", Int::I16 => "i16", Int::U16 => "u16",
            Int::I32 => "i32", Int::U32 => "u32", Int::I64 => "i64", Int::U64 => "u64",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Arg {
    Int(Int, Option<(i128, i128)>),
    /// `f32` or `f64` (bits).
    Float(u32),
    Buf(usize),
    Str(usize),
}

#[derive(Clone, Debug, PartialEq)]
enum Ret {
    Void,
    Bool,
    Int(Int),
    Float(u32),
    Ptr(usize),
}

#[derive(Clone, Debug)]
struct Case {
    name: String,
    ret: Ret,
    args: Vec<Arg>,
}

fn parse_corpus(src: &str) -> Vec<Case> {
    let mut cases = Vec::new();
    for (n, line) in src.lines().enumerate() {
        let Some(spec) = line.trim().strip_prefix("// @diff ") else { continue };
        let bad = |what: &str| -> ! { panic!("corpus.c:{}: {what}: {line}", n + 1) };
        let (name, sig) = spec.split_once(':').unwrap_or_else(|| bad("expected `name: ret(args)`"));
        let (ret, args) = sig.trim().split_once('(').unwrap_or_else(|| bad("missing `(`"));
        let args = args.trim().strip_suffix(')').unwrap_or_else(|| bad("missing `)`"));
        let ret = match ret.trim() {
            "void" => Ret::Void,
            "bool" => Ret::Bool,
            "f32" => Ret::Float(32),
            "f64" => Ret::Float(64),
            r => match r.strip_prefix("ptr:") {
                Some(k) => Ret::Ptr(k.parse().unwrap_or_else(|_| bad("bad ptr:K"))),
                None => Ret::Int(Int::parse(r).unwrap_or_else(|| bad("bad return type"))),
            },
        };
        let args = args
            .split(',')
            .map(str::trim)
            .filter(|a| !a.is_empty())
            .map(|a| {
                if let Some(n) = a.strip_prefix("buf:") {
                    return Arg::Buf(n.parse().unwrap_or_else(|_| bad("bad buf:N")));
                }
                if let Some(n) = a.strip_prefix("str:") {
                    return Arg::Str(n.parse().unwrap_or_else(|_| bad("bad str:N")));
                }
                match a {
                    "f32" => return Arg::Float(32),
                    "f64" => return Arg::Float(64),
                    _ => {}
                }
                let (ty, range) = match a.split_once(':') {
                    Some((ty, r)) => {
                        let (lo, hi) = r.split_once("..").unwrap_or_else(|| bad("bad range"));
                        let lo: i128 = lo.parse().unwrap_or_else(|_| bad("bad range"));
                        let hi: i128 = hi.parse().unwrap_or_else(|_| bad("bad range"));
                        assert!(lo < hi, "corpus.c:{}: empty range", n + 1);
                        (ty, Some((lo, hi)))
                    }
                    None => (a, None),
                };
                Arg::Int(Int::parse(ty).unwrap_or_else(|| bad("bad argument type")), range)
            })
            .collect::<Vec<_>>();
        if let Ret::Ptr(k) = ret {
            assert!(matches!(args.get(k), Some(Arg::Buf(_) | Arg::Str(_))), "corpus.c:{}: ptr:{k} must name a buffer argument", n + 1);
        }
        cases.push(Case { name: name.trim().to_string(), ret, args });
    }
    cases
}

// ---------------------------------------------------------------------------
// Decompiling the linked corpus

/// One function's decompiled source, or why it didn't lift.
struct Decompiled {
    case: Case,
    result: Result<[String; 2], String>,
}

const SYSV: [&str; 6] = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"];

/// Decompile the program the way the CLI does: every function, so calls see their
/// callees' signatures, with the corpus cases selected for emission, and data the
/// code points into as statics. Also returns each mode's prelude (the `extern "C"`
/// block for callees that aren't emitted, and the statics).
fn decompile(file: &[u8], cases: &[Case], dwarf: bool) -> (Vec<Decompiled>, [String; 2]) {
    let bin = Binary::parse(file).expect("compiler output should parse");
    let inputs = bin
        .funcs
        .iter()
        .map(|fb| {
            // A case's `.cold` part is a function of its own that the case jumps to.
            let name = fb.name.strip_suffix(".cold").unwrap_or(&fb.name);
            let selected = cases.iter().any(|c| c.name == name);
            Input { name: fb.name.clone(), ident: fb.name.replace(".", "_"), addr: fb.addr, bytes: fb.bytes, selected }
        })
        .collect();
    let used: HashSet<String> = bin.funcs.iter().map(|f| f.name.clone()).collect();
    let by_addr: HashMap<u64, &str> = bin.funcs.iter().map(|f| (f.addr, f.name.as_str())).collect();
    let globals = Globals::new(&bin, used, &by_addr);
    let global_of = |addr: u64| globals.expr(addr);
    let global_end = |base: u64, at: u64| globals.end_expr(base, at);
    let program = Program::build_with(inputs, Some(file), false, BuildOptions { dwarf, model: None });
    let opts = Options { global_of: &global_of, global_end: &global_end, ..Options::default() };
    let [fast, safe] = [Mode::Fast, Mode::Safe].map(|m| program.emit_all_with(m, &opts));
    let decompiled = cases
        .iter()
        .map(|case| {
            let result = match program.funcs.iter().position(|f| f.name == case.name) {
                None => Err("no such symbol in the object file".to_string()),
                Some(i) => match &program.funcs[i].ir {
                    // An IR that fails verification is a lifter bug, not a gap: fail loudly.
                    Err(e) if e.contains("failed verification") => panic!("{}: {e}", case.name),
                    Err(e) => Err(match e.strip_prefix("unsupported instruction ") {
                        Some(rest) => format!("unsupported {}", rest.split(' ').next().unwrap_or(rest)),
                        None => e.split(" at ").next().unwrap_or(e).to_string(),
                    }),
                    Ok(_) => Ok([fast[i].as_ref().unwrap().0.clone(), safe[i].as_ref().unwrap().0.clone()]),
                },
            };
            Decompiled { case: case.clone(), result }
        })
        .collect();
    let mut prelude = program.prelude();
    let mut parts = [String::new(), String::new()];
    for (i, f) in program.funcs.iter().enumerate() {
        if f.selected && !cases.iter().any(|c| c.name == f.name) {
            for (part, out) in parts.iter_mut().zip([&fast, &safe]) {
                part.push_str(&out[i].as_ref().expect("a selected function that lifts").0);
            }
        }
    }
    // The statics the emitted functions use, and the ones those point to.
    let mut statics = BTreeSet::new();
    for f in program.funcs.iter().filter(|f| f.selected) {
        if let Ok(ir) = &f.ir {
            let mut items = Vec::new();
            globals.referenced(ir, &mut items);
            statics.extend(items);
        }
    }
    if !statics.is_empty() {
        prelude.push_str(PRELUDE);
        let mut todo: Vec<_> = statics.iter().copied().collect();
        let mut more = Vec::new();
        while !todo.is_empty() {
            for item in &todo {
                globals.emit_static(item, &mut prelude, &mut more);
            }
            more.sort();
            more.dedup();
            todo = more.drain(..).filter(|i| statics.insert(*i)).collect();
        }
    }
    let [fast_parts, safe_parts] = parts;
    (decompiled, [prelude.clone() + &fast_parts, prelude + &safe_parts])
}

// ---------------------------------------------------------------------------
// Generating the runner program

/// How to pass each parameter of an emitted function, from its signature line, e.g.
/// `pub unsafe fn get8(mut rdi: u64, rsi: i32) -> u64 {`. Parameters are named after
/// their register (`rdi`, `rdi_ref`, `rdi_p`, `arg6`) or, with debug info, after
/// the C parameter, in order. Errs when the harness can't call it (a slice for an
/// argument that the C code takes as an integer).
fn call_args(src: &str, case: &Case) -> Result<String, String> {
    let sig = src.lines().find(|l| l.starts_with("pub ")).ok_or("no signature in emitted code")?;
    let close = sig.rfind(") -> ").or_else(|| sig.rfind(") {")).ok_or("unexpected signature")?;
    let params = &sig[sig.find('(').unwrap() + 1..close];
    let mut out = Vec::new();
    for (pos, p) in params.split(", ").filter(|p| !p.is_empty()).enumerate() {
        let p = p.strip_prefix("mut ").unwrap_or(p);
        let (name, ty) = p.split_once(": ").ok_or_else(|| format!("unexpected parameter `{p}`"))?;
        // `_rdx`, `_n`: an argument the decompiled function takes but doesn't use
        let name = name.strip_prefix('_').unwrap_or(name);
        // float argument j: the j-th `f32`/`f64` of the C function
        if let Some(j) = name.strip_prefix("xmm").and_then(|j| j.parse::<usize>().ok()) {
            let i = case.args.iter().enumerate().filter(|(_, a)| matches!(a, Arg::Float(_))).nth(j).map(|(i, _)| i);
            out.push(match i {
                Some(i) => format!("f64::from_bits(a.reg({i}))"),
                None => "f64::from_bits(a.junk())".to_string(),
            });
            continue;
        }
        let reg = name.strip_suffix("_ref").or_else(|| name.strip_suffix("_p")).unwrap_or(name);
        let by_reg = SYSV.iter().position(|&r| r == reg).or_else(|| reg.strip_prefix("arg").and_then(|n| n.parse().ok()));
        let named = by_reg.is_none() && reg != "rsp";
        let idx = if named { Some(pos) } else { by_reg }.filter(|&i| i < case.args.len());
        let buf = |i: usize| matches!(case.args[i], Arg::Buf(_) | Arg::Str(_));
        // integer argument registers count the integer arguments only
        let idx = match (named, idx) {
            (false, Some(k)) => case.args.iter().enumerate().filter(|(_, a)| !matches!(a, Arg::Float(_))).nth(k).map(|(i, _)| i),
            _ => idx,
        };
        out.push(match (ty, idx) {
            (_, None) if reg == "rsp" => "a.stack()".to_string(),
            (_, None) => format!("a.junk() as {}", if ty.starts_with('*') { "_" } else { ty }),
            ("&[u8]", Some(i)) if buf(i) => format!("a.shared({i})"),
            ("&mut [u8]", Some(i)) if buf(i) => format!("a.slice({i})"),
            ("Option<&[u8]>", Some(i)) if buf(i) => format!("Some(a.shared({i}))"),
            ("Option<&mut [u8]>", Some(i)) if buf(i) => format!("Some(a.slice({i}))"),
            // a slice of wider elements: the buffer's whole elements
            (t, Some(i)) if buf(i) && t.starts_with("&mut [") => format!("a.elems_mut({i})"),
            (t, Some(i)) if buf(i) && t.starts_with("&[") => format!("a.elems({i})"),
            (t, Some(i)) if buf(i) && t.starts_with("Option<&mut [") => format!("Some(a.elems_mut({i}))"),
            (t, Some(i)) if buf(i) && t.starts_with("Option<&[") => format!("Some(a.elems({i}))"),
            // a struct the buffer holds
            (t, Some(i)) if buf(i) && t.starts_with("&mut ") => format!("&mut *(a.reg({i}) as *mut _)"),
            (t, Some(i)) if buf(i) && t.starts_with('&') => format!("&*(a.reg({i}) as *const _)"),
            (t, Some(i)) if buf(i) && t.starts_with("Option<&mut ") => format!("Some(&mut *(a.reg({i}) as *mut _))"),
            (t, Some(i)) if buf(i) && t.starts_with("Option<&") => format!("Some(&*(a.reg({i}) as *const _))"),
            (t, Some(_)) if t.starts_with('&') || t.starts_with("Option<") => {
                return Err(format!("safe mode takes `{reg}` as `{ty}`, but the C function has no pointer argument there"))
            }
            ("bool", Some(i)) => format!("a.reg({i}) as u8 != 0"),
            (t, Some(i)) if t.starts_with('*') => format!("a.reg({i}) as _"),
            (t, Some(i)) => format!("a.reg({i}) as {t}"),
        });
    }
    Ok(out.join(", "))
}

fn arg_spec(a: &Arg) -> String {
    match a {
        Arg::Int(t, r) => {
            let (lo, hi) = match r {
                Some((lo, hi)) => (*lo, *hi),
                None if t.signed() => (-(1i128 << (t.bits() - 1)), 1i128 << (t.bits() - 1)),
                None => (0, 1i128 << t.bits()),
            };
            format!("A::Int {{ bits: {}, signed: {}, lo: {lo}, hi: {hi}, full: {} }}", t.bits(), t.signed(), r.is_none())
        }
        Arg::Float(b) => format!("A::Float({b})"),
        Arg::Buf(n) => format!("A::Buf({n})"),
        Arg::Str(n) => format!("A::Str({n})"),
    }
}

/// Source of the runner: `run <case> <fast|safe>` checks one function in one mode.
fn runner_source(funcs: &[&Decompiled], prelude: &[String; 2]) -> (String, String, String) {
    let (mut fast, mut safe) = (String::from(ALLOW), String::from(ALLOW));
    fast.push_str(&prelude[0]);
    safe.push_str(&prelude[1]);
    let mut externs = String::new();
    let mut arms = String::new();
    for d in funcs {
        let c = &d.case;
        let [f_src, s_src] = d.result.as_ref().unwrap();
        fast.push_str(f_src);
        safe.push_str(s_src);
        let params: Vec<String> = c
            .args
            .iter()
            .enumerate()
            .map(|(i, a)| match a {
                Arg::Int(t, _) => format!("a{i}: {}", t.rust()),
                Arg::Float(b) => format!("a{i}: f{b}"),
                _ => format!("a{i}: *mut u8"),
            })
            .collect();
        let ret_ty = match &c.ret {
            Ret::Void => String::new(),
            Ret::Bool => " -> u8".into(),
            Ret::Int(t) => format!(" -> {}", t.rust()),
            Ret::Float(b) => format!(" -> f{b}"),
            Ret::Ptr(_) => " -> *mut u8".into(),
        };
        let _ = writeln!(externs, "        pub fn {}({}){ret_ty};", c.name, params.join(", "));
        let c_args: Vec<String> = c
            .args
            .iter()
            .enumerate()
            .map(|(i, a)| match a {
                Arg::Int(t, _) => format!("a.val({i}) as {}", t.rust()),
                Arg::Float(b) => format!("f{b}::from_bits(a.val({i}) as u{b})"),
                _ => format!("a.reg({i}) as *mut u8"),
            })
            .collect();
        let c_call = format!("c::{}({})", c.name, c_args.join(", "));
        let c_call = match c.ret {
            Ret::Void => format!("{{ {c_call}; 0 }}"),
            Ret::Float(_) => format!("{c_call}.to_bits() as u64"),
            _ => format!("{c_call} as u64"),
        };
        let ret = match c.ret {
            Ret::Void => "R::Void".to_string(),
            Ret::Bool => "R::Bits(8)".to_string(),
            Ret::Int(t) => format!("R::Bits({})", t.bits()),
            Ret::Float(b) => format!("R::Bits({b})"),
            Ret::Ptr(k) => format!("R::Ptr({k})"),
        };
        let spec: Vec<String> = c.args.iter().map(arg_spec).collect();
        let call = |m: &str, src: &str| match call_args(src, c) {
            // a function that returns nothing (void) is called for its effects
            Ok(args) if !src.lines().find(|l| l.starts_with("pub ")).unwrap_or("").contains(") -> ") => {
                format!("Ok(|a: &mut Args| unsafe {{ {m}::{}({args}); 0 }})", c.name)
            }
            Ok(args) => format!("Ok(|a: &mut Args| unsafe {{ Ret::bits({m}::{}({args})) }})", c.name),
            Err(e) => format!("Err({e:?})"),
        };
        let _ = writeln!(
            arms,
            "        ({n:?}, \"fast\") => check(&[{spec}], {ret}, |a: &mut Args| unsafe {{ {c_call} }}, {fcall}),\n        ({n:?}, \"safe\") => check(&[{spec}], {ret}, |a: &mut Args| unsafe {{ {c_call} }}, {scall}),",
            n = c.name,
            spec = spec.join(", "),
            fcall = call("fast", f_src),
            scall = call("safe", s_src),
        );
    }
    let main = RUNNER.replace("@EXTERNS@", &externs).replace("@ARMS@", &arms);
    (main, fast, safe)
}

const ALLOW: &str = "#[allow(unused_mut, unused_variables, unused_assignments, unreachable_code, non_snake_case, non_camel_case_types, unused_parens, unused_unsafe, dead_code)]\n";

const RUNNER: &str = r##"#![allow(unused_unsafe, unused_parens, dead_code, unreachable_code)]
mod fast { include!("fast.rs"); }
mod safe { include!("safe.rs"); }
mod c {
    extern "C" {
@EXTERNS@    }
}

use std::panic::{catch_unwind, AssertUnwindSafe};

/// A decompiled function's return value as the bits it leaves in rax.
trait Ret { fn bits(self) -> u64; }
macro_rules! ret { ($($t:ty => $u:ty),*) => { $(impl Ret for $t { fn bits(self) -> u64 { self as $u as u64 } })* } }
ret!(u8 => u8, u16 => u16, u32 => u32, u64 => u64, i8 => u8, i16 => u16, i32 => u32, i64 => u64, bool => u8);
impl<T> Ret for *mut T { fn bits(self) -> u64 { self as u64 } }
impl<T> Ret for *const T { fn bits(self) -> u64 { self as u64 } }
impl Ret for (u64, u64) { fn bits(self) -> u64 { self.0 } }
/// A float result: an f32's bits are the low half, as in xmm0.
impl Ret for f64 { fn bits(self) -> u64 { self.to_bits() } }

enum A { Int { bits: u32, signed: bool, lo: i128, hi: i128, full: bool }, Float(u32), Buf(usize), Str(usize) }
#[derive(Clone, Copy)]
enum R { Void, Bits(u32), Ptr(usize) }

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 { self.next() % n }
}

/// One set of inputs. Each side of the comparison gets its own clone, so its own buffers.
#[derive(Clone)]
struct Args {
    /// The C value of each integer argument, sign- or zero-extended to 64 bits.
    vals: Vec<u64>,
    /// What the decompiled code sees in the argument register: the C value, with
    /// random bits above 32 for narrower types (the ABI leaves them undefined).
    regs: Vec<u64>,
    bufs: Vec<Vec<u8>>,
    stack: Vec<u8>,
    junk: u64,
}

impl Args {
    fn gen(rng: &mut Rng, spec: &[A]) -> Args {
        let mut a = Args { vals: vec![], regs: vec![], bufs: vec![], stack: vec![0xcc; 8192], junk: rng.next() | 1 };
        for s in spec {
            let (v, buf) = match *s {
                A::Int { bits, signed, lo, hi, full } => {
                    let v: i128 = if full && rng.below(3) == 0 {
                        // edge values
                        let edges = [0i128, 1, 2, -1, 7, 8, 63, 64, 97, 255, lo, hi - 1, lo + 1, hi - 2, (lo + hi) / 2];
                        edges[rng.below(edges.len() as u64) as usize].clamp(lo, hi - 1)
                    } else if full && rng.below(2) == 0 {
                        (rng.below(300) as i128).clamp(lo, hi - 1) * if signed && rng.below(2) == 0 { -1 } else { 1 }
                    } else {
                        lo + (rng.next() as u128 % (hi - lo) as u128) as i128
                    };
                    let v = v.clamp(lo, hi - 1);
                    let _ = bits;
                    (v as i64 as u64, Vec::new())
                }
                A::Float(bits) => {
                    // mostly small values with a fraction, some special ones, some random bits
                    let x: f64 = match rng.below(8) {
                        0 => [0.0, -0.0, 1.0, -1.5, 0.5, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1e300, -3e-310][rng.below(10) as usize],
                        1 => f64::from_bits(rng.next()),
                        _ => (rng.below(4001) as f64 - 2000.0) / 8.0,
                    };
                    let v = if bits == 32 { (x as f32).to_bits() as u64 } else { x.to_bits() };
                    (v, Vec::new())
                }
                A::Buf(n) => {
                    let mut b: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
                    // sometimes small values, so data-dependent branches go both ways
                    if rng.below(2) == 0 {
                        for x in b.iter_mut() { *x %= 8; }
                    }
                    (0, b)
                }
                A::Str(n) => {
                    const ABC: &[u8] = b"ab-0129";
                    let len = (rng.below(n as u64) as usize).min(if rng.below(2) == 0 { 4 } else { n - 1 });
                    let mut b: Vec<u8> = (0..n).map(|_| (rng.next() as u8) | 1).collect();
                    for x in &mut b[..len] { *x = ABC[rng.below(ABC.len() as u64) as usize]; }
                    b[len] = 0;
                    (0, b)
                }
            };
            let reg = match *s {
                A::Int { bits, .. } if bits < 64 => (v & 0xffff_ffff) | (rng.next() << 32),
                // the rest of an f32's register is undefined too
                A::Float(32) => (v & 0xffff_ffff) | (rng.next() << 32),
                _ => v,
            };
            a.vals.push(v);
            a.regs.push(reg);
            a.bufs.push(buf);
        }
        a
    }
    fn val(&mut self, i: usize) -> u64 { self.vals[i] }
    fn reg(&mut self, i: usize) -> u64 {
        if self.bufs[i].is_empty() { self.regs[i] } else { self.bufs[i].as_mut_ptr() as u64 }
    }
    fn slice(&mut self, i: usize) -> &'static mut [u8] {
        // The buffers outlive the call and no argument names the same one twice.
        unsafe { std::slice::from_raw_parts_mut(self.bufs[i].as_mut_ptr(), self.bufs[i].len()) }
    }
    fn shared(&mut self, i: usize) -> &'static [u8] { self.slice(i) }
    fn elems_mut<T>(&mut self, i: usize) -> &'static mut [T] {
        let b = &mut self.bufs[i];
        assert!(b.as_ptr() as usize % std::mem::align_of::<T>() == 0, "unaligned buffer");
        unsafe { std::slice::from_raw_parts_mut(b.as_mut_ptr() as *mut T, b.len() / std::mem::size_of::<T>()) }
    }
    fn elems<T>(&mut self, i: usize) -> &'static [T] { self.elems_mut(i) }
    fn stack(&mut self) -> u64 { self.stack.as_mut_ptr() as u64 + 4096 }
    fn junk(&mut self) -> u64 { self.junk }
    fn base(&self, i: usize) -> u64 { self.bufs[i].as_ptr() as u64 }
    fn show(&self) -> String {
        let mut s = String::new();
        for i in 0..self.vals.len() {
            if self.bufs[i].is_empty() {
                s += &format!(" a{i}={:#x}", self.vals[i]);
            } else {
                s += &format!(" a{i}={:02x?}", self.bufs[i]);
            }
        }
        s
    }
}

fn ret_key(r: R, a: &Args, v: u64) -> Option<u64> {
    match r {
        R::Void => None,
        R::Bits(64) => Some(v),
        R::Bits(b) => Some(v & ((1u64 << b) - 1)),
        R::Ptr(k) => Some(if v == 0 { u64::MAX } else { v.wrapping_sub(a.base(k)) }),
    }
}

fn check(spec: &[A], ret: R, orig: impl Fn(&mut Args) -> u64, dec: Result<impl Fn(&mut Args) -> u64, &str>) -> Result<String, String> {
    let dec = dec.map_err(|e| format!("can't call: {e}"))?;
    let trials: u64 = std::env::var("TRIALS").ok().and_then(|s| s.parse().ok()).unwrap_or(500);
    let mut rng = Rng(std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(0x9E37_79B9_7F4A_7C15) | 1);
    for t in 0..trials {
        let input = Args::gen(&mut rng, spec);
        let mut want = input.clone();
        let mut got = input.clone();
        let w = orig(&mut want);
        let g = match catch_unwind(AssertUnwindSafe(|| dec(&mut got))) {
            Ok(g) => g,
            Err(p) => {
                let msg = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                return Err(format!("trial {t}: panicked ({msg}) on{}", input.show()));
            }
        };
        let (wk, gk) = (ret_key(ret, &want, w), ret_key(ret, &got, g));
        if wk != gk {
            let show = |k: Option<u64>| k.map_or("-".to_string(), |k| format!("{k:#x}"));
            return Err(format!("trial {t}: returned {}, original {}, on{}", show(gk), show(wk), input.show()));
        }
        for i in 0..spec.len() {
            if want.bufs[i] != got.bufs[i] {
                return Err(format!("trial {t}: a{i} afterwards {:02x?}, original {:02x?}, on{}", got.bufs[i], want.bufs[i], input.show()));
            }
        }
    }
    Ok(format!("{trials} trials"))
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let args: Vec<String> = std::env::args().collect();
    let r = match (args[1].as_str(), args[2].as_str()) {
@ARMS@        _ => Err("no such case".to_string()),
    };
    match r {
        Ok(s) => println!("ok {s}"),
        Err(e) => {
            println!("FAIL {e}");
            std::process::exit(1);
        }
    }
}
"##;

// ---------------------------------------------------------------------------
// Driving it

#[derive(Clone, Debug, PartialEq)]
enum Outcome {
    Pass,
    NotLifted(String),
    Wrong(String),
}

struct Variant {
    label: String,
    cc: String,
    opt: &'static str,
    /// Compile with `-g` and use the debug info.
    dwarf: bool,
}

fn compilers() -> Vec<String> {
    if let Ok(list) = std::env::var("CHUNGUSITE_DIFF_CC") {
        return list.split_whitespace().map(String::from).collect();
    }
    ["cc", "clang"]
        .into_iter()
        .filter(|c| Command::new(c).arg("--version").output().is_ok_and(|o| o.status.success()))
        .map(String::from)
        .collect()
}

/// `cc` is usually gcc under another name; label it by what it is.
fn compiler_label(cc: &str) -> String {
    let out = Command::new(cc).arg("--version").output().unwrap();
    let v = String::from_utf8_lossy(&out.stdout).to_lowercase();
    if cc == "cc" && v.contains("clang") {
        "clang".into()
    } else if cc == "cc" && (v.contains("gcc") || v.contains("free software foundation")) {
        "gcc".into()
    } else {
        Path::new(cc).file_name().unwrap().to_string_lossy().into_owned()
    }
}

fn run_with_timeout(mut cmd: Command) -> Result<(bool, String), String> {
    let mut child = cmd.stdout(Stdio::piped()).stderr(Stdio::null()).spawn().map_err(|e| e.to_string())?;
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            let mut out = String::new();
            std::io::Read::read_to_string(child.stdout.as_mut().unwrap(), &mut out).unwrap();
            if out.is_empty() {
                return Err(format!("crashed ({status})"));
            }
            return Ok((status.success(), out.trim().to_string()));
        }
        if start.elapsed() > TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("timed out after {}s", TIMEOUT.as_secs()));
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn check_variant(v: &Variant, cases: &[Case], corpus: &Path, root: &Path) -> Vec<(String, [Outcome; 2])> {
    let dir = root.join(&v.label);
    std::fs::create_dir_all(&dir).unwrap();
    let obj = dir.join("corpus.o");
    let g: &[&str] = if v.dwarf { &["-g"] } else { &[] };
    let out = Command::new(&v.cc).arg(v.opt).args(g).args(["-c", "-o"]).arg(&obj).arg(corpus).output().unwrap();
    assert!(out.status.success(), "{} {} failed on corpus.c:\n{}", v.cc, v.opt, String::from_utf8_lossy(&out.stderr));

    // Decompile the object linked into a program, as a decompiler would see it:
    // relocations applied, and .rodata (lookup and jump tables) at real addresses.
    // The runner links the object itself, which is the same code.
    let linked = dir.join("corpus.elf");
    let out = Command::new(&v.cc).arg("-nostartfiles").arg("-o").arg(&linked).arg(&obj).output().unwrap();
    assert!(out.status.success(), "{} failed to link corpus.o:\n{}", v.cc, String::from_utf8_lossy(&out.stderr));
    let (decompiled, prelude) = decompile(&std::fs::read(&linked).unwrap(), cases, v.dwarf);
    let lifted: Vec<&Decompiled> = decompiled.iter().filter(|d| d.result.is_ok()).collect();
    let (main, fast, safe) = runner_source(&lifted, &prelude);
    std::fs::write(dir.join("fast.rs"), fast).unwrap();
    std::fs::write(dir.join("safe.rs"), safe).unwrap();
    std::fs::write(dir.join("main.rs"), main).unwrap();
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let bin = dir.join("run");
    let out = Command::new(rustc)
        .args(["--edition", "2021", "-A", "warnings", "-C", "debuginfo=0", "-o"])
        .arg(&bin)
        .arg(dir.join("main.rs"))
        .arg("-C")
        .arg(format!("link-arg={}", obj.display()))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "the decompiled code for {} failed to build (sources in {}):\n{}",
        v.label,
        dir.display(),
        String::from_utf8_lossy(&out.stderr)
    );

    let trials = std::env::var("CHUNGUSITE_DIFF_TRIALS").unwrap_or_else(|_| "500".into());
    let seed = std::env::var("CHUNGUSITE_DIFF_SEED").ok();
    decompiled
        .par_iter()
        .map(|d| {
            let name = d.case.name.clone();
            if let Err(e) = &d.result {
                let o = Outcome::NotLifted(e.clone());
                return (name, [o.clone(), o]);
            }
            let run = |mode: &str| {
                let mut cmd = Command::new(&bin);
                cmd.args([&name, mode]).env("TRIALS", &trials);
                if let Some(s) = &seed {
                    cmd.env("SEED", s);
                }
                match run_with_timeout(cmd) {
                    Ok((true, _)) => Outcome::Pass,
                    Ok((false, out)) => Outcome::Wrong(out.strip_prefix("FAIL ").unwrap_or(&out).to_string()),
                    Err(e) => Outcome::Wrong(e),
                }
            };
            let outcomes = [run("fast"), run("safe")];
            (name, outcomes)
        })
        .collect()
}

#[test]
fn decompiled_c_matches_the_original() {
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/differential/corpus.c");
    let mut cases = parse_corpus(&std::fs::read_to_string(&corpus).unwrap());
    assert!(cases.len() > 40, "corpus.c should have @diff lines");
    if let Ok(only) = std::env::var("CHUNGUSITE_DIFF_ONLY") {
        cases.retain(|c| c.name.contains(&only));
    }

    let ccs = compilers();
    if ccs.is_empty() {
        assert!(std::env::var_os("CI").is_none(), "no C compiler found (tried cc and clang)");
        eprintln!("differential: no C compiler found, skipping");
        return;
    }
    let mut variants = Vec::new();
    for cc in &ccs {
        let label = compiler_label(cc);
        for opt in OPT_LEVELS {
            for dwarf in [false, true] {
                let g = if dwarf { "+g" } else { "" };
                variants.push(Variant { label: format!("{label}{opt}{g}"), cc: cc.clone(), opt, dwarf });
            }
        }
    }

    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("differential");
    let results: Vec<Vec<(String, [Outcome; 2])>> =
        variants.par_iter().map(|v| check_variant(v, &cases, &corpus, &root)).collect();

    // Report: one row per function, one column per variant.
    let mut report = String::new();
    let width = cases.iter().map(|c| c.name.len()).max().unwrap_or(0);
    let _ = write!(report, "{:width$}", "");
    for v in &variants {
        let _ = write!(report, "  {:>9}", v.label);
    }
    report.push('\n');
    let mut failures = Vec::new();
    let mut fixed = Vec::new();
    let (mut pass, mut lifted, mut total) = (0, 0, 0);
    let mut not_lifted: BTreeMap<String, usize> = BTreeMap::new();
    for (ci, c) in cases.iter().enumerate() {
        let _ = write!(report, "{:width$}", c.name);
        for (vi, v) in variants.iter().enumerate() {
            let (name, outcomes) = &results[vi][ci];
            assert_eq!(name, &c.name);
            total += 1;
            let cell = match outcomes {
                [Outcome::Pass, Outcome::Pass] => "ok",
                [Outcome::NotLifted(_), _] => "-",
                [Outcome::Pass, _] => "safe FAIL",
                [_, Outcome::Pass] => "fast FAIL",
                _ => "FAIL",
            };
            let _ = write!(report, "  {cell:>9}");
            match &outcomes[0] {
                Outcome::NotLifted(why) => {
                    *not_lifted.entry(why.clone()).or_default() += 1;
                    continue;
                }
                _ => lifted += 1,
            }
            if cell == "ok" {
                pass += 1;
            }
            for (mode, o) in ["fast", "safe"].iter().zip(outcomes) {
                let id = format!("{}/{}/{mode}", v.label, c.name);
                let known = KNOWN_BAD.iter().find(|(pat, _)| id.contains(pat));
                match (o, known) {
                    (Outcome::Wrong(why), None) => failures.push(format!("{id}: {why}")),
                    (Outcome::Pass, Some((pat, _))) => fixed.push(format!("{id} passes now; drop KNOWN_BAD entry {pat:?}")),
                    _ => {}
                }
            }
        }
        report.push('\n');
    }
    let pct = |n: usize| 100.0 * n as f64 / total.max(1) as f64;
    let _ = writeln!(
        report,
        "\n{pass} of {total} ({:.1}%) pass in both modes; {lifted} lift ({:.1}%), {} lift but disagree with the original.",
        pct(pass),
        pct(lifted),
        lifted - pass
    );
    let mut causes: Vec<_> = not_lifted.into_iter().collect();
    causes.sort_by(|a, b| b.1.cmp(&a.1));
    let _ = writeln!(report, "Not lifted, by cause:");
    for (cause, n) in causes {
        let _ = writeln!(report, "  {n:5}  {cause}");
    }
    for (vi, v) in variants.iter().enumerate() {
        for (name, outcomes) in &results[vi] {
            for (mode, o) in ["fast", "safe"].iter().zip(outcomes) {
                if let Outcome::Wrong(why) = o {
                    let id = format!("{}/{name}/{mode}", v.label);
                    let known = KNOWN_BAD.iter().any(|(pat, _)| id.contains(pat));
                    let _ = writeln!(report, "{id}: {why}{}", if known { " (known bad)" } else { "" });
                }
            }
        }
    }
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("report.txt"), &report).unwrap();
    eprintln!("{report}\n(report: {})", root.join("report.txt").display());
    for f in &fixed {
        eprintln!("note: {f}");
    }
    assert!(
        failures.is_empty(),
        "{} decompiled functions disagree with the original:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
