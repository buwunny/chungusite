//! The `chungusite` command: decompile x86_64 functions from a binary to Rust.
//!
//!     chungusite ./prog                         # every function, fast mode
//!     chungusite ./prog --mode safe -f parse    # one function, safe mode
//!     chungusite ./prog --list                  # what's in there, and what lifts
//!     chungusite --hex "8b 47 08 03 47 10 c3"   # raw bytes, no file
use chungusite::{
    borrow::{analyze, Class},
    dump::dump,
    emit::{emit_function, EmitStats, Mode},
    ir::Function,
    lift::{LiftError, Lifter},
    load::{Binary, FuncBytes},
    opt::clean,
    verify::verify,
};
use clap::{Parser, ValueEnum};
use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(version, about = "Decompile x86_64 machine code to Rust")]
struct Cli {
    /// ELF, Mach-O or PE file to decompile.
    #[arg(required_unless_present = "hex", conflicts_with = "hex")]
    input: Option<PathBuf>,

    /// Decompile these hex bytes (loaded at 0x1000) instead of a file.
    #[arg(long)]
    hex: Option<String>,

    /// fast: raw pointers in unsafe. safe: &[u8] / &mut [u8] where borrow inference proves it.
    #[arg(long, value_enum, default_value_t = CliMode::Fast)]
    mode: CliMode,

    /// Only these functions, by symbol name (repeatable).
    #[arg(short, long = "function", value_name = "NAME")]
    functions: Vec<String>,

    /// Only the function starting at this address, e.g. 0x401136 (repeatable).
    #[arg(short, long = "addr", value_name = "ADDR", value_parser = parse_addr)]
    addrs: Vec<u64>,

    /// Byte length for an --addr that no symbol covers.
    #[arg(long, value_parser = parse_addr)]
    size: Option<u64>,

    /// List functions and whether each one lifts, instead of emitting code.
    #[arg(short, long)]
    list: bool,

    /// What to print.
    #[arg(long, value_enum, default_value_t = Emit::Rust)]
    emit: Emit,

    /// Write the output here instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Skip functions that fail to lift instead of emitting a todo!() stub for them.
    #[arg(long)]
    skip_failed: bool,
}

#[derive(Copy, Clone, ValueEnum)]
enum CliMode {
    Fast,
    Safe,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Emit {
    /// Rust source.
    Rust,
    /// The cleaned SSA IR that the emitter sees.
    Ir,
    /// The IR straight out of the lifter, before cleanup.
    RawIr,
    /// What safe mode's borrow inference concludes for each argument.
    Borrows,
}

fn parse_addr(s: &str) -> Result<u64, String> {
    let r = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(h) => u64::from_str_radix(h, 16),
        None => s.parse(),
    };
    r.map_err(|e| format!("bad address {s:?}: {e}"))
}

fn parse_hex(s: &str) -> Result<Vec<u8>, String> {
    let digits: Vec<u8> = s.bytes().filter(u8::is_ascii_hexdigit).collect();
    if !digits.len().is_multiple_of(2) {
        return Err("odd number of hex digits".into());
    }
    Ok(digits
        .chunks(2)
        .map(|c| u8::from_str_radix(std::str::from_utf8(c).unwrap(), 16).unwrap())
        .collect())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("chungusite: {e}");
            ExitCode::from(2)
        }
    }
}

fn run(cli: &Cli) -> Result<ExitCode, String> {
    let data;
    let hex_bytes;
    let bin;
    let (funcs, source): (Vec<FuncBytes>, String) = match (&cli.hex, &cli.input) {
        (Some(h), _) => {
            hex_bytes = parse_hex(h)?;
            (vec![FuncBytes { name: "func".into(), addr: 0x1000, bytes: &hex_bytes }], "hex input".into())
        }
        (None, Some(path)) => {
            data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
            bin = Binary::parse(&data).map_err(|e| format!("{}: {e}", path.display()))?;
            (select(&bin, cli)?, path.display().to_string())
        }
        (None, None) => unreachable!("clap requires one"),
    };
    let name_of = |addr: u64| -> Option<String> {
        funcs.iter().find(|f| f.addr == addr).map(|f| f.name.clone())
    };

    let mode = match cli.mode {
        CliMode::Fast => Mode::Fast,
        CliMode::Safe => Mode::Safe,
    };
    let mode_name = if mode == Mode::Fast { "fast" } else { "safe" };

    let mut out = String::new();
    if cli.emit == Emit::Rust && !cli.list {
        let _ = writeln!(out, "// Decompiled by chungusite from {source} (--mode {mode_name}).");
        out.push_str("#![allow(unused_mut, unused_variables, unused_assignments, unreachable_code, non_snake_case, unused_parens, unused_unsafe, clippy::all)]\n");
    }

    let mut lifter = Lifter::new();
    let mut func = Function::with_capacity(256, 16);
    let mut idents = HashSet::new();
    let mut failures: BTreeMap<String, usize> = BTreeMap::new();
    let (mut ok, mut total) = (0, EmitStats::default());

    let mut raw_ir = String::new();
    for fb in &funcs {
        let ident = rust_ident(&fb.name, fb.addr, &mut idents);
        let lifted = lifter
            .lift(fb.bytes, fb.addr, &mut func)
            .map_err(|e| describe(&e))
            .and_then(|()| verify(&func).map_err(|e| format!("lifted IR failed verification: {e:?}")))
            .and_then(|()| {
                if cli.emit == Emit::RawIr {
                    raw_ir = dump(&func);
                }
                clean(&mut func);
                verify(&func).map_err(|e| format!("cleaned IR failed verification: {e:?}"))
            });
        let header = format!("{} @ {:#x}, {} bytes", fb.name, fb.addr, fb.bytes.len());

        let err = match lifted {
            Ok(()) => None,
            Err(e) => {
                *failures.entry(failure_kind(&e)).or_default() += 1;
                Some(e)
            }
        };
        if cli.list {
            match &err {
                None => {
                    let _ = writeln!(out, "ok    {header}");
                }
                Some(e) => {
                    let _ = writeln!(out, "FAIL  {header}: {e}");
                }
            }
            if err.is_none() {
                ok += 1;
            }
            continue;
        }
        match (err, cli.emit) {
            (Some(e), _) if cli.skip_failed => eprintln!("skipped {header}: {e}"),
            (Some(e), Emit::Rust) => {
                let _ = writeln!(out, "\n// {header}\n// not lifted: {e}");
                let _ = writeln!(out, "pub fn {ident}() -> u64 {{\n    todo!({:?})\n}}", format!("not lifted: {e}"));
            }
            (Some(e), _) => {
                let _ = writeln!(out, "\n; {header}\n; not lifted: {e}");
            }
            (None, Emit::Ir) => {
                ok += 1;
                let _ = writeln!(out, "\n; {header}\n{}", dump(&func));
            }
            (None, Emit::RawIr) => {
                ok += 1;
                let _ = writeln!(out, "\n; {header}\n{raw_ir}");
            }
            (None, Emit::Borrows) => {
                ok += 1;
                let _ = writeln!(out, "\n; {header}");
                borrows(&func, &mut out);
            }
            (None, Emit::Rust) => {
                ok += 1;
                let mut body = String::new();
                let s = emit_function(&func, &ident, mode, &name_of, &mut body);
                let _ = write!(out, "\n// {header}");
                if s.checked + s.raw > 0 {
                    let _ = write!(out, "; {} of {} memory accesses bounds-checked", s.checked, s.checked + s.raw);
                }
                out.push('\n');
                out.push_str(&body);
                total.checked += s.checked;
                total.raw += s.raw;
                total.todo += s.todo;
                total.state_machines += s.state_machines;
            }
        }
    }

    match &cli.output {
        Some(p) => std::fs::write(p, &out).map_err(|e| format!("{}: {e}", p.display()))?,
        None => print!("{out}"),
    }

    eprintln!("chungusite: lifted {ok} of {} functions ({mode_name} mode)", funcs.len());
    if total.checked + total.raw > 0 {
        eprintln!("  memory accesses: {} bounds-checked, {} raw", total.checked, total.raw);
    }
    if total.state_machines > 0 {
        eprintln!("  {} with irreducible control flow, kept as a `loop {{ match bb }}` state machine", total.state_machines);
    }
    if total.todo > 0 {
        eprintln!("  {} todo!() left where the emitter can't express an instruction yet", total.todo);
    }
    if !failures.is_empty() {
        let mut v: Vec<_> = failures.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        eprintln!("  failures by cause:");
        for (kind, n) in v {
            eprintln!("    {n:5}  {kind}");
        }
    }
    Ok(if ok == funcs.len() { ExitCode::SUCCESS } else { ExitCode::from(1) })
}

/// One line per argument: what safe mode would make of it.
fn borrows(func: &Function, out: &mut String) {
    const REG: [&str; 16] =
        ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"];
    for p in &analyze(func).params {
        let class = match p.class {
            Class::NotPointer => "integer",
            Class::Shared if p.nullable => "Option<&T>",
            Class::Mut if p.nullable => "Option<&mut T>",
            Class::Shared => "&T",
            Class::Mut => "&mut T",
            Class::Raw => "raw pointer (escapes)",
        };
        let fields: Vec<String> =
            p.fields.iter().map(|&(o, w)| format!("{o:+}{}", if w { " (written)" } else { "" })).collect();
        let _ = writeln!(
            out,
            "  {}: {class}{}{}{}",
            REG[p.reg as usize & 15],
            if fields.is_empty() { String::new() } else { format!(", fields at {}", fields.join(", ")) },
            if p.indexed { ", indexed" } else { "" },
            if p.returned && p.class != Class::NotPointer { ", return value borrows from it" } else { "" },
        );
    }
}

/// The functions the user asked for, or all of them.
fn select<'a>(bin: &Binary<'a>, cli: &Cli) -> Result<Vec<FuncBytes<'a>>, String> {
    let copy = |f: &FuncBytes<'a>| FuncBytes { name: f.name.clone(), addr: f.addr, bytes: f.bytes };
    if cli.functions.is_empty() && cli.addrs.is_empty() {
        if bin.funcs.is_empty() {
            return Err(format!(
                "no function symbols found (stripped binary?); use --addr with --size, e.g. --addr {:#x} --size 0x100",
                bin.entry()
            ));
        }
        return Ok(bin.funcs.iter().map(copy).collect());
    }
    let mut out = Vec::new();
    for name in &cli.functions {
        match bin.funcs.iter().find(|f| &f.name == name) {
            Some(f) => out.push(copy(f)),
            None => return Err(format!("no function named {name:?} (try --list)")),
        }
    }
    for &addr in &cli.addrs {
        let f = match (bin.func_at(addr), cli.size) {
            (Some(f), None) => copy(f),
            (found, Some(size)) => {
                let bytes = bin.code_at(addr, size).ok_or(format!("{addr:#x} is not in a code section"))?;
                let name = found.map_or(format!("sub_{addr:x}"), |f| f.name.clone());
                FuncBytes { name, addr, bytes }
            }
            (None, None) => return Err(format!("no function starts at {addr:#x}; pass --size to lift it anyway")),
        };
        out.push(f);
    }
    Ok(out)
}

fn describe(e: &LiftError) -> String {
    match e {
        LiftError::Unsupported { ip, mnemonic } => format!("unsupported instruction {mnemonic:?} at {ip:#x}"),
        LiftError::FlagsNotInBlock { ip } => format!("branch at {ip:#x} reads flags set in another block"),
        LiftError::BranchOutOfRange { ip, target } => format!("branch at {ip:#x} leaves the function (to {target:#x})"),
        LiftError::TargetInsideInstruction { target } => format!("branch into the middle of an instruction at {target:#x}"),
    }
}

/// Group failures for the summary: by mnemonic for unsupported instructions.
fn failure_kind(e: &str) -> String {
    match e.strip_prefix("unsupported instruction ") {
        Some(rest) => format!("unsupported {}", rest.split(' ').next().unwrap_or(rest)),
        None => e.split(" at ").next().unwrap_or(e).replace(|c: char| c.is_ascii_digit(), "").to_string(),
    }
}

/// A unique Rust identifier for a symbol name.
fn rust_ident(name: &str, addr: u64, used: &mut HashSet<String>) -> String {
    let mut s: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect();
    if s.is_empty() || s.starts_with(|c: char| c.is_ascii_digit()) || is_keyword(&s) {
        s.insert_str(0, "f_");
    }
    if !used.insert(s.clone()) {
        s = format!("{s}_{addr:x}");
        used.insert(s.clone());
    }
    s
}

fn is_keyword(s: &str) -> bool {
    matches!(
        s,
        "as" | "break" | "const" | "continue" | "crate" | "else" | "enum" | "extern" | "false" | "fn" | "for"
            | "if" | "impl" | "in" | "let" | "loop" | "match" | "mod" | "move" | "mut" | "pub" | "ref"
            | "return" | "self" | "Self" | "static" | "struct" | "super" | "trait" | "true" | "type"
            | "unsafe" | "use" | "where" | "while" | "async" | "await" | "dyn" | "abstract" | "become"
            | "box" | "do" | "final" | "macro" | "override" | "priv" | "typeof" | "unsized" | "virtual"
            | "yield" | "try" | "gen" | "_"
    )
}
