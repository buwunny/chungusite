//! The `chungusite` command: decompile x86_64 functions from a binary to Rust.
//!
//!     chungusite ./prog                         # every function, fast mode
//!     chungusite ./prog --mode safe -f parse    # one function, safe mode
//!     chungusite ./prog --list                  # what's in there, and what lifts
//!     chungusite --hex "8b 47 08 03 47 10 c3"   # raw bytes, no file
use chungusite::{
    borrow::{analyze, Analysis, Class, Root},
    check,
    dump::dump,
    emit::{EmitStats, Mode},
    globals::{Globals, Item, PRELUDE},
    ir::{Function, Idx},
    load::{Binary, FuncBytes},
    names::rust_ident,
    program::{Input, Options, Program},
    sources::SOURCES,
};
use clap::{Parser, ValueEnum};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
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

    /// Safe mode: compile the output with rustc (in batches) and emit every
    /// function it rejects in fast mode instead, until everything compiles.
    #[arg(long)]
    check: bool,

    /// Worker threads (default: one per CPU, or RAYON_NUM_THREADS).
    #[arg(short, long)]
    jobs: Option<usize>,
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

/// `--check` rounds: each one recompiles after sending the failures to fast mode.
const MAX_CHECK_ROUNDS: usize = 4;

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
    if let Some(n) = cli.jobs {
        rayon::ThreadPoolBuilder::new().num_threads(n).build_global().map_err(|e| e.to_string())?;
    }
    let data;
    let hex_bytes;
    let bin;
    // `all` is every function in the binary: signatures depend on the callees,
    // whether or not they are printed. `funcs` is what was asked for.
    let (all, funcs, source, bin): (Vec<FuncBytes>, Vec<FuncBytes>, String, Option<&Binary>) = match (&cli.hex, &cli.input) {
        (Some(h), _) => {
            hex_bytes = parse_hex(h)?;
            data = Vec::new();
            let f = || FuncBytes { name: "func".into(), demangled: None, addr: 0x1000, bytes: &hex_bytes };
            (vec![f()], vec![f()], "hex input".into(), None)
        }
        (None, Some(path)) => {
            data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
            bin = Binary::parse(&data).map_err(|e| format!("{}: {e}", path.display()))?;
            let all = bin.funcs.iter().map(copy).collect();
            (all, select(&bin, cli)?, path.display().to_string(), Some(&bin))
        }
        (None, None) => unreachable!("clap requires one"),
    };

    let mode = match cli.mode {
        CliMode::Fast => Mode::Fast,
        CliMode::Safe => Mode::Safe,
    };
    let mode_name = if mode == Mode::Fast { "fast" } else { "safe" };
    let rust = cli.emit == Emit::Rust && !cli.list;

    // Identifiers are assigned in address order before anything runs in parallel,
    // so they (and the output) don't depend on scheduling.
    let mut used = HashSet::new();
    let idents: Vec<String> = funcs.iter().map(|f| rust_ident(f.pretty(), f.addr, &mut used)).collect();
    // Function pointers in data are named by their identifier in the output.
    let by_addr: HashMap<u64, &str> = funcs.iter().zip(&idents).map(|(f, i)| (f.addr, i.as_str())).collect();
    let globals = bin.filter(|_| rust).map(|b| Globals::new(b, used, &by_addr));
    let global_of = |addr: u64| globals.as_ref().and_then(|g| g.expr(addr));

    // Lift everything, recover signatures, and emit, in parallel (`program.rs`).
    // A selected function is one of `all` (by address) or bytes of its own (--size).
    let mut inputs: Vec<Input> = all
        .iter()
        .map(|f| Input { name: f.name.clone(), ident: String::new(), addr: f.addr, bytes: f.bytes, selected: false })
        .collect();
    let mut index = Vec::with_capacity(funcs.len());
    for (fb, ident) in funcs.iter().zip(&idents) {
        match inputs.iter().position(|x| x.addr == fb.addr && x.bytes.len() == fb.bytes.len() && !x.selected) {
            Some(i) => {
                inputs[i].ident = ident.clone();
                inputs[i].selected = true;
                index.push(i);
            }
            None => {
                index.push(inputs.len());
                inputs.push(Input { name: fb.name.clone(), ident: ident.clone(), addr: fb.addr, bytes: fb.bytes, selected: true });
            }
        }
    }
    let file = bin.map(|_| &data[..]);
    let program = Program::build(inputs, file, cli.emit == Emit::RawIr);
    let global_slice = |addr: u64| globals.as_ref().and_then(|g| g.slice(addr));
    // Function pointers in data, other than GOT slots (calls through those are
    // direct calls; `program.rs` sees which ones are loaded for other uses).
    let address_taken: Vec<u64> = bin.map_or(Vec::new(), |b| {
        let got = |at: u64| b.data_section_at(at).is_some_and(|s| matches!(b.data[s].name.as_str(), ".got" | ".got.plt" | "__got"));
        b.pointers.iter().filter(|(&at, _)| !got(at)).map(|(_, &t)| t).collect()
    });
    let opts = Options { global_of: &global_of, global_slice: &global_slice, fast: &[], address_taken: &address_taken };
    let mut emitted = if rust { program.emit_all_with(mode, &opts) } else { Vec::new() };
    let analyses = if cli.emit == Emit::Borrows { program.analyses(&opts) } else { Vec::new() };

    // The statics the emitted functions point into.
    let mut statics = BTreeSet::new();
    let mut statics_src = String::new();
    let mut statics_stub = String::new();
    if let Some(g) = &globals {
        for &i in &index {
            if let Ok(ir) = &program.funcs[i].ir {
                let mut items = Vec::new();
                g.referenced(ir, &mut items);
                statics.extend(items);
            }
        }
        if !statics.is_empty() {
            statics_src.push_str("\n// Data the functions above point into, from the binary's data sections.\n");
            statics_src.push_str(PRELUDE);
            // Pointers inside the statics pull in more statics, until none are new.
            let mut todo: Vec<Item> = statics.iter().copied().collect();
            let mut more = Vec::new();
            while !todo.is_empty() {
                for item in &todo {
                    g.emit_static(item, &mut statics_src, &mut more);
                }
                more.sort();
                more.dedup();
                todo = more.drain(..).filter(|i| statics.insert(*i)).collect();
            }
            statics_stub.push_str(PRELUDE);
            for item in &statics {
                g.emit_static_stub(item, &mut statics_stub);
            }
        }
    }

    // --check: what rustc rejects goes back to fast mode, and everything is
    // emitted again (callers of a function that now takes integers change too).
    let mut fast = vec![false; program.funcs.len()];
    let mut check_errors = Vec::new();
    if rust && cli.check && mode == Mode::Safe {
        let shared = format!("{}{statics_stub}", program.prelude());
        let dir = std::env::temp_dir().join(format!("chungusite-check-{}", std::process::id()));
        for _round in 0..MAX_CHECK_ROUNDS {
            let sel: Vec<usize> = index.iter().copied().filter(|&i| emitted[i].is_some()).collect();
            let srcs: Vec<&str> = sel.iter().map(|&i| emitted[i].as_ref().unwrap().0.as_str()).collect();
            let r = check::check(&shared, &srcs, &dir).map_err(|e| format!("--check: running rustc: {e}"))?;
            check_errors = r.other.clone();
            check_errors.extend(r.messages.iter().filter(|(k, _)| fast[sel[*k]]).map(|(k, m)| format!("{}: {m}", program.funcs[sel[*k]].ident)));
            let new: Vec<usize> = r.failing.iter().map(|&k| sel[k]).filter(|&i| !fast[i]).collect();
            if new.is_empty() {
                if !r.failing.is_empty() {
                    eprintln!("chungusite: --check: {} functions fail to compile even in fast mode", r.failing.len());
                }
                break;
            }
            for i in new {
                fast[i] = true;
            }
            let opts = Options { fast: &fast, ..opts };
            emitted = program.emit_all_with(mode, &opts);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    let mut out = String::new();
    if rust {
        let _ = writeln!(out, "// Decompiled by chungusite from {source} (--mode {mode_name}).");
        out.push_str("#![allow(unused_mut, unused_variables, unused_assignments, unreachable_code, non_snake_case, non_upper_case_globals, unused_parens, unused_unsafe, clippy::all)]\n");
        out.push_str(&program.prelude());
    }
    let mut failures: BTreeMap<String, usize> = BTreeMap::new();
    let (mut ok, mut total) = (0, EmitStats::default());
    let (mut no_raw, mut safe_fns, mut emitted_fns) = (0, 0, 0);
    for (fb, &i) in funcs.iter().zip(&index) {
        let pf = &program.funcs[i];
        let header = format!("{} @ {:#x}, {} bytes", fb.pretty(), fb.addr, fb.bytes.len());
        match &pf.ir {
            Ok(_) => ok += 1,
            Err(e) => *failures.entry(failure_kind(e)).or_default() += 1,
        }
        if cli.list {
            let _ = match &pf.ir {
                Ok(_) => writeln!(out, "ok    {header}"),
                Err(e) => writeln!(out, "FAIL  {header}: {e}"),
            };
            continue;
        }
        match (&pf.ir, cli.emit) {
            (Err(e), _) if cli.skip_failed => eprintln!("skipped {header}: {e}"),
            (Err(e), Emit::Rust) => {
                let _ = writeln!(out, "\n// {header}\n// not lifted: {e}");
                let _ = writeln!(out, "pub fn {}() -> u64 {{\n    todo!({:?})\n}}", pf.ident, format!("not lifted: {e}"));
            }
            (Err(e), _) => {
                let _ = writeln!(out, "\n; {header}\n; not lifted: {e}");
            }
            (Ok(ir), Emit::Ir) => {
                let _ = writeln!(out, "\n; {header}\n{}", dump(ir));
            }
            (Ok(_), Emit::RawIr) => {
                let _ = writeln!(out, "\n; {header}\n{}", pf.raw_ir.as_deref().unwrap_or(""));
            }
            (Ok(ir), Emit::Borrows) => {
                let _ = writeln!(out, "\n; {header}");
                match &analyses[i] {
                    Some(a) => borrows(ir, a, &mut out),
                    None => borrows(ir, &analyze(ir), &mut out),
                }
            }
            (Ok(ir), Emit::Rust) => {
                let (body, s) = emitted[i].as_ref().expect("emitted");
                emitted_fns += 1;
                let _ = write!(out, "\n// {header}");
                if s.checked + s.raw > 0 {
                    let _ = write!(out, "; {} of {} memory accesses bounds-checked", s.checked, s.checked + s.raw);
                }
                out.push('\n');
                out.push_str(body);
                total.checked += s.checked;
                total.raw += s.raw;
                for k in 0..4 {
                    total.raw_by[k] += s.raw_by[k];
                }
                no_raw += (s.raw == 0) as usize;
                safe_fns += body.starts_with("pub fn") as usize;
                total.todo += s.todo;
                total.state_machines += s.state_machines;
                let _ = ir;
            }
        }
    }
    if rust && !cli.list {
        out.push_str(&statics_src);
    }

    match &cli.output {
        Some(p) => std::fs::write(p, &out).map_err(|e| format!("{}: {e}", p.display()))?,
        None => print!("{out}"),
    }

    eprintln!("chungusite: lifted {ok} of {} functions ({mode_name} mode)", funcs.len());
    if total.checked + total.raw > 0 {
        let mem_raw: usize = total.raw_by.iter().sum();
        eprintln!("  memory accesses: {} bounds-checked, {} raw", total.checked, mem_raw);
        let by: Vec<String> = SOURCES.iter().zip(total.raw_by).map(|(n, c)| format!("{n} {c}")).collect();
        eprintln!("  raw accesses by source: {}", by.join(", "));
    }
    if emitted_fns > 0 {
        eprintln!("  {no_raw} of {emitted_fns} functions have no raw pointer; {safe_fns} are safe `fn`s");
    }
    if !statics.is_empty() {
        let bytes: u64 = statics.iter().map(|i| i.len).sum();
        eprintln!("  {} statics ({bytes} bytes) from the binary's data sections", statics.len());
    }
    let downgraded = fast.iter().filter(|&&x| x).count();
    if cli.check && mode == Mode::Safe {
        eprintln!("  --check: {downgraded} functions didn't compile in safe mode and are emitted in fast mode");
        if !check_errors.is_empty() {
            eprintln!("  --check: {} errors that fast mode doesn't fix:\n{}", check_errors.len(), check::describe(&check_errors));
        }
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

/// One line per argument: what safe mode would make of it, then the other
/// objects the function points into.
fn borrows(func: &Function, a: &Analysis, out: &mut String) {
    const REG: [&str; 16] =
        ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"];
    let _ = func;
    for (k, p) in a.params.iter().enumerate() {
        let class = match p.class {
            Class::NotPointer => "integer".to_string(),
            Class::Shared if p.nullable => "Option<&T>".to_string(),
            Class::Mut if p.nullable => "Option<&mut T>".to_string(),
            Class::Shared => "&T".to_string(),
            Class::Mut => "&mut T".to_string(),
            Class::Raw => format!("raw pointer ({})", a.why.get(k).copied().unwrap_or("escapes")),
        };
        let fields: Vec<String> =
            p.fields.iter().map(|&(o, w)| format!("{o:+}{}", if w { " (written)" } else { "" })).collect();
        let _ = writeln!(
            out,
            "  {}: {class}{}{}{}",
            if p.reg >= 16 { "stack".to_string() } else { REG[p.reg as usize].to_string() },
            if fields.is_empty() { String::new() } else { format!(", fields at {}", fields.join(", ")) },
            if p.indexed { ", indexed" } else { "" },
            if p.returned && p.class != Class::NotPointer { ", return value borrows from it" } else { "" },
        );
    }
    for (r, root) in a.roots.iter().enumerate() {
        let name = match root {
            Root::Param(_) => continue,
            Root::Frame => "frame".to_string(),
            Root::Global(c) => format!("global {c:#x}"),
            Root::Alloc(v) => format!("allocation v{}", v.index()),
        };
        let what = match (a.safe[r], a.written[r], root) {
            (true, _, Root::Alloc(_)) => "Box<[u8]>".to_string(),
            (true, true, _) => "&mut [u8]".to_string(),
            (true, false, _) => "&[u8]".to_string(),
            (false, _, _) => format!("raw ({})", a.why[r]),
        };
        let _ = writeln!(out, "  {name}: {what}");
    }
    if a.loan_errors + a.move_errors > 0 {
        let _ = writeln!(out, "  downgraded: {} conflicting borrows, {} uses after free", a.loan_errors, a.move_errors);
    }
}

fn copy<'a>(f: &FuncBytes<'a>) -> FuncBytes<'a> {
    FuncBytes { name: f.name.clone(), demangled: f.demangled.clone(), addr: f.addr, bytes: f.bytes }
}

/// The functions the user asked for, or all of them.
fn select<'a>(bin: &Binary<'a>, cli: &Cli) -> Result<Vec<FuncBytes<'a>>, String> {
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
        match bin.funcs.iter().find(|f| &f.name == name || f.demangled.as_ref() == Some(name)) {
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
                let demangled = found.and_then(|f| f.demangled.clone());
                FuncBytes { name, demangled, addr, bytes }
            }
            (None, None) => return Err(format!("no function starts at {addr:#x}; pass --size to lift it anyway")),
        };
        out.push(f);
    }
    Ok(out)
}

/// Group failures for the summary: by mnemonic for unsupported instructions.
fn failure_kind(e: &str) -> String {
    match e.strip_prefix("unsupported instruction ") {
        Some(rest) => format!("unsupported {}", rest.split(' ').next().unwrap_or(rest)),
        None => e.split(" at ").next().unwrap_or(e).replace(|c: char| c.is_ascii_digit(), "").to_string(),
    }
}
