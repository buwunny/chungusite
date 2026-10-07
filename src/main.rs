//! The `chungusite` command: decompile x86_64 functions from a binary to Rust.
//!
//!     chungusite ./prog                         # every function, fast mode
//!     chungusite ./prog --mode safe -f parse    # one function, safe mode
//!     chungusite ./prog --list                  # what's in there, and what lifts
//!     chungusite --hex "8b 47 08 03 47 10 c3"   # raw bytes, no file
use chungusite::{
    borrow::{analyze, Class},
    dump::dump,
    emit::{EmitStats, Mode},
    globals::{Globals, Item, PRELUDE},
    ir::Function,
    load::{Binary, FuncBytes},
    names::rust_ident,
    program::{Input, Program},
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

/// The input file, mapped rather than read: only the pages the decompiler
/// touches (headers, symbols, the code it lifts, the data it emits) are loaded.
fn map(path: &std::path::Path) -> std::io::Result<memmap2::Mmap> {
    let file = std::fs::File::open(path)?;
    // Safety: the map is read-only and lives until the end of `run`. If another
    // process truncates the file meanwhile, reads fault; that is the usual
    // trade-off of mapping input files, and the same as most binary tools make.
    unsafe { memmap2::Mmap::map(&file) }
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
            data = None;
            let f = || FuncBytes { name: "func".into(), demangled: None, addr: 0x1000, bytes: &hex_bytes };
            (vec![f()], vec![f()], "hex input".into(), None)
        }
        (None, Some(path)) => {
            data = Some(map(path).map_err(|e| format!("{}: {e}", path.display()))?);
            bin = Binary::parse(data.as_deref().unwrap()).map_err(|e| format!("{}: {e}", path.display()))?;
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
    let file = data.as_deref();
    let program = Program::build(inputs, file, cli.emit == Emit::RawIr);
    // Function pointers in data are named by their identifier in the output; a
    // function --skip-failed leaves out keeps its slot's bytes from the file.
    let by_addr: HashMap<u64, &str> = funcs
        .iter()
        .zip(&idents)
        .zip(&index)
        .filter(|(_, &i)| !cli.skip_failed || program.funcs[i].ir.is_ok())
        .map(|((f, id), _)| (f.addr, id.as_str()))
        .collect();
    let globals = bin.filter(|_| rust).map(|b| Globals::new(b, used, &by_addr));
    let global_of = |addr: u64| globals.as_ref().and_then(|g| g.expr(addr));
    let emitted = if rust { program.emit_all(mode, &global_of) } else { Vec::new() };

    let mut out = String::new();
    if rust {
        let _ = writeln!(out, "// Decompiled by chungusite from {source} (--mode {mode_name}).");
        out.push_str("#![allow(unused_mut, unused_variables, unused_assignments, unreachable_code, non_snake_case, non_upper_case_globals, unused_parens, unused_unsafe, clippy::all)]\n");
        out.push_str(&program.prelude());
    }
    let mut failures: BTreeMap<String, usize> = BTreeMap::new();
    let mut statics = BTreeSet::new();
    let (mut ok, mut total) = (0, EmitStats::default());
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
                borrows(ir, &mut out);
            }
            (Ok(ir), Emit::Rust) => {
                let (body, s) = emitted[i].as_ref().expect("emitted");
                let _ = write!(out, "\n// {header}");
                if s.checked + s.raw > 0 {
                    let _ = write!(out, "; {} of {} memory accesses bounds-checked", s.checked, s.checked + s.raw);
                }
                out.push('\n');
                out.push_str(body);
                total.checked += s.checked;
                total.raw += s.raw;
                total.todo += s.todo;
                total.state_machines += s.state_machines;
                if let Some(g) = &globals {
                    let mut items = Vec::new();
                    g.referenced(ir, &mut items);
                    statics.extend(items);
                }
            }
        }
    }
    if let Some(g) = &globals {
        if !statics.is_empty() {
            out.push_str("\n// Data the functions above point into, from the binary's data sections.\n");
            out.push_str(PRELUDE);
            // Pointers inside the statics pull in more statics, until none are new.
            let mut todo: Vec<Item> = statics.iter().copied().collect();
            let mut more = Vec::new();
            while !todo.is_empty() {
                for item in &todo {
                    g.emit_static(item, &mut out, &mut more);
                }
                more.sort();
                more.dedup();
                todo = more.drain(..).filter(|i| statics.insert(*i)).collect();
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
    if !statics.is_empty() {
        let bytes: u64 = statics.iter().map(|i| i.len).sum();
        eprintln!("  {} statics ({bytes} bytes) from the binary's data sections", statics.len());
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
