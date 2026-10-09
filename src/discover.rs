//! Find functions in a binary that has no symbol table (a stripped build).
//!
//! Sources, most trusted first:
//! * unwind tables: `.eh_frame` FDEs (ELF, Mach-O) and `.pdata` (PE) give the
//!   exact start and length of every function compiled with unwind info, which
//!   is nearly all of them on x86_64;
//! * the dynamic symbols (exports), which `strip` keeps;
//! * Mach-O's `LC_FUNCTION_STARTS`, which `strip` keeps too: every start, no sizes;
//! * the entry point and the start of each code section (`_init`, `_fini`);
//! * targets of direct calls, `lea reg, [rip+x]` into code (function pointers,
//!   `main` in `_start`), and code addresses stored in data (vtables, tables);
//! * code that nothing above covers, after alignment padding.
//!
//! Functions without unwind info end where control flow from their start stops
//! (the last instruction reached before the next known start, following jump
//! tables and stopping at calls that don't return), so the padding after them
//! is not included and whatever follows the padding is checked for a prologue.
//! A call doesn't return if it goes to an import like `__stack_chk_fail` or
//! `abort` (through the PLT or the GOT), to a function with such a name, or to
//! a function every path through which ends in a call that doesn't return or
//! a trap (`ud2`, `hlt`, `int3`). A candidate strictly inside an FDE's range is
//! rejected: unwind info is authoritative, and such addresses are mostly the
//! cold half of a split function or a misread of something that isn't a call.
use iced_x86::{Code, ConditionCode, Decoder, DecoderOptions, FlowControl, Instruction, Mnemonic, OpKind, Register};
use object::{Object, ObjectSection, ObjectSegment, ObjectSymbol, ObjectSymbolTable, RelocationTarget, SectionKind};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// A function found without the symbol table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Found {
    pub addr: u64,
    pub size: u64,
    /// For the few that can be named without symbols: `_start`, `main`, `_init`, `_fini`.
    pub name: Option<&'static str>,
}

struct Text<'a> {
    addr: u64,
    bytes: &'a [u8],
}

/// Discover the functions in `file`. `known` are functions that are already
/// known with their sizes (dynamic exports); `pointers` are the slots the loader
/// fills in and the values it writes there (`Binary::pointers`), some of which
/// are function addresses.
pub fn functions(file: &object::File, known: &[(u64, u64)], pointers: &BTreeMap<u64, u64>) -> Vec<Found> {
    let mut code: Vec<Text> = file
        .sections()
        .filter(|s| s.kind() == SectionKind::Text && s.size() > 0 && !is_stub_section(s.name().unwrap_or("")))
        .filter_map(|s| Some(Text { addr: s.address(), bytes: s.data().ok().filter(|d| d.len() as u64 == s.size())? }))
        .collect();
    code.sort_by_key(|c| c.addr);
    let in_code = |a: u64| code.iter().position(|c| a >= c.addr && a < c.addr + c.bytes.len() as u64);
    let mut img = Image::new(file);
    img.slots = pointers.iter().map(|(&s, &a)| (s, a)).collect();

    // Exact ranges: unwind tables and exported symbols.
    let mut exact: BTreeMap<u64, u64> = BTreeMap::new();
    for (start, len) in unwind_ranges(file) {
        if len > 0 && in_code(start).is_some() {
            exact.entry(start).or_insert(len);
        }
    }
    for &(a, n) in known {
        if n > 0 && in_code(a).is_some() {
            exact.entry(a).or_insert(n);
        }
    }
    let exact_ranges: Vec<(u64, u64)> = exact.iter().map(|(&a, &n)| (a, a + n)).collect();
    // Strictly inside a function whose extent is exact.
    let interior = |a: u64| {
        let i = exact_ranges.partition_point(|r| r.0 <= a);
        i > 0 && a > exact_ranges[i - 1].0 && a < exact_ranges[i - 1].1
    };

    let mut names: BTreeMap<u64, &'static str> = BTreeMap::new();
    let mut starts: BTreeSet<u64> = exact.keys().copied().collect();
    let seed = |a: u64, starts: &mut BTreeSet<u64>| {
        if in_code(a).is_some() && !interior(a) {
            starts.insert(a);
        }
    };
    seed(file.entry(), &mut starts);
    for c in &code {
        seed(c.addr, &mut starts);
    }
    for &p in pointers.values() {
        seed(p, &mut starts);
    }
    for a in function_starts(file) {
        seed(a, &mut starts);
    }
    if file.entry() != 0 {
        names.insert(file.entry(), "_start");
    }
    for s in file.sections() {
        match s.name() {
            Ok(".init") => names.insert(s.address(), "_init"),
            Ok(".fini") => names.insert(s.address(), "_fini"),
            _ => None,
        };
    }

    // Grow to a fixpoint: walk every function, add what it calls or points to,
    // then look for code in what is still uncovered.
    let mut walked: BTreeMap<u64, Walk> = BTreeMap::new();
    // Labels of computed `goto`s first taken for functions.
    let mut labels: BTreeSet<u64> = BTreeSet::new();
    let seeds = starts.clone();
    loop {
        let mut new = Vec::new();
        let mut later = Vec::new();
        let list: Vec<u64> = starts.iter().copied().collect();
        for (k, &a) in list.iter().enumerate() {
            let c = &code[in_code(a).expect("starts are in code")];
            let section_end = c.addr + c.bytes.len() as u64;
            let limit = match exact.get(&a) {
                Some(&n) => a + n,
                None => list.get(k + 1).copied().unwrap_or(section_end).min(section_end),
            };
            let w = match walked.get(&a) {
                Some(w) if w.limit == limit => w,
                _ => {
                    let bytes = &c.bytes[(a - c.addr) as usize..(limit - c.addr) as usize];
                    walked.insert(a, walk(&img, bytes, a, limit, exact.contains_key(&a)));
                    &walked[&a]
                }
            };
            for &t in &w.refs {
                if !starts.contains(&t) && !labels.contains(&t) && in_code(t).is_some() && !interior(t) && !(t > a && t < w.end) {
                    // Only jumped to: wait until everything called, pointed to
                    // or found between functions is walked, so a jump out of a computed `goto`'s label
                    // back into its function doesn't split it first.
                    let jumped = w.jumps.iter().filter(|&&j| j == t).count();
                    if jumped == w.refs.iter().filter(|&&r| r == t).count() {
                        later.push(t);
                    } else {
                        new.push(t);
                    }
                }
            }
        }
        // A function's computed `goto` table holds labels, not functions, when
        // nothing known to start a function lies between it and them and the
        // code at one of them jumps back into it. Start over without them.
        let mut more = Vec::new();
        for (&a, w) in starts.iter().map(|a| (a, &walked[a])) {
            let (Some(&first), Some(&last)) = (w.labels.iter().min(), w.labels.iter().max()) else { continue };
            let fresh = w.labels.iter().any(|l| !labels.contains(l));
            let clear = exact.range(a + 1..=last).next().is_none() && in_code(first) == in_code(a) && in_code(last) == in_code(a);
            let back = w.labels.iter().any(|l| starts.contains(l) && walked.get(l).is_some_and(|lw| lw.jumps.iter().any(|&t| t >= a && t < *l)));
            if fresh && clear && back {
                more.extend(w.labels.iter().copied());
            }
        }
        if !more.is_empty() {
            labels.extend(more);
            starts = seeds.iter().copied().filter(|a| !labels.contains(a)).collect();
            continue;
        }
        if new.is_empty() {
            // Functions found not to return end their callers' paths: walk
            // those again before looking at what is left between functions.
            let found: HashSet<u64> = starts.iter().copied().filter(|a| !walked[a].returns && img.noreturn.insert(*a)).collect();
            if !found.is_empty() {
                walked.retain(|_, w| !w.calls(&img).any(|t| found.contains(&t)));
                continue;
            }
            // Prologues in the gaps between functions.
            let mut covered: Vec<(u64, u64)> = starts.iter().map(|a| (*a, walked[a].end)).collect();
            covered.sort_unstable();
            for c in &code {
                let mut at = c.addr;
                let end = c.addr + c.bytes.len() as u64;
                let i = covered.partition_point(|r| r.0 < end);
                for &(s, e) in covered[..i].iter().filter(|r| r.1 > c.addr).chain(std::iter::once(&(end, end))) {
                    if s > at {
                        if let Some(p) = code_after_padding(&c.bytes[(at - c.addr) as usize..(s - c.addr) as usize], at) {
                            if !interior(p) && !labels.contains(&p) {
                                new.push(p);
                            }
                        }
                    }
                    at = at.max(e);
                }
            }
        }
        if new.is_empty() {
            new = later;
        }
        if new.is_empty() {
            break;
        }
        starts.extend(new);
    }

    // `_start` passes `main` to `__libc_start_main` in rdi.
    if let Some(m) = walked.get(&file.entry()).and_then(|w| w.rdi_code_ref) {
        if starts.contains(&m) {
            names.entry(m).or_insert("main");
        }
    }

    starts
        .iter()
        .map(|&a| {
            let size = exact.get(&a).copied().unwrap_or(walked[&a].end - a);
            Found { addr: a, size, name: names.get(&a).copied() }
        })
        .filter(|f| f.size > 0)
        .collect()
}

/// PLT and stub sections hold trampolines to imports, which `program.rs` names
/// from the relocations; they aren't functions to decompile.
fn is_stub_section(name: &str) -> bool {
    matches!(name, ".plt" | ".plt.sec" | ".plt.got" | ".iplt" | "__stubs" | "__stub_helper")
}

/// Immediates below this are never taken as code addresses (non-PIE x86_64
/// code starts at 0x400000; PIE code at a few KiB, where it collides with integers).
const MIN_IMM_ADDR: u64 = 0x10000;

struct Walk {
    /// The `limit` this was computed for; it changes when a new start appears before it.
    limit: u64,
    /// One past the last byte of the last instruction reached.
    end: u64,
    /// Call targets, code addresses taken with `lea`, and jumps out of the function.
    refs: Vec<u64>,
    /// The last code address loaded into rdi before the first indirect call
    /// (`main`, when this is `_start`).
    rdi_code_ref: Option<u64>,
    /// Some path leaves the function normally: through a `ret`, a tail call to
    /// a function that returns, an indirect jump that isn't a jump table, or by
    /// running into `limit` or bytes that don't decode (where it goes is unknown).
    /// False if every path ends in a call that doesn't return or a trap
    /// (`ud2`, `hlt`, `int3`).
    returns: bool,
    /// The memory operands of `call [rip+slot]` and `jmp [rip+slot]` (GOT slots).
    slots: Vec<u64>,
    /// Entries of the tables a computed `goto` jumps through (`label_table`).
    labels: Vec<u64>,
    /// Targets of jumps that leave `[start, limit)`.
    jumps: Vec<u64>,
}

impl Walk {
    /// What the calls and tail calls go to, directly or through a GOT slot
    /// (and what the loader puts in the slot): if one of them turns out not to
    /// return, this walk may end sooner.
    fn calls<'a>(&'a self, img: &'a Image) -> impl Iterator<Item = u64> + 'a {
        let slots = self.slots.iter().flat_map(|s| std::iter::once(*s).chain(img.slots.get(s).copied()));
        self.refs.iter().copied().chain(slots)
    }
}

/// Follow control flow from `start` without leaving `[start, limit)`. If
/// `exact`, the range is known to be one function, so it is decoded in full.
fn walk(img: &Image, bytes: &[u8], start: u64, limit: u64, exact: bool) -> Walk {
    let mut dec = Decoder::with_ip(64, bytes, start, DecoderOptions::NONE);
    let mut seen = BTreeSet::new();
    let mut todo = vec![start];
    let mut refs = Vec::new();
    let mut end = start;
    let mut rdi = None;
    let mut rdi_done = false;
    // rip-relative addresses taken with `lea`: candidate jump table bases.
    let mut leas = Vec::new();
    let mut returns = false;
    let mut slots = Vec::new();
    let mut labels = Vec::new();
    let mut jumps = Vec::new();
    let mut insn = Instruction::default();
    while let Some(at) = todo.pop() {
        let mut ip = at;
        // `cmp idx, n` (or clang's `sub idx, n`) just before, and the number of
        // cases its `ja` (`jae`) leaves
        let mut cmp_imm: Option<u64> = None;
        let mut cases = None;
        // Cleared where the path ends in a known way: at a jump, a return, a
        // call that doesn't return, a trap, or code already walked.
        let mut open = true;
        while ip < limit {
            if !seen.insert(ip) {
                open = false;
                break;
            }
            dec.set_ip(ip);
            if dec.set_position((ip - start) as usize).is_err() {
                break;
            }
            dec.decode_out(&mut insn);
            if insn.is_invalid() || insn.next_ip() > limit {
                break;
            }
            end = end.max(insn.next_ip());
            ip = insn.next_ip();
            if insn.mnemonic() == Mnemonic::Lea && insn.is_ip_rel_memory_operand() {
                let t = insn.ip_rel_memory_address();
                refs.push(t);
                leas.push(t);
                if matches!(insn.op0_register(), Register::RDI | Register::EDI) && !rdi_done {
                    rdi = Some(t);
                }
            }
            // Non-PIE code loads function addresses as immediates (`mov edi, offset main`).
            // Small values are left alone: in a PIE they are integers, not addresses.
            if matches!(insn.code(), Code::Mov_r32_imm32 | Code::Mov_rm64_imm32 | Code::Mov_r64_imm64)
                && insn.op0_kind() == OpKind::Register
            {
                let t = insn.immediate(1);
                if t >= MIN_IMM_ADDR {
                    refs.push(t);
                    if matches!(insn.op0_register(), Register::RDI | Register::EDI) && !rdi_done {
                        rdi = Some(t);
                    }
                }
            }
            let imm = (matches!(insn.mnemonic(), Mnemonic::Cmp | Mnemonic::Sub) && insn.op_count() == 2 && matches!(insn.op0_kind(), OpKind::Register | OpKind::Memory))
                .then(|| insn.op_kind(1))
                .filter(|&k| matches!(k, OpKind::Immediate8 | OpKind::Immediate8to32 | OpKind::Immediate8to64 | OpKind::Immediate32 | OpKind::Immediate32to64))
                .map(|_| insn.immediate(1));
            if insn.flow_control() == FlowControl::ConditionalBranch {
                cases = match (cmp_imm, insn.condition_code()) {
                    (Some(n), ConditionCode::a) => Some(n.saturating_add(1)),
                    (Some(n), ConditionCode::ae) => Some(n),
                    _ => cases,
                };
            }
            cmp_imm = imm;
            let slot = insn.is_ip_rel_memory_operand().then(|| insn.ip_rel_memory_address());
            match insn.flow_control() {
                FlowControl::Call if insn.op0_kind() == OpKind::NearBranch64 => {
                    let t = insn.near_branch_target();
                    refs.push(t);
                    if img.noreturn.contains(&t) {
                        open = false;
                        break;
                    }
                }
                FlowControl::IndirectCall => {
                    rdi_done = true;
                    slots.extend(slot);
                    if slot.is_some_and(|s| img.slot_noreturn(s)) {
                        open = false;
                        break;
                    }
                }
                FlowControl::IndirectBranch => {
                    let mut cases = jump_table(&insn, img, &leas, start, limit, cases);
                    // A computed `goto` (`&&label` in GNU C): a jump through a
                    // register loaded from a table of code addresses after the
                    // function's start. Those inside the range are walked; the
                    // caller decides whether the rest are labels or functions.
                    let through_reg = insn.op0_kind() == OpKind::Register
                        || insn.op0_kind() == OpKind::Memory && !matches!(insn.memory_base(), Register::None | Register::RIP);
                    if cases.is_empty() && through_reg {
                        if let Some(t) = label_table(img, &leas, start) {
                            cases = t.iter().copied().filter(|&l| l < limit).collect();
                            labels.extend(t);
                        }
                    }
                    slots.extend(slot);
                    // Without a table, a tail call through a register or the GOT.
                    returns |= cases.is_empty() && !slot.is_some_and(|s| img.slot_noreturn(s));
                    todo.extend(cases);
                    open = false;
                    break;
                }
                FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch
                    if insn.op0_kind() == OpKind::NearBranch64 =>
                {
                    let t = insn.near_branch_target();
                    // a `jmp` to the next instruction, or over nothing but
                    // padding, goes to the next function: inside one, the
                    // code would fall through
                    let next = insn.next_ip();
                    let over_padding = insn.flow_control() == FlowControl::UnconditionalBranch
                        && t >= next
                        && t < limit
                        && is_padding(&bytes[(next - start) as usize..(t - start) as usize]);
                    if t >= start && t < limit && !over_padding {
                        todo.push(t);
                    } else {
                        refs.push(t); // tail call
                        jumps.push(t);
                        returns |= !img.noreturn.contains(&t);
                    }
                    if insn.flow_control() == FlowControl::UnconditionalBranch {
                        open = false;
                        break;
                    }
                }
                FlowControl::Return => {
                    returns = true;
                    open = false;
                    break;
                }
                FlowControl::Interrupt => {
                    // `int3` is a trap; `int n` is a system call, which may return.
                    returns |= insn.mnemonic() != Mnemonic::Int3;
                    open = false;
                    break;
                }
                FlowControl::Exception if insn.mnemonic() == Mnemonic::Ud2 => {
                    open = false;
                    break;
                }
                _ if insn.mnemonic() == Mnemonic::Hlt => {
                    open = false;
                    break;
                }
                _ => {}
            }
        }
        returns |= open;
    }
    if exact {
        end = limit;
    }
    Walk { limit, end, refs, rdi_code_ref: rdi, returns, slots, labels, jumps }
}

/// Where a call never returns, for the lifter: the PLT stubs and GOT slots of
/// imports that don't return (`noreturn`), the functions among `funcs` (start,
/// bytes, symbol name) that don't, by name or because every path through them
/// ends in a call that doesn't return or a trap, and the GOT slots the loader
/// fills with one of those functions (`slots`: slot -> address). Rust calls
/// `handle_alloc_error` and the panic functions through such slots.
pub fn noreturn_calls(file: &object::File, funcs: &[(u64, &[u8], &str)], slots: &HashMap<u64, u64>) -> HashSet<u64> {
    let mut img = Image::new(file);
    img.slots = slots.clone();
    img.noreturn.extend(funcs.iter().filter(|f| noreturn(f.2)).map(|f| f.0));
    let walk_one = |img: &Image, &(a, bytes, _): &(u64, &[u8], &str)| walk(img, bytes, a, a + bytes.len() as u64, true);
    let walks: Vec<Walk> = funcs.par_iter().map(|f| walk_one(&img, f)).collect();
    // Callee or slot -> the functions that call it, to know what to walk again.
    let mut callers: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, w) in walks.iter().enumerate() {
        for t in w.calls(&img) {
            callers.entry(t).or_default().push(i);
        }
    }
    let by_addr: HashMap<u64, usize> = funcs.iter().enumerate().map(|(i, f)| (f.0, i)).collect();
    let mut via_slot: HashMap<u64, Vec<u64>> = HashMap::new();
    for (&s, &t) in &img.slots {
        if by_addr.contains_key(&t) {
            via_slot.entry(t).or_default().push(s);
        }
    }
    let mut found: Vec<usize> = (0..funcs.len()).filter(|&i| !walks[i].returns).collect();
    while !found.is_empty() {
        let mut dirty = BTreeSet::new();
        for &i in &found {
            let a = funcs[i].0;
            if img.noreturn.insert(a) {
                for t in std::iter::once(a).chain(via_slot.get(&a).into_iter().flatten().copied()) {
                    dirty.extend(callers.get(&t).into_iter().flatten().copied());
                }
            }
        }
        let dirty: Vec<usize> = dirty.into_iter().filter(|&i| !img.noreturn.contains(&funcs[i].0)).collect();
        found = dirty.into_par_iter().filter(|&i| !walk_one(&img, &funcs[i]).returns).collect();
    }
    let slots: Vec<u64> = img.slots.iter().filter(|(_, t)| img.noreturn.contains(t)).map(|(&s, _)| s).collect();
    img.noreturn.extend(slots);
    img.noreturn
}

/// The cases of a jump table behind the indirect `jmp` `insn`, as far as they
/// land in `[start, limit)`: `jmp [table + idx*8]` with absolute entries
/// (non-PIE), or `jmp reg` after adding a 32-bit entry to a table base the
/// function `lea`s (PIE). The table has `cases` entries when a bounds check
/// before the jump says so; otherwise (Rust's `match` on a discriminant has
/// none) it is read until an entry lands outside the function. Empty for a tail
/// call through a register or the GOT.
fn jump_table(insn: &Instruction, img: &Image, leas: &[u64], start: u64, limit: u64, cases: Option<u64>) -> Vec<u64> {
    let inside = |t: u64| t >= start && t < limit;
    let n = cases.unwrap_or(1024).min(1024);
    let mut out = Vec::new();
    if insn.op0_kind() == OpKind::Memory && insn.memory_base() == Register::None && insn.memory_index_scale() == 8 {
        let table = insn.memory_displacement64();
        for k in 0..n {
            match img.read(table + 8 * k, 8).map(|b| u64::from_le_bytes(b.try_into().unwrap())) {
                Some(t) if inside(t) => out.push(t),
                _ => break,
            }
        }
    } else if insn.op0_kind() == OpKind::Register {
        for &table in leas.iter().rev() {
            for k in 0..n {
                match img.read(table + 4 * k, 4).map(|b| table.wrapping_add(i32::from_le_bytes(b.try_into().unwrap()) as u64)) {
                    Some(t) if inside(t) => out.push(t),
                    _ => break,
                }
            }
            if !out.is_empty() {
                break;
            }
        }
    }
    out
}

/// The table of code addresses a computed `goto` jumps through: the last base
/// the function `lea`s that holds at least two 8-byte addresses after `start`
/// (filled in by the loader in a PIE), read until an entry isn't one.
fn label_table(img: &Image, leas: &[u64], start: u64) -> Option<Vec<u64>> {
    let entry = |a: u64| img.slots.get(&a).copied().or_else(|| img.read(a, 8).map(|b| u64::from_le_bytes(b.try_into().unwrap())));
    leas.iter().rev().find_map(|&table| {
        let t: Vec<u64> = (0..1024).map(|k| entry(table + 8 * k)).map_while(|e| e.filter(|&e| e > start && img.read(e, 1).is_some())).collect();
        (t.len() >= 2).then_some(t)
    })
}

/// What `walk` needs from the rest of the binary.
struct Image<'a> {
    /// Loaded sections, sorted by address, for reading jump tables.
    mem: Vec<(u64, &'a [u8])>,
    /// PLT stubs and GOT slots of imports that don't return, and functions
    /// known not to return.
    noreturn: HashSet<u64>,
    /// Slots the loader fills with an address in the binary (GOT entries).
    slots: HashMap<u64, u64>,
}

impl<'a> Image<'a> {
    fn new(file: &object::File<'a>) -> Image<'a> {
        let mut mem: Vec<(u64, &[u8])> = file
            .sections()
            .filter(|s| s.address() != 0 && !matches!(s.kind(), SectionKind::UninitializedData | SectionKind::Metadata))
            .filter_map(|s| Some((s.address(), s.data().ok().filter(|d| !d.is_empty())?)))
            .collect();
        mem.sort_by_key(|s| s.0);
        let got = got_names(file);
        let plt = plt_names(file, &got);
        let noreturn = plt.iter().chain(&got).filter(|(_, n)| noreturn(n)).map(|(&a, _)| a).collect();
        Image { mem, noreturn, slots: HashMap::new() }
    }

    /// `call [rip+slot]` doesn't return: the slot holds an import or a function that doesn't.
    fn slot_noreturn(&self, slot: u64) -> bool {
        self.noreturn.contains(&slot) || self.slots.get(&slot).is_some_and(|t| self.noreturn.contains(t))
    }

    fn read(&self, addr: u64, n: usize) -> Option<&[u8]> {
        let k = self.mem.partition_point(|s| s.0 <= addr).checked_sub(1)?;
        let (a, d) = self.mem[k];
        d.get((addr - a) as usize..)?.get(..n)
    }
}

/// Functions that never return, by symbol name: a call to one ends its block,
/// and code after it belongs to the next function. C library imports, and the
/// Rust standard library's panic and allocation-failure functions (`-> !`),
/// which a Rust binary defines itself and often calls through the GOT.
pub fn noreturn(name: &str) -> bool {
    let name = name.split('@').next().unwrap_or(name);
    if matches!(
        name,
        "abort" | "exit" | "_exit" | "_Exit" | "quick_exit" | "__stack_chk_fail" | "__assert_fail"
            | "__assert_perror_fail" | "__fortify_fail" | "__chk_fail" | "err" | "errx" | "verr" | "verrx"
            | "longjmp" | "siglongjmp" | "__longjmp_chk" | "pthread_exit" | "__cxa_throw" | "__cxa_rethrow"
            | "__cxa_bad_cast" | "__cxa_bad_typeid" | "__cxa_throw_bad_array_new_length" | "_Unwind_Resume"
            | "__cxa_pure_virtual" | "__cxa_call_unexpected" | "_ZSt9terminatev" | "rust_begin_unwind"
    ) {
        return true;
    }
    // Identifiers appear verbatim in both Rust manglings, so only names that
    // contain one of the last path segments are demangled.
    if !RUST_NORETURN.iter().any(|p| name.contains(p.rsplit("::").next().unwrap())) && !name.contains("panicking") {
        return false;
    }
    let Some(path) = crate::load::demangle(name) else { return false };
    // Generic arguments: `core::panicking::assert_failed::<u32, u32>`.
    let path = path.split("::<").next().unwrap_or(&path);
    path.starts_with("core::panicking::") || RUST_NORETURN.contains(&path)
}

/// Rust standard library functions that return `!`, besides all of `core::panicking`.
const RUST_NORETURN: &[&str] = &[
    "core::option::unwrap_failed",
    "core::option::expect_failed",
    "core::result::unwrap_failed",
    "core::cell::panic_already_borrowed",
    "core::cell::panic_already_mutably_borrowed",
    "core::slice::index::slice_index_fail",
    "core::slice::index::slice_start_index_len_fail",
    "core::slice::index::slice_end_index_len_fail",
    "core::slice::index::slice_index_order_fail",
    "core::slice::index::slice_start_index_overflow_fail",
    "core::slice::index::slice_end_index_overflow_fail",
    "core::slice::copy_from_slice_impl::len_mismatch_fail",
    "core::str::slice_error_fail",
    "core::str::slice_error_fail_rt",
    "core::str::slice_error_fail_ct",
    "alloc::alloc::handle_alloc_error",
    "alloc::raw_vec::handle_error",
    "alloc::raw_vec::capacity_overflow",
    "std::alloc::rust_oom",
    "std::process::exit",
    "std::process::abort",
    "std::panicking::begin_panic",
    "std::panicking::begin_panic_handler",
    "std::panicking::rust_panic_with_hook",
    "std::panicking::rust_panic",
    "__rustc::rust_begin_unwind",
];

/// Mach-O's `LC_FUNCTION_STARTS`: the start of every function, as ULEB128
/// deltas from the `__TEXT` segment. `strip` keeps it. Empty for other formats.
fn function_starts(file: &object::File) -> Vec<u64> {
    use object::read::macho::LoadCommandVariant;
    let object::File::MachO64(m) = file else { return Vec::new() };
    let Some(text) = file.segments().find(|s| s.name().ok().flatten() == Some("__TEXT")) else { return Vec::new() };
    let mut out = Vec::new();
    let Ok(mut cmds) = m.macho_load_commands() else { return out };
    while let Ok(Some(cmd)) = cmds.next() {
        let Ok(LoadCommandVariant::LinkeditData(l)) = cmd.variant() else { continue };
        let Ok(starts) = l.function_starts(m.endian(), m.data(), text.address()) else { continue };
        out.extend(starts.map_while(Result::ok));
    }
    out
}

/// GOT slot -> the symbol the dynamic loader puts there, from the dynamic relocations.
pub fn got_names(file: &object::File) -> HashMap<u64, String> {
    let mut got = HashMap::new();
    let dynsyms = file.dynamic_symbol_table();
    for (at, r) in file.dynamic_relocations().into_iter().flatten() {
        let RelocationTarget::Symbol(i) = r.target() else { continue };
        let Some(sym) = dynsyms.as_ref().and_then(|t| t.symbol_by_index(i).ok()) else { continue };
        // a weak symbol nothing defines (`__gmon_start__`) leaves its slot
        // null; the code tests it before calling through it
        if sym.is_weak() && sym.is_undefined() {
            continue;
        }
        if let Ok(n) = sym.name() {
            if !n.is_empty() {
                got.insert(at, n.to_string());
            }
        }
    }
    got
}

/// PLT stub -> the import it jumps to. A stub is `[endbr64;] jmp [rip+slot]`
/// with `slot` in `got` (from `got_names`).
pub fn plt_names(file: &object::File, got: &HashMap<u64, String>) -> HashMap<u64, String> {
    let mut plt = HashMap::new();
    for sec in file.sections() {
        if !is_stub_section(sec.name().unwrap_or("")) {
            continue;
        }
        let Ok(code) = sec.data() else { continue };
        let mut entry = None;
        for i in Decoder::with_ip(64, code, sec.address(), DecoderOptions::NONE).iter() {
            if i.mnemonic() == Mnemonic::Endbr64 {
                entry = Some(i.ip());
                continue;
            }
            if i.flow_control() == FlowControl::IndirectBranch && i.is_ip_rel_memory_operand() {
                if let Some(n) = got.get(&i.ip_rel_memory_address()) {
                    plt.insert(entry.unwrap_or(i.ip()), n.clone());
                }
            }
            entry = None;
        }
    }
    plt
}

/// Skip alignment padding (`nop`s, `int3`, zero bytes) at the start of `gap`;
/// if valid code follows, its address. The gap starts where the function before
/// it ends, which follows jump tables and stops at calls that don't return, so
/// nothing reaches this code from there: it is a function nothing calls
/// directly. Compilers don't always align functions or give them a prologue
/// (clang -Os packs leaf functions with at most a few bytes between them).
fn code_after_padding(gap: &[u8], addr: u64) -> Option<u64> {
    let mut dec = Decoder::with_ip(64, gap, addr, DecoderOptions::NONE);
    let mut insn = Instruction::default();
    while dec.can_decode() {
        let pos = dec.position();
        if gap[pos] == 0 {
            // `add [rax], al` is never padding-adjacent code; zero fill is.
            if dec.set_position(pos + 1).is_err() {
                return None;
            }
            dec.set_ip(addr + pos as u64 + 1);
            continue;
        }
        dec.decode_out(&mut insn);
        if insn.is_invalid() {
            return None;
        }
        // `xchg ax, ax` and `data16 cs nop` forms decode as Nop; anything else is code.
        if !matches!(insn.mnemonic(), Mnemonic::Nop | Mnemonic::Int3) {
            return decodes(&gap[pos..], insn.ip()).then_some(insn.ip());
        }
    }
    None
}

/// `b` is all padding: nops, `int3` or zeros.
fn is_padding(b: &[u8]) -> bool {
    let mut dec = Decoder::new(64, b, DecoderOptions::NONE);
    let mut insn = Instruction::default();
    while dec.can_decode() {
        let pos = dec.position();
        if b[pos] == 0 {
            if dec.set_position(pos + 1).is_err() {
                return true;
            }
            continue;
        }
        dec.decode_out(&mut insn);
        if !matches!(insn.mnemonic(), Mnemonic::Nop | Mnemonic::Int3) {
            return false;
        }
    }
    true
}

/// The first few instructions of `b` are valid code, not padding.
fn decodes(b: &[u8], ip: u64) -> bool {
    for i in Decoder::with_ip(64, b, ip, DecoderOptions::NONE).iter().take(4) {
        if i.is_invalid() || i.mnemonic() == Mnemonic::Int3 {
            return false;
        }
        if matches!(i.flow_control(), FlowControl::Return | FlowControl::UnconditionalBranch) {
            break;
        }
    }
    true
}

/// (start, length) of every function with unwind info.
fn unwind_ranges(file: &object::File) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    for s in file.sections() {
        let Ok(data) = s.data() else { continue };
        match s.name() {
            Ok(".eh_frame" | "__eh_frame") => eh_frame(data, s.address(), &mut out),
            Ok(".pdata") => {
                // RUNTIME_FUNCTION: begin RVA, end RVA, unwind info RVA.
                let base = file.relative_address_base();
                for e in data.chunks_exact(12) {
                    let b = u32::from_le_bytes(e[0..4].try_into().unwrap()) as u64;
                    let end = u32::from_le_bytes(e[4..8].try_into().unwrap()) as u64;
                    if b != 0 && end > b {
                        out.push((base + b, end - b));
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// FDE ranges from an `.eh_frame` section loaded at `base`.
pub fn eh_frame(data: &[u8], base: u64, out: &mut Vec<(u64, u64)>) {
    // CIE offset -> pointer encoding of its FDEs.
    let mut cies: BTreeMap<usize, u8> = BTreeMap::new();
    let mut pos = 0;
    while pos + 4 <= data.len() {
        let mut r = Reader { data, pos, base };
        let Some(len) = r.u32() else { break };
        if len == 0 {
            break; // terminator
        }
        let (len, hdr) = if len == 0xffff_ffff { (r.u64().unwrap_or(0) as usize, 12) } else { (len as usize, 4) };
        let body = pos + hdr;
        let next = body.saturating_add(len);
        if next > data.len() {
            break;
        }
        let id_at = r.pos;
        let id = if hdr == 12 { r.u64() } else { r.u32().map(u64::from) };
        match id {
            Some(0) => {
                if let Some(enc) = cie_encoding(&mut r) {
                    cies.insert(pos, enc);
                }
            }
            Some(id) => {
                let cie = id_at.checked_sub(id as usize);
                if let Some(&enc) = cie.and_then(|c| cies.get(&c)) {
                    if let (Some(begin), Some(range)) = (r.pointer(enc), r.pointer(enc & 0x0f)) {
                        out.push((begin, range));
                    }
                }
            }
            None => break,
        }
        pos = next;
    }
}

/// The FDE pointer encoding (`R` augmentation) of the CIE `r` is positioned in
/// (after its id). `DW_EH_PE_absptr` if it has none.
fn cie_encoding(r: &mut Reader) -> Option<u8> {
    let version = r.u8()?;
    let aug_start = r.pos;
    while r.u8()? != 0 {}
    let aug = &r.data[aug_start..r.pos - 1];
    if aug.contains(&b'h') {
        return None; // "eh" (gcc 2.x): an extra pointer; not worth supporting
    }
    r.uleb()?; // code alignment
    r.sleb()?; // data alignment
    if version == 1 {
        r.u8()?;
    } else {
        r.uleb()?; // return address register
    }
    if aug.first() != Some(&b'z') {
        return Some(0);
    }
    r.uleb()?; // augmentation data length
    for &c in &aug[1..] {
        match c {
            b'R' => return r.u8(),
            b'P' => {
                let e = r.u8()?;
                r.pointer(e)?;
            }
            b'L' => {
                r.u8()?;
            }
            b'S' | b'B' => {}
            _ => return None,
        }
    }
    Some(0)
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    /// Address of `data[0]`, for pc-relative pointers.
    base: u64,
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let b = self.data.get(self.pos..self.pos + N)?.try_into().ok()?;
        self.pos += N;
        Some(b)
    }
    fn u8(&mut self) -> Option<u8> {
        self.take::<1>().map(|b| b[0])
    }
    fn u16(&mut self) -> Option<u16> {
        self.take().map(u16::from_le_bytes)
    }
    fn u32(&mut self) -> Option<u32> {
        self.take().map(u32::from_le_bytes)
    }
    fn u64(&mut self) -> Option<u64> {
        self.take().map(u64::from_le_bytes)
    }
    fn uleb(&mut self) -> Option<u64> {
        let (mut v, mut shift) = (0u64, 0);
        loop {
            let b = self.u8()?;
            if shift < 64 {
                v |= u64::from(b & 0x7f) << shift;
            }
            shift += 7;
            if b & 0x80 == 0 {
                return Some(v);
            }
        }
    }
    fn sleb(&mut self) -> Option<i64> {
        let (mut v, mut shift) = (0i64, 0);
        loop {
            let b = self.u8()?;
            if shift < 64 {
                v |= i64::from(b & 0x7f) << shift;
            }
            shift += 7;
            if b & 0x80 == 0 {
                if shift < 64 && b & 0x40 != 0 {
                    v |= -1 << shift;
                }
                return Some(v);
            }
        }
    }
    /// A `DW_EH_PE_*` encoded pointer. Only absolute and pc-relative
    /// applications are understood (the only ones x86_64 toolchains emit in FDEs).
    fn pointer(&mut self, enc: u8) -> Option<u64> {
        if enc == 0xff {
            return None;
        }
        let at = self.base + self.pos as u64;
        let v = match enc & 0x0f {
            0x00 | 0x04 => self.u64()?,
            0x01 => self.uleb()?,
            0x02 => u64::from(self.u16()?),
            0x03 => u64::from(self.u32()?),
            0x09 => self.sleb()? as u64,
            0x0a => self.u16()? as i16 as u64,
            0x0b => self.u32()? as i32 as u64,
            0x0c => self.u64()?,
            _ => return None,
        };
        match enc & 0x70 {
            0x00 => Some(v),
            0x10 => Some(at.wrapping_add(v)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_fdes() {
        // One CIE ("zR", pcrel|sdata4) and one FDE for 0x20 bytes at 0x1000,
        // with the section loaded at 0x2000.
        let mut d = Vec::new();
        let cie = [
            0u8, 0, 0, 0, // id
            1, b'z', b'R', 0, // version, augmentation
            1, 0x78, 16, // code align 1, data align -8, return reg 16
            1, 0x1b, // augmentation data: FDE encoding
            0, 0, 0, // padding (DW_CFA_nop)
        ];
        d.extend((cie.len() as u32).to_le_bytes());
        d.extend(cie);
        let fde_at = d.len();
        let mut fde = Vec::new();
        fde.extend(((fde_at + 4) as u32).to_le_bytes()); // CIE pointer
        let field = 0x2000 + fde_at as i64 + 8;
        fde.extend(((0x1000 - field) as i32).to_le_bytes());
        fde.extend(0x20u32.to_le_bytes());
        fde.extend([0, 0, 0, 0]); // no augmentation data, nops
        d.extend((fde.len() as u32).to_le_bytes());
        d.extend(fde);
        d.extend([0, 0, 0, 0]);
        let mut out = Vec::new();
        eh_frame(&d, 0x2000, &mut out);
        assert_eq!(out, vec![(0x1000, 0x20)]);
    }

    /// A stripped x86_64 Mach-O executable: `__TEXT,__text` at 0x100000100
    /// holding `xor eax, eax; jz next` and then `next: mov rax, rdi; ret`, and,
    /// if `starts`, an `LC_FUNCTION_STARTS` that lists both.
    fn macho(starts: bool) -> Vec<u8> {
        let code = [0x31, 0xc0, 0x74, 0x00, 0x48, 0x89, 0xf8, 0xc3];
        let (base, code_at, blob_at) = (0x1_0000_0000u64, 0x100usize, 0x180usize);
        let mut d = Vec::new();
        let u32s = |d: &mut Vec<u8>, v: &[u32]| v.iter().for_each(|x| d.extend(x.to_le_bytes()));
        let name = |d: &mut Vec<u8>, s: &str| d.extend(s.bytes().chain(std::iter::repeat(0)).take(16));
        let (ncmds, cmds_size) = if starts { (2, 152 + 16) } else { (1, 152) };
        // mach_header_64: magic, x86_64, MH_EXECUTE, flags, reserved
        u32s(&mut d, &[0xfeed_facf, 0x0100_0007, 3, 2, ncmds, cmds_size, 0, 0]);
        // LC_SEGMENT_64 __TEXT, from the start of the file, with one section
        u32s(&mut d, &[0x19, 152]);
        name(&mut d, "__TEXT");
        for v in [base, 0x1000, 0, (code_at + code.len()) as u64] {
            d.extend(v.to_le_bytes());
        }
        u32s(&mut d, &[5, 5, 1, 0]);
        name(&mut d, "__text");
        name(&mut d, "__TEXT");
        for v in [base + code_at as u64, code.len() as u64] {
            d.extend(v.to_le_bytes());
        }
        // offset, align, reloff, nreloc, S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS, reserved
        u32s(&mut d, &[code_at as u32, 4, 0, 0, 0x8000_0400, 0, 0, 0]);
        // ULEB128 offsets from __TEXT: 0x100, then +4; zero ends the list
        let blob = [0x80, 0x02, 0x04, 0x00, 0, 0, 0, 0];
        if starts {
            u32s(&mut d, &[0x26, 16, blob_at as u32, blob.len() as u32]);
        }
        d.resize(code_at, 0);
        d.extend(code);
        d.resize(blob_at, 0);
        d.extend(blob);
        d
    }

    #[test]
    fn macho_function_starts() {
        let found = |starts: bool| {
            let data = macho(starts);
            let file = object::File::parse(&data[..]).unwrap();
            assert_eq!(function_starts(&file), if starts { vec![0x1_0000_0100, 0x1_0000_0104] } else { vec![] });
            functions(&file, &[], &BTreeMap::new()).iter().map(|f| (f.addr, f.size)).collect::<Vec<_>>()
        };
        // Without the list, the jump to `next` stays inside one function.
        assert_eq!(found(false), vec![(0x1_0000_0100, 8)]);
        // With it, the jump is a tail call to the second function.
        assert_eq!(found(true), vec![(0x1_0000_0100, 4), (0x1_0000_0104, 4)]);
    }

    #[test]
    fn rust_noreturn_names() {
        assert!(noreturn("_ZN4core9panicking9panic_fmt17h0123456789abcdefE"));
        assert!(noreturn("_RNvNtCscI6d9CVNmLh_4core6option13unwrap_failed"));
        assert!(noreturn("_RNvNtCs40k4W9msRzi_5alloc7raw_vec12handle_error"));
        assert!(noreturn("_ZN5alloc5alloc18handle_alloc_error17h0123456789abcdefE"));
        assert!(noreturn("abort@GLIBC_2.2.5"));
        // `Fallibility::capacity_overflow` returns an error when allocation may fail.
        assert!(!noreturn("_ZN9hashbrown3raw11Fallibility17capacity_overflow17h0123456789abcdefE"));
        assert!(!noreturn("_ZN4core3fmt5write17h0123456789abcdefE"));
        assert!(!noreturn("main"));
    }

    #[test]
    fn code_after_nops() {
        // nopw 0(%rax,%rax); int3; push rbp
        let gap = [0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00, 0xcc, 0x55, 0x48, 0x89, 0xe5];
        assert_eq!(code_after_padding(&gap, 0x100), Some(0x107));
        // nop; mov rax, rdi; ret
        assert_eq!(code_after_padding(&[0x90, 0x48, 0x89, 0xf8, 0xc3], 0x100), Some(0x101));
        // padding, then bytes that aren't code
        assert_eq!(code_after_padding(&[0x90, 0x06, 0x07], 0x100), None);
        assert_eq!(code_after_padding(&[0x90, 0xcc, 0x00, 0x00], 0x100), None);
    }
}
