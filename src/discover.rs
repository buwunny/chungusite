//! Find functions in a binary that has no symbol table (a stripped build).
//!
//! Sources, most trusted first:
//! * unwind tables: `.eh_frame` FDEs (ELF, Mach-O) and `.pdata` (PE) give the
//!   exact start and length of every function compiled with unwind info, which
//!   is nearly all of them on x86_64;
//! * the dynamic symbols (exports), which `strip` keeps;
//! * the entry point and the start of each code section (`_init`, `_fini`);
//! * targets of direct calls, `lea reg, [rip+x]` into code (function pointers,
//!   `main` in `_start`), and code addresses stored in data (vtables, tables);
//! * code that nothing above covers, after alignment padding.
//!
//! Functions without unwind info end where control flow from their start stops
//! (the last instruction reached before the next known start, following jump
//! tables and stopping at calls to imports that don't return, like
//! `__stack_chk_fail`), so the padding
//! after them is not included and whatever follows the padding is checked for
//! a prologue. A candidate strictly inside an FDE's range is rejected: unwind
//! info is authoritative, and such addresses are mostly the cold half of a
//! split function or a misread of something that isn't a call.
use iced_x86::{Code, ConditionCode, Decoder, DecoderOptions, FlowControl, Instruction, Mnemonic, OpKind, Register};
use object::{Object, ObjectSection, ObjectSymbol, ObjectSymbolTable, RelocationTarget, SectionKind};
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
/// known with their sizes (dynamic exports); `pointers` are values the loader
/// writes into data (`Binary::pointers`), some of which are function addresses.
pub fn functions(file: &object::File, known: &[(u64, u64)], pointers: impl IntoIterator<Item = u64>) -> Vec<Found> {
    let mut code: Vec<Text> = file
        .sections()
        .filter(|s| s.kind() == SectionKind::Text && s.size() > 0 && !is_stub_section(s.name().unwrap_or("")))
        .filter_map(|s| Some(Text { addr: s.address(), bytes: s.data().ok().filter(|d| d.len() as u64 == s.size())? }))
        .collect();
    code.sort_by_key(|c| c.addr);
    let in_code = |a: u64| code.iter().position(|c| a >= c.addr && a < c.addr + c.bytes.len() as u64);
    let img = Image::new(file);

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
    for p in pointers {
        seed(p, &mut starts);
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
    loop {
        let mut new = Vec::new();
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
                if !starts.contains(&t) && in_code(t).is_some() && !interior(t) && !(t > a && t < w.end) {
                    new.push(t);
                }
            }
        }
        if new.is_empty() {
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
                            if !interior(p) {
                                new.push(p);
                            }
                        }
                    }
                    at = at.max(e);
                }
            }
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
    let mut insn = Instruction::default();
    while let Some(at) = todo.pop() {
        let mut ip = at;
        // `cmp idx, n` (or clang's `sub idx, n`) just before, and the number of
        // cases its `ja` (`jae`) leaves
        let mut cmp_imm: Option<u64> = None;
        let mut cases = None;
        while ip < limit && seen.insert(ip) {
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
            match insn.flow_control() {
                FlowControl::Call if insn.op0_kind() == OpKind::NearBranch64 => {
                    let t = insn.near_branch_target();
                    refs.push(t);
                    if img.noreturn.contains(&t) {
                        break;
                    }
                }
                FlowControl::IndirectCall => {
                    rdi_done = true;
                    if insn.is_ip_rel_memory_operand() && img.noreturn.contains(&insn.ip_rel_memory_address()) {
                        break;
                    }
                }
                FlowControl::IndirectBranch => {
                    todo.extend(jump_table(&insn, img, &leas, start, limit, cases));
                    break;
                }
                FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch
                    if insn.op0_kind() == OpKind::NearBranch64 =>
                {
                    let t = insn.near_branch_target();
                    if t >= start && t < limit {
                        todo.push(t);
                    } else {
                        refs.push(t); // tail call
                    }
                    if insn.flow_control() == FlowControl::UnconditionalBranch {
                        break;
                    }
                }
                FlowControl::Return | FlowControl::Interrupt => break,
                FlowControl::Exception if insn.mnemonic() == Mnemonic::Ud2 => break,
                _ if insn.mnemonic() == Mnemonic::Hlt => break,
                _ => {}
            }
        }
    }
    if exact {
        end = limit;
    }
    Walk { limit, end, refs, rdi_code_ref: rdi }
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

/// What `walk` needs from the rest of the binary.
struct Image<'a> {
    /// Loaded sections, sorted by address, for reading jump tables.
    mem: Vec<(u64, &'a [u8])>,
    /// PLT stubs and GOT slots of imports that don't return.
    noreturn: HashSet<u64>,
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
        Image { mem, noreturn }
    }

    fn read(&self, addr: u64, n: usize) -> Option<&[u8]> {
        let k = self.mem.partition_point(|s| s.0 <= addr).checked_sub(1)?;
        let (a, d) = self.mem[k];
        d.get((addr - a) as usize..)?.get(..n)
    }
}

/// Imports that don't return: code after a call to one belongs to the next function.
/// Imports that never return: a call to one ends its block.
pub fn noreturn(name: &str) -> bool {
    let name = name.split('@').next().unwrap_or(name);
    matches!(
        name,
        "abort" | "exit" | "_exit" | "_Exit" | "quick_exit" | "__stack_chk_fail" | "__assert_fail"
            | "__assert_perror_fail" | "__fortify_fail" | "__chk_fail" | "err" | "errx" | "verr" | "verrx"
            | "longjmp" | "siglongjmp" | "__longjmp_chk" | "pthread_exit" | "__cxa_throw" | "__cxa_rethrow"
            | "__cxa_bad_cast" | "__cxa_bad_typeid" | "__cxa_throw_bad_array_new_length" | "_Unwind_Resume"
            | "__cxa_pure_virtual" | "__cxa_call_unexpected" | "_ZSt9terminatev"
    )
}

/// GOT slot -> the symbol the dynamic loader puts there, from the dynamic relocations.
pub fn got_names(file: &object::File) -> HashMap<u64, String> {
    let mut got = HashMap::new();
    let dynsyms = file.dynamic_symbol_table();
    for (at, r) in file.dynamic_relocations().into_iter().flatten() {
        let RelocationTarget::Symbol(i) = r.target() else { continue };
        let Some(sym) = dynsyms.as_ref().and_then(|t| t.symbol_by_index(i).ok()) else { continue };
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
