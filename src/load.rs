//! Read x86_64 machine code out of an object file (ELF, Mach-O or PE) with `object`,
//! and find the functions and data in it from the symbol table, or, for a stripped
//! binary, from unwind tables and control flow (`discover.rs`).
use object::{
    Architecture, BinaryFormat, Object, ObjectKind, ObjectSection, ObjectSymbol, ObjectSymbolTable, RelocationFlags, RelocationTarget,
    SectionKind, SymbolKind,
};
use std::borrow::Cow;
use std::collections::BTreeMap;
use rayon::prelude::*;

/// A function found in the binary: its name, load address and bytes.
pub struct FuncBytes<'a> {
    /// The symbol as it appears in the binary (mangled).
    pub name: String,
    /// The demangled name without hash or parameter list, if `name` is a Rust or C++ symbol.
    pub demangled: Option<String>,
    pub addr: u64,
    pub bytes: &'a [u8],
}

impl FuncBytes<'_> {
    /// The name to show people: demangled if possible.
    pub fn pretty(&self) -> &str {
        self.demangled.as_deref().unwrap_or(&self.name)
    }
}

/// An allocated data section: `.data`, `.rodata`, `.bss`, `__const`, `.rdata`...
pub struct DataSection<'a> {
    pub name: String,
    pub addr: u64,
    pub size: u64,
    /// The file contents; `None` for zero-initialized sections (`.bss`). Owned
    /// only for the thread-local block (`Tls`), which is assembled.
    pub bytes: Option<Cow<'a, [u8]>>,
    /// Writable at run time (`.data`, `.bss`), so its static must be `static mut`.
    pub writable: bool,
    /// Code (`.text`): only the data in it a label marks is a static, such as
    /// a table after a function in hand-written assembly.
    pub code: bool,
}

/// A data object symbol (`STT_OBJECT`): a named static, string or table.
pub struct DataSym {
    pub name: String,
    pub demangled: Option<String>,
    pub addr: u64,
    /// 0 when the symbol table doesn't say.
    pub size: u64,
    /// Index into `Binary::data`.
    pub section: usize,
}

impl DataSym {
    pub fn pretty(&self) -> &str {
        self.demangled.as_deref().unwrap_or(&self.name)
    }
}

/// A symbol a shared library defines, `addend` bytes in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Import {
    pub name: String,
    /// Data (`stdout`) rather than a function.
    pub data: bool,
    pub addend: i64,
}

/// The largest thread-local block loaded; a bigger one is left out.
const MAX_TLS: u64 = 1 << 28;

pub struct Binary<'a> {
    file: object::File<'a>,
    /// Defined functions, sorted by address, one per address.
    pub funcs: Vec<FuncBytes<'a>>,
    /// Allocated data sections, sorted by address. Empty for relocatable objects,
    /// whose section addresses are all 0 and mean nothing until linked.
    pub data: Vec<DataSection<'a>>,
    /// Data symbols inside `data`, sorted by address, one per address.
    pub data_syms: Vec<DataSym>,
    /// Pointer-sized slots in `data` that the dynamic loader fills in (GOT entries,
    /// vtables, pointer tables in a PIE), and what they point to.
    pub pointers: BTreeMap<u64, u64>,
    /// Data a shared library defines that the dynamic loader copies into the
    /// binary at start (`R_X86_64_COPY`: `stdin`, `stderr`, `environ`), by address,
    /// with the library symbol's name.
    pub copied: BTreeMap<u64, String>,
    /// Pointer-sized slots the dynamic loader fills with the address of a symbol
    /// another library defines (`stdout`, `optarg`, or `free` passed as a callback),
    /// by address. Weak imports, null when nothing defines them, are left out.
    pub imports: BTreeMap<u64, Import>,
    /// How many of `funcs` came from `discover::functions` rather than a symbol
    /// (nonzero only when the static symbol table has no functions).
    pub discovered: usize,
}

#[derive(Debug)]
pub enum LoadError {
    Parse(object::Error),
    /// Only x86_64 code can be lifted.
    Arch(Architecture),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Parse(e) => write!(f, "not a readable object file: {e}"),
            LoadError::Arch(a) => write!(f, "unsupported architecture {a:?}; only x86_64 is supported"),
        }
    }
}

impl<'a> Binary<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Binary<'a>, LoadError> {
        let file = object::File::parse(data).map_err(LoadError::Parse)?;
        if file.architecture() != Architecture::X86_64 {
            return Err(LoadError::Arch(file.architecture()));
        }

        // Start address of every symbol, per section, for symbols without a
        // size. An assembler's local labels (`.loop`, `foo.done`) are inside
        // functions, not the starts of new ones.
        let mut starts: Vec<(usize, u64)> = file
            .symbols()
            .filter(|s| s.kind() == SymbolKind::Text || !s.is_local())
            .filter_map(|s| Some((s.section_index()?.0, s.address())))
            .collect();
        starts.sort_unstable();
        starts.dedup();
        // and of every symbol, labels included, for the extent of a label
        let mut all_starts: Vec<(usize, u64)> = file.symbols().filter_map(|s| Some((s.section_index()?.0, s.address()))).collect();
        all_starts.sort_unstable();
        all_starts.dedup();

        let mut funcs = Vec::new();
        // The static symbol table first; a stripped binary still has its dynamic exports.
        for sym in file.symbols().chain(file.dynamic_symbols()) {
            // A global symbol without a type in code is a function too: hand-written
            // assembly often doesn't say (`global f` without `:function`).
            let untyped = sym.kind() == SymbolKind::Unknown && !sym.is_local();
            if !(sym.kind() == SymbolKind::Text || untyped) || !sym.is_definition() {
                continue;
            }
            let Some(idx) = sym.section_index() else { continue };
            let Ok(sec) = file.section_by_index(idx) else { continue };
            if sec.kind() != SectionKind::Text {
                continue;
            }
            let size = if sym.size() > 0 { sym.size() } else { size_to_next_symbol(&starts, sym.address(), &sec) };
            if size == 0 {
                continue;
            }
            let Ok(Some(bytes)) = sec.data_range(sym.address(), size) else { continue };
            let name = sym.name().unwrap_or("").to_string();
            let name = if name.is_empty() { format!("sub_{:x}", sym.address()) } else { name };
            funcs.push(FuncBytes { name, demangled: None, addr: sym.address(), bytes });
        }
        funcs.sort_by_key(|f| f.addr);
        funcs.dedup_by_key(|f| f.addr);

        let mut sections = Vec::new();
        if file.kind() != ObjectKind::Relocatable {
            for sec in file.sections() {
                let writable = match sec.kind() {
                    SectionKind::Data | SectionKind::UninitializedData => true,
                    SectionKind::ReadOnlyData | SectionKind::ReadOnlyDataWithRel | SectionKind::ReadOnlyString | SectionKind::Text => false,
                    _ => continue,
                };
                if sec.address() == 0 || sec.size() == 0 {
                    continue;
                }
                let bytes = match sec.kind() {
                    SectionKind::UninitializedData => None,
                    _ => match sec.data() {
                        Ok(d) if d.len() as u64 == sec.size() => Some(Cow::Borrowed(d)),
                        _ => continue,
                    },
                };
                let name = sec.name().unwrap_or("").to_string();
                let code = sec.kind() == SectionKind::Text;
                sections.push((sec.index().0, DataSection { name, addr: sec.address(), size: sec.size(), bytes, writable, code }));
            }
            sections.sort_by_key(|(_, s)| s.addr);
        }

        let mut data_syms = Vec::new();
        for sym in file.symbols().chain(file.dynamic_symbols()) {
            let Some(idx) = sym.section_index() else { continue };
            let Some(section) = sections.iter().position(|(i, _)| *i == idx.0) else { continue };
            let s = &sections[section].1;
            // In code, a local label without a type: what an assembler makes of
            // `table:` (or `.tab:`), which may be data, up to the next symbol.
            let label = s.code && sym.kind() == SymbolKind::Unknown && sym.is_local();
            if !(sym.kind() == SymbolKind::Data && !s.code || label) || !sym.is_definition() {
                continue;
            }
            let (addr, end) = (sym.address(), s.addr + s.size);
            if addr < s.addr || addr >= end {
                continue;
            }
            let Ok(name) = sym.name() else { continue };
            if name.is_empty() {
                continue;
            }
            let size = match label {
                true => all_starts.get(all_starts.partition_point(|&a| a <= (idx.0, addr))).filter(|a| a.0 == idx.0).map_or(end, |a| a.1) - addr,
                false => sym.size().min(end - addr),
            };
            data_syms.push(DataSym { name: name.to_string(), demangled: None, addr, size, section });
        }
        data_syms.sort_by(|a, b| a.addr.cmp(&b.addr).then(b.size.cmp(&a.size)));
        data_syms.dedup_by_key(|s| s.addr);

        // Demangling is the slowest part of loading a big binary; it is per symbol.
        funcs.par_iter_mut().for_each(|f| f.demangled = demangle(&f.name));
        data_syms.par_iter_mut().for_each(|s| s.demangled = demangle(&s.name));

        let mut data: Vec<DataSection> = sections.into_iter().map(|(_, s)| s).collect();
        // The thread-local block goes above everything else, so it stays last.
        // (a corrupt header can claim any size; real blocks are kilobytes)
        if let Some(t) = tls(&file).filter(|t| t.size <= MAX_TLS && t.image.len() as u64 <= t.size) {
            let mut block = vec![0; t.size as usize];
            block[..t.image.len()].copy_from_slice(t.image);
            data.push(DataSection { name: TLS_SECTION.into(), addr: t.addr, size: t.size, bytes: Some(Cow::Owned(block)), writable: true, code: false });
        }
        let pointers = if data.is_empty() { BTreeMap::new() } else { pointers(&file, &data) };
        let copied = copied(&file);
        let imports = if data.is_empty() { BTreeMap::new() } else { imports(&file) };

        // Stripped: no function in the static symbol table, at most the dynamic exports.
        let stripped = file.kind() != ObjectKind::Relocatable
            && !file.symbols().any(|s| s.kind() == SymbolKind::Text && s.is_definition());
        let mut discovered = 0;
        if stripped {
            let known: Vec<(u64, u64)> = funcs.iter().map(|f| (f.addr, f.bytes.len() as u64)).collect();
            for d in crate::discover::functions(&file, &known, &pointers) {
                if let Some(f) = funcs.iter_mut().find(|f| f.addr == d.addr) {
                    // An export: keep its name. Its size without a static symbol table ran
                    // to the next export; the unwind table's extent is exact.
                    f.bytes = code_in(&file, d.addr, d.size).unwrap_or(f.bytes);
                    continue;
                }
                let Some(bytes) = code_in(&file, d.addr, d.size) else { continue };
                let name = d.name.map_or_else(|| format!("sub_{:x}", d.addr), str::to_string);
                funcs.push(FuncBytes { name, demangled: None, addr: d.addr, bytes });
                discovered += 1;
            }
            funcs.sort_by_key(|f| f.addr);
        }
        Ok(Binary { file, funcs, data, data_syms, pointers, copied, imports, discovered })
    }

    pub fn entry(&self) -> u64 {
        self.file.entry()
    }

    /// Why the binary looks packed (its code compressed or encrypted, unpacked
    /// at run time), if it does: a decompiler then sees only the unpacking
    /// stub. `data` is the whole file.
    pub fn packed(&self, data: &[u8]) -> Option<String> {
        let names: Vec<&str> = self.file.sections().filter_map(|s| s.name().ok()).collect();
        let upx_sections = names.iter().any(|n| n.starts_with("UPX") || *n == ".upx");
        // UPX's header, near the start or after the stub, with no section table left
        let upx_magic = data.windows(4).take(1024).any(|w| w == b"UPX!") || data.windows(4).any(|w| w == b"UPX!") && names.is_empty();
        if upx_sections || upx_magic {
            return Some("it is packed with UPX, so only the unpacking stub can be decompiled; unpack it with `upx -d` first".into());
        }
        // Executable segments that are mostly random-looking bytes.
        use object::{ObjectSegment, SegmentFlags};
        let exec = |f: SegmentFlags| match f {
            SegmentFlags::Elf { p_flags, .. } => p_flags.0 & object::elf::PF_X.0 != 0,
            SegmentFlags::MachO { initprot, .. } => initprot.0 & object::macho::VM_PROT_EXECUTE.0 != 0,
            SegmentFlags::Coff { characteristics } => characteristics.0 & object::pe::IMAGE_SCN_MEM_EXECUTE.0 != 0,
            _ => false,
        };
        let mut code: Vec<&[u8]> = self.file.segments().filter(|g| exec(g.flags())).filter_map(|g| g.data().ok()).collect();
        if code.is_empty() {
            code = self.file.sections().filter(|s| s.kind() == SectionKind::Text).filter_map(|s| s.data().ok()).collect();
        }
        let n: usize = code.iter().map(|c| c.len()).sum();
        if n < 4096 {
            return None;
        }
        let mut counts = [0u64; 256];
        for c in &code {
            for &b in c.iter() {
                counts[b as usize] += 1;
            }
        }
        let bits: f64 = counts.iter().filter(|&&c| c > 0).map(|&c| c as f64 / n as f64).map(|p| -p * p.log2()).sum();
        // Machine code is about 5.5 to 6.5 bits a byte; compressed data close to 8.
        (bits > 7.4).then(|| format!(
            "its code looks compressed or encrypted ({bits:.1} bits of entropy a byte), as a packer leaves it; unpack it first, or decompile a memory dump taken after it unpacks"
        ))
    }

    /// The function at `addr`, by exact start address.
    pub fn func_at(&self, addr: u64) -> Option<&FuncBytes<'a>> {
        self.funcs.binary_search_by_key(&addr, |f| f.addr).ok().map(|i| &self.funcs[i])
    }

    /// Up to `len` bytes of code starting at `addr` (clipped to the end of its
    /// section), for functions the symbol table misses.
    pub fn code_at(&self, addr: u64, len: u64) -> Option<&'a [u8]> {
        code_in(&self.file, addr, len)
    }

    /// Name of the function that starts at `addr`, if any.
    pub fn name_at(&self, addr: u64) -> Option<&str> {
        self.func_at(addr).map(|f| f.name.as_str())
    }

    /// The data section containing `addr`.
    pub fn data_section_at(&self, addr: u64) -> Option<usize> {
        let i = self.data.partition_point(|s| s.addr <= addr).checked_sub(1)?;
        (addr < self.data[i].addr + self.data[i].size).then_some(i)
    }
}

fn code_in<'a>(file: &object::File<'a>, addr: u64, len: u64) -> Option<&'a [u8]> {
    let sec = file
        .sections()
        .find(|s| s.kind() == SectionKind::Text && addr >= s.address() && addr < s.address() + s.size())?;
    let len = len.min(sec.address() + sec.size() - addr);
    sec.data_range(addr, len).ok().flatten()
}

/// Slots the loader fills with an address: 64-bit dynamic relocations whose
/// target is known (`R_X86_64_RELATIVE`, or a symbol the binary defines), plus,
/// for a binary without dynamic relocations (static, non-PIE), every nonzero GOT slot.
fn copied(file: &object::File) -> BTreeMap<u64, String> {
    let mut out = BTreeMap::new();
    let Some(dynsyms) = file.dynamic_symbol_table() else { return out };
    for (at, r) in file.dynamic_relocations().into_iter().flatten() {
        let (RelocationFlags::Elf { r_type: object::elf::R_X86_64_COPY }, RelocationTarget::Symbol(i)) = (r.flags(), r.target()) else { continue };
        let Some(name) = dynsyms.symbol_by_index(i).ok().and_then(|s| s.name().ok().map(str::to_string)) else { continue };
        // `stderr@GLIBC_2.2.5` in some tables
        let name = name.split('@').next().unwrap_or_default().to_string();
        if !name.is_empty() {
            out.insert(at, name);
        }
    }
    out
}

fn imports(file: &object::File) -> BTreeMap<u64, Import> {
    use object::elf::{R_X86_64_64, R_X86_64_GLOB_DAT};
    let mut out = BTreeMap::new();
    let Some(dynsyms) = file.dynamic_symbol_table() else { return out };
    for (at, r) in file.dynamic_relocations().into_iter().flatten() {
        let (RelocationFlags::Elf { r_type: R_X86_64_64 | R_X86_64_GLOB_DAT }, RelocationTarget::Symbol(i)) = (r.flags(), r.target()) else { continue };
        let Ok(sym) = dynsyms.symbol_by_index(i) else { continue };
        let data = match sym.kind() {
            SymbolKind::Data => true,
            SymbolKind::Text => false,
            _ => continue, // `__gmon_start__`, untyped: nothing to say what it is
        };
        if sym.is_definition() || sym.is_weak() {
            continue;
        }
        let Ok(name) = sym.name() else { continue };
        let name = name.split('@').next().unwrap_or_default();
        if !name.is_empty() {
            out.insert(at, Import { name: name.to_string(), data, addend: r.addend() });
        }
    }
    out
}

fn pointers(file: &object::File, data: &[DataSection]) -> BTreeMap<u64, u64> {
    use object::elf::{R_X86_64_64, R_X86_64_GLOB_DAT, R_X86_64_JUMP_SLOT, R_X86_64_RELATIVE};
    let mut out = BTreeMap::new();
    let dynsyms = file.dynamic_symbol_table();
    let mut relocated = false;
    for (at, r) in file.dynamic_relocations().into_iter().flatten() {
        relocated = true;
        let RelocationFlags::Elf { r_type } = r.flags() else { continue };
        if !matches!(r_type, R_X86_64_64 | R_X86_64_GLOB_DAT | R_X86_64_JUMP_SLOT | R_X86_64_RELATIVE) {
            continue;
        }
        let target = match r.target() {
            RelocationTarget::Absolute => r.addend() as u64,
            RelocationTarget::Symbol(i) => {
                let Some(sym) = dynsyms.as_ref().and_then(|t| t.symbol_by_index(i).ok()) else { continue };
                if !sym.is_definition() {
                    continue; // an import: resolved from another library at run time
                }
                sym.address().wrapping_add(r.addend() as u64)
            }
            _ => continue,
        };
        out.insert(at, target);
    }
    for sec in data {
        if relocated || !matches!(sec.name.as_str(), ".got" | ".got.plt" | "__got") {
            continue;
        }
        let Some(bytes) = &sec.bytes else { continue };
        for (k, w) in bytes.chunks_exact(8).enumerate() {
            let at = sec.addr + 8 * k as u64;
            let v = u64::from_le_bytes(w.try_into().unwrap());
            if v != 0 {
                out.insert(at, v);
            }
        }
    }
    out
}

/// Symbols without a size (common in hand-written assembly): run to the next
/// symbol in the same section, or to the end of the section. `starts` is every
/// (section index, symbol address), sorted.
fn size_to_next_symbol(starts: &[(usize, u64)], addr: u64, sec: &object::Section) -> u64 {
    let end = sec.address().saturating_add(sec.size());
    let key = (sec.index().0, addr);
    let i = starts.partition_point(|&s| s <= key);
    let next = starts.get(i).filter(|s| s.0 == key.0).map_or(end, |s| s.1);
    next.min(end).saturating_sub(addr)
}

/// Demangle a Rust (legacy or v0) or Itanium C++ symbol, without the Rust hash
/// or the C++ parameter list: `_ZN4core3ptr13drop_in_place17h0123E` becomes
/// `core::ptr::drop_in_place`, `_ZN3foo3barEi` becomes `foo::bar`. Mach-O's
/// extra leading underscore is accepted. `None` for anything else.
pub fn demangle(sym: &str) -> Option<String> {
    let s = if sym.starts_with("__Z") || sym.starts_with("__R") { &sym[1..] } else { sym };
    if !(s.starts_with("_Z") || s.starts_with("_R")) {
        return None;
    }
    if let Ok(d) = rustc_demangle::try_demangle(s) {
        return Some(format!("{d:#}"));
    }
    let d = cpp_demangle::Symbol::new(s).ok()?;
    let opts = cpp_demangle::DemangleOptions::new().no_params().no_return_type();
    d.demangle_with_options(&opts).ok()
}

/// Name of the data section `Binary::parse` makes for the thread-local block.
pub const TLS_SECTION: &str = ".tls";

/// An executable's thread-local block (`.tdata`, then `.tbss`) as the decompiled
/// code sees it. The x86-64 ABI puts the main program's block just below the
/// thread pointer, which `fs:0` holds (variant II), so a thread-local is read as
/// `fs:[-k]`. The lifter turns those into addresses in this block, which becomes
/// one `static`: every thread of the decompiled program shares it, which is right
/// for one thread (as the atomics are).
///
/// `.tbss` has no addresses of its own (it overlaps the sections after `.tdata`),
/// so the block is placed above every section.
pub struct Tls<'a> {
    pub addr: u64,
    /// The block, then the 8-byte word at the thread pointer.
    pub size: u64,
    /// `.tdata`'s bytes, at the start of the block; the rest is zero.
    pub image: &'a [u8],
    /// What `fs:0` holds: the end of the block.
    pub thread_pointer: u64,
}

/// The thread-local block of a linked ELF file that has one.
pub fn tls<'a>(file: &object::File<'a>) -> Option<Tls<'a>> {
    if file.format() != BinaryFormat::Elf || file.kind() == ObjectKind::Relocatable {
        return None;
    }
    let is_tls = |k: SectionKind| matches!(k, SectionKind::Tls | SectionKind::UninitializedTls);
    let secs: Vec<_> = file.sections().filter(|s| is_tls(s.kind()) && s.size() != 0).collect();
    let start = secs.iter().map(|s| s.address()).min()?;
    let end = secs.iter().map(|s| s.address().saturating_add(s.size())).max()?;
    let align = secs.iter().map(|s| s.align()).max().unwrap_or(1).max(16);
    if align > 1 << 16 || end < start {
        return None; // a corrupt header
    }
    // The initialized part: one `.tdata` at the start (what linkers produce).
    let image = match secs.iter().filter(|s| s.kind() == SectionKind::Tls).collect::<Vec<_>>()[..] {
        [] => &[][..],
        [d] if d.address() == start => d.data().ok().filter(|b| b.len() as u64 == d.size())?,
        _ => return None,
    };
    // The block's size rounded up to its alignment is how far below the thread
    // pointer it starts (glibc's `l_tls_offset` for the executable).
    let offset = (end - start).checked_next_multiple_of(align)?;
    let top = file.sections().map(|s| s.address().saturating_add(s.size())).max()?;
    let addr = top.checked_add(0x1000)?.checked_next_multiple_of(align.max(0x1000))?;
    Some(Tls { addr, size: offset.checked_add(8)?, image, thread_pointer: addr.checked_add(offset)? })
}

#[cfg(test)]
mod tests {
    use super::demangle;

    #[test]
    fn demangles_rust_and_cpp() {
        let d = |s| demangle(s).unwrap_or_default();
        assert_eq!(d("_ZN4core3ptr13drop_in_place17h0123456789abcdefE"), "core::ptr::drop_in_place");
        assert_eq!(d("__ZN4core3ptr13drop_in_place17h0123456789abcdefE"), "core::ptr::drop_in_place");
        assert_eq!(d("_RNvCs1234_7mycrate3foo"), "mycrate::foo");
        assert_eq!(d("_ZN3foo3barEi"), "foo::bar");
        assert_eq!(d("_ZNSt6vectorIiSaIiEE9push_backERKi"), "std::vector<int, std::allocator<int> >::push_back");
        assert_eq!(demangle("main"), None);
        assert_eq!(demangle("_Zgarbage"), None);
    }
}
