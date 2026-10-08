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

        // Start address of every symbol, per section, for symbols without a size.
        let mut starts: Vec<(usize, u64)> = file
            .symbols()
            .filter_map(|s| Some((s.section_index()?.0, s.address())))
            .collect();
        starts.sort_unstable();
        starts.dedup();

        let mut funcs = Vec::new();
        // The static symbol table first; a stripped binary still has its dynamic exports.
        for sym in file.symbols().chain(file.dynamic_symbols()) {
            if sym.kind() != SymbolKind::Text || !sym.is_definition() {
                continue;
            }
            let Some(idx) = sym.section_index() else { continue };
            let Ok(sec) = file.section_by_index(idx) else { continue };
            if sec.kind() != SectionKind::Text {
                continue;
            }
            let size = if sym.size() > 0 { sym.size() } else { size_to_next_symbol(&starts, sym.address(), &sec) };
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
                    SectionKind::ReadOnlyData | SectionKind::ReadOnlyDataWithRel | SectionKind::ReadOnlyString => false,
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
                sections.push((sec.index().0, DataSection { name, addr: sec.address(), size: sec.size(), bytes, writable }));
            }
            sections.sort_by_key(|(_, s)| s.addr);
        }

        let mut data_syms = Vec::new();
        for sym in file.symbols().chain(file.dynamic_symbols()) {
            if sym.kind() != SymbolKind::Data || !sym.is_definition() {
                continue;
            }
            let Some(idx) = sym.section_index() else { continue };
            let Some(section) = sections.iter().position(|(i, _)| *i == idx.0) else { continue };
            let s = &sections[section].1;
            let (addr, end) = (sym.address(), s.addr + s.size);
            if addr < s.addr || addr >= end {
                continue;
            }
            let Ok(name) = sym.name() else { continue };
            if name.is_empty() {
                continue;
            }
            let size = sym.size().min(end - addr);
            data_syms.push(DataSym { name: name.to_string(), demangled: None, addr, size, section });
        }
        data_syms.sort_by(|a, b| a.addr.cmp(&b.addr).then(b.size.cmp(&a.size)));
        data_syms.dedup_by_key(|s| s.addr);

        // Demangling is the slowest part of loading a big binary; it is per symbol.
        funcs.par_iter_mut().for_each(|f| f.demangled = demangle(&f.name));
        data_syms.par_iter_mut().for_each(|s| s.demangled = demangle(&s.name));

        let mut data: Vec<DataSection> = sections.into_iter().map(|(_, s)| s).collect();
        // The thread-local block goes above everything else, so it stays last.
        if let Some(t) = tls(&file) {
            let mut block = vec![0; t.size as usize];
            block[..t.image.len()].copy_from_slice(t.image);
            data.push(DataSection { name: TLS_SECTION.into(), addr: t.addr, size: t.size, bytes: Some(Cow::Owned(block)), writable: true });
        }
        let pointers = if data.is_empty() { BTreeMap::new() } else { pointers(&file, &data) };

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
        Ok(Binary { file, funcs, data, data_syms, pointers, discovered })
    }

    pub fn entry(&self) -> u64 {
        self.file.entry()
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
    let end = sec.address() + sec.size();
    let key = (sec.index().0, addr);
    let i = starts.partition_point(|&s| s <= key);
    let next = starts.get(i).filter(|s| s.0 == key.0).map_or(end, |s| s.1);
    next.min(end) - addr
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
    let end = secs.iter().map(|s| s.address() + s.size()).max()?;
    let align = secs.iter().map(|s| s.align()).max().unwrap_or(1).max(16);
    // The initialized part: one `.tdata` at the start (what linkers produce).
    let image = match secs.iter().filter(|s| s.kind() == SectionKind::Tls).collect::<Vec<_>>()[..] {
        [] => &[][..],
        [d] if d.address() == start => d.data().ok().filter(|b| b.len() as u64 == d.size())?,
        _ => return None,
    };
    // The block's size rounded up to its alignment is how far below the thread
    // pointer it starts (glibc's `l_tls_offset` for the executable).
    let offset = (end - start).next_multiple_of(align);
    let top = file.sections().map(|s| s.address() + s.size()).max()?;
    let addr = top.checked_add(0x1000)?.next_multiple_of(align.max(0x1000));
    Some(Tls { addr, size: offset + 8, image, thread_pointer: addr + offset })
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
