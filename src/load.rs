//! Read x86_64 machine code out of an object file (ELF, Mach-O or PE) with `object`,
//! and find the functions in it from the symbol table.
use object::{Architecture, Object, ObjectSection, ObjectSymbol, SectionKind, SymbolKind};

/// A function found in the binary: its name, load address and bytes.
pub struct FuncBytes<'a> {
    pub name: String,
    pub addr: u64,
    pub bytes: &'a [u8],
}

pub struct Binary<'a> {
    file: object::File<'a>,
    /// Defined functions, sorted by address, one per address.
    pub funcs: Vec<FuncBytes<'a>>,
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
            let size = if sym.size() > 0 { sym.size() } else { size_to_next_symbol(&file, sym.address(), &sec) };
            let Ok(Some(bytes)) = sec.data_range(sym.address(), size) else { continue };
            let name = sym.name().unwrap_or("").to_string();
            let name = if name.is_empty() { format!("sub_{:x}", sym.address()) } else { name };
            funcs.push(FuncBytes { name, addr: sym.address(), bytes });
        }
        funcs.sort_by_key(|f| f.addr);
        funcs.dedup_by_key(|f| f.addr);
        Ok(Binary { file, funcs })
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
        let sec = self.file.sections().find(|s| {
            s.kind() == SectionKind::Text && addr >= s.address() && addr < s.address() + s.size()
        })?;
        let len = len.min(sec.address() + sec.size() - addr);
        sec.data_range(addr, len).ok().flatten()
    }

    /// Name of the function that starts at `addr`, if any.
    pub fn name_at(&self, addr: u64) -> Option<&str> {
        self.func_at(addr).map(|f| f.name.as_str())
    }
}

/// Symbols without a size (common in hand-written assembly): run to the next
/// symbol in the same section, or to the end of the section.
fn size_to_next_symbol(file: &object::File, addr: u64, sec: &object::Section) -> u64 {
    let end = sec.address() + sec.size();
    let next = file
        .symbols()
        .filter(|s| s.section_index() == Some(sec.index()) && s.address() > addr)
        .map(|s| s.address())
        .min()
        .unwrap_or(end);
    next.min(end) - addr
}
