//! Globals: constant addresses into the binary's data sections become `static`s.
//!
//! The lifter turns a RIP-relative (or absolute) memory operand into
//! `IntToPtr(Const(addr))`. Left alone, the emitted Rust would read `addr` in its
//! own process, which holds nothing of the original's. Instead each such address
//! is mapped to the data item that contains it and emitted as that item's address
//! plus an offset, and the item becomes a `static` holding the bytes from the file.
//!
//! An item is the data symbol covering the address when there is one, sized by the
//! symbol table. Otherwise (string literals, anonymous constants) it is the whole
//! unsymbolized gap between the neighbouring symbols in that section, so indexing
//! from the referenced address stays inside the static. Writable sections (`.data`,
//! `.bss`) become `static mut`.
//!
//! Pointer slots the dynamic loader fills in (`Binary::pointers`: GOT entries,
//! vtables, tables of string pointers) are emitted as pointers to the target's
//! static or function, so code that loads an address out of the GOT and follows it
//! lands in the right static. Each GOT slot is its own 8-byte item, so only the
//! slots the code uses are emitted. A slot whose target is outside the output is
//! null if it is an import, and keeps its bytes from the file if it is a function
//! of the binary that isn't emitted (not selected, or skipped by `--skip-failed`).
use crate::ir::{Function, Idx, InstKind};
use crate::load::{Binary, TLS_SECTION};
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

/// One `static`: `len` bytes at `start`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Item {
    pub start: u64,
    pub len: u64,
    /// Index into `Binary::data`.
    pub section: usize,
    /// Index into `Binary::data_syms`, if a symbol names it.
    pub sym: Option<usize>,
}

pub struct Globals<'b, 'a> {
    bin: &'b Binary<'a>,
    /// Identifier for each data symbol.
    sym_idents: Vec<String>,
    /// Every identifier taken (functions and data symbols), for anonymous items.
    used: HashSet<String>,
    /// Identifier of each function in the output, by address, for function pointers.
    funcs: &'b HashMap<u64, &'b str>,
}

/// The types statics have: plain bytes, or 8-byte words when some of them are
/// pointers. Aligned like the most demanding SSE access.
pub const PRELUDE: &str = "\
#[repr(C, align(16))]
pub struct Bytes<const N: usize> {
    pub b: [u8; N],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union Word {
    pub b: [u8; 8],
    pub p: *const u8,
}
#[repr(C, align(16))]
pub struct Words<const N: usize> {
    pub w: [Word; N],
}
unsafe impl<const N: usize> Sync for Words<N> {}
";

impl<'b, 'a> Globals<'b, 'a> {
    /// `used` holds the function identifiers already taken; data symbols get
    /// identifiers that don't clash with them.
    pub fn new(bin: &'b Binary<'a>, mut used: HashSet<String>, funcs: &'b HashMap<u64, &'b str>) -> Self {
        let sym_idents = bin
            .data_syms
            .iter()
            .map(|s| crate::names::rust_ident(s.pretty(), s.addr, &mut used))
            .collect();
        Globals { bin, sym_idents, used, funcs }
    }

    /// The item containing `addr`, if it is in a data section.
    pub fn item_at(&self, addr: u64) -> Option<Item> {
        let section = self.bin.data_section_at(addr)?;
        let sec = &self.bin.data[section];
        let syms = &self.bin.data_syms;
        let i = syms.partition_point(|s| s.addr <= addr);
        let prev = i.checked_sub(1).map(|k| (k, &syms[k])).filter(|(_, s)| s.section == section);
        if let Some((k, s)) = prev {
            if addr < s.addr + s.size {
                return Some(Item { start: s.addr, len: s.size, section, sym: Some(k) });
            }
        }
        // code reaches every thread-local from the thread pointer, so the block
        // can't be split
        if sec.name == TLS_SECTION {
            return Some(Item { start: sec.addr, len: sec.size, section, sym: None });
        }
        if matches!(sec.name.as_str(), ".got" | ".got.plt" | "__got") {
            let start = sec.addr + (addr - sec.addr) / 8 * 8;
            return Some(Item { start, len: 8.min(sec.addr + sec.size - start), section, sym: None });
        }
        let start = prev.map_or(sec.addr, |(_, s)| s.addr + s.size);
        let end = syms.get(i).filter(|s| s.section == section).map_or(sec.addr + sec.size, |s| s.addr);
        Some(Item { start, len: end - start, section, sym: None })
    }

    pub fn ident(&self, item: &Item) -> String {
        match item.sym {
            Some(k) => self.sym_idents[k].clone(),
            None => {
                let mut s = match self.bin.data[item.section].name.as_str() {
                    TLS_SECTION => "THREAD_LOCALS".to_string(),
                    ".got" | ".got.plt" | "__got" => format!("GOT_{:x}", item.start),
                    _ => format!("ANON_{:x}", item.start),
                };
                while self.used.contains(&s) {
                    s.push('_');
                }
                s
            }
        }
    }

    /// A Rust expression for `addr` as a `u64`: the address of its static plus
    /// the offset into it.
    pub fn expr(&self, addr: u64) -> Option<String> {
        let item = self.item_at(addr)?;
        let base = format!("({} as u64)", self.addr_of(&item));
        Some(match addr - item.start {
            0 => base,
            off => format!("{base}.wrapping_add({off:#x})"),
        })
    }

    /// The static holding `addr`, if safe code can read it as a slice: a
    /// read-only item emitted as `Bytes<N>` (no pointer slots in it).
    pub fn slice(&self, addr: u64) -> Option<String> {
        let item = self.item_at(addr)?;
        let sec = &self.bin.data[item.section];
        let words = self.bin.pointers.range(item.start..item.start + item.len).next().is_some();
        (!sec.writable && !words).then(|| self.ident(&item))
    }

    /// Is `item` emitted as `Words` (it holds pointer slots), not `Bytes`?
    fn words(&self, item: &Item) -> bool {
        let end = item.start + item.len;
        let mut slots = self.bin.pointers.range(item.start..end).peekable();
        slots.peek().is_some() && item.start.is_multiple_of(8) && slots.all(|(&a, _)| a.is_multiple_of(8) && a + 8 <= end)
    }

    /// `item`'s static with the right name and type but zero contents, which
    /// rustc checks much faster than the real initializer (`--check`).
    pub fn emit_static_stub(&self, item: &Item, out: &mut String) {
        let m = if self.bin.data[item.section].writable { "mut " } else { "" };
        let name = self.ident(item);
        let n = item.len;
        let _ = match self.words(item) {
            true => writeln!(out, "pub static {m}{name}: Words<{k}> = Words {{ w: [Word {{ b: [0; 8] }}; {k}] }};", k = n.div_ceil(8)),
            false => writeln!(out, "pub static {m}{name}: Bytes<{n}> = Bytes {{ b: [0; {n}] }};"),
        };
    }

    /// A raw pointer to `item`'s static.
    fn addr_of(&self, item: &Item) -> String {
        let name = self.ident(item);
        match self.bin.data[item.section].writable {
            true => format!("unsafe {{ core::ptr::addr_of_mut!({name}) }}"),
            false => format!("core::ptr::addr_of!({name})"),
        }
    }

    /// The initializer of a pointer slot holding `target`: the function or static
    /// it points to (adding that static to `more`), or null; `None` for a function
    /// of the binary that isn't in the output, whose slot keeps its file bytes.
    fn pointer(&self, target: u64, more: &mut Vec<Item>) -> Option<String> {
        if let Some(f) = self.funcs.get(&target) {
            return Some(format!("{f} as *const u8"));
        }
        let Some(item) = self.item_at(target) else {
            return self.bin.func_at(target).is_none().then(|| "core::ptr::null()".into());
        };
        more.push(item);
        Some(match target - item.start {
            0 => format!("{} as *const u8", self.addr_of(&item)),
            off => format!("({} as *const u8).wrapping_add({off:#x})", self.addr_of(&item)),
        })
    }

    /// Items that `f` uses as pointers, the same addresses `expr` rewrites.
    pub fn referenced(&self, f: &Function, out: &mut Vec<Item>) {
        for (_, blk) in f.blocks.iter() {
            for &id in blk.insts.get(&f.value_pool) {
                let InstKind::IntToPtr(v) = f.insts[id].kind else { continue };
                if let InstKind::Const(c) = f.insts[v].kind {
                    out.extend(self.item_at(f.consts[c.index()] as u64));
                }
            }
        }
    }

    /// The `static` for `item`, with a comment saying where it came from. Statics
    /// that its pointer slots point to are added to `more`.
    pub fn emit_static(&self, item: &Item, out: &mut String, more: &mut Vec<Item>) {
        let sec = &self.bin.data[item.section];
        let name = self.ident(item);
        let n = item.len;
        let what = match item.sym {
            Some(k) => self.bin.data_syms[k].pretty().to_string(),
            None => "no symbol".to_string(),
        };
        let _ = write!(out, "\n// {} {:#x}, {n} bytes ({what})", sec.name, item.start);
        let bytes = sec.bytes.as_deref().map(|b| &b[(item.start - sec.addr) as usize..][..n as usize]);
        if let Some(text) = bytes.and_then(preview) {
            let _ = write!(out, ": {text:?}");
        }
        out.push('\n');
        let m = if sec.writable { "mut " } else { "" };
        let end = item.start + n;
        let slots: Vec<(u64, u64)> = self.bin.pointers.range(item.start..end).map(|(&a, &t)| (a, t)).collect();
        if !self.words(item) {
            let init = match bytes {
                Some(b) if b.iter().any(|&c| c != 0) => byte_string(b),
                _ => format!("[0; {n}]"),
            };
            let _ = writeln!(out, "pub static {m}{name}: Bytes<{n}> = Bytes {{ b: {init} }};");
            return;
        }
        // Whole words, with the pointer slots as pointers; a ragged tail is zero-padded.
        let k = n.div_ceil(8);
        let _ = write!(out, "pub static {m}{name}: Words<{k}> = Words {{ w: [");
        let mut slots = slots.into_iter().peekable();
        for j in 0..k {
            let at = item.start + 8 * j;
            out.push_str(if j % 4 == 0 { "\n    " } else { " " });
            if slots.peek().is_some_and(|&(a, _)| a == at) {
                let (_, t) = slots.next().unwrap();
                if let Some(p) = self.pointer(t, more) {
                    let _ = write!(out, "Word {{ p: {p} }},");
                    continue;
                }
            }
            let mut w = [0u8; 8];
            if let Some(b) = bytes {
                let have = &b[8 * j as usize..b.len().min(8 * j as usize + 8)];
                w[..have.len()].copy_from_slice(have);
            }
            let _ = write!(out, "Word {{ b: {} }},", byte_string(&w));
        }
        out.push_str("\n] };\n");
    }
}

/// The start of a C string, if the bytes begin with one.
fn preview(b: &[u8]) -> Option<String> {
    let end = b.iter().position(|&c| c == 0)?;
    let s = std::str::from_utf8(&b[..end]).ok()?;
    if s.chars().count() < 3 || s.starts_with(char::is_control) || !s.chars().all(|c| !c.is_control() || c == '\n' || c == '\t') {
        return None;
    }
    Some(match s.char_indices().nth(60) {
        Some((i, _)) => format!("{}...", &s[..i]),
        None => s.to_string(),
    })
}

/// `*b"..."` with lines of at most ~100 characters.
fn byte_string(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2 + 8);
    s.push_str("*b\"");
    let mut line = 0;
    for &c in b {
        if line >= 100 {
            // A backslash-newline continues the literal and skips the next
            // line's leading whitespace, so whitespace after it is escaped below.
            s.push_str("\\\n");
            line = 0;
        }
        let before = s.len();
        match c {
            b'"' => s.push_str("\\\""),
            b'\\' => s.push_str("\\\\"),
            b'\n' => s.push_str("\\n"),
            b'\t' if line > 0 => s.push_str("\\t"),
            0 => s.push_str("\\0"),
            b' ' if line > 0 => s.push(' '),
            0x21..=0x7e => s.push(c as char),
            _ => {
                let _ = write!(s, "\\x{c:02x}");
            }
        }
        line += s.len() - before;
    }
    s.push('"');
    s
}

#[cfg(test)]
mod tests {
    use super::byte_string;

    #[test]
    fn byte_strings_escape() {
        assert_eq!(byte_string(b"hi \"x\"\\\n\0\xff"), r#"*b"hi \"x\"\\\n\0\xff""#);
        assert_eq!(byte_string(b" \t"), r#"*b"\x20\t""#);
    }
}
