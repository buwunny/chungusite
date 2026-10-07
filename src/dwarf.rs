//! Debug info (DWARF) as a `TypeModel`: each function's argument names and types,
//! its return type, and the layout of the structs its pointer arguments point
//! to. Like any model it only proposes; `types::infer` checks every proposal
//! against the code.
//!
//! Arguments are matched to registers the System V way: integers, pointers and
//! enums take rdi, rsi, rdx, rcx, r8, r9 in order, floats go in xmm registers and
//! are skipped. A function with an argument or return value that is a struct by
//! value, or wider than 8 bytes, is skipped entirely, because the classification
//! rules for those (hidden return pointer, splitting into two registers) are not
//! modelled. That also covers the Rust ABI for what is left: rustc passes scalars
//! and thin pointers in registers in order, and slices, `&str` and `&dyn` are
//! structs in DWARF.
//!
//! In a relocatable object (`.o`) the debug sections still need their relocations
//! applied (string offsets, function addresses); `load` does that.
#![allow(non_upper_case_globals)] // gimli's DW_* constants

use crate::types::{CType, Field, FuncRef, IntTy, ParamProposal, Proposal, StructTy, TypeModel};
use gimli::{constants::*, AttributeValue, EndianSlice, LittleEndian, UnitOffset};
use object::{Object, ObjectKind, ObjectSection, ObjectSymbol, RelocationKind, RelocationTarget};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

type R<'a> = EndianSlice<'a, LittleEndian>;

/// Function signatures from DWARF, by entry address.
#[derive(Default)]
pub struct DebugInfo {
    funcs: HashMap<u64, Vec<FnDebug>>,
}

struct FnDebug {
    names: Vec<String>,
    proposal: Proposal,
}

impl DebugInfo {
    /// Read the debug info of an object file. Empty if it has none, or none this
    /// reader understands.
    pub fn parse(data: &[u8]) -> DebugInfo {
        let Ok(file) = object::File::parse(data) else { return DebugInfo::default() };
        if file.section_by_name(".debug_info").is_none() {
            return DebugInfo::default();
        }
        let relocatable = file.kind() == ObjectKind::Relocatable;
        let load = |id: gimli::SectionId| -> Result<Cow<[u8]>, gimli::Error> {
            let Some(sec) = file.section_by_name(id.name()) else { return Ok(Cow::Borrowed(&[])) };
            let Ok(data) = sec.uncompressed_data() else { return Ok(Cow::Borrowed(&[])) };
            if !relocatable {
                return Ok(data);
            }
            let mut d = data.into_owned();
            for (off, r) in sec.relocations() {
                let RelocationTarget::Symbol(i) = r.target() else { continue };
                let Ok(sym) = file.symbol_by_index(i) else { continue };
                if r.kind() != RelocationKind::Absolute {
                    continue;
                }
                let v = sym.address().wrapping_add(r.addend() as u64);
                let at = off as usize;
                match r.size() {
                    32 if at + 4 <= d.len() => d[at..at + 4].copy_from_slice(&(v as u32).to_le_bytes()),
                    64 if at + 8 <= d.len() => d[at..at + 8].copy_from_slice(&v.to_le_bytes()),
                    _ => {}
                }
            }
            Ok(Cow::Owned(d))
        };
        let Ok(sections) = gimli::DwarfSections::load(load) else { return DebugInfo::default() };
        let dwarf = sections.borrow(|s| EndianSlice::new(s, LittleEndian));
        let mut out = DebugInfo::default();
        let mut units = dwarf.units();
        while let Ok(Some(header)) = units.next() {
            let Ok(unit) = dwarf.unit(header) else { continue };
            let mut subs = Vec::new();
            let mut lang_ok = false;
            {
                let mut cur = unit.entries();
                while let Ok(Some(e)) = cur.next_dfs() {
                    match e.tag() {
                        DW_TAG_compile_unit | DW_TAG_partial_unit => {
                            lang_ok = matches!(
                                e.attr_value(DW_AT_language),
                                Some(AttributeValue::Language(
                                    DW_LANG_C89 | DW_LANG_C | DW_LANG_C99 | DW_LANG_C11 | DW_LANG_C17 | DW_LANG_C_plus_plus
                                        | DW_LANG_C_plus_plus_03 | DW_LANG_C_plus_plus_11 | DW_LANG_C_plus_plus_14
                                        | DW_LANG_C_plus_plus_17 | DW_LANG_C_plus_plus_20 | DW_LANG_Rust
                                ))
                            );
                            if !lang_ok {
                                break;
                            }
                        }
                        DW_TAG_subprogram => {
                            if let Some(v) = e.attr_value(DW_AT_low_pc) {
                                if let Ok(Some(a)) = dwarf.attr_address(&unit, v) {
                                    subs.push((e.offset(), a));
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            if !lang_ok {
                continue;
            }
            let mut cx = Ctx { dwarf: &dwarf, unit: &unit, memo: HashMap::new() };
            for (off, addr) in subs {
                if let Some(fd) = cx.function(off) {
                    out.funcs.entry(addr).or_default().push(fd);
                }
            }
        }
        out
    }

    /// Functions described.
    pub fn len(&self) -> usize {
        self.funcs.values().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.funcs.is_empty()
    }
}

impl TypeModel for DebugInfo {
    fn name(&self) -> &str {
        "dwarf"
    }

    fn propose(&self, func: &FuncRef) -> Option<Proposal> {
        let cands = self.funcs.get(&func.addr)?;
        // In an object file several sections start at 0: match by name.
        let fd = cands
            .iter()
            .find(|c| c.names.iter().any(|n| n == func.name))
            .or_else(|| (cands.len() == 1 && cands[0].names.is_empty()).then(|| &cands[0]))
            .or_else(|| (cands.len() == 1).then(|| &cands[0]))?;
        Some(fd.proposal.clone())
    }
}

struct Ctx<'d, 'a> {
    dwarf: &'d gimli::Dwarf<R<'a>>,
    unit: &'d gimli::Unit<R<'a>>,
    /// (type DIE offset, pointer depth) -> type; `None` is `void`.
    memo: HashMap<(usize, u8), Option<CType>>,
}

type Die<'a> = gimli::DebuggingInformationEntry<R<'a>>;

impl<'a> Ctx<'_, 'a> {
    fn entry(&self, off: UnitOffset) -> Option<Die<'a>> {
        self.unit.entry(off).ok()
    }

    fn string(&self, e: &Die<'a>, at: gimli::DwAt) -> Option<String> {
        let v = e.attr_value(at)?;
        let s = self.dwarf.attr_string(self.unit, v).ok()?;
        Some(String::from_utf8_lossy(s.slice()).into_owned())
    }

    fn reference(e: &Die<'a>, at: gimli::DwAt) -> Option<UnitOffset> {
        match e.attr_value(at)? {
            AttributeValue::UnitRef(o) => Some(o),
            _ => None,
        }
    }

    fn udata(e: &Die<'a>, at: gimli::DwAt) -> Option<u64> {
        e.attr_value(at)?.udata_value()
    }

    /// The DIE itself, then what it completes (`DW_AT_abstract_origin`,
    /// `DW_AT_specification`), up to three levels.
    fn chain(&self, e: Die<'a>) -> Vec<Die<'a>> {
        let mut v = vec![e];
        while v.len() < 4 {
            let last = v.last().unwrap();
            let Some(o) = Self::reference(last, DW_AT_abstract_origin).or_else(|| Self::reference(last, DW_AT_specification)) else {
                break;
            };
            match self.entry(o) {
                Some(n) => v.push(n),
                None => break,
            }
        }
        v
    }

    fn first<T>(&self, chain: &[Die<'a>], f: impl Fn(&Die<'a>) -> Option<T>) -> Option<T> {
        chain.iter().find_map(f)
    }

    fn children(&self, off: UnitOffset) -> Vec<Die<'a>> {
        let mut out = Vec::new();
        let Ok(mut tree) = self.unit.entries_tree(Some(off)) else { return out };
        let Ok(root) = tree.root() else { return out };
        let mut it = root.children();
        while let Ok(Some(node)) = it.next() {
            out.push(node.entry().clone());
        }
        out
    }

    fn function(&mut self, off: UnitOffset) -> Option<FnDebug> {
        let e = self.entry(off)?;
        let chain = self.chain(e);
        let mut names = Vec::new();
        for at in [DW_AT_linkage_name, DW_AT_MIPS_linkage_name, DW_AT_name] {
            if let Some(n) = self.first(&chain, |d| self.string(d, at)) {
                names.push(n);
            }
        }
        let ret = match self.first(&chain, |d| Self::reference(d, DW_AT_type)) {
            None => None,
            Some(t) => match self.resolve(t, 0, None) {
                None => None,
                Some(t @ (CType::Int(_) | CType::Ptr(_))) if t.size() <= 8 => Some(t),
                Some(CType::Float { .. }) => None,
                // a struct returned through a hidden pointer in rdi shifts every argument
                Some(_) => return None,
            },
        };
        // The DIE's own parameters, else the declaration's.
        let mut params_of = None;
        for d in &chain {
            let ps: Vec<Die<'a>> =
                self.children(d.offset()).into_iter().filter(|c| c.tag() == DW_TAG_formal_parameter).collect();
            if !ps.is_empty() {
                params_of = Some(ps);
                break;
            }
        }
        let mut params = Vec::new();
        for p in params_of.unwrap_or_default() {
            let pchain = self.chain(p);
            let name = self.first(&pchain, |d| self.string(d, DW_AT_name));
            let t = self.first(&pchain, |d| Self::reference(d, DW_AT_type))?;
            match self.resolve(t, 0, None)? {
                CType::Float { .. } => continue,
                t @ (CType::Int(_) | CType::Ptr(_)) if t.size() <= 8 => params.push(ParamProposal { name, ty: Some(t) }),
                _ => return None,
            }
        }
        Some(FnDebug { names, proposal: Proposal { params, ret } })
    }

    /// The type at `off`. `depth` counts pointers followed from an argument: the
    /// pointee of an argument is resolved, pointers inside it stay opaque.
    fn resolve(&mut self, off: UnitOffset, depth: u8, hint: Option<&str>) -> Option<CType> {
        let key = (off.0, depth);
        if let Some(t) = self.memo.get(&key) {
            return t.clone();
        }
        // Guards against cycles through malformed input; by-value nesting can't loop.
        self.memo.insert(key, Some(CType::Opaque { bytes: 0 }));
        let t = self.resolve_uncached(off, depth, hint);
        self.memo.insert(key, t.clone());
        t
    }

    fn resolve_uncached(&mut self, off: UnitOffset, depth: u8, hint: Option<&str>) -> Option<CType> {
        let e = self.entry(off)?;
        let size = Self::udata(&e, DW_AT_byte_size);
        let target = Self::reference(&e, DW_AT_type);
        match e.tag() {
            DW_TAG_base_type => {
                let bytes = size.unwrap_or(0);
                let enc = match e.attr_value(DW_AT_encoding) {
                    Some(AttributeValue::Encoding(x)) => x,
                    _ => DW_ATE_unsigned,
                };
                Some(match (enc, bytes) {
                    (DW_ATE_float, 4 | 8) => CType::Float { bytes: bytes as u8 },
                    (DW_ATE_signed | DW_ATE_signed_char, 1 | 2 | 4 | 8) => CType::Int(IntTy::new(bytes as u8, true)),
                    (DW_ATE_unsigned | DW_ATE_unsigned_char | DW_ATE_boolean | DW_ATE_UTF | DW_ATE_address, 1 | 2 | 4 | 8) => {
                        CType::Int(IntTy::new(bytes as u8, false))
                    }
                    _ => CType::Opaque { bytes },
                })
            }
            DW_TAG_typedef => {
                let name = self.string(&e, DW_AT_name);
                self.resolve(target?, depth, name.as_deref())
            }
            DW_TAG_const_type | DW_TAG_volatile_type | DW_TAG_restrict_type | DW_TAG_atomic_type
            | DW_TAG_immutable_type | DW_TAG_packed_type => match target {
                Some(t) => self.resolve(t, depth, hint),
                None => None,
            },
            DW_TAG_pointer_type | DW_TAG_reference_type | DW_TAG_rvalue_reference_type => {
                if depth > 0 {
                    return Some(CType::Ptr(None));
                }
                let pointee = match target {
                    Some(t) => self.resolve(t, depth + 1, None),
                    None => None,
                };
                Some(CType::Ptr(pointee.map(Arc::new)))
            }
            DW_TAG_enumeration_type => {
                let signed = match target.and_then(|t| self.resolve(t, depth, None)) {
                    Some(CType::Int(t)) => t.signed,
                    _ => false,
                };
                match size {
                    Some(b @ (1 | 2 | 4 | 8)) => Some(CType::Int(IntTy::new(b as u8, signed))),
                    _ => Some(CType::Opaque { bytes: size.unwrap_or(0) }),
                }
            }
            DW_TAG_array_type => {
                let elem = self.resolve(target?, depth, None)?;
                let mut dims = Vec::new();
                for c in self.children(off) {
                    if c.tag() != DW_TAG_subrange_type {
                        continue;
                    }
                    let n = match (Self::udata(&c, DW_AT_count), Self::udata(&c, DW_AT_upper_bound)) {
                        (Some(n), _) => n,
                        (None, Some(u)) => (u + 1).saturating_sub(Self::udata(&c, DW_AT_lower_bound).unwrap_or(0)),
                        _ => 0,
                    };
                    dims.push(n);
                }
                if dims.is_empty() {
                    dims.push(0);
                }
                let mut t = elem;
                for &n in dims.iter().rev() {
                    t = CType::Array { elem: Arc::new(t), len: n };
                }
                Some(t)
            }
            DW_TAG_structure_type | DW_TAG_class_type => {
                let size = size.unwrap_or(0);
                if e.attr_value(DW_AT_declaration).is_some() {
                    return Some(CType::Opaque { bytes: 0 });
                }
                let name = self
                    .string(&e, DW_AT_name)
                    .or_else(|| hint.map(str::to_string))
                    .unwrap_or_else(|| format!("anon_{:x}", off.0));
                let mut fields = Vec::new();
                for (i, c) in self.children(off).into_iter().enumerate() {
                    match c.tag() {
                        // a Rust enum: which fields exist depends on the discriminant
                        DW_TAG_variant_part => return Some(CType::Opaque { bytes: size }),
                        DW_TAG_member | DW_TAG_inheritance => {}
                        _ => continue,
                    }
                    let Some(at) = Self::udata(&c, DW_AT_data_member_location) else { continue };
                    let Some(t) = Self::reference(&c, DW_AT_type) else { continue };
                    let Some(ty) = self.resolve(t, depth, None) else { continue };
                    let ty = if c.attr_value(DW_AT_bit_size).is_some() { CType::Opaque { bytes: ty.size() } } else { ty };
                    let fname = match self.string(&c, DW_AT_name) {
                        Some(n) if c.tag() == DW_TAG_member => n,
                        _ => format!("base{i}"),
                    };
                    fields.push(Field { name: fname, off: at, ty });
                }
                Some(CType::Struct(Arc::new(layout(name, size, fields))))
            }
            DW_TAG_union_type => Some(CType::Opaque { bytes: size.unwrap_or(0) }),
            DW_TAG_unspecified_type => None,
            _ => Some(CType::Opaque { bytes: size.unwrap_or(0) }),
        }
    }
}

/// Fields sorted, with zero-sized ones dropped, names made Rust identifiers and
/// unique, and overlapping ones (bit-fields sharing a unit, anonymous unions)
/// merged into opaque bytes.
fn layout(name: String, size: u64, mut fields: Vec<Field>) -> StructTy {
    fields.retain(|f| f.ty.size() > 0 && f.off + f.ty.size() <= size.max(f.off + f.ty.size()));
    fields.sort_by_key(|f| f.off);
    let mut out: Vec<Field> = Vec::new();
    for f in fields {
        if let Some(last) = out.last_mut() {
            let end = last.off + last.ty.size();
            if f.off < end {
                let new_end = end.max(f.off + f.ty.size());
                last.ty = CType::Opaque { bytes: new_end - last.off };
                last.name = format!("bits{}", last.off);
                continue;
            }
        }
        out.push(f);
    }
    let size = size.max(out.last().map_or(0, |f| f.off + f.ty.size()));
    let mut seen = std::collections::HashSet::new();
    for f in &mut out {
        let mut n = field_ident(&f.name);
        while !seen.insert(n.clone()) {
            n.push('_');
        }
        f.name = n;
    }
    StructTy { name, size, fields: out }
}

/// A field name as a Rust identifier (`type` becomes `type_`).
fn field_ident(name: &str) -> String {
    let s = crate::names::sanitize(name);
    match s.strip_prefix("f_") {
        Some(rest) if !name.starts_with("f_") && !rest.is_empty() && !rest.starts_with(|c: char| c.is_ascii_digit()) => format!("{rest}_"),
        _ => s,
    }
}
