//! DWARF debug info: the prototypes (parameter names and types, return type) of
//! the functions in a binary built with `-g`, and the layouts of the structs they
//! take pointers to. `types.rs` prefers these over what it infers.
//!
//! Only what maps cleanly onto the IR is kept. A type that has no Rust equivalent
//! here (unions, Rust enums, bitfields, 128-bit integers) becomes bytes of the
//! right size, and a function whose prototype can't be matched to registers (a
//! struct or float passed by value, varargs, a compiler clone such as
//! `f.constprop.0`) is left to inference by `types.rs`.
use crate::ir::*;
use gimli::{constants, AttributeValue, EndianSlice, LittleEndian, RelocateReader, UnitOffset};
use object::{Object, ObjectSection};
use std::collections::HashMap;

/// A function's prototype, from its `DW_TAG_subprogram`.
#[derive(Clone, Debug)]
pub struct DebugFn {
    pub name: String,
    pub linkage: Option<String>,
    /// In declaration order. Zero-sized parameters, which take no register, are left out.
    pub params: Vec<(String, TyId)>,
    /// `None` for `void`.
    pub ret: Option<TyId>,
    pub variadic: bool,
}

/// Prototypes by entry address.
#[derive(Default)]
pub struct DebugInfo {
    pub funcs: HashMap<u64, DebugFn>,
}

#[derive(Copy, Clone, Debug)]
struct Relocs<'a>(&'a object::read::RelocationMap);

impl gimli::read::Relocate<usize> for Relocs<'_> {
    fn relocate_address(&self, offset: usize, value: u64) -> gimli::Result<u64> {
        Ok(self.0.relocate(offset as u64, value))
    }
    fn relocate_offset(&self, offset: usize, value: usize) -> gimli::Result<usize> {
        <usize as gimli::ReaderOffset>::from_u64(self.0.relocate(offset as u64, value as u64))
    }
}

type R<'a> = RelocateReader<EndianSlice<'a, LittleEndian>, Relocs<'a>>;

/// The prototypes in `data`'s debug info of the functions at `wanted` addresses,
/// with their types interned into `table`. `None` if there is no DWARF (or it
/// can't be read).
pub fn read(data: &[u8], table: &mut TyTable, wanted: &dyn Fn(u64) -> bool) -> Option<DebugInfo> {
    let file = object::File::parse(data).ok()?;
    file.section_by_name(".debug_info")?;
    // Section bytes and relocations first, so the readers can borrow them.
    let empty = object::read::RelocationMap::default();
    let mut sections: HashMap<gimli::SectionId, (std::borrow::Cow<[u8]>, object::read::RelocationMap)> = HashMap::new();
    use gimli::SectionId as S;
    for id in [
        S::DebugAbbrev, S::DebugAddr, S::DebugInfo, S::DebugLine, S::DebugLineStr, S::DebugLoc, S::DebugLocLists,
        S::DebugRanges, S::DebugRngLists, S::DebugStr, S::DebugStrOffsets, S::DebugTypes,
    ] {
        if let Some(s) = file.section_by_name(id.name()) {
            let bytes = s.uncompressed_data().unwrap_or_default();
            let relocs = s.relocation_map().unwrap_or_default();
            sections.insert(id, (bytes, relocs));
        }
    }
    let load = |id: gimli::SectionId| -> Result<R, gimli::Error> {
        Ok(match sections.get(&id) {
            Some((b, r)) => RelocateReader::new(EndianSlice::new(b, LittleEndian), Relocs(r)),
            None => RelocateReader::new(EndianSlice::new(&[], LittleEndian), Relocs(&empty)),
        })
    };
    let dwarf = gimli::Dwarf::load(load).ok()?;
    let mut info = DebugInfo::default();
    let mut by_name = HashMap::new();
    let first = table.structs.len();
    let mut units = dwarf.units();
    while let Ok(Some(header)) = units.next() {
        let Ok(unit) = dwarf.unit(header) else { continue };
        // Rust gives different types the same name (generic instances, enum
        // variants), so only C and C++ structs are shared between units by name.
        let lang = unit.entry(unit.header.root_offset()).ok().and_then(|e| match e.attr_value(constants::DW_AT_language) {
            Some(AttributeValue::Language(l)) => Some(l),
            _ => None,
        });
        let dedupe = lang != Some(constants::DW_LANG_Rust);
        let mut c = Conv { dwarf: &dwarf, unit: &unit, table, memo: HashMap::new(), by_name: &mut by_name, queue: Vec::new(), dedupe };
        let _ = c.functions(&mut info, wanted);
    }
    // Layout (packed or not) once every struct has its members.
    let mut done = vec![false; table.structs.len()];
    for i in first..table.structs.len() {
        set_packed(table, StructId::from_u32(i as u32), &mut done);
    }
    // anonymous structs no typedef named
    for i in first..table.structs.len() {
        let s = StructId::from_u32(i as u32);
        if table.structs[s].name.is_empty() {
            table.structs[s].name = table.fresh_struct_name("anon");
        }
    }
    Some(info)
}

struct Conv<'a, 'd> {
    dwarf: &'a gimli::Dwarf<R<'d>>,
    unit: &'a gimli::Unit<R<'d>>,
    table: &'a mut TyTable,
    memo: HashMap<UnitOffset, TyId>,
    /// Named structs already read, by name and size: each compilation unit has
    /// its own copy of a header's types.
    by_name: &'a mut HashMap<(String, u32), TyId>,
    /// Structs registered but whose members aren't read yet. Reading them from a
    /// queue instead of recursively keeps the stack shallow: Rust binaries have
    /// long chains of structs pointing at structs.
    queue: Vec<(UnitOffset, StructId)>,
    dedupe: bool,
}

type Die<'d> = gimli::DebuggingInformationEntry<R<'d>>;

impl<'d> Conv<'_, 'd> {
    fn functions(&mut self, info: &mut DebugInfo, wanted: &dyn Fn(u64) -> bool) -> gimli::Result<()> {
        let mut cursor = self.unit.entries();
        let mut subprograms = Vec::new();
        while let Some(e) = cursor.next_dfs()? {
            if e.tag() == constants::DW_TAG_subprogram {
                if let Some(lo) = e.attr_value(constants::DW_AT_low_pc) {
                    if let Ok(Some(addr)) = self.dwarf.attr_address(self.unit, lo) {
                        if wanted(addr) {
                            subprograms.push((addr, e.offset()));
                        }
                    }
                }
            }
        }
        for (addr, off) in subprograms {
            let f = self.function(off);
            self.members();
            if let Some(f) = f {
                info.funcs.entry(addr).or_insert(f);
            }
        }
        Ok(())
    }

    /// The declaration a concrete subprogram (or parameter) refers to, if any.
    fn origin(&self, e: &Die<'d>) -> Option<Die<'d>> {
        for at in [constants::DW_AT_abstract_origin, constants::DW_AT_specification] {
            if let Some(AttributeValue::UnitRef(o)) = e.attr_value(at) {
                return self.unit.entry(o).ok();
            }
        }
        None
    }

    /// An attribute of `e`, or of what it refers to (two levels: a concrete
    /// instance of an out-of-line C++ member function).
    fn attr(&self, e: &Die<'d>, at: constants::DwAt) -> Option<AttributeValue<R<'d>>> {
        if let Some(v) = e.attr_value(at) {
            return Some(v);
        }
        let o = self.origin(e)?;
        o.attr_value(at).or_else(|| self.origin(&o)?.attr_value(at))
    }

    fn string(&self, v: AttributeValue<R<'d>>) -> Option<String> {
        let s = self.dwarf.attr_string(self.unit, v).ok()?;
        Some(String::from_utf8_lossy(s.inner().slice()).into_owned())
    }

    fn name(&self, e: &Die<'d>) -> Option<String> {
        self.attr(e, constants::DW_AT_name).and_then(|v| self.string(v))
    }

    fn function(&mut self, off: UnitOffset) -> Option<DebugFn> {
        let e = self.unit.entry(off).ok()?;
        let name = self.name(&e)?;
        let linkage = self
            .attr(&e, constants::DW_AT_linkage_name)
            .or_else(|| self.attr(&e, constants::DW_AT_MIPS_linkage_name))
            .and_then(|v| self.string(v));
        let ret = match self.attr(&e, constants::DW_AT_type) {
            Some(AttributeValue::UnitRef(t)) => Some(self.ty(t)),
            _ => None,
        };
        // Parameters from the declaration when there is one: a concrete instance
        // may leave out the ones that were optimized away.
        let decl = match self.origin(&e) {
            Some(o) => self.origin(&o).unwrap_or(o),
            None => e,
        };
        let mut params = Vec::new();
        let mut variadic = false;
        let mut tree = self.unit.entries_tree(Some(decl.offset())).ok()?;
        let root = tree.root().ok()?;
        let mut kids = root.children();
        let mut raw = Vec::new();
        while let Ok(Some(k)) = kids.next() {
            let k = k.entry();
            match k.tag() {
                constants::DW_TAG_formal_parameter => raw.push(k.clone()),
                constants::DW_TAG_unspecified_parameters => variadic = true,
                _ => {}
            }
        }
        for (i, p) in raw.iter().enumerate() {
            let Some(AttributeValue::UnitRef(t)) = self.attr(p, constants::DW_AT_type) else { return None };
            let ty = self.ty(t);
            if self.table.size_of(ty) == 0 {
                continue;
            }
            let pname = self.name(p).unwrap_or_else(|| format!("arg{i}"));
            params.push((pname, ty));
        }
        Some(DebugFn { name, linkage, params, ret, variadic })
    }

    /// The type at `off`, interned. Never fails: what can't be expressed is bytes.
    fn ty(&mut self, off: UnitOffset) -> TyId {
        if let Some(&t) = self.memo.get(&off) {
            return t;
        }
        // Typedefs and qualifiers: follow the chain to the type itself. The
        // first typedef names an anonymous struct (`typedef struct { .. } name`).
        let mut at = off;
        let mut typedef = None;
        for _ in 0..32 {
            let Ok(e) = self.unit.entry(at) else { break };
            match e.tag() {
                constants::DW_TAG_typedef
                | constants::DW_TAG_const_type
                | constants::DW_TAG_volatile_type
                | constants::DW_TAG_restrict_type
                | constants::DW_TAG_atomic_type => {
                    if e.tag() == constants::DW_TAG_typedef && typedef.is_none() {
                        typedef = self.name(&e);
                    }
                    match e.attr_value(constants::DW_AT_type) {
                        Some(AttributeValue::UnitRef(t)) => at = t,
                        _ => {
                            // `const void`
                            self.memo.insert(off, TyId::B1);
                            return TyId::B1;
                        }
                    }
                }
                _ => break,
            }
        }
        let t = match self.memo.get(&at) {
            Some(&t) => t,
            None => {
                let t = self.ty_uncached(at).unwrap_or(TyId::B1);
                self.memo.insert(at, t);
                t
            }
        };
        if let (Some(n), Ty::Struct(s)) = (typedef, self.table.tys[t]) {
            if self.table.structs[s].name.is_empty() {
                self.table.structs[s].name = self.table.fresh_struct_name(&n);
            }
        }
        self.memo.insert(off, t);
        t
    }

    fn target(&mut self, e: &Die<'d>) -> Option<TyId> {
        match e.attr_value(constants::DW_AT_type) {
            Some(AttributeValue::UnitRef(t)) => Some(self.ty(t)),
            _ => None,
        }
    }

    fn size(&self, e: &Die<'d>) -> Option<u32> {
        e.attr_value(constants::DW_AT_byte_size)?.udata_value().map(|s| s as u32)
    }

    fn ty_uncached(&mut self, off: UnitOffset) -> Option<TyId> {
        let e = self.unit.entry(off).ok()?;
        let blob = |t: &mut TyTable, n: u32| match n {
            1 | 2 | 4 | 8 => TyId::unknown(n as usize),
            0 => t.intern(Ty::Array { elem: TyId::B1, len: 0 }),
            n => t.intern(Ty::Array { elem: TyId::B1, len: n }),
        };
        Some(match e.tag() {
            constants::DW_TAG_base_type => {
                let n = self.size(&e).unwrap_or(0);
                let enc = match e.attr_value(constants::DW_AT_encoding) {
                    Some(AttributeValue::Encoding(x)) => x,
                    _ => return Some(blob(self.table, n)),
                };
                match (enc, n) {
                    (constants::DW_ATE_boolean, 1) => TyId::BOOL,
                    (constants::DW_ATE_float, 4) => self.table.intern(Ty::F32),
                    (constants::DW_ATE_float, 8) => self.table.intern(Ty::F64),
                    (constants::DW_ATE_signed | constants::DW_ATE_signed_char, 1 | 2 | 4 | 8) => self.table.int(n as usize, true),
                    (
                        constants::DW_ATE_unsigned | constants::DW_ATE_unsigned_char | constants::DW_ATE_UTF | constants::DW_ATE_boolean,
                        1 | 2 | 4 | 8,
                    ) => self.table.int(n as usize, false),
                    _ => blob(self.table, n),
                }
            }
            constants::DW_TAG_pointer_type | constants::DW_TAG_reference_type | constants::DW_TAG_rvalue_reference_type => {
                let (pointee, mutbl) = match e.attr_value(constants::DW_AT_type) {
                    Some(AttributeValue::UnitRef(t)) => {
                        let konst = self.is_const(t);
                        (self.ty(t), if konst { Mutbl::Not } else { Mutbl::Mut })
                    }
                    _ => (TyId::B1, Mutbl::Mut), // void *
                };
                // a pointer to a function or to an unsized thing is a byte pointer
                let pointee = if self.table.size_of(pointee) == 0 { TyId::B1 } else { pointee };
                self.table.ptr(pointee, mutbl)
            }
            constants::DW_TAG_enumeration_type => {
                let n = self.size(&e).unwrap_or(4);
                let signed = match self.target(&e) {
                    Some(t) => matches!(self.table.tys[t], Ty::Int { signed: true, .. }),
                    None => true,
                };
                match n {
                    1 | 2 | 4 | 8 => self.table.int(n as usize, signed),
                    n => blob(self.table, n),
                }
            }
            constants::DW_TAG_array_type => {
                let elem = self.target(&e)?;
                let mut len = None;
                let mut tree = self.unit.entries_tree(Some(off)).ok()?;
                let root = tree.root().ok()?;
                let mut kids = root.children();
                let mut dims = 0;
                while let Ok(Some(k)) = kids.next() {
                    let k = k.entry();
                    if k.tag() != constants::DW_TAG_subrange_type {
                        continue;
                    }
                    dims += 1;
                    let n = match (k.attr_value(constants::DW_AT_count), k.attr_value(constants::DW_AT_upper_bound)) {
                        (Some(c), _) => c.udata_value(),
                        (None, Some(u)) => u.udata_value().map(|u| u + 1),
                        _ => None,
                    };
                    len = Some(len.unwrap_or(1u64) * n.unwrap_or(0));
                }
                let len = len.unwrap_or(0);
                if dims != 1 {
                    // flatten multi-dimensional arrays to bytes
                    let n = self.table.size_of(elem) as u64 * len;
                    return Some(blob(self.table, n.min(u32::MAX as u64) as u32));
                }
                self.table.intern(Ty::Array { elem, len: len.min(u32::MAX as u64) as u32 })
            }
            constants::DW_TAG_structure_type | constants::DW_TAG_class_type => return self.structure(&e),
            constants::DW_TAG_union_type => blob(self.table, self.size(&e).unwrap_or(0)),
            // function types, `void`, unspecified (`nullptr_t`) ...
            _ => blob(self.table, self.size(&e).unwrap_or(0)),
        })
    }

    fn is_const(&self, mut off: UnitOffset) -> bool {
        for _ in 0..8 {
            let Ok(e) = self.unit.entry(off) else { return false };
            match e.tag() {
                constants::DW_TAG_const_type => return true,
                constants::DW_TAG_volatile_type | constants::DW_TAG_restrict_type | constants::DW_TAG_atomic_type => {
                    match e.attr_value(constants::DW_AT_type) {
                        Some(AttributeValue::UnitRef(t)) => off = t,
                        _ => return false,
                    }
                }
                _ => return false,
            }
        }
        false
    }

    /// Registers the struct at `e` (members are read later, by `members`).
    fn structure(&mut self, e: &Die<'d>) -> Option<TyId> {
        let size = self.size(e)?; // a declaration (`struct foo;`) has no size: bytes
        let c_name = self.name(e);
        if let Some(&t) = c_name.as_ref().filter(|_| self.dedupe).and_then(|n| self.by_name.get(&(n.clone(), size))) {
            return Some(t);
        }
        let name = c_name.as_ref().map(|n| self.table.fresh_struct_name(n)).unwrap_or_default();
        let (sid, ty) = self.table.add_struct(StructDef { name, fields: Vec::new(), size, packed: false, debug: true });
        self.memo.insert(e.offset(), ty);
        if let Some(n) = c_name.filter(|_| self.dedupe) {
            self.by_name.insert((n, size), ty);
        }
        self.queue.push((e.offset(), sid));
        Some(ty)
    }

    /// Reads the members of every struct registered so far, and of the structs
    /// those refer to.
    fn members(&mut self) {
        while let Some((off, sid)) = self.queue.pop() {
            let fields = self.struct_fields(off, self.table.structs[sid].size).unwrap_or_default();
            self.table.structs[sid].fields = fields;
        }
    }

    fn struct_fields(&mut self, off: UnitOffset, size: u32) -> Option<Vec<Field>> {
        let mut members = Vec::new();
        let mut tree = self.unit.entries_tree(Some(off)).ok()?;
        let root = tree.root().ok()?;
        let mut kids = root.children();
        while let Ok(Some(k)) = kids.next() {
            let k = k.entry();
            match k.tag() {
                constants::DW_TAG_member | constants::DW_TAG_inheritance => members.push(k.clone()),
                // a Rust enum: its variants overlap, so it stays bytes
                constants::DW_TAG_variant_part => return None,
                _ => {}
            }
        }
        let mut fields: Vec<Field> = Vec::new();
        for m in &members {
            if m.attr_value(constants::DW_AT_bit_size).is_some() || m.attr_value(constants::DW_AT_data_bit_offset).is_some() {
                continue; // bitfields stay padding
            }
            if m.attr_value(constants::DW_AT_external).is_some() {
                continue; // a static member
            }
            let off = match m.attr_value(constants::DW_AT_data_member_location) {
                Some(v) => match v.udata_value() {
                    Some(o) => o as u32,
                    None => continue, // a location expression (virtual base)
                },
                None => 0,
            };
            let Some(fty) = self.target(m) else { continue };
            let fsize = self.table.size_of(fty);
            if fsize == 0 || off.checked_add(fsize).is_none_or(|end| end > size) {
                continue;
            }
            let base = if m.tag() == constants::DW_TAG_inheritance { "base".to_string() } else { self.name(m).unwrap_or_default() };
            fields.push(Field { name: base, off, ty: fty });
        }
        fields.sort_by_key(|f| f.off);
        // drop members that overlap the one before (unions inside, reordered input)
        let mut kept: Vec<Field> = Vec::new();
        for f in fields {
            if kept.last().is_some_and(|l| l.off + self.table.size_of(l.ty) > f.off) {
                continue;
            }
            kept.push(f);
        }
        unique_field_names(&mut kept);
        Some(kept)
    }
}

/// Decides `packed` for struct `s` after the structs it contains by value.
fn set_packed(t: &mut TyTable, s: StructId, done: &mut Vec<bool>) {
    if done.len() < t.structs.len() {
        done.resize(t.structs.len(), false);
    }
    if std::mem::replace(&mut done[s.index()], true) {
        return;
    }
    let inner: Vec<StructId> = t.structs[s]
        .fields
        .iter()
        .filter_map(|f| {
            let mut ty = f.ty;
            while let Ty::Array { elem, .. } = t.tys[ty] {
                ty = elem;
            }
            match t.tys[ty] {
                Ty::Struct(x) => Some(x),
                _ => None,
            }
        })
        .collect();
    for x in inner {
        set_packed(t, x, done);
    }
    let def = &t.structs[s];
    let packed = !natural_layout(t, &def.fields, def.size);
    t.structs[s].packed = packed;
}

/// Rust identifiers for the field names, unique within the struct.
fn unique_field_names(fields: &mut [Field]) {
    let mut seen = std::collections::HashSet::new();
    for f in fields.iter_mut() {
        let mut n = if f.name.is_empty() { format!("_anon{}", f.off) } else { crate::names::sanitize(&f.name) };
        if n.starts_with("_pad") || !seen.insert(n.clone()) {
            n = format!("{n}_{}", f.off);
            seen.insert(n.clone());
        }
        f.name = n;
    }
}

/// Does `#[repr(C)]` with explicit padding between the fields reproduce these
/// offsets and this size? (No for packed structs and odd sizes.)
pub fn natural_layout(t: &TyTable, fields: &[Field], size: u32) -> bool {
    let mut align = 1;
    for f in fields {
        let a = t.align_of(f.ty);
        if f.off % a != 0 {
            return false;
        }
        align = align.max(a);
    }
    size % align == 0
}
