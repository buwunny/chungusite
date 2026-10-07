//! chungusite IR sketch. One SSA IR, three tiers of instructions.
//! fast mode: lift -> SSA -> emit (Tier::Raw stays raw).
//! safe mode: lift -> SSA -> analyses -> rewrite Raw into Place ops where proven -> emit.
#![allow(dead_code)]
use std::marker::PhantomData;
use std::mem::size_of;
use std::num::NonZeroU32;

// ---------- dense typed indices (u32, never Box/Rc) ----------
/// Stored as index+1 in a NonZeroU32 so Option<Id> is still 4 bytes.
macro_rules! idx { ($($n:ident),*) => {$(
    #[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
    pub struct $n(pub NonZeroU32);
)*}}
idx!(ValueId, BlockId, LocalId, PlaceId, TyId, ConstId, GlobalId, FuncId, RegionId, StructId, SigId, Symbol);

/// Vec indexed by a typed id. The only container nodes live in.
pub struct Arena<I, T> { data: Vec<T>, _i: PhantomData<I> }

/// A slice of a per-function pool (call args, phi inputs, projections).
/// Replaces Vec<T> inside nodes so every node stays fixed-size and Copy.
#[derive(Copy, Clone, Debug)]
pub struct ListRef { pub start: u32, pub len: u32 }

// ---------- types (interned once per binary, shared by all functions) ----------
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Mutbl { Not, Mut }

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Ty {
    Int { bits: u8, signed: bool },
    Bool, F32, F64,
    /// Width known, meaning not (yet). Analyses refine it; emit falls back to uN.
    Unknown { bytes: u16 },
    RawPtr { pointee: TyId, mutbl: Mutbl },                  // fast mode / fallback
    Ref { pointee: TyId, mutbl: Mutbl, region: RegionId },    // safe mode only
    Array { elem: TyId, len: u32 },
    Slice { elem: TyId },
    Struct(StructId),
    Fn(SigId),
}

// ---------- function body ----------
pub struct Function {
    pub name: Symbol,
    pub sig: SigId,
    pub entry: BlockId,
    pub blocks: Arena<BlockId, Block>,
    pub insts: Arena<ValueId, Inst>,     // SSA: ValueId == the inst that defines it
    pub locals: Arena<LocalId, Local>,   // recovered stack slots / variables
    pub places: Arena<PlaceId, Place>,   // only populated in safe mode
    pub value_pool: Vec<ValueId>,        // backing store for ListRef of values
    pub proj_pool: Vec<Proj>,            // backing store for ListRef of projections
    pub consts: Vec<u128>,               // ConstId -> bits
    pub origin: Vec<u64>,                // ValueId -> x86 address (debug/side table)
}

pub struct Block {
    pub insts: ListRef,          // into value_pool, in order
    pub params: ListRef,         // block params instead of phi nodes
    pub term: Terminator,
}

#[derive(Copy, Clone, Debug)]
pub struct Local { pub ty: TyId, pub frame_offset: i32, pub size: u32, pub name: Option<Symbol> }

#[derive(Copy, Clone, Debug)]
pub struct Inst { pub kind: InstKind, pub ty: TyId }

#[derive(Copy, Clone, Debug)]
pub enum InstKind {
    // ---- Tier::Pure: shared by both modes ----
    Const(ConstId),
    Param(u32),
    Bin { op: BinOp, lhs: ValueId, rhs: ValueId },
    Un { op: UnOp, v: ValueId },
    Cmp { cc: Cond, lhs: ValueId, rhs: ValueId },
    Cast { kind: CastKind, v: ValueId },
    Select { c: ValueId, t: ValueId, f: ValueId },
    FuncRef(FuncId),             // direct callee as a value
    ImportRef(Symbol),           // libc/WinAPI import as a value
    Call { callee: ValueId, args: ListRef },

    // ---- Tier::Raw: what the lifter produces; fast mode emits these directly ----
    /// base + index*scale + disp, i.e. an x86 effective address. Pointer typed.
    PtrOffset { base: ValueId, index: Option<ValueId>, scale: u8, disp: i32 },
    AddrOfLocal(LocalId),        // &raw mut local
    AddrOfGlobal(GlobalId),
    IntToPtr(ValueId),
    PtrToInt(ValueId),
    Load { ptr: ValueId, align: u8, volatile: bool },
    Store { ptr: ValueId, val: ValueId, align: u8 },
    MemCopy { dst: ValueId, src: ValueId, len: ValueId },

    // ---- Tier::Safe: only created by safe-mode rewrites ----
    Copy(PlaceId),               // read a Copy value out of a place
    Move(PlaceId),               // move out (ownership transfer)
    Assign { place: PlaceId, val: ValueId },
    Borrow { place: PlaceId, mutbl: Mutbl, region: RegionId },
    Aggregate { ty: TyId, fields: ListRef },   // struct/array literal

    // ---- escape hatch ----
    Opaque { addr_idx: u32 },    // inline asm / unliftable instruction, always unsafe
}

/// A Rust place expression: base + projections, e.g. (*p).items[i].len
#[derive(Copy, Clone, Debug)]
pub struct Place { pub base: PlaceBase, pub proj: ListRef }

#[derive(Copy, Clone, Debug)]
pub enum PlaceBase { Local(LocalId), Global(GlobalId), Deref(ValueId) /* a &T / &mut T value */ }

#[derive(Copy, Clone, Debug)]
pub enum Proj { Field(u32), Index(ValueId), ConstIndex(u32), Deref, Subslice { from: u32, to: u32 } }

#[derive(Copy, Clone, Debug)]
pub enum Terminator {
    Jump { to: BlockId, args: ListRef },
    Branch { c: ValueId, t: BlockId, f: BlockId, args: ListRef /* t args then f args */ },
    Switch { v: ValueId, table: ListRef /* BlockIds in value_pool */, default: BlockId },
    Return(Option<ValueId>),
    TailCall { callee: ValueId, args: ListRef },
    Unreachable,
}

#[derive(Copy, Clone, Debug)] pub enum BinOp { Add, Sub, Mul, UDiv, SDiv, URem, SRem, And, Or, Xor, Shl, LShr, AShr, RotL, RotR }
#[derive(Copy, Clone, Debug)] pub enum UnOp { Neg, Not, Bswap, Popcnt, Ctz, Clz }
#[derive(Copy, Clone, Debug)] pub enum Cond { Eq, Ne, Ult, Ule, Ugt, Uge, Slt, Sle, Sgt, Sge }
#[derive(Copy, Clone, Debug)] pub enum CastKind { Trunc, ZExt, SExt, Bitcast, IntToFloat, FloatToInt }

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Tier { Pure, Raw, Safe }

impl InstKind {
    /// The emitter wraps maximal runs of Tier::Raw in one `unsafe { }`.
    pub fn tier(&self) -> Tier {
        use InstKind::*;
        match self {
            Const(_) | Param(_) | Bin { .. } | Un { .. } | Cmp { .. } | Cast { .. }
            | Select { .. } | FuncRef(_) | ImportRef(_) | Call { .. } => Tier::Pure,
            PtrOffset { .. } | AddrOfLocal(_) | AddrOfGlobal(_) | IntToPtr(_) | PtrToInt(_)
            | Load { .. } | Store { .. } | MemCopy { .. } | Opaque { .. } => Tier::Raw,
            Copy(_) | Move(_) | Assign { .. } | Borrow { .. } | Aggregate { .. } => Tier::Safe,
        }
    }
}

// ---------- safe-mode side tables (never allocated in fast mode) ----------
/// What the analyses proved a pointer-typed ValueId points at.
#[derive(Copy, Clone, Debug)]
pub enum PtrFact {
    Unknown,                                         // stays raw, stays unsafe
    Place(PlaceId),                                  // points exactly at a place
    Owned { alloc_site: ValueId, ty: TyId },         // malloc/new result -> Box<T>
    SliceOf { base: PlaceId, len: Option<ValueId> }, // base+i*size pattern -> &[T]
}

#[derive(Copy, Clone, Debug)]
pub struct RegionFact { pub live_from: ValueId, pub live_to: ValueId, pub exclusive: bool }

pub struct SafeFacts {
    pub ptr: Vec<PtrFact>,         // indexed by ValueId.0, dense
    pub regions: Vec<RegionFact>,  // indexed by RegionId.0
}

// Keep nodes small: InstKind fits 16 bytes, Inst 20 bytes, so 3 per cache line.
const _: () = assert!(size_of::<InstKind>() <= 16);
const _: () = assert!(size_of::<Inst>() <= 20);
const _: () = assert!(size_of::<Terminator>() <= 24);

