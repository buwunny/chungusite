//! chungusite IR sketch. One SSA IR, three tiers of instructions.
//! fast mode: lift -> SSA -> emit (Tier::Raw stays raw).
//! safe mode: lift -> SSA -> analyses -> rewrite Raw into Place ops where proven -> emit.
use std::marker::PhantomData;
use std::mem::size_of;
use std::num::NonZeroU32;

// ---------- dense typed indices (u32, never Box/Rc) ----------
/// A dense index type usable as an `Arena` key.
pub trait Idx: Copy {
    fn new(i: usize) -> Self;
    fn index(self) -> usize;
}

/// Stored as index+1 in a NonZeroU32 so Option<Id> is still 4 bytes.
macro_rules! idx { ($($n:ident),*) => {$(
    #[derive(Copy, Clone, PartialEq, Eq, Hash)]
    pub struct $n(pub NonZeroU32);
    /// Prints the index, not the stored index+1: `ValueId(0)` is the first value.
    impl std::fmt::Debug for $n {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, concat!(stringify!($n), "({})"), self.0.get() - 1)
        }
    }
    impl $n {
        #[inline]
        pub const fn from_u32(i: u32) -> Self {
            match NonZeroU32::new(i + 1) { Some(n) => Self(n), None => panic!("id overflow") }
        }
    }
    impl Idx for $n {
        #[inline] fn new(i: usize) -> Self { Self::from_u32(i as u32) }
        #[inline] fn index(self) -> usize { (self.0.get() - 1) as usize }
    }
)*}}
idx!(ValueId, BlockId, LocalId, PlaceId, TyId, ConstId, GlobalId, FuncId, RegionId, StructId, SigId, Symbol);

/// Vec indexed by a typed id. The only container nodes live in.
pub struct Arena<I, T> { data: Vec<T>, _i: PhantomData<I> }

impl<I: Idx, T> Arena<I, T> {
    pub fn with_capacity(n: usize) -> Self { Self { data: Vec::with_capacity(n), _i: PhantomData } }
    #[inline]
    pub fn push(&mut self, t: T) -> I { let id = I::new(self.data.len()); self.data.push(t); id }
    #[inline] pub fn len(&self) -> usize { self.data.len() }
    #[inline] pub fn is_empty(&self) -> bool { self.data.is_empty() }
    /// Drops the contents but keeps the capacity, so the next function reuses it.
    #[inline] pub fn clear(&mut self) { self.data.clear() }
    pub fn iter(&self) -> impl Iterator<Item = (I, &T)> { self.data.iter().enumerate().map(|(i, t)| (I::new(i), t)) }
}

impl<I: Idx + std::fmt::Debug, T: std::fmt::Debug> std::fmt::Debug for Arena<I, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<I: Idx, T> std::ops::Index<I> for Arena<I, T> {
    type Output = T;
    #[inline] fn index(&self, i: I) -> &T { &self.data[i.index()] }
}
impl<I: Idx, T> std::ops::IndexMut<I> for Arena<I, T> {
    #[inline] fn index_mut(&mut self, i: I) -> &mut T { &mut self.data[i.index()] }
}

/// A slice of a per-function pool (call args, phi inputs, projections).
/// Replaces Vec<T> inside nodes so every node stays fixed-size and Copy.
#[derive(Copy, Clone, Debug)]
pub struct ListRef { pub start: u32, pub len: u32 }

impl ListRef {
    pub const EMPTY: ListRef = ListRef { start: 0, len: 0 };
    #[inline]
    pub fn get<T>(self, pool: &[T]) -> &[T] { &pool[self.start as usize..(self.start + self.len) as usize] }
}

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

/// Types every function needs, at fixed ids so the lifter never hashes to find them.
impl TyId {
    pub const B1: TyId = TyId::from_u32(0);   // Unknown { bytes: 1 }
    pub const B2: TyId = TyId::from_u32(1);
    pub const B4: TyId = TyId::from_u32(2);
    pub const B8: TyId = TyId::from_u32(3);
    pub const BOOL: TyId = TyId::from_u32(4);
    pub const PTR: TyId = TyId::from_u32(5);  // *mut B1, i.e. *mut u8
    pub const UNIT: TyId = TyId::from_u32(6); // stores, which define no value
    pub const PAIR: TyId = TyId::from_u32(7); // [B8; 2]: a 16-byte rax:rdx return value

    #[inline]
    pub fn unknown(bytes: usize) -> TyId {
        match bytes { 1 => Self::B1, 2 => Self::B2, 4 => Self::B4, _ => Self::B8 }
    }
}

/// Per-binary type interner, pre-seeded with the `TyId` constants above.
pub struct TyTable { pub tys: Arena<TyId, Ty> }

impl TyTable {
    pub fn new() -> Self {
        let mut tys = Arena::with_capacity(256);
        for t in [
            Ty::Unknown { bytes: 1 }, Ty::Unknown { bytes: 2 }, Ty::Unknown { bytes: 4 }, Ty::Unknown { bytes: 8 },
            Ty::Bool, Ty::RawPtr { pointee: TyId::B1, mutbl: Mutbl::Mut }, Ty::Array { elem: TyId::B1, len: 0 },
            Ty::Array { elem: TyId::B8, len: 2 },
        ] { tys.push(t); }
        TyTable { tys }
    }
}

impl Default for TyTable { fn default() -> Self { Self::new() } }

// ---------- function body ----------
#[derive(Debug)]
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

impl Function {
    /// Pre-size once; after `clear` the same buffers serve every later function.
    pub fn with_capacity(insts: usize, blocks: usize) -> Self {
        Function {
            name: Symbol::from_u32(0),
            sig: SigId::from_u32(0),
            entry: BlockId::from_u32(0),
            blocks: Arena::with_capacity(blocks),
            insts: Arena::with_capacity(insts),
            locals: Arena::with_capacity(16),
            places: Arena::with_capacity(0),
            value_pool: Vec::with_capacity(insts * 2),
            proj_pool: Vec::with_capacity(0),
            consts: Vec::with_capacity(insts / 4),
            origin: Vec::with_capacity(insts),
        }
    }

    pub fn clear(&mut self) {
        self.blocks.clear();
        self.insts.clear();
        self.locals.clear();
        self.places.clear();
        self.value_pool.clear();
        self.proj_pool.clear();
        self.consts.clear();
        self.origin.clear();
    }
}

#[derive(Debug)]
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
    /// A value the ABI leaves undefined: a register a call clobbered, a non-argument
    /// register read on entry, an uninitialized stack slot. Emitted as zero.
    Undef,
    Param(u32),
    /// Live-in value of a block (SSA block parameter). The u8 is the x86 GPR number,
    /// kept so the emitter can name it and the lifter can match edge arguments.
    BlockParam(u8),
    Bin { op: BinOp, lhs: ValueId, rhs: ValueId },
    Un { op: UnOp, v: ValueId },
    Cmp { cc: Cond, lhs: ValueId, rhs: ValueId },
    Cast { kind: CastKind, v: ValueId },
    Select { c: ValueId, t: ValueId, f: ValueId },
    FuncRef(FuncId),             // direct callee as a value
    ImportRef(Symbol),           // libc/WinAPI import as a value
    Call { callee: ValueId, args: ListRef },
    /// Caller-saved register `reg` (x86 number) after the `Call` named by `call`.
    /// The ABI leaves it undefined, except rdx, which holds the high half of a
    /// 16-byte (rax:rdx) result. Compilers that see the callee rely on more: gcc's
    /// interprocedural register allocation keeps a value in a register the callee
    /// is known not to touch. `abi::apply` resolves each one from the callee's
    /// signature.
    CallOut { call: ValueId, reg: u8 },
    /// The caller-saved registers at a return (`lift::EXIT_REGS` order), recorded
    /// when `Lifter::track_exits` is set, so `abi` can tell which registers a
    /// function preserves and whether it returns rdx too. `abi::apply` removes it.
    Exit { regs: ListRef },

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
    /// A jump table: case `k` goes to block `table[k]` (block ids stored in
    /// `value_pool`, see `BlockId::as_value`), any other value to `default`. Its
    /// edges carry no arguments, so every successor of a `Switch` has no block
    /// parameters (the lifter routes each case through a parameterless block).
    Switch { v: ValueId, table: ListRef, default: BlockId },
    Return(Option<ValueId>),
    TailCall { callee: ValueId, args: ListRef },
    Unreachable,
}

impl BlockId {
    /// A block id stored in `value_pool`, for `Switch` tables.
    #[inline]
    pub fn as_value(self) -> ValueId { ValueId(self.0) }
    #[inline]
    pub fn from_value(v: ValueId) -> BlockId { BlockId(v.0) }
}

impl Terminator {
    /// Successor blocks, each once for a `Switch` (its default first), and in
    /// edge-argument order for the others (Branch: true edge, then false edge).
    /// `pool` is the function's `value_pool`, which holds `Switch` tables.
    #[inline]
    pub fn successors(self, pool: &[ValueId]) -> Succs<'_> {
        let (two, table) = match self {
            Terminator::Jump { to, .. } => ([Some(to), None], &[][..]),
            Terminator::Branch { t, f, .. } => ([Some(t), Some(f)], &[][..]),
            Terminator::Switch { table, default, .. } => ([Some(default), None], table.get(pool)),
            _ => ([None, None], &[][..]),
        };
        Succs { two, i: 0, table, k: 0 }
    }
}

/// Iterator over a terminator's successors; see `Terminator::successors`.
pub struct Succs<'a> {
    two: [Option<BlockId>; 2],
    i: usize,
    table: &'a [ValueId],
    k: usize,
}

impl Iterator for Succs<'_> {
    type Item = BlockId;
    fn next(&mut self) -> Option<BlockId> {
        while self.i < 2 {
            self.i += 1;
            if let Some(b) = self.two[self.i - 1] {
                return Some(b);
            }
        }
        // switch cases: each target the first time it appears, unless it's the default
        while self.k < self.table.len() {
            let v = self.table[self.k];
            self.k += 1;
            if Some(BlockId::from_value(v)) != self.two[0] && !self.table[..self.k - 1].contains(&v) {
                return Some(BlockId::from_value(v));
            }
        }
        None
    }
}

/// `UMulHi`/`SMulHi`: the high 64 bits of the full 128-bit product (one-operand `MUL`/`IMUL`).
#[derive(Copy, Clone, Debug)] pub enum BinOp { Add, Sub, Mul, UMulHi, SMulHi, UDiv, SDiv, URem, SRem, And, Or, Xor, Shl, LShr, AShr, RotL, RotR }
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
            Const(_) | Undef | Param(_) | BlockParam(_) | Bin { .. } | Un { .. } | Cmp { .. } | Cast { .. }
            | Select { .. } | FuncRef(_) | ImportRef(_) | Call { .. } | CallOut { .. } | Exit { .. } => Tier::Pure,
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

