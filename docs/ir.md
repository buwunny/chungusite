# chungusite IR design

**Core idea: one IR, not two.** Both modes share a single SSA IR whose instructions come in three tiers. The lifter only ever produces `Pure` and `Raw` instructions. Fast mode emits them as-is. Safe mode runs analyses, then *rewrites* `Raw` instructions into `Safe` ones wherever it can prove ownership and lifetimes, and leaves the rest raw. The emitter is shared and wraps each maximal run of `Raw` instructions in one `unsafe { }`.

So `--mode fast` is just `--mode safe` with zero rewrites applied. That gives you:

- One lifter, one emitter, one test corpus for both modes.
- Graceful degradation: when safe-mode analysis can't prove a pointer, that spot falls back to the fast-mode output instead of failing the whole function.
- A measurable "how safe is this output" metric (fraction of `Raw` instructions remaining).

## Pipeline

```
x86_64 bytes
  └─ decode + lift ─────────► Pure + Raw IR (SSA, block params)       [both modes]
       └─ cheap cleanup (const fold, copy prop, flag elimination)       [both modes]
            ├─ --mode fast ─► emit  → raw pointers + unsafe { }
            └─ --mode safe ─► analyses → SafeFacts side tables
                               └─ rewrite Raw → Safe where proven
                                    └─ emit → &T / &mut T, Box, slices; unsafe only for leftovers
```

## Worked example

```asm
; int sum2(struct S *s, long i)  -> s->count + s->items[i]
mov  eax, [rdi+8]
add  eax, [rdi+rsi*4+16]
ret
```

Lifted IR (identical in both modes up to here):

```
v0 = Param(0)                         : *mut u8   (Unknown pointee)
v1 = Param(1)                         : i64
v2 = PtrOffset { base: v0, disp: 8 }
v3 = Load v2                          : i32
v4 = PtrOffset { base: v0, index: v1, scale: 4, disp: 16 }
v5 = Load v4                          : i32
v6 = Bin Add v3, v5                   : i32
Return v6
```

**Fast mode** emits the Raw tier directly:

```rust
pub unsafe fn sum2(a0: *mut u8, a1: i64) -> i32 {
    let v3 = *(a0.add(8) as *const i32);
    let v5 = *(a0.offset(a1 * 4 + 16) as *const i32);
    v3.wrapping_add(v5)
}
```

**Safe mode**: type recovery sees two disjoint access patterns off `v0` (a scalar at +8, an `i32` array at +16 indexed by `v1`) and infers `struct S { _pad: u64, count: i32, _pad2: u32, items: [i32; N] }`. Escape and lifetime analysis sees `v0` is only read and never escapes, so `PtrFact::Place` + a shared region. The rewrite replaces `PtrOffset`+`Load` pairs with `Copy(place)`:

```
p0 = Place { base: Deref(v0), proj: [Field(1)] }
p1 = Place { base: Deref(v0), proj: [Field(3), Index(v1)] }
v3 = Copy p0
v5 = Copy p1
```

```rust
pub fn sum2(s: &S, i: usize) -> i32 {
    s.count.wrapping_add(s.items[i])
}
```

If the analysis had failed on `items[i]` (say `i` might be negative), only `v5` stays `Raw` and the output becomes `s.count.wrapping_add(unsafe { *(…) })`.

## How it stays fast

1. **Flat arenas, u32 ids, no pointers between nodes.** Every instruction, block, place and type lives in a `Vec` indexed by a 4-byte id. No `Box<Expr>` trees, no `Rc<RefCell<…>>`, no per-node allocation. Building a function is a handful of `Vec::push` calls, and freeing it is dropping a few `Vec`s.
2. **Fixed-size, `Copy` nodes.** Variable-length data (call args, phi/block args, projections, switch tables) goes into per-function pools referenced by `ListRef { start, len }`, so every `InstKind` is 16 bytes and `Inst` is 20. Three instructions per cache line, and passes are linear scans. The `const` size asserts at the bottom of the sketch fail the build if someone adds a fat variant.
3. **Ids are `NonZeroU32`.** `Option<ValueId>` stays 4 bytes, which is what keeps `PtrOffset` (an x86 effective address in one node) inside 16 bytes.
4. **SSA with block parameters, not phi nodes.** Construction is a single pass using the Braun et al. "simple and efficient SSA construction" algorithm over the CFG, with no dominance frontiers needed.
5. **Safe-mode data lives in side tables, not in the nodes.** `SafeFacts` is dense `Vec`s indexed by `ValueId`. Fast mode never allocates it, so it pays nothing for safe mode's existence, not even bigger nodes. The `places` arena is empty in fast mode too.
6. **Rewrites are in place.** Turning `Load { ptr }` into `Copy(place)` overwrites one 16-byte slot; ids stay stable, so no remapping pass and no re-hashing.
7. **Types are interned once per binary.** `TyId` is a 4-byte handle into a shared table, so type equality is integer comparison.
8. **Functions are independent units.** After call-graph discovery, lift and emit per function with `rayon`. Safe mode's interprocedural part (signatures, struct layouts) runs as a fixpoint over summaries, not over function bodies.
9. **No strings until emit.** Names are `Symbol` ids; the emitter writes straight into one `String` buffer per function.

For decoding, use a table-driven decoder such as `iced-x86` (or `yaxpeax-x86`) in no-alloc mode, and lift each instruction straight into the arena without an intermediate per-instruction AST.

## Design notes and trade-offs

- **Flags.** The lifter models `rflags` bits as ordinary SSA values (`Cmp`, `Bin`) and lets dead-code elimination drop unused ones. That is simpler than lazy flag tracking, and DCE removes the flag values nothing reads.
- **`Unknown { bytes }` type.** Early on, a register is just "8 bytes". Fast mode emits `u64` for it; safe mode refines it. This keeps the lifter from ever having to guess.
- **`Move` vs `Copy`.** Safe mode uses `Move` only when type recovery says the type isn't `Copy` (e.g. it contains an owned `Box`/`Vec` recovered from allocator calls). Otherwise reads are `Copy`.
- **`Opaque`** holds inline asm and anything unliftable (e.g. `rdtsc`, `cpuid`). It's always `Tier::Raw` and becomes `core::arch::asm!` inside `unsafe`.
- **What's deliberately missing:** exceptions/SEH unwinding and SIMD vector types. Both slot in as new `Ty` and `InstKind` variants without changing the structure, but they'll need the 16-byte budget watched (SIMD constants go in the `consts` pool, not inline).

## The sketch

Also saved as [ir.rs](ir.rs). It compiles on rustc 1.97 with no warnings, and the size asserts pass (`InstKind` = 16 B, `Inst` = 20 B, `Terminator` = 24 B, `Place` = 16 B).

```rust
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

```
