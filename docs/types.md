# Types

Before this pass every value was an unsigned integer and every pointer a `u64`, or a `&[u8]` in safe mode. [`src/types.rs`](../src/types.rs) recovers:

- **integer widths** of arguments and return values: `fn clamp(edi: i32, esi: i32, edx: i32) -> i32` instead of three `u64`s;
- **signedness** of every value, so `x <= y` on `i32`s instead of `(x as i32) <= (y as i32)` on `u32`s;
- **what pointer arguments point to**: a struct with named fields (`&rec`, `r.count`), one scalar (`&u64`), or an array (`&[i32]`).

Types come from the code itself (the *facts*), from debug info when the binary has it, and later from the ML type model. Debug info and models only *propose* types. A proposal is used only if the facts allow it, and otherwise rejected with a reason (`--emit types` prints both). The IR doesn't change: the pass produces a side table, `FnTypes`, that the emitter looks up.

## Facts

`Facts::gather` collects four facts per function, all sound. They run after `abi::apply`, on the function's real signature.

**Demanded bits.** A backward pass computes how many low bits of each value something uses. Addition, subtraction, multiplication, bitwise operations, left shifts, truncation and the address arithmetic of `lea` only need the low bits of their operands that their own result needs. Everything else (comparisons, right shifts, division, memory addresses, call arguments, stored values) needs all of them. An argument whose demand is at most 32 bits is taken as a narrower integer, named after the register part it uses (`edi`, `si`, `dil`), and the body's 64-bit view is rebuilt from it (`let rdi: u64 = edi as u32 as u64;`), so nothing else changes. The upper bits it drops are bits nothing reads.

**Zero-extended returns.** A forward pass computes how many low bits each value can be non-zero in: `ZExt` from a narrow value, constants, masks, narrow loads, and block parameters that only merge such values. If every return value fits in N ≤ 32 bits, the function returns an N-bit integer. Callers zero-extend it back (`f(..) as u32 as u64`), which gives exactly the register value the original returned. A function that tail-calls doesn't qualify, because it returns whatever the callee does.

**Signedness.** Values that must have the same Rust type form one class: the operands and result of an arithmetic or bitwise operation, the two operands of a comparison, `select` arms, and a block parameter with its incoming arguments. Each class votes:

| Signed | Unsigned |
|---|---|
| `jl`, `jg`, `cmovge`... (signed compares) | `jb`, `ja`, `setae`... (unsigned compares) |
| `sar`, `idiv` | `shr`, `div` |
| `movsx`, `movsxd` (on the source) | `movzx` (on the source) |

A 32-bit write zero-extending into the 64-bit register (`mov eax, ..`) is not evidence: every 32-bit instruction does it. Addresses and call results are always unsigned. Signedness only changes how a value is spelled. Every operation keeps its x86 meaning, so an unsigned compare of two `i32`s casts them (`(a as u32) < (b as u32)`) and an `i32` written to a 64-bit register is zero-extended (`as u32 as u64`). A wrong vote makes the output read worse, never compute something else.

**Accesses.** For each load and store, `borrow`'s origins say which argument the address derives from and at what offset. A computed offset also gets an alignment (`residues`): `p + 4*i` and `p += 4` in a loop are 4-aligned relative to `p`, so that access is an element of a 4-byte array.

## Inferred pointees

With nothing proposed, a pointer argument's accesses decide its type:

- all at constant offsets, not overlapping: a struct with a field per offset, `S_<function>_<register>` with fields `f0`, `f8`, ... and padding between them;
- only offset 0: one scalar, `&u64`;
- one element size, some at computed but aligned offsets: a slice, `&[u32]`.

Struct definitions are `#[repr(C, packed)]` with explicit padding. The Rust layout is byte-for-byte the C one, but with alignment 1. So a reference can be made to any address the binary used, and fields are integers (pointers are `u64`, `bool` is `u8`), so any bytes are a valid value.

In safe mode an argument becomes `&S`, `&mut S`, `&[T]` or `&T` (in `Option` when null-checked) if borrow inference allows a reference and *every* access through it is described by the type. Accesses become `r.count`, `*p`, `p[3]` or `p[(addr - p_base) / 4]`, still bounds-checked. Otherwise it stays a byte slice. In fast mode, and in safe mode for arguments that stay raw, a struct pointer is `*mut S` and the accesses it describes are `(*p).count`. Address arithmetic that only fed such accesses is no longer printed.

## Proposals: debug info and models

```rust
pub trait TypeModel: Sync {
    fn name(&self) -> &str;
    fn propose(&self, func: &FuncRef) -> Option<Proposal>;
}
```

A `Proposal` lists C types (`CType`: integers, floats, pointers, structs with their layout, arrays) and names for the integer-class arguments in calling-convention order, and the return type. `types::infer` asks each model in order of trust: debug info first, then the model in `TypeOptions::model`. For each argument it takes the first proposal and checks it:

| Proposal | Accepted when | Example rejection |
|---|---|---|
| integer of N bits | the argument isn't dereferenced, and its demanded bits fit in N | `rejected i16 for rdi: the code uses 32 bits of it` |
| pointer to a struct | every constant-offset access lands exactly on a scalar field (nested structs and arrays included) | `rejected struct two * for rdi: 8-byte access at +0 matches no field` |
| pointer to a scalar | every access has the element's size, at an aligned offset | `rejected u32 * for rsi: 1-byte access to 4-byte elements` |
| return of N bits | every return value fits in N bits | `rejected u8 for the return value: the code returns 32 bits` |
| argument count | the proposal lists at least as many arguments as the code takes | `rejected all arguments: it lists 2 integer arguments, the code takes 3` |

An argument's name is used only if its type was not rejected. A rejected argument falls back to what the facts infer. Names that would clash with a function, a static, a keyword or a name the emitter makes up get a `_` suffix (`self` becomes `self_`).

`NoModel` proposes nothing and is the default. **The ML type model plugs in here**: implement `TypeModel` with `refine::to_ids(func.ir, ..)` and `refine::infer::Refiner` (docs/ml-runtime.md), turn the predicted labels into a `Proposal`, and pass it as `TypeOptions { model: Some(&m), .. }` to `Program::build_with`. Its proposals are checked exactly like debug info's, so a wrong prediction costs a rejection note, not a wrong program. `tests/types.rs` has a stand-in model that exercises all of this.

### DWARF

[`src/dwarf.rs`](../src/dwarf.rs) reads `.debug_info` with `gimli` and implements `TypeModel`. It covers C, C++ and Rust compile units: functions by entry address (and by name, for object files where every section starts at 0), argument names and types through `DW_AT_abstract_origin` and `DW_AT_specification`, typedefs, qualifiers, enums, arrays, nested structs, and C++ base classes. Bit-fields and unions become opaque bytes, and so do Rust enums with data (`DW_TAG_variant_part`). In a relocatable object (`.o`) the debug sections' relocations are applied first, since string offsets and addresses aren't filled in yet.

Arguments are matched to registers the System V way. Integers, pointers and enums take rdi, rsi, rdx, rcx, r8, r9 in order, and floating-point arguments are skipped because they travel in xmm registers. A function with a struct passed or returned by value is skipped entirely, because the rules for those (a hidden return pointer in rdi, splitting into two registers) aren't modelled. That rule also keeps the Rust ABI safe: rustc passes scalars and thin pointers in order, and slices, `&str` and `&dyn` are structs in DWARF. Rust type names lose their module paths, so `Vec<chungusite::load::FuncBytes, alloc::alloc::Global>` becomes `Vec_FuncBytes__Global`.

`--no-debug-info` turns it off.

## What it does on real code

On chungusite's own debug build (23,503 functions, 21,293 lifted, with DWARF), type recovery narrows 2,480 integer arguments and 1,122 return values, types 12,763 pointer arguments, and names 4,525 arguments from debug info. In safe mode, 7,460 arguments used to be byte slices: now 7,227 of them are `&S`, `&T` or `&[T]` and 233 stay `&[u8]`. The output type-checks in both modes.

`tests/differential.rs` checks that the typed code still computes what the original does. It runs every corpus function at `-O1`, `-O2` and `-Os` and, for debug-info types, at `-O2 -g`, with gcc and clang. `tests/types.rs` covers each fact, the inferred pointees, the proposal checks, debug info from C, and random programs in both modes.

## Not done yet

- Pointers stay `u64` inside function bodies. Only arguments get pointer types.
- Values loaded from memory aren't typed by what is done with them: a loaded pointer's own pointee (`p->next->count`) needs points-to facts per field (ownership.md, stage 6).
- Arrays of structs (`items[i].x`) are seen as indexed accesses of mixed sizes and stay byte slices.
- Callers don't pass their argument types to callees, and callees don't tell callers which bits they demand. Both would narrow more with an interprocedural fixpoint like `abi`'s.
- Statics are still `Bytes<N>` (roadmap step 6).
