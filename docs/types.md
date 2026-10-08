# Type recovery

`src/types.rs` gives the emitter real types in place of `u64` everywhere: integer widths and signedness, pointer arguments, and structs with fields. It runs after signature recovery ([calls.md](calls.md)) and changes nothing in the IR. The IR keeps the lifter's storage types (`B1`..`B8`, `PTR`); what type recovery produces is a side table per function (`FnTypes`: a Rust type per value, a pointee per value used as a pointer, argument and return types) plus the struct definitions in the program's `TyTable`.

Before, on `long bump(struct node *p, int d)` from `gcc -O2`:

```rust
pub unsafe fn bump(mut rdi: u64, mut rsi: u64) -> u64 {
    let v1: u64 = rdi.wrapping_add(0x8);
    let v2: u64 = unsafe { (v1 as *const u64).read_unaligned() };
```

After, with `-g`:

```rust
#[repr(C)]
pub struct Node {
    pub key: i32,
    pub _pad4: [u8; 4],
    pub count: i64,
    pub next: *mut Node,
    pub flag: u8,
    pub _pad25: [u8; 7],
}

pub unsafe fn bump(p: *mut Node, d: i32) -> i64 {
    let v8: i64 = unsafe { (*p).count };
```

and without debug info, from the accesses alone:

```rust
#[repr(C, packed)]
pub struct S1 {
    pub f0: i32,
    pub _pad4: [u8; 4],
    pub f8: i64,
    pub _pad16: [u8; 8],
    pub f24: u8,
}

pub unsafe fn bump(rdi_p: *mut S1, rsi: i32) -> i64 {
```

In safe mode the same argument becomes `p: &mut Node` and the accesses `p.count = v9;`.

## Where types come from

Strongest first, per function:

1. **DWARF** (`src/dwarf.rs`, `gimli`), unless `--no-dwarf`. Prototypes are matched to functions by entry address: parameter names, parameter and return types, and the layouts of the structs they point to, with field names. Relocatable objects (`.o`) work too. A prototype is used only when the name matches the symbol, it isn't variadic, every parameter fits in an integer register, and there are at least as many parameters as signature recovery found arguments; otherwise the function falls through to inference (struct or float arguments by value, compiler clones like `f.constprop.0`). Types with no clean Rust equivalent (unions, Rust enums, bitfields, 128-bit integers, odd sizes) become byte arrays of the right size. A struct whose recorded layout isn't the natural C layout is `packed`. A `void *` or `char *` parameter whose code clearly uses a struct takes the inferred struct instead.
2. **A type model**, optional (see below).
3. **Inference**, always on:
   - *Signedness*: values that flow into each other share a class (union-find over arithmetic, block parameters and their edge arguments, selects and compares). Signed compares, `SDiv`/`SRem`, `SAR`, `MOVSX` and small negative constants vote signed; unsigned compares, `UDiv`/`URem` and `SHR` vote unsigned; a class used as an address is unsigned.
   - *Narrow arguments and returns*: an argument whose every use truncates it is `i32`/`u32` (or narrower); a return value that is always zero-extended from 32 bits, or a `SETcc` result, returns `i32`/`u32` or `bool`.
   - *Pointees*, Steensgaard style: values that flow into each other point at the same type. Each load or store `*(base + d)` adds a field at offset `d`, and the value stored or loaded there is unified with that field, so `p = p->next` makes `next: *mut S2` point at its own struct. Non-overlapping fields make a struct (`#[repr(C, packed)]`, since the real alignment isn't known, with fields `f{offset}`); indexed accesses of one width make a scalar pointee (`*const u32`). Only dereferences make a pointer: `lea` arithmetic on an integer argument doesn't.

Every type is a claim the emitted code relies on, so each one keeps what the machine code does: a narrowed argument is only ever used truncated, a narrowed return value has zero upper bits, and a field access happens at exactly the offset and width the instruction used. Anything uncertain stays `u64`. Each emitted struct has a `const _: () = assert!(size_of::<S>() == N);`.

## In the emitted code

- Signatures carry the types and debug names. The prologue converts typed arguments back to the `u64` register variables the body was written against, so bodies need no type-directed rewriting; values inside a body are typed integers and pointer values are still `u64` addresses.
- 8- and 16-bit arguments are widened by their own signedness, matching clang's caller-side extension, so a debug `i8` parameter sees the same register value the original did.
- In fast mode an access that is exactly a field of a typed pointer argument reads `(*p).field`; in safe mode an argument `borrow::analyze` classifies as a reference, all of whose accesses are exact fields, is `&S`/`&mut S` and accesses are `p.field`. A field access never replaces a bounds-checked slice access.
- Calls between decompiled functions pass and receive the callee's types.

The stderr summary adds a line: `types: A of B arguments typed, N structs inferred from field accesses, K prototypes from debug info`. On chungusite's own debug build: 19,922 of 43,024 arguments typed, 957 inferred structs, 5,362 prototypes from debug info; without debug info, 15,872 typed. The output type-checks in both modes.

## The model hook

`types::TypeModel` is where a learned type model plugs in ([ml-runtime.md](ml-runtime.md)). Nothing in this crate implements it yet.

```rust
pub trait TypeModel: Sync {
    fn propose(&self, f: &Function, vars: &[Var]) -> Vec<Option<Proposal>>;
}
```

It is asked about each argument and the return value (`Var::Arg`, `Var::Ret`) of a function without usable debug info, and answers with a C type label as DWARF spells it (`int`, `unsigned char`, `char *`, `size_t`) and a score. `parse_label` turns the label into a width, signedness, bool or pointer, and `accept` checks it against the facts: a proposal may say more than the code shows (an `int` for an argument only used as a byte), never less (a byte for a value used at 32 bits) or something else (an integer for a dereferenced value, unsigned for a value compared as signed). A rejected proposal changes nothing, and `TypeStats` counts both. Pass a model with `program::Options { model: Some(&m), .. }` and `Program::build_with`.

## Tests

`tests/types.rs` checks prototypes and structs from `gcc -O2 -g`, safe-mode struct references, `--no-dwarf` inference, `parse_label` and the gate. `tests/differential.rs` runs each corpus build both without and with `-g`, so a wrong type from either source shows up as a wrong result. 427 of 468 functions pass in each variant (854 of 936), the same as before type recovery.

## Next

- Pointer values inside bodies as real pointers rather than `u64`.
- Interprocedural pointees: a callee's `*mut S` should type what the caller passes.
- Indexed arguments in safe mode as `&[T]` rather than `&[u8]`.
- Typed statics (`[u32; N]`, strings) from how globals are accessed.
