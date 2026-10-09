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
   - *Pointees*, Steensgaard style: values that flow into each other point at the same type. Each load or store `*(base + d)` adds a field at offset `d`, and the value stored or loaded there is unified with that field, so `p = p->next` makes `next: *mut S2` point at its own struct. Classes span the whole program: a value passed to a decompiled function joins the class of the parameter it arrives in, and a call's result the class of what the callee returns, so each function's partial view of a struct adds up to one struct. Two classes join only if their fields agree (no overlap at different widths, also for the classes their common pointer fields would join), neither holds a stack, global or constant address, and indexed and field accesses don't mix; otherwise the call leaves them apart. A prototype's pointer types type the classes they share with other functions, but only where every access fits the prototype's type: a struct and its first field have the same address, so a callee's `&self.items` can join `self`'s class. Mutability (`*const`/`*mut`) stays per function. Non-overlapping fields make a struct (`#[repr(C, packed)]`, since the real alignment isn't known, with fields `f{offset}`); indexed accesses of one width make a scalar pointee (`*const u32`). Only dereferences make a pointer: `lea` arithmetic on an integer argument doesn't.

Every type is a claim the emitted code relies on, so each one keeps what the machine code does: a narrowed argument is only ever used truncated, a narrowed return value has zero upper bits, and a field access happens at exactly the offset and width the instruction used. Anything uncertain stays `u64`. Each emitted struct has a `const _: () = assert!(size_of::<S>() == N);`.

## In the emitted code

- Signatures carry the types and debug names. The prologue converts typed arguments back to the `u64` register variables the body was written against, so bodies need no type-directed rewriting; values inside a body are typed integers.
- A value in the body that points at a struct is a `*mut S` when it is loaded from memory or carried by a block parameter and every use is one the emitter converts: an access through it, an address computed from it, a compare, a call argument, an edge copy or a return. Its field reads are `(*v7).next`, and a compare with 0 is `v7.is_null()`. Arguments stay `u64` registers in the body. On chungusite's own debug build 10,922 locals are typed pointers, with 867 `is_null()` tests.
- 8- and 16-bit arguments are widened by their own signedness, matching clang's caller-side extension, so a debug `i8` parameter sees the same register value the original did.
- In fast mode an access that is exactly a field of a typed pointer argument reads `(*p).field`; in safe mode an argument `borrow::analyze` classifies as a reference, all of whose accesses are exact fields, is `&S`/`&mut S` and accesses are `p.field`, unless decompiled code calls the function or it lends the argument to a callee as a slice (callers lend byte slices, not structs). A field access never replaces a bounds-checked slice access.
- In safe mode a slice argument whose pointee is a 2-, 4- or 8-byte integer is `&[T]`/`&mut [T]` when every access through it reads or writes one whole element at an offset from the argument that is provably a multiple of the element size (a residue-mod-`w` dataflow over the body), nothing copies or fills it, and it isn't passed to a call. Accesses index elements: `a[v8 as usize]` when the address is plainly `a + 4 * v8`, `a[(p - a_base) / 4]` for a pointer walk. The same gate as structs applies (no decompiled caller, since callers lend bytes). On chungusite's own debug build 78 functions take one.
- Calls between decompiled functions pass and receive the callee's types.

The stderr summary adds a line: `types: A of B arguments typed, N structs inferred from field accesses, K prototypes from debug info`. On chungusite's own debug build: 45,587 of 63,944 arguments typed, 1,938 inferred structs, 7,175 prototypes from debug info; without debug info, 42,546 typed. Before pointee classes crossed calls these were 29,392 and 23,922, and 72,892 accesses printed as struct fields, now 73,998. The fast-mode output type-checks.

## The model hook

`types::TypeModel` is where a learned type model plugs in. `refine::model::TypeClassifier` implements it for an ONNX classifier, and `--refine <dir>` uses it ([ml-runtime.md](ml-runtime.md#--refine)).

```rust
pub trait TypeModel: Sync {
    fn propose(&self, f: &Function, vars: &[Var]) -> Vec<Option<Proposal>>;
}
```

It is asked about each argument and the return value (`Var::Arg`, `Var::Ret`) of a function without usable debug info, and answers with a C type label as DWARF spells it (`int`, `unsigned char`, `char *`, `size_t`) and a score. `parse_label` turns the label into a width, signedness, bool or pointer, and `accept` checks it against the facts: a proposal may say more than the code shows (an `int` for an argument only used as a byte), never less (a byte for a value used at 32 bits) or something else (an integer for a dereferenced value, unsigned for a value compared as signed). A rejected proposal changes nothing, and `TypeStats` counts both (the stderr summary prints them as `type model: N proposals used, M turned down by the facts`). The questions don't depend on each other, so `recover` asks them for every function in parallel before the sequential part that builds the types. Pass a model with `program::BuildOptions { model: Some(&m), .. }` and `Program::build_with`.

## Tests

`tests/types.rs` checks prototypes and structs from `gcc -O2 -g`, safe-mode struct references, `--no-dwarf` inference, one struct across callers and callees (with and without the callees' debug info), typed pointer locals, `&[T]` arguments, `parse_label` and the gate. `tests/differential.rs` runs each corpus build both without and with `-g`, so a wrong type from either source shows up as a wrong result; 1,918 of 1,920 pairs pass and none disagree.

## Next

- Typed statics (`[u32; N]`, strings) from how globals are accessed.
