# Using the `chungusite` command

```
cargo install --path .            # or: cargo run --release -- <args>

chungusite ./prog                          # every function, fast mode, to stdout
chungusite ./prog --mode safe -o prog.rs   # safe mode, to a file
chungusite ./prog -f parse -f main         # only these symbols (mangled or demangled)
chungusite ./prog --addr 0x401136          # the function starting there
chungusite ./prog --addr 0x401136 --size 0x40   # no symbol there: lift exactly these bytes
chungusite ./prog --list                   # which functions lift, and why the others don't
chungusite --hex "48 8b 47 08 c3"          # raw bytes, loaded at 0x1000, no file needed
chungusite ./prog -j 1                     # one worker thread (default: one per CPU)
```

Input is any x86_64 ELF, Mach-O or PE file (`object` crate). Functions come from the symbol table, falling back to the dynamic symbols, so a stripped binary needs `--addr` with `--size`.

Rust (legacy and v0) and Itanium C++ symbols are demangled, without the Rust hash or the C++ parameter list: `_ZN4core3fmt5write17h…E` shows as `core::fmt::write` in `--list` and the headers, and becomes the identifier `core__fmt__write`. `-f` accepts either form.

Functions are lifted, cleaned and emitted in parallel with `rayon`, one `Lifter` and `Function` per worker thread. Identifiers are assigned before the parallel part and the results are joined in address order, so the output is byte-for-byte the same at any `-j`.

`--emit` picks what is printed:

| `--emit` | Output |
|---|---|
| `rust` (default) | Rust source, one `pub fn` per function |
| `ir` | cleaned SSA IR, which is what the emitter sees |
| `raw-ir` | IR straight from the lifter, before `opt::clean` |
| `borrows` | safe mode's verdict for each argument (`&T`, `Option<&mut T>`, raw...) |

A summary goes to stderr: how many functions lifted, how many memory accesses are bounds-checked versus raw, and the failures grouped by cause, most common first. That table is the to-do list for the lifter. The exit code is 0 if every selected function lifted, 1 if some did not, and 2 for usage or file errors.

A function that fails to lift still appears in the output as a stub whose body is `todo!("not lifted: <reason>")`, so the file always compiles. `--skip-failed` leaves the stubs out.

## What the Rust looks like

The emitter is [`src/emit.rs`](../src/emit.rs). Every IR value is a Rust integer (`u8`..`u64`, or `bool` for comparisons), and pointers are `u64` addresses. That way the output type-checks however the binary mixes pointers and integers. Signatures are recovered for the whole program at once ([calls.md](calls.md)): arguments are named after the registers they arrive in, in System V order (`rdi, rsi, rdx, rcx, r8, r9`, then `arg6`, `arg7`, ... from the stack), and a function returns `u64`, `(u64, u64)` (rax:rdx) or nothing. Calls to other decompiled functions use their names, and everything else they call is declared in a `mod ffi` at the top of the file. Stack slots become `let` bindings; a function that takes the address of one keeps a `frame` array.

**Fast mode** is an `unsafe fn` with every load and store as an unaligned raw access:

```rust
pub unsafe fn get_count(mut rdi: u64) -> u64 {
    return unsafe { (rdi.wrapping_add(8) as *const u64).read_unaligned() };
}
```

**Safe mode** asks `borrow::analyze` about each argument. An argument it classifies as `&T` or `&mut T` arrives as a byte slice (`&[u8]` or `&mut [u8]`), wrapped in `Option` when the code null-checks it. Any access whose pointer derives from exactly one such argument becomes a bounds-checked slice access. So a wrong size guess panics instead of reading out of bounds. Every other access stays raw, and the function is `unsafe` only if one does:

```rust
pub fn maybe_count(rdi_ref: Option<&[u8]>) -> u64 {
    let rdi_base: u64 = rdi_ref.as_deref().map_or(0, |s| s.as_ptr() as u64);
    let mut rdi: u64 = rdi_base;
    ...
    return u64::from_le_bytes(rdi_ref.unwrap()[rdi.wrapping_add(8).wrapping_sub(rdi_base) as usize..][..8].try_into().unwrap());
```

Each function header says how many of its accesses are checked. Byte slices are a stopgap until struct recovery can name the pointee type and turn these into `&S` with real fields.

Control flow is structured ([`src/structure.rs`](../src/structure.rs)): branches become `if`/`else`, loops become `while` or `loop` with `break`, `continue` and early `return`, and block parameters become mutable variables assigned on each edge. A loop's exits are emitted after it, so leaving it is a `break`. A value used once, in the block that computes it, is written into its use instead of getting a `let` (a load only when nothing between the two writes memory or calls), and constants are literals, so a loop whose test comes first reads as a `while`:

```rust
pub unsafe fn find(mut rdi: u64, mut rsi: u64, mut rdx: u64) -> u64 {
    let mut v6: u64 = 0;
    let mut v36: u64 = 0;
    if rsi == 0 {
        v36 = rsi;
    } else {
        v6 = 0_u64;
        while (unsafe { (rdi.wrapping_add(v6.wrapping_mul(8)) as *const u64).read_unaligned() }) != rdx {
            let v13: u64 = v6.wrapping_add(1); // 0x17
            if rsi == v13 {
                return rsi;
            }
            v6 = v13;
        }
        v36 = v6;
    }
    return v36;
}
```

A block that only tests a condition joins its predecessor's test, so `a || b` and `a && b` stay one `if`. Where the nesting still needs it (two paths with code of their own into the same block), a labeled block (`'b7: { .. break 'b7; .. }`) stands in. An irreducible cycle (one with two entries) has no nesting: just its blocks become a `loop { match bb { ... } }`, and edges into it set `bb`; the rest of the function is structured around it. The summary on stderr counts the functions that have one.

Every statement ends with the address of the instruction it came from.

## Globals

A RIP-relative or absolute memory operand is a constant address. If it falls in a data section (`.data`, `.rodata`, `.bss`, `.got`, `__const`, ...), the emitted code takes the address of a `static` instead, so the output reads the binary's data rather than whatever lives at that address in its own process:

```rust
let v2: u64 = (core::ptr::addr_of!(TABLE) as u64).wrapping_add(0x8); // 0x1139
...
// .rodata 0x3f60, 16 bytes (TABLE)
pub static TABLE: Bytes<16> = Bytes { b: *b"\n\0\0\0\x14\0\0\0\x1e\0\0\0(\0\0\0" };
```

Each static is the data symbol covering the address, sized by the symbol table. Without one (string literals, anonymous constants) it is the whole gap between the neighbouring symbols, so indexing from the referenced address stays inside the static. Writable sections become `static mut`, `.bss` is `[0; N]`, and a comment gives the section, the address and, for text, the string.

Slots the dynamic loader fills with an address (GOT entries, vtables, pointer tables, from the binary's dynamic relocations) are emitted as pointers to the static or function they point to, and those statics are emitted too. Each GOT slot is its own 8-byte static, so only the slots the code uses appear. A slot pointing at an import or at a function not in the output is null. Only data the selected functions reach is emitted; `tests/globals.rs` builds a cdylib, decompiles it and runs the result against the original's statics.

## Tests

`tests/emit.rs` compiles what the emitter prints with `rustc`:

- the `sum` loop from `tests/common`, in both modes, linked into a program that runs it and checks the results, including that safe mode panics on a too-short slice;
- 400 random programs built from supported instructions, in both modes, which must all type-check;
- 300 random register-only programs emitted both structured and as the whole-function state machine, run on the same random inputs, which must return the same values (programs that don't terminate run out of fuel and are skipped); the irreducible ones exercise the per-region `match bb`;
- the real binary on a small ELF built in the test: `--list`, both modes, `-f`, `--emit ir` and the exit codes.

`tests/globals.rs` decompiles a cdylib that reads a `static`, writes a `static mut` (both through the GOT) and has a mangled function, then links the output into a program that checks the values. It also checks that `-j 1` and `-j 4` print the same thing.
