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
    let v1: u64 = rdi.wrapping_add(0x8); // 0x1129
    let v2: u64 = unsafe { (v1 as *const u64).read_unaligned() }; // 0x1129
    return v2 as u64;
}
```

**Safe mode** asks `borrow::analyze` about each argument. An argument it classifies as `&T` or `&mut T` arrives as a byte slice (`&[u8]` or `&mut [u8]`), wrapped in `Option` when the code null-checks it. Any access whose pointer derives from exactly one such argument becomes a bounds-checked slice access. So a wrong size guess panics instead of reading out of bounds. Every other access stays raw, and the function is `unsafe` only if one does:

```rust
pub fn maybe_count(rdi_ref: Option<&[u8]>) -> u64 {
    let rdi_base: u64 = rdi_ref.as_deref().map_or(0, |s| s.as_ptr() as u64);
    let mut rdi: u64 = rdi_base;
    ...
    let v7: u64 = u64::from_le_bytes(rdi_ref.unwrap()[v6.wrapping_sub(rdi_base) as usize..][..8].try_into().unwrap());
```

Each function header says how many of its accesses are checked. Byte slices are a stopgap until struct recovery can name the pointee type and turn these into `&S` with real fields.

Control flow is structured ([`src/structure.rs`](../src/structure.rs)): branches become `if`/`else`, loops become `while` or `loop` with `break`, `continue` and early `return`, and block parameters become mutable variables assigned on each edge. A value used once, in the block that computes it, is written inside the expression that uses it rather than as a `let` of its own, unless that would move a load past a store or call, or a call past any memory access. A block that only tests a condition joins the test before it in `&&` or `||`, and constants (including casts of constants) are written as literals. This is `count_outside` from a small C file, compiled by gcc at `-O0`:

```c
long count_outside(const long *a, long n, long lo, long hi) {
    long c = 0;
    for (long i = 0; i < n && a[i] != 0; i++)
        if (a[i] < lo || a[i] > hi)
            c++;
    return c;
}
```

```rust
pub unsafe fn count_outside(mut rdi: u64, mut rsi: u64, mut rdx: u64, mut rcx: u64) -> u64 {
    let mut v174: u64 = 0;
    let mut v161: u64 = 0;
    (v174, v161) = (0_u64, 0_u64);
    while (v161 as i64) < (rsi as i64) && (unsafe { (rdi.wrapping_add(v161.wrapping_mul(8)) as *const u64).read_unaligned() }) != 0_u64 {
        if (rdx as i64) > ((unsafe { (rdi.wrapping_add(v161.wrapping_mul(8)) as *const u64).read_unaligned() }) as i64) || (rcx as i64) < ((unsafe { (rdi.wrapping_add(v161.wrapping_mul(8)) as *const u64).read_unaligned() }) as i64) {
            (v174, v161) = (v174.wrapping_add(1_u64), v161.wrapping_add(1_u64));
        } else {
            v161 = v161.wrapping_add(1_u64);
        }
    }
    return v174;
}
```

Where the nesting still needs it (a loop with several exits that lead to different code, or two paths into the same code from different depths), a labeled block (`'b7: { .. break 'b7; .. }`) stands in. A function whose CFG is irreducible (a cycle with two entries) gets a `bb` variable for that cycle only: each edge into the cycle sets `bb` to the block it enters, and a `loop` at the top of the cycle tests it (`if bb == 2 { .. } else { .. }`). Everything outside the cycle, and inside it, stays structured. The summary on stderr counts those functions.

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
- 300 random register-only programs emitted both structured and as the whole-function state machine (`structure: false`), run on the same random inputs, which must return the same values (programs that don't terminate run out of fuel and are skipped);
- the real binary on a small ELF built in the test: `--list`, both modes, `-f`, `--emit ir` and the exit codes.

`tests/globals.rs` decompiles a cdylib that reads a `static`, writes a `static mut` (both through the GOT) and has a mangled function, then links the output into a program that checks the values. It also checks that `-j 1` and `-j 4` print the same thing.
