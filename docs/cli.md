# Using the `chungusite` command

```
cargo install --path .            # or: cargo run --release -- <args>

chungusite ./prog                          # every function, fast mode, to stdout
chungusite ./prog --mode safe -o prog.rs   # safe mode, to a file
chungusite ./prog -f parse -f main         # only these symbols
chungusite ./prog --addr 0x401136          # the function starting there
chungusite ./prog --addr 0x401136 --size 0x40   # no symbol there: lift exactly these bytes
chungusite ./prog --list                   # which functions lift, and why the others don't
chungusite --hex "48 8b 47 08 c3"          # raw bytes, loaded at 0x1000, no file needed
```

Input is any x86_64 ELF, Mach-O or PE file (`object` crate). Functions come from the symbol table, falling back to the dynamic symbols, so a stripped binary needs `--addr` with `--size`.

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

The emitter is [`src/emit.rs`](../src/emit.rs). Every IR value is a Rust integer (`u8`..`u64`, or `bool` for comparisons), and pointers are `u64` addresses. That way the output type-checks however the binary mixes pointers and integers. Arguments are named after the registers they arrive in, in System V order (`rdi, rsi, rdx, rcx, r8, r9`), followed by any other register the function reads on entry. Every function returns `rax` as a `u64`.

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

Control flow is structured ([`src/structure.rs`](../src/structure.rs)): branches become `if`/`else`, loops become `loop` with `break`, `continue` and early `return`, and block parameters become mutable variables assigned on each edge. A loop's exits are emitted after it, so leaving it is a `break`:

```rust
pub unsafe fn find(mut rdi: u64, mut rsi: u64, mut rdx: u64) -> u64 {
    let mut v2: u64 = 0;
    ...
    v2 = v1;
    loop {
        let v4: bool = v2 >= rsi; // 0x5
        if v4 {
            break;
        }
        let v7: u64 = rdi.wrapping_add(v2.wrapping_mul(8)); // 0x7
        let v8: u64 = unsafe { (v7 as *const u64).read_unaligned() }; // 0x7
        let v10: bool = v8 == rdx; // 0xe
        if v10 {
            break;
        }
        ...
        v2 = v13;
    }
    return v2 as u64;
}
```

Where the nesting needs it (two paths into the same `else`, as in `if a || b`), a labeled block (`'b7: { .. break 'b7; .. }`) stands in. A function whose CFG is irreducible (a cycle with two entries) has no such nesting and stays a `loop { match bb { ... } }` state machine; the summary on stderr counts those.

Every statement ends with the address of the instruction it came from.

## Tests

`tests/emit.rs` compiles what the emitter prints with `rustc`:

- the `sum` loop from `tests/common`, in both modes, linked into a program that runs it and checks the results, including that safe mode panics on a too-short slice;
- 400 random programs built from supported instructions, in both modes, which must all type-check;
- 300 random register-only programs emitted both structured and as the state machine, run on the same random inputs, which must return the same values (programs that don't terminate run out of fuel and are skipped);
- the real binary on a small ELF built in the test: `--list`, both modes, `-f`, `--emit ir` and the exit codes.
