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
chungusite ./prog --no-debug-info          # ignore DWARF: types only from the code
```

Input is any x86_64 ELF, Mach-O or PE file (`object` crate). Functions come from the symbol table. A stripped binary (no function in the static symbol table) works the same way: functions are discovered instead (`src/discover.rs`), see [Stripped binaries](#stripped-binaries). `--addr` with `--size` lifts bytes that nothing finds.

Rust (legacy and v0) and Itanium C++ symbols are demangled, without the Rust hash or the C++ parameter list: `_ZN4core3fmt5write17h…E` shows as `core::fmt::write` in `--list` and the headers, and becomes the identifier `core__fmt__write`. `-f` accepts either form.

Functions are lifted, cleaned and emitted in parallel with `rayon`, one `Lifter` and `Function` per worker thread. Identifiers are assigned before the parallel part and the results are joined in address order, so the output is byte-for-byte the same at any `-j`.

`--emit` picks what is printed:

| `--emit` | Output |
|---|---|
| `rust` (default) | Rust source, one `pub fn` per function |
| `ir` | cleaned SSA IR, which is what the emitter sees |
| `raw-ir` | IR straight from the lifter, before `opt::clean` |
| `borrows` | safe mode's verdict for each argument (`&T`, `Option<&mut T>`, raw...) |
| `types` | each argument's and the return value's recovered type, and which debug-info proposals were rejected and why ([types.md](types.md)) |

A summary goes to stderr: how many functions lifted, how many memory accesses are bounds-checked versus raw, and the failures grouped by cause, most common first. That table is the to-do list for the lifter. The exit code is 0 if every selected function lifted, 1 if some did not, and 2 for usage or file errors.

A function that fails to lift still appears in the output as a stub whose body is `todo!("not lifted: <reason>")`, so the file always compiles. `--skip-failed` leaves the stubs out.

## What the Rust looks like

The emitter is [`src/emit.rs`](../src/emit.rs). Every IR value is a Rust integer (`u8`..`i64`, or `bool` for comparisons), signed where the code treats it as signed, and pointers inside a function are `u64` addresses. That way the output type-checks however the binary mixes pointers and integers. Signatures are recovered for the whole program at once ([calls.md](calls.md)): arguments are named after the registers they arrive in, in System V order (`rdi, rsi, rdx, rcx, r8, r9`, then `arg6`, `arg7`, ... from the stack), and a function returns `u64`, `(u64, u64)` (rax:rdx) or nothing. Type recovery ([types.md](types.md)) narrows that: an argument the code only uses the low 32 bits of is an `i32` or `u32` named after its register part (`edi`), a return value that is always zero-extended from 32 bits is an `i32`, and a pointer argument is a pointer to the struct, scalar or array its accesses describe. With DWARF, arguments and structs get their source names and types (`fn area(r: &rect) -> i32`). Calls to other decompiled functions use their names, and everything else they call is declared in a `mod ffi` at the top of the file. Stack slots become `let` bindings; a function that takes the address of one keeps a `frame` array.

**Fast mode** is an `unsafe fn` with every load and store as a raw access. A struct pointer argument is a `*mut S`, and the accesses it describes read fields; any other access is an unaligned read or write at the address. For `mov rax, [rdi+8]; ret`:

```rust
#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct S_get_count_rdi {
    _pad0: [u8; 8],
    pub f8: u64,
}

pub unsafe fn get_count(rdi_ref: *mut S_get_count_rdi) -> u64 {
    let v2: u64 = unsafe { (*rdi_ref).f8 }; // 0x1129
    return v2 as u64;
}
```

Struct definitions come first in the file, one per distinct layout. They are packed, with explicit padding, so their layout is the C one but any address can be borrowed as one.

**Safe mode** asks `borrow::analyze` about each argument. An argument it classifies as `&T` or `&mut T` becomes a reference to its recovered type (`&S`, `&T`, `&[T]`), wrapped in `Option` when the code null-checks it, if that type describes every access through it; otherwise it arrives as a byte slice (`&[u8]` or `&mut [u8]`). Accesses through it become field reads and writes, or bounds-checked indexing, so a wrong size guess panics instead of reading out of bounds. Every other access stays raw, and the function is `unsafe` only if one does:

```rust
pub fn maybe_count(mut rdi_ref: Option<&mut S_maybe_count_rdi>, mut rsi: u64) -> u64 {
    let rdi_base: u64 = rdi_ref.as_deref().map_or(0, |s| s as *const S_maybe_count_rdi as u64);
    let mut rdi: u64 = rdi_base;
    ...
    rdi_ref.as_deref_mut().unwrap().f8 = rsi; // 0x1005
    let v8: u64 = rdi_ref.as_deref().unwrap().f16; // 0x1009
    return v8 as u64;
}
```

Each function header says how many of its accesses are checked.

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

## Stripped binaries

Without a symbol table, `src/discover.rs` finds functions from, most trusted first:

1. unwind tables: the FDEs in `.eh_frame` (ELF, Mach-O) or `.pdata` (PE), which give the exact start and length of every function compiled with unwind info, the default on x86_64;
2. the dynamic symbols (exports), which `strip` keeps and which keep their names;
3. the entry point and the start of each code section;
4. targets of direct calls and jumps out of a function, `lea reg, [rip+x]` into code, `mov reg, imm` into code in a non-PIE binary, and code addresses the loader writes into data (vtables, function tables);
5. code that none of the above covers, if it starts with a prologue (`endbr64`, `push rbp`/`rbx`/`r12`-`r15`, `sub rsp, ..`) or at a 16-byte boundary, after any alignment padding.

This repeats until nothing new turns up. A function without unwind info ends at the last instruction its control flow reaches before the next known start. A candidate strictly inside an FDE's range is rejected. Discovered functions are named `sub_<addr>`, except `_start` (the entry point), `main` (what `_start` passes to `__libc_start_main` in `rdi`), and `_init`/`_fini` (the `.init`/`.fini` sections). Calls into imports are named as before, from the PLT stubs and GOT relocations, which `strip` keeps. The summary on stderr says how many functions were discovered.

On chungusite's own debug build, `strip` keeps none of its 20,025 function symbols, and discovery finds all 20,025 starts. 20,021 sizes match the symbol table exactly; the other 4 are crtstuff's hand-written functions, whose symbol sizes include their trailing padding. Lifting and type-checking give the same results with and without symbols. Known gap: without unwind tables, discovery doesn't know which calls don't return, so a function ending in a call to `__stack_chk_fail` or `abort` can run into the next one (gcc -O2 with `-fno-asynchronous-unwind-tables` merges 1 of the corpus's 88 functions this way).

## Tests

`tests/emit.rs` compiles what the emitter prints with `rustc`:

- the `sum` loop from `tests/common`, in both modes, linked into a program that runs it and checks the results, including that safe mode panics on a too-short slice;
- 400 random programs built from supported instructions, in both modes, which must all type-check;
- 300 random register-only programs emitted both structured and as the state machine, run on the same random inputs, which must return the same values (programs that don't terminate run out of fuel and are skipped);
- the real binary on a small ELF built in the test: `--list`, both modes, `-f`, `--emit ir` and the exit codes.

`tests/discover.rs` links `tests/differential/corpus.c` with each C compiler found, with and without unwind tables and as a non-PIE, strips a copy, and checks that `--list` on the stripped copy finds the same function starts as the symbol table (and the same sizes, except the crt functions), names `main`, and that the stripped copy's output type-checks in both modes.

`tests/types.rs` checks type recovery ([types.md](types.md)): narrow and signed arguments and returns, structs and slices inferred from accesses in both modes, callers adapting to a callee's narrow types, a stand-in `TypeModel` whose proposals are accepted or rejected, DWARF from C compiled with `-g`, and 120 random typed programs that must type-check in both modes. `tests/differential.rs` runs its corpus at `-O2 -g` too, so types from debug info are checked against the original's behaviour.

`tests/globals.rs` decompiles a cdylib that reads a `static`, writes a `static mut` (both through the GOT) and has a mangled function, then links the output into a program that checks the values. It also checks that `-j 1` and `-j 4` print the same thing.
