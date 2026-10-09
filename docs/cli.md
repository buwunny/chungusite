# Using the `chungusite` command

```
cargo install --path .            # or: cargo run --release -- <args>

chungusite ./prog                          # every function, fast mode, to stdout
chungusite ./prog --mode safe -o prog.rs   # safe mode, to a file
chungusite ./prog --mode safe --check      # ... and compile it with rustc, fast mode for what fails
chungusite ./prog -f parse -f main         # only these symbols (mangled or demangled)
chungusite ./prog --addr 0x401136          # the function starting there
chungusite ./prog --addr 0x401136 --size 0x40   # no symbol there: lift exactly these bytes
chungusite ./prog --list                   # which functions lift, and why the others don't
chungusite --hex "48 8b 47 08 c3"          # raw bytes, loaded at 0x1000, no file needed
chungusite ./prog --cargo prog-rs          # a Cargo project: cd prog-rs && cargo run -- ARGS
chungusite ./prog -j 1                     # one worker thread (default: one per CPU)
chungusite ./prog --no-dwarf               # ignore debug info: infer every type
chungusite ./prog --no-asm                 # fail a function with an instruction the lifter can't model, instead of keeping it as asm!
chungusite ./prog --refine models/types     # ask a trained type model (--features ml)
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
| `borrows` | safe mode's verdict for each argument (`&T`, `Option<&mut T>`, raw and why; "may keep the pointers in it" when the function may hold on to a pointer loaded from it) and for the frame (each of its objects: `frame from N`), globals and allocations |
| `dataset` | training data for the type and name models: one JSON line per argument and return value of each function with a DWARF prototype (`file`, `func`, `addr`, `var`, `value`, `label`, `name`, and `text`, the row `--refine` would show the model). Needs debug info, so not with `--no-dwarf`. [tools/train](../tools/train/README.md) builds a corpus from it |

A summary goes to stderr: how many functions lifted, how many memory accesses are safe (a bounds-checked slice access or a struct field) versus raw (and what the raw ones go through: the frame, a global, an argument, or another pointer), how many functions have no raw pointer, how many arguments got a type, and the failures grouped by cause, most common first. That table is the to-do list for the lifter. The exit code is 0 if every selected function lifted, 1 if some did not, and 2 for usage or file errors.

A function that fails to lift still appears in the output as a stub whose body is `todo!("not lifted: <reason>")`, so the file always compiles. `--skip-failed` leaves the stubs out.

## What the Rust looks like

The emitter is [`src/emit.rs`](../src/emit.rs). Every IR value is a Rust integer with its recovered width and signedness (`u8`..`u64`, `i8`..`i64`, or `bool` for comparisons), and pointer values inside a body are `u64` addresses. That way the output type-checks however the binary mixes pointers and integers. Signatures are recovered for the whole program at once ([calls.md](calls.md)), and their types by [`src/types.rs`](../src/types.rs) ([types.md](types.md)): from the debug info when the binary has it (with the source's parameter names and structs), otherwise from how each value is used. Arguments without a debug name are named after the registers they arrive in, in System V order (`rdi, rsi, rdx, rcx, r8, r9`, then `arg6`, `arg7`, ... from the stack); a pointer argument is `rdi_p: *mut S1`. A function returns its recovered type, `(u64, u64)` (rax:rdx) or nothing. Calls to other decompiled functions use their names, and everything else they call is declared in a `mod ffi` at the top of the file. Stack slots become `let` bindings; a function that takes the address of one keeps a `frame` array.

**Fast mode** is an `unsafe fn`. An access to a field of a recovered struct reads the field; every other load and store is an unaligned raw access:

```rust
/// 16 bytes, inferred from field accesses.
#[repr(C, packed)]
pub struct S1 {
    pub _pad0: [u8; 8],
    pub f8: u64,
}

pub unsafe fn get_count(rdi_p: *const S1) -> u64 {
    let mut rdi: u64 = rdi_p as u64;
    return unsafe { (*rdi_p).f8 };
}
```

With debug info (`gcc -g`) the struct, its field names and the parameter names come from the source: `pub unsafe fn bump(p: *mut Node, d: i32) -> i64` reads `(*p).count`.

**Safe mode** asks the borrow analysis ([ownership.md](ownership.md)) which objects each pointer points into, across the whole program. An argument it classifies as `&T` or `&mut T` arrives as `&S` or `&mut S` if every access through it is a field of its recovered struct and no decompiled code calls the function; its accesses become field reads and writes (`p.count = v9;`). Any other such argument arrives as a byte slice (`&[u8]` or `&mut [u8]`), wrapped in `Option` when the code null-checks it; the stack frame becomes a byte array, a read-only global its static's bytes, and a `malloc`'d buffer a `Box<[u8]>`. Any access whose pointer derives from exactly one such object becomes a bounds-checked slice access. So a wrong size guess panics instead of reading out of bounds. Calls lend slices to callees that take them; a caller that can't calls the callee's *raw twin*, `name_raw`, the same function in fast mode. Every other access stays raw, and the function is `unsafe` only if one does:

```rust
pub fn maybe_count(rdi_ref: Option<&[u8]>) -> u64 {
    let rdi_base: u64 = rdi_ref.as_deref().map_or(0, |s| s.as_ptr() as u64);
    let mut rdi: u64 = rdi_base;
    ...
    return u64::from_le_bytes(rdi_ref.unwrap()[rdi.wrapping_add(8).wrapping_sub(rdi_base) as usize..][..8].try_into().unwrap());
```

Each function header says how many of its accesses are safe. Byte slices remain for arguments that are indexed, whose accesses don't all match a struct field, or that callers lend; typed slices (`&[T]`) are next.

`--check` compiles the safe-mode output with rustc (in parallel batches; about two minutes for 18,000 functions on 4 cores) and emits every function rustc rejects in fast mode instead, which always compiles, then checks again.

Control flow is structured ([`src/structure.rs`](../src/structure.rs)): branches become `if`/`else`, loops become `while` or `loop` with `break`, `continue` and early `return`, and block parameters become mutable variables assigned on each edge. An arm that does nothing but assign them (with values that need no call or memory access) runs before the `if` instead, so the other arm needs no `else`: `v = b; if a != 0 { *p = a; v = c; }`, as the source would have it. A loop's exits are emitted after it, so leaving it is a `break`. A value used once, in the block that computes it, is written into its use instead of getting a `let` (a load only when nothing between the two writes memory or calls), and constants are literals, so a loop whose test comes first reads as a `while`:

```rust
pub unsafe fn find(rdi_p: *const u64, mut rsi: u64, mut rdx: u64) -> u64 {
    let mut rdi: u64 = rdi_p as u64;
    let mut v6: u64 = 0;
    if rsi == 0 {
        return rsi;
    }
    v6 = 0_u64;
    while (unsafe { (rdi.wrapping_add(v6.wrapping_mul(8)) as *const u64).read_unaligned() }) != rdx {
        let v13: u64 = v6.wrapping_add(1); // 0x1110
        if rsi == v13 {
            return rsi;
        }
        v6 = v13;
    }
    return v6;
}
```

A value used once is written into its use when that doesn't move it past anything it could observe: pure arithmetic, a load with no store or call in between, and a call's result when only pure values come between it and the use and the use runs on every path the call did (`return f(x);`, `if f(x) != 0`, not one arm's assignment or an `if`-expression). Variables are declared up front (`let mut v6`) only when they need to be: one assigned by a single statement and used only after it in the same scope is declared there instead (`let v13`). And compilers merge the returns of a function into one epilogue, so a small block that returns (at most 16 instructions, no stores) is copied back into each path that reaches it, so each path returns on its own (`if rsi == 0 { return rsi; }`) instead of assigning a variable for a shared `return`, and needs no labeled block to get there.

A block that only tests a condition joins its predecessor's test, so `a || b` and `a && b` stay one `if`. Two paths with code of their own into the same block are nested as `if`/`else` where that only needs a little code copied: `if a && b { X } else { Y }` rather than leaving early to skip `Y`, and the code after a loop copied into the loop's exits when one leaves past it. Where the nesting still needs it (and it isn't a small return), a labeled block (`'b7: { .. break 'b7; .. }`) stands in. An irreducible cycle (one with two entries) has no nesting, so before structuring it gets a header of its own (`src/dispatch.rs`): a block that takes the number of the entry to go to and branches on it, so that only edges into an entry go through that test and the rest of the cycle nests like any loop. Where that isn't possible (the cycle contains the function's entry, or a value would no longer be defined on every path to its use), just its blocks become a `loop { match bb { ... } }`, and edges into it set `bb`; the rest of the function is structured around it. The summary on stderr counts the functions that have one.

Every statement ends with the address of the instruction it came from.

## Globals

A RIP-relative or absolute memory operand is a constant address. If it falls in a data section (`.data`, `.rodata`, `.bss`, `.got`, `__const`, ...), the emitted code takes the address of a `static` instead, so the output reads the binary's data rather than whatever lives at that address in its own process:

```rust
let v2: u64 = (core::ptr::addr_of!(TABLE) as u64).wrapping_add(0x8); // 0x1139
...
// .rodata 0x3f60, 16 bytes (TABLE)
pub static TABLE: Bytes<16> = Bytes { b: *b"\n\0\0\0\x14\0\0\0\x1e\0\0\0(\0\0\0" };
```

Each static is the data symbol covering the address, sized by the symbol table. Without one (string literals, anonymous constants) it is the whole gap between the neighbouring symbols, so indexing from the referenced address stays inside the static. Writable sections become `static mut`, `.bss` is `[0; N]`, and a comment gives the section, the address and, for text, the string. An address just past the end of a static, compared with a pointer into it (a loop over an array stops at the address of whatever follows it), is that static's address plus its length, since the output's statics aren't laid out next to each other. Likewise an address up to 16 bytes before the start (a loop walking an array down stops one element early, or `table[i - 1]` folded into the base) is the static's address minus the gap, also where that address falls in the tail of the previous item. Pointer tables the loader relocates (`R_X86_64_RELATIVE`) are read with their addends applied. The C library's data that a program reads through a copy relocation (`stdin`, `stderr` and `optarg` in a PIE) is the library's own: `extern "C" { #[link_name = "stderr"] pub static mut stderr_GLIBC_2_2_5: [u8; 8]; }`. The address of a function of the output (a callback stored in a struct) is that function: `(default_bzfree as u64)`.

Slots the dynamic loader fills with an address (GOT entries, vtables, pointer tables, from the binary's dynamic relocations) are emitted as pointers to the static or function they point to, and those statics are emitted too. Each GOT slot is its own 8-byte static, so only the slots the code uses appear. A slot holding another library's symbol (`stdout`, `optarg`, `free` stored as a callback) points to that symbol, declared `extern` in the initializer, so the output's own loader fills it in; a weak import (`__gmon_start__`) stays null, and one pointing at a function of the binary that isn't in the output (not selected, or left out by `--skip-failed`) keeps its bytes from the file, so the output compiles either way. Thread-locals (`fs:[-k]`, see [lift.md](lift.md#bit-instructions-atomics-and-the-rest)) are one `static mut THREAD_LOCALS` holding the whole thread-local block, `.tdata`'s bytes and then zeros, since code reaches them all from the thread pointer at its end; the decompiled program's threads share it. Only data the selected functions reach is emitted; `tests/globals.rs` builds a cdylib, decompiles it and runs the result against the original's statics.

## Stripped binaries

Without a symbol table, `src/discover.rs` finds functions from, most trusted first:

1. unwind tables: the FDEs in `.eh_frame` (ELF, Mach-O) or `.pdata` (PE), which give the exact start and length of every function compiled with unwind info, the default on x86_64;
2. the dynamic symbols (exports), which `strip` keeps and which keep their names;
3. Mach-O's `LC_FUNCTION_STARTS`, which `strip` keeps too: the start of every function (as ULEB128 offsets from `__TEXT`), without sizes;
4. the entry point and the start of each code section;
5. targets of direct calls and jumps out of a function, `lea reg, [rip+x]` into code, `mov reg, imm` into code in a non-PIE binary, and code addresses the loader writes into data (vtables, function tables);
6. code that none of the above covers, after any alignment padding: since a function ends where its control flow ends, nothing reaches that code, so it is a function nothing calls directly. Compilers don't always align functions or give them a prologue (clang -Os packs leaf functions back to back).

This repeats until nothing new turns up. A function without unwind info ends at the last instruction its control flow reaches before the next known start. A candidate strictly inside an FDE's range is rejected. Discovered functions are named `sub_<addr>`, except `_start` (the entry point), `main` (what `_start` passes to `__libc_start_main` in `rdi`), and `_init`/`_fini` (the `.init`/`.fini` sections). Calls into imports are named as before, from the PLT stubs and GOT relocations, which `strip` keeps. The summary on stderr says how many functions were discovered.

On chungusite's own debug build, `strip` keeps none of its 20,025 function symbols, and discovery finds all 20,025 starts. 20,021 sizes match the symbol table exactly; the other 4 are crtstuff's hand-written functions, whose symbol sizes include their trailing padding. Lifting and type-checking give the same results with and without symbols. Without unwind tables, a function's end is where its control flow ends: discovery follows jump tables and the label tables of computed `goto`s (whose labels, found first as pointers, are dropped as functions once the code at one jumps back into the function that uses the table), treats a `jmp` to the next instruction or over nothing but padding as a tail call into the next function (inside a function the code would fall through), and stops at calls that don't return, so such a function doesn't run into the next one. A call doesn't return if it goes to an import that doesn't (`__stack_chk_fail`, `abort`, `exit`, `__cxa_throw`, `_Unwind_Resume`, ...), to one of Rust's `-> !` functions (all of `core::panicking`, `unwrap_failed`, `handle_alloc_error`, `alloc::raw_vec::handle_error`, the slice and `str` index failures, `std::process::exit`, ...), or to a function found not to return because every path through it ends in such a call, `ud2`, `hlt` or `int3`. This goes to a fixpoint, so a `die()` that calls `exit` stops its callers, and their callers if they only call it. The call can be direct, through the PLT, or through a GOT slot, including one the loader fills with a function in the binary itself (`R_X86_64_RELATIVE`), which is how a Rust PIE calls the standard library. The lifter uses the same set for every linked binary, stripped or not (in an object file, only the names), and ends the block at such a call.

## Hand-written, obfuscated and packed code

A global symbol in code without a type (`global f` in nasm, `.globl f` without `.type` in gas) is a function too, and an assembler's local labels between functions (`.loop:`, `table:`) don't split them: a function without a size runs to the next function or global symbol. A label in code that the code reads data from (a table after the `ret`) becomes a `static`, like data in a data section. Functions whose bytes don't decode from start to end (junk after a `jmp`, overlapping instructions) are lifted along their control flow instead ([lift.md](lift.md#code-that-hides-from-a-linear-sweep)).

An instruction the lifter has no model of (`rdtsc`, `crc32`, SSSE3's `pshufb`, AES-NI, `rcl`, a `div` with a real 128-bit dividend) no longer fails its function: it is kept as it is, in an `asm!` that runs it on the values of the registers it uses ([lift.md](lift.md#instructions-kept-as-inline-assembly)). The output is then x86_64-only, as the original was:

```rust
let v8_asm: [u64; 1] = unsafe { let mut asm1 = v7; core::arch::asm!("crc32 rax, qword ptr [{0}]", in(reg) v5, inout("rax") asm1, options(nostack)); [asm1] };
v9 = v8_asm[0];
```

The summary counts them, and `--no-asm` turns this off.

A packed binary carries its real code compressed or encrypted and unpacks it at run time, so all there is to decompile is the unpacking stub. chungusite warns when a binary looks packed: UPX's sections or header, or executable segments whose bytes look random (more than 7.4 bits of entropy a byte; machine code has about 6). It doesn't unpack anything itself: run `upx -d` first, or decompile a memory dump of the running process.

`tests/hostile.rs` builds a program from hand-written assembly with a junk byte after a `jmp`, a jump into its own instruction, a table after a `ret` and an always-false branch to garbage, all with untyped symbols and local labels, and checks that the decompiled Cargo project prints what the original does in both modes; the same for a program whose hand-written functions use instructions only inline assembly can run (`crc32` on the heap and the stack, `pshufb`, `rcl`/`rcr` reading and setting the carry, `rdtsc`, rbx and bh as operands); and that a binary with 64 KiB of random bytes in its code gets the packed warning while an ordinary one doesn't.

## Cargo projects

`--cargo DIR` writes the output as a Cargo project instead of one file ([`src/project.rs`](../src/project.rs)), in either mode and with `--check`:

```
DIR/Cargo.toml                 package named after the binary, edition 2021, no dependencies
DIR/src/main.rs                fn main(): argc, argv and envp from the process, then the decompiled main
DIR/src/decompiled/mod.rs      lint allows, `simd`, and `pub use` of every module below
DIR/src/decompiled/types.rs    recovered structs
DIR/src/decompiled/ffi.rs      the externs, and todo!() stand-ins for functions of the binary that aren't in the output
DIR/src/decompiled/data.rs     the statics
DIR/src/decompiled/crt.rs      _start, _init, frame_dummy and the rest of the C runtime's code
DIR/src/decompiled/*.rs        the functions
```

Functions are grouped by, in order: the C runtime's names (`crt`); the namespace of a Rust or C++ symbol, its first two path segments (`core::fmt::write` and `<core::fmt::Arguments as Display>::fmt` in `core_fmt`); the compilation unit in the debug info (`src/list.c` in `list_c`); and for the rest, address order, 64 functions to a module (`code`, or `code_1`, `code_2`, ...), since linkers keep each object file's functions together. Every module starts with `use super::*;`, so the code is the same as in one file. Identifiers are unique across the whole output, so the globs never clash, and a module name never matches a struct's or `core`/`std`/`alloc`.

The generated `main` calls the decompiled `main` (`_main` on Mach-O) with as many of argc, argv and envp as its signature takes, and exits with what it returns. In safe mode it calls `main`'s raw twin, which takes integers: `--cargo` counts `main` as address-taken, as it is in the binary (`_start` passes it to `__libc_start_main`). Without a decompiled `main` (`-f` without it, `--hex`, a library) the crate is a library, `src/lib.rs`.

Linking needs two things the single file leaves to whoever links it. A call to a function of the binary that isn't in the output (not selected, or not lifted) is an extern in the single file; in the project it is a `todo!()` in `ffi.rs` with the same signature, so it links and panics if called. And every shared library the binary needs (`DT_NEEDED`) other than the ones Rust's std links already (libc, libm, libpthread, libdl, librt, libutil, libgcc_s) gets a `#[link(name = "libselinux.so.1", modifiers = "+verbatim")]`, so it links against the installed library without its development package. The C runtime's code refers to weak symbols (`__gmon_start__`, `_ITM_registerTMCloneTable`); nothing calls it, so the linker drops it.

Running it again replaces `src/` and `Cargo.toml`; it refuses a directory whose `Cargo.toml` it didn't write. `tests/cargo.rs` builds a three-file C program with `-g`, writes a project in both modes, and checks the modules, that `cargo build` has no warnings, and that the program prints and returns the same as the original.

Small programs run as they did, and so does `/bin/ls`: across 66 option sets in five directories its output matches the original's, except where it prints its own process id. `tests/x87.rs` covers what it needed besides the [Globals](#globals) slots: `long double` code, lifted as f64 (the x87 stack is eight register slots, its depth tracked per block; 80-bit values are converted at loads and stores, and `fist` rounds as the control word says), and stack arguments of a function whose frame offsets can't be followed (`alloca`). A `long double` returned in `st0` still fails the function.

## Tests

`tests/emit.rs` compiles what the emitter prints with `rustc`:

- the `sum` loop from `tests/common`, in both modes, linked into a program that runs it and checks the results, including that safe mode panics on a too-short slice;
- 400 random programs built from supported instructions, in both modes, which must all type-check;
- 300 random register-only programs emitted both structured and as the whole-function state machine, run on the same random inputs, which must return the same values (programs that don't terminate run out of fuel and are skipped); the irreducible ones exercise the per-region `match bb`;
- the real binary on a small ELF built in the test: `--list`, both modes, `-f`, `--emit ir` and the exit codes.

`tests/discover.rs` links `tests/differential/corpus.c` with each C compiler found, with and without unwind tables and as a non-PIE, strips a copy, and checks that `--list` on the stripped copy finds the same function starts as the symbol table (and the same sizes, except the crt functions), names `main`, and that the stripped copy's output type-checks in both modes. A second test builds a program without unwind tables at -O0 to -Os in which two functions end in a call to the program's own `die()` (which ends in `exit` or `abort`) and one function nothing calls follows one of them, and checks that the stripped copy has all of them with their exact sizes; it also has two hand-written functions that branch on flags after a call through a GOT slot to `die` and to `abort`, which must lift. A unit test in `src/discover.rs` reads `LC_FUNCTION_STARTS` from a hand-assembled Mach-O.

`tests/types.rs` compiles a small C file with `gcc -O2 -g` and checks the recovered prototypes and structs with debug info, in safe mode, and with `--no-dwarf`, and the gate that checks a type model's proposals.

`tests/differential.rs` runs every corpus build twice, without and with `-g`, so wrong types from either source show up as wrong results.

`tests/globals.rs` decompiles a cdylib that reads a `static`, writes a `static mut` (both through the GOT) and has a mangled function, then links the output into a program that checks the values. It also checks that `-j 1` and `-j 4` print the same thing.
