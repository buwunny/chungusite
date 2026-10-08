# What's left for a working decompiler

The pipeline runs end to end today: `chungusite <binary>` reads an ELF, Mach-O or PE file, lifts each function, cleans the SSA, and emits Rust that compiles in both modes ([cli.md](cli.md)). What limits it is coverage. On chungusite's own debug build (20,025 functions), 19,637 lift (up from 18,131 before jump tables, 16-byte SSE copies and one-operand `MUL`, and 1,920 before the lifter learned calls, the stack, conditional moves and memory operands), and the output for all of them type-checks, with or without `--skip-failed`. Calls are real calls with recovered signatures, and stack slots are local variables (steps 2 and 3), which removed about half of all raw memory accesses: 101,454 remain in fast mode. Safe mode bounds-checks a third of all accesses and leaves 63% of functions without a raw pointer (step 7; measured on the larger build that includes safe mode itself: 83,554 of 251,662 accesses, 15,833 of 25,156 functions). Lifting, signature inference and emission run in parallel (`-j`); the whole binary, 4.2 MB of lifted machine code, takes about 2.3 s on 4 cores. The input file is memory-mapped, and a function with an instruction the lifter doesn't know fails in the lifter's first pass, before any IR is built.

The ML refinement layer (`src/refine`, behind the `ml` feature; see docs/ml-runtime.md) is left out of this list on purpose. It plugs in at step 5 through `types::TypeModel` and is optional: everything below works without it. The CLI does not call it yet; wiring it in is a `--refine` flag that runs after safe mode, once the models are trained.

`chungusite <binary> --list` prints failures grouped by cause, which is the lifter's to-do list. The order below follows that table.

## 1. Lifter coverage (the blocker for real code)

`CALL`, `PUSH`/`POP`, the 8/16-bit `MOV` forms, `CMOVcc`/`SETcc`, memory operands, `MOVZX`/`MOVSX`, shifts, `IMUL`, one-operand `MUL` and `DIV` at every width, `INC`/`DEC`/`NEG`/`NOT`, 16-byte SSE copies, zeroing and bitwise ops, jump tables (`Terminator::Switch`), `ADC`/`SBB`, the bit instructions, rotates, double shifts, the atomics, `rep movs`, flags read in another block and thread-locals are lifted now ([lift.md](lift.md)). On chungusite's own debug build (29,735 functions), 29,650 lift (99.7%). What still fails (85 functions), by the first unsupported instruction in each function:

| Cause | Functions | What's needed |
|---|---|---|
| SSE vectors and floats (`PMOVMSKB`, `PCMPGTB`, `PUNPCKLBW`, `UCOMISD`, `PADDQ`, `PCMPEQB`, `PINSRW`, `CVTSI2SS`, AVX moves) | 80 | xmm values across blocks, then vector ops; scalar float (`F32`/`F64` exist in `Ty`) |
| Flags read after a call through the GOT that doesn't return (`handle_alloc_error`), so the code after it is never reached | 2 | noreturn detection for calls through a pointer (step 8) |
| A conditional branch into another function (a `.cold` part) | 1 | a tail call on one side of a branch |
| `CPUID`, `XGETBV` | 2 | an opaque intrinsic |

## 2. Calls and signatures (done)

Whole-program signature recovery ([calls.md](calls.md)): arguments, stack arguments, one or two return registers, and which caller-saved registers a function preserves (gcc's interprocedural register allocation depends on it). Direct calls, tail calls and imports through relocations, the PLT and the GOT are emitted by name, with `extern "C"` declarations for imports. On the sample binary this took the `todo!()` count from 37,048 to the 983 functions that don't lift. Still open:

- Indirect calls (function pointers, vtables) guess their arguments from the call site. Recovering vtables (step 6) would give them targets.
- Floating-point and vector arguments and returns (xmm registers) aren't tracked yet, which matters once SSE arithmetic lifts (step 1); 16-byte copies don't need it.
- ~~In safe mode, functions that other decompiled functions call take integers.~~ Done in step 7: callers lend slices, and a caller that can't calls the callee's raw twin.

## 3. Stack frames (done)

`frame::promote` turns stack slots whose address doesn't escape into SSA values and stack arguments into parameters ([calls.md](calls.md)). On the sample binary, 2,893 of 10,665 functions still need a `frame` array for slots whose address escapes. Still open:

- ~~Borrow escaped slots (`&mut frame[..]`) so safe mode can bounds-check them.~~ Done in step 7: a safe frame is a byte array, indexed and lent to callees.
- Split the frame into one local per object instead of one array.

## 4. Control-flow structuring

Done: `src/structure.rs` turns every CFG into `if`/`else`, `while`, `loop` with `break`/`continue`, and early `return` (Ramsey's dominator-tree construction, then a cleanup pass). Blocks that only test a condition fold into `a || b` / `a && b`, single-use values are inlined into their use, and an irreducible cycle becomes a `loop { match bb }` over just its own blocks. On chungusite's own debug build that took the fast-mode output from 985k to 439k lines and its `let`s from 582k to 74k; the 177 functions with irreducible flow went from 9,930 `match` arms to 716. Left for readability:

- Values used only in a folded condition block (`if a || p[i] == 0`) are still computed into a variable first, because inlining is decided per block.
- Labeled blocks remain where two paths with their own code meet (3,524 in functions that were structured before, from 3,687).
- Copies on edges into a block with one predecessor (`(v20, v16) = (v71, v27);`) could reuse the predecessor's variables.

## 5. Types (done, first pass)

[types.md](types.md): integer widths and signedness from how values are used, prototypes, parameter names and structs from DWARF when present, structs inferred from field accesses otherwise (with recursive pointers like `next: *mut S2`), `&S`/`&mut S` struct arguments in safe mode, and a `TypeModel` hook whose proposals are checked against the facts before they are used. On chungusite's debug build 19,922 of 43,024 arguments get a type (15,872 without debug info), with 5,362 prototypes from DWARF and 957 inferred structs. Still open:

- Pointer values inside bodies are still `u64` addresses; only arguments and fields are typed pointers.
- Interprocedural pointee types: a callee's `*mut S` should type the caller's value it is passed.
- Indexed arguments in safe mode as `&[T]` instead of `&[u8]`.
- Wiring a trained model into the CLI through `TypeModel` (`--refine`).

## 6. Globals and data

Done for the common case ([cli.md](cli.md#globals)): constant addresses into data sections become `static`s named from data symbols, and loader-filled pointer slots (GOT, vtables) point at the right static or function. Still to do:

- Typed statics: today every static is `Bytes<N>` or `Words<N>`. Once types (step 5) also cover globals and know an access is a `u32` table or a `&str`, emit `[u32; N]` or a string.
- Thread-locals reached through a register (the initial-exec model in shared objects) and `__tls_get_addr` (dynamic TLS) are unsupported; thread-locals of an executable are one `static mut THREAD_LOCALS` shared by all threads.
- Mach-O chained fixups and PE base relocations aren't read, so pointer slots in those formats keep their file bytes.
- Relocatable objects (`.o`) have no addresses, so their data, jump tables included, isn't read. The differential harness links the corpus into a program (`-nostartfiles`) to get around it; laying out an object's sections and applying its relocations would make `.o` input work directly.

## 7. Safe mode across calls (done)

Stages 5 to 7 in [ownership.md](ownership.md) are in: the frame, read-only globals and `malloc`'d allocations are roots like arguments, and pointers stored in the frame or the heap are followed; call summaries let callers lend slices to callees (`split_at_mut` for two at once), and a caller that can't calls the callee's raw twin; `malloc`/`free` become `Box<[u8]>` and a drop, `memcpy`/`memset` slice operations; conflicting loans and uses after free are found with Polonius-style rules on `datafrog` and downgraded; and `--check` compiles the output with rustc and emits whatever it rejects in fast mode. On chungusite's own debug build, 46% of memory accesses are bounds-checked (from 14%), and 65% of functions have no raw pointer (from 27%); the table is in ownership.md. Still open:

- Splitting the frame into one local per object (step 3), so that one escaping object doesn't make the whole frame raw. Frames are the largest remaining source of raw accesses.
- Passing a `Box` to a callee that frees it (by value), and returning one.
- Returning references (`&'a [u8]`) instead of addresses, which needs the `subset` facts the loan rules already support.
- Measuring the inferred signatures against DWARF on a Rust corpus.

## 8. Binary handling

- ~~Stripped binaries: discover functions from the entry point, call targets and `.eh_frame`, instead of requiring `--addr`/`--size`.~~ Done ([cli.md](cli.md#stripped-binaries)): on chungusite's stripped debug build all 20,025 function starts are found. Still open: noreturn calls (`__stack_chk_fail`, `abort`) for binaries without unwind tables, and Mach-O `LC_FUNCTION_STARTS`.
- ~~Demangle C++ and Rust symbol names.~~ Done.
- ~~Lift functions in parallel with `rayon`, one `Lifter` and `Function` per thread, as ir.md plans.~~ Done (`-j`).

## 9. Checking the output means the same thing

`tests/emit.rs` already runs one decompiled function in both modes and compares results. Generalize that into differential testing: compile small C functions, decompile them, call both on random inputs, and compare. That is the test that says the decompiler is *correct*, not just that its output compiles.

Started: `tests/differential.rs` compiles `tests/differential/corpus.c` (118 functions, including calls, stack arguments, address-taken locals, wide multiplies, 16-byte copies, jump tables, bit instructions and atomics) with gcc and clang at -O1, -O2 and -Os, links it into a program so its data and jump tables have addresses, decompiles each function in both modes with its data as statics, and runs original and decompiled code side by side on random inputs, comparing return values and buffer contents. A function that doesn't lift yet is counted; one that lifts and computes something different fails the test unless `KNOWN_BAD` lists it. The pass rate it prints is a second to-do list next to `--list`: it shows which missing instructions cost the most real code. Add a function to the corpus with a `// @diff name: ret(args)` line above it. It passes 1,390 of 1,416 (function, build) pairs, and no function that lifts gives a wrong result.
