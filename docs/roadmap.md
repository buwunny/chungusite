# What's left for a working decompiler

The pipeline runs end to end today: `chungusite <binary>` reads an ELF, Mach-O or PE file, lifts each function, cleans the SSA, and emits Rust that compiles in both modes ([cli.md](cli.md)). What limits it is coverage. On chungusite's own debug build (11,600 functions), 10,600 lift (up from 1,920 before the lifter learned calls, the stack, conditional moves and memory operands), and the output for all of them type-checks. Calls are real calls with recovered signatures, and stack slots are local variables (steps 2 and 3), which removed about half of all raw memory accesses: 101,454 remain in fast mode, and in safe mode 10,666 of 101,454 accesses come out bounds-checked. Lifting, signature inference and emission run in parallel (`-j`); the whole binary takes about 1 s on 4 cores.

The ML refinement layer (`src/refine`, behind the `ml` feature; see docs/ml-runtime.md) is left out of this list on purpose. It plugs in at step 5 and is optional: everything below works without it. The CLI does not call it yet; wiring it in is a `--refine` flag that runs after safe mode, once the models are trained.

`chungusite <binary> --list` prints failures grouped by cause, which is the lifter's to-do list. The order below follows that table.

## 1. Lifter coverage (the blocker for real code)

`CALL`, `PUSH`/`POP`, the 8/16-bit `MOV` forms, `CMOVcc`/`SETcc`, memory operands, `MOVZX`/`MOVSX`, shifts, `IMUL`, `INC`/`DEC`/`NEG`/`NOT` and `DIV`/`IDIV` after `CQO` or `xor edx, edx` are lifted now ([lift.md](lift.md)). What still fails on the 11,580-function sample, by the first unsupported instruction in each function:

| Cause | Functions | What's needed |
|---|---|---|
| SSE (`MOVUPS`, `XORPS`, `MOVDQA`, `MOVAPS`, `MOVD`, ...) | ~600 | 16-byte copies first (`MOVUPS` pairs are mostly memcpy of two qwords), then scalar float (`F32`/`F64` exist in `Ty`), then vectors |
| Indirect `JMP` | 136 | jump-table recovery into `Terminator::Switch` (the emitter needs a case for it too) |
| `MUL` (one operand) | 46 | a 128-bit product: `rax` is the low half, `rdx` the high half, OF/CF = high != 0 |
| `SBB`, `ADC`, `XADD`, `CMPXCHG`, `BSR`, `TZCNT`, `BT`, `BSWAP`, `ROL`, `MOVSQ`, ... | ~150 | one at a time; atomics need an `Atomic` op or `Opaque` |
| Flags across blocks | 20 | materialize the flag values at the block exit when a successor reads them (`LiftError::FlagsNotInBlock`) |

## 2. Calls and signatures (done)

Whole-program signature recovery ([calls.md](calls.md)): arguments, stack arguments, one or two return registers, and which caller-saved registers a function preserves (gcc's interprocedural register allocation depends on it). Direct calls, tail calls and imports through relocations, the PLT and the GOT are emitted by name, with `extern "C"` declarations for imports. On the sample binary this took the `todo!()` count from 37,048 to the 983 functions that don't lift. Still open:

- Indirect calls (function pointers, vtables) guess their arguments from the call site. Recovering vtables (step 6) would give them targets.
- Floating-point and vector arguments and returns (xmm registers) aren't tracked yet, which matters once SSE lifts (step 1).
- In safe mode, functions that other decompiled functions call take integers, because callers have addresses, not slices. Passing slices across calls needs the call summaries of step 7.

## 3. Stack frames (done)

`frame::promote` turns stack slots whose address doesn't escape into SSA values and stack arguments into parameters ([calls.md](calls.md)). On the sample binary, 2,893 of 10,665 functions still need a `frame` array for slots whose address escapes. Still open:

- Borrow escaped slots (`&mut frame[..]`) so safe mode can bounds-check them; today frame accesses are raw.
- Split the frame into one local per object instead of one array.

## 4. Control-flow structuring

Done: `src/structure.rs` turns every reducible CFG into `if`/`else`, `loop` with `break`/`continue`, and early `return` (Ramsey's dominator-tree construction, then a cleanup pass); irreducible CFGs keep the state machine. On chungusite's own debug build all 1,962 lifted functions come out structured. Left for readability:

- Short-circuit conditions: `if a || b` currently needs a labeled block, because two paths reach the same `else`.
- `while cond { .. }`: the condition is computed in statements before the `if`, so loops print as `loop { let c = ..; if !c { break; } .. }`. Inlining single-use pure values into their use would fix this and shorten most code.
- Irreducible regions are handled per function, not per region: one bad cycle turns the whole function back into a state machine.

## 5. Types

- Structs from the per-argument field facts (`ParamBorrow::fields`, `indexed`). This turns safe mode's `&[u8]` plus byte offsets into `&S` with named fields, and indexed access into `&[T]`.
- Value widths and signedness from how values are used (signed compares, `SAR`, `MOVSX`), instead of `u64` everywhere.
- Pointers versus integers, so pointer values stop being `u64`.
- This is where the ML type and naming model plugs in: it proposes, and the facts above accept or reject.

## 6. Globals and data

Done for the common case ([cli.md](cli.md#globals)): constant addresses into data sections become `static`s named from data symbols, and loader-filled pointer slots (GOT, vtables) point at the right static or function. Still to do:

- Typed statics: today every static is `Bytes<N>` or `Words<N>`. Once types (step 5) know an access is a `u32` table or a `&str`, emit `[u32; N]` or a string.
- Thread-locals (`fs:`-relative accesses) are still unsupported in the lifter.
- Mach-O chained fixups and PE base relocations aren't read, so pointer slots in those formats keep their file bytes.

## 7. Remaining safe-mode stages

Stages 5 to 7 in [ownership.md](ownership.md): call summaries and moves (`malloc`/`free` into `Box`), loans and lifetimes, and the final pass that runs rustc on the output and downgrades whatever fails to borrow-check back to raw pointers.

## 8. Binary handling

- ~~Stripped binaries: discover functions from the entry point, call targets and `.eh_frame`, instead of requiring `--addr`/`--size`.~~ Done ([cli.md](cli.md#stripped-binaries)): on chungusite's stripped debug build all 20,025 function starts are found. Still open: noreturn calls (`__stack_chk_fail`, `abort`) for binaries without unwind tables, and Mach-O `LC_FUNCTION_STARTS`.
- ~~Demangle C++ and Rust symbol names.~~ Done.
- ~~Lift functions in parallel with `rayon`, one `Lifter` and `Function` per thread, as ir.md plans.~~ Done (`-j`).

## 9. Checking the output means the same thing

`tests/emit.rs` already runs one decompiled function in both modes and compares results. Generalize that into differential testing: compile small C functions, decompile them, call both on random inputs, and compare. That is the test that says the decompiler is *correct*, not just that its output compiles.

Started: `tests/differential.rs` compiles `tests/differential/corpus.c` (78 functions, including calls, stack arguments and address-taken locals) with gcc and clang at -O1, -O2 and -Os, decompiles each function in both modes, and runs original and decompiled code side by side on random inputs, comparing return values and buffer contents. A function that doesn't lift yet is counted; one that lifts and computes something different fails the test unless `KNOWN_BAD` lists it. The pass rate it prints is a second to-do list next to `--list`: it shows which missing instructions cost the most real code. Add a function to the corpus with a `// @diff name: ret(args)` line above it.
