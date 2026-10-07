# What's left for a working decompiler

The pipeline runs end to end today: `chungusite <binary>` reads an ELF, Mach-O or PE file, lifts each function, cleans the SSA, and emits Rust that compiles in both modes ([cli.md](cli.md)). What limits it is coverage. On chungusite's own debug build (11,600 functions), 1,920 lift, and the output for all of them type-checks. In safe mode 2,259 of 7,914 memory accesses come out bounds-checked. Release-mode decompilation of the whole binary takes under 0.1 s.

The ML refinement layer (`src/refine`, behind the `ml` feature; see docs/ml-runtime.md) is left out of this list on purpose. It plugs in at step 5 and is optional: everything below works without it. The CLI does not call it yet; wiring it in is a `--refine` flag that runs after safe mode, once the models are trained.

`chungusite <binary> --list` prints failures grouped by cause, which is the lifter's to-do list. The order below follows that table.

## 1. Lifter coverage (the blocker for real code)

Failures on the 11,600-function sample, by the first unsupported instruction in each function, most common first:

| Cause | Functions | What's needed |
|---|---|---|
| `CALL` | 4,605 | a `Call` instruction plus the System V clobber set (rax, rcx, rdx, rsi, rdi, r8-r11 become unknown after it) |
| `PUSH`/`POP` | 1,603 | RSP tracking: `push` is `rsp -= 8; store`, `pop` the reverse. Prologues and epilogues are in nearly every non-leaf function |
| `MOV` forms | 1,152 | 8/16-bit register writes (merge into the old value), segment and other operand forms |
| `CMOVcc`, `SETcc` | ~880 | `Select` and `Cmp` from the pending flags, which the IR already has |
| `CMP`/`TEST` with memory operands | ~600 | load the operand, then the existing flag logic |
| `MOVZX`, `MOVSX`, `MOVSXD` | ~180 | `Cast` ZExt/SExt |
| `SHL`/`SHR`/`SAR`, `IMUL`, `MUL`, `INC`/`DEC`, `NEG`/`NOT` | ~350 | `Bin`/`Un`, plus their flags |
| ALU ops with memory operands | ~40 | load, op, store |
| Indirect `JMP` | 57 | jump-table recovery into `Terminator::Switch` (the emitter needs a case for it too) |
| SSE (`MOVUPS`, `MOVAPS`, `MOVSD`, ...) | ~150 | scalar float first (`F32`/`F64` exist in `Ty`), then vectors |

Flags that cross a block boundary (`LiftError::FlagsNotInBlock`) also need handling: materialize the flag values at the block exit when a successor reads them.

`CALL` plus `PUSH`/`POP` alone accounts for most failures. Those two together are the single biggest win.

## 2. Calls and signatures

- Lift `CALL` into `InstKind::Call`, and emit direct calls to other decompiled functions by name.
- Argument recovery: which of `rdi, rsi, rdx, rcx, r8, r9` a callee actually reads. Today every register read on entry becomes a parameter, so a `void` function still "returns" `rax`, and callee-saved registers and `rsp` show up as arguments.
- Return recovery: whether `rax` is written on every path to a return.
- Imports through the PLT and GOT become `extern "C"` declarations with their symbol names.
- Tail calls (`jmp` out of the function) currently emit `todo!()`. With argument recovery they become `return callee(args)`.

## 3. Stack frames

`rsp` is a pointer root in the borrow analysis already, and `Analysis::stack_slots` finds the slots (stage 4 in [ownership.md](ownership.md)). Still to do:

- Give the emitted function a real frame (a local byte array) instead of taking `rsp` as an argument.
- Promote slots whose address isn't taken to SSA values, so locals become `let` bindings.
- Keep address-taken slots as locals and borrow them (`&mut local`).

## 4. Control-flow structuring

Every function with a branch is currently a `loop { match bb { ... } }` state machine. That is correct for any CFG but hard to read. Structuring recovers `if`/`else`, `while`/`loop` with `break`, and early `return` from the dominator tree and loop nesting (`cfg.rs` has dominators). Irreducible regions keep the state machine.

## 5. Types

- Structs from the per-argument field facts (`ParamBorrow::fields`, `indexed`). This turns safe mode's `&[u8]` plus byte offsets into `&S` with named fields, and indexed access into `&[T]`.
- Value widths and signedness from how values are used (signed compares, `SAR`, `MOVSX`), instead of `u64` everywhere.
- Pointers versus integers, so pointer values stop being `u64`.
- This is where the ML type and naming model plugs in: it proposes, and the facts above accept or reject.

## 6. Globals and data

RIP-relative accesses are lifted as constant addresses, so they read the original process's memory. Map them to `static`s built from the binary's `.data`/`.rodata` (strings, tables, vtables), and name them from symbols.

## 7. Remaining safe-mode stages

Stages 5 to 7 in [ownership.md](ownership.md): call summaries and moves (`malloc`/`free` into `Box`), loans and lifetimes, and the final pass that runs rustc on the output and downgrades whatever fails to borrow-check back to raw pointers.

## 8. Binary handling

- Stripped binaries: discover functions from the entry point, call targets and `.eh_frame`, instead of requiring `--addr`/`--size`.
- Demangle C++ and Rust symbol names.
- Lift functions in parallel with `rayon`, one `Lifter` and `Function` per thread, as ir.md plans.

## 9. Checking the output means the same thing

`tests/emit.rs` already runs one decompiled function in both modes and compares results. Generalize that into differential testing: compile small C functions, decompile them, call both on random inputs, and compare. That is the test that says the decompiler is *correct*, not just that its output compiles.
