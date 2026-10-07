# Lifting x86_64 with iced-x86

[`src/lift.rs`](../src/lift.rs) turns decoded `iced_x86::Instruction`s into the IR from [ir.md](ir.md). It covers:

- data movement: `MOV` in every register, immediate and memory form, including 8 and 16-bit writes (merged into the old value) and `AH`-style high bytes; `MOVZX`, `MOVSX`, `MOVSXD`, `LEA`, `CDQE`/`CWDE`;
- arithmetic: `ADD/SUB/AND/OR/XOR/CMP/TEST`, `INC/DEC/NEG/NOT`, `SHL/SHR/SAR` (immediate or `CL` count), two and three-operand `IMUL`, and `DIV`/`IDIV` when `rdx` only extends `rax` (`xor edx, edx` or `CQO`/`CDQ` first). Any of them can take a memory operand: the lifter loads it, and a memory destination is stored back;
- flags: `Jcc`, `CMOVcc` (`Select`) and `SETcc` (`Cmp` zero-extended to a byte) all read the same lazy flags, including carry and overflow after `ADD`/`SUB`/`CMP`;
- the stack: `PUSH`, `POP` and `LEAVE` move `rsp` with a `PtrOffset` and a store or load;
- calls: `CALL` (direct, register or memory) becomes `InstKind::Call` with the System V argument registers; `JMP [RIP+x]` is a tail call through the GOT;
- `JMP`, `RET`, `NOP`/`ENDBR64`, and `UD2`/`INT3`/`HLT`, which end the block as `Unreachable`.

Anything else returns `LiftError::Unsupported` so the caller can decide what to do. It never guesses.

## Calls

A call is `rax = callee(rdi, rsi, rdx, rcx, r8, r9, rsp, rax, r10, r11)`: every register the callee might read, because which ones it does read isn't known yet. Afterwards each caller-saved register (`rcx`, `rdx`, `rsi`, `rdi`, `r8`-`r11`) is a `CallOut { call, reg }` placeholder, and the flags are clobbered. Whether a `CallOut` is the high half of a 16-byte result (rdx), the register's value from before the call (the callee preserves it) or undefined depends on the callee, so `abi::apply` resolves it once signatures are known ([calls.md](calls.md)). A tail call (`jmp` out of the function) lists the same ten registers.

With `Lifter::track_exits` set, every return also records the caller-saved registers in an `Exit` instruction, which is how signature inference sees what a function leaves in them. `abi::apply` removes the `Exit`s.

The argument lists go into `value_pool` in `finalize`, after every block's instruction list, because a block's instructions must stay one contiguous run of the pool.

## Example

```asm
    mov  rax, rdi
    mov  [rax+8], rcx
    xor  edx, edx
top:
    cmp  rdx, rsi
    jae  done
    mov  r8, [rdi+rdx*8+16]
    add  rax, r8
    add  rdx, 1
    jmp  top
done:
    ret
```

lifts to the following IR. This exact text is checked in `tests/lift.rs`.

```
bb0(v1, v18, v0):                      ; live-ins: rcx, rsi, rdi
  v2 = ptr v0 + 8
  store v2 <- v1
  v4 = const 0x0                       ; xor edx, edx
  v5 = ZExt v4                         ; 32-bit write zero-extends
  jump bb1(v0, v5, v18, v0)
bb1(v19, v6, v7, v20):                 ; rax, rdx, rsi, rdi
  v8 = cmp.Uge v6, v7                  ; cmp + jae fused
  br v8 bb3(v19) bb2(v19, v6, v7, v20)
bb2(v13, v10, v21, v9):
  v11 = ptr v9 + v10*8 + 16            ; whole x86 address mode, one node
  v12 = load v11
  v14 = Add v13, v12
  v15 = const 0x1
  v16 = Add v10, v15
  jump bb1(v14, v16, v21, v9)
bb3(v17):
  ret v17
```

## Why it doesn't allocate

`tests/no_alloc.rs` installs a counting global allocator and checks that lifting the same function 1000 more times after a warm-up makes zero heap allocations. The techniques that make this hold:

1. **Reuse, don't rebuild.** `Lifter` and `Function` are long-lived. `lift` calls `clear()` on every `Vec` and arena, which keeps their capacity. One pair per worker thread lifts every function that thread is given.
2. **One `Instruction`, decoded in place.** `Decoder::decode_out(&mut self.insn)` overwrites a single 40-byte `Instruction` and never builds a list of instructions. `Decoder` itself borrows the byte slice and doesn't allocate.
3. **Two passes instead of a hash map.** Pass 1 collects block leaders (entry, branch targets, fall-throughs) into a reused `Vec<u64>` and sorts it. Pass 2 notices a block boundary by comparing the current address with the next leader, which costs O(1). Branch targets are resolved with a binary search. No `HashMap`, no hashing.
4. **The register file is a fixed array.** Each block has `[Option<ValueId>; 16]` for its exit values and another for its live-ins. `NonZeroU32` ids make each array 64 bytes, exactly one cache line. A register-to-register `MOV` emits nothing at all. It just copies a slot.
5. **Lazy flags.** `CMP` emits nothing. It records `Flags::Sub { lhs, rhs }`, and the `Jcc` that reads it emits a single `Cmp`. Flag results that nothing reads are never built, so there's nothing for DCE to clean up later.
6. **Idioms fold during lifting.** `xor r, r` becomes a constant, and `test r, r` reads `r` directly with no `And`.
7. **SSA wiring is a fix-up pass, not a graph.** Reading an undefined register creates a `BlockParam`. `finalize` propagates live-ins backwards to a fixpoint over the two arrays above, then writes parameter lists and edge arguments into the existing `value_pool`.

## Running and checking it

`chungusite --hex "<bytes>" --emit raw-ir` lifts your own hex bytes (loaded at `0x1000`) and prints the IR before cleanup; `--emit ir` shows it after. See [cli.md](cli.md) for the rest of the command. The `mov eax, 1; add eax, 2` spike (`--hex "b8 01 00 00 00 83 c0 02" --emit raw-ir`):

```
bb0():
  v0 = const 0x1
  v1 = ZExt v0          ; eax after the mov, as rax
  v2 = const 0x2
  v3 = Add v0, v2       ; reads the 32-bit value back, no Trunc(ZExt(..))
  v4 = ZExt v3
  Unreachable           ; the bytes end without a ret
```

`Lifter::reg_out(block, reg)` returns the value a register holds at a block's exit. It accepts any view of the register (`RAX`, `EAX`, `AX`, `AL` all map to the same slot).

[`src/verify.rs`](../src/verify.rs) checks the structure of a lifted function: no dangling value or block ids, every edge passes exactly as many arguments as its target has parameters, nothing uses a store as a value, and every value belongs to a block. Dominance isn't checked yet. The tests run it on everything they lift:

- `tests/spike.rs` checks the spike's exact IR and register mapping, and that each of the 16 GPRs lands in its own slot.
- `tests/robust.rs` covers bad branch targets: into the middle of an instruction, past the end, a fall-through off the end, and flags coming from another block. Each returns a `LiftError`. It also lifts 20,000 random byte strings without a panic, and 2,000 random programs built from supported instructions with random jumps, all of which must pass the verifier.

After lifting, `opt::clean` removes trivial block params and dead code; see [ownership.md](ownership.md) for that pass and the safe-mode analyses built on it.

## Not handled yet

- SSE and AVX (`MOVUPS`, `MOVAPS`, `MOVSD`, `XORPS`, ...), the largest group of failures left.
- Indirect jumps other than `JMP [RIP+x]`, which are mostly jump tables to recover into `Terminator::Switch`.
- One-operand `MUL`/`IMUL` and `DIV` with a real 128-bit dividend, which need a 128-bit product.
- `ADC`/`SBB`, atomics (`LOCK XADD`, `CMPXCHG`), `BSR`/`TZCNT`/`BSWAP`/`BT`, rotates, and string ops (`MOVSQ`).
- Flags that cross blocks (`LiftError::FlagsNotInBlock`), and parity, plus the signed conditions after `ADD`.
- FS/GS (TLS) accesses.
