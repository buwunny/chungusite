# Lifting x86_64 with iced-x86

[`src/lift.rs`](../src/lift.rs) turns decoded `iced_x86::Instruction`s into the IR from [ir.md](ir.md). It covers `NOP`/`ENDBR64`, register and immediate `MOV`s, loads and stores (`MOV [RAX+8], RCX`), `LEA`, `ADD/SUB/AND/OR/XOR/CMP/TEST`, `Jcc`, `JMP` and `RET`. Anything else returns `LiftError::Unsupported` so the caller can decide what to do. It never guesses.

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

- 8 and 16-bit register writes, which need a merge with the old value, and `AH`-style high-byte registers.
- Flags that cross blocks (`LiftError::FlagsNotInBlock`), plus conditions other than ZF/SF after non-`CMP` ops.
- `CALL`, indirect jumps (jump tables), stack frame and `RSP` tracking, and FS/GS (TLS) accesses.
