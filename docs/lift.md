# Lifting x86_64 with iced-x86

[`src/lift.rs`](../src/lift.rs) turns decoded `iced_x86::Instruction`s into the IR from [ir.md](ir.md). It covers:

- data movement: `MOV` in every register, immediate and memory form, including 8 and 16-bit writes (merged into the old value) and `AH`-style high bytes; `MOVZX`, `MOVSX`, `MOVSXD`, `LEA`, `CDQE`/`CWDE`;
- arithmetic: `ADD/SUB/AND/OR/XOR/CMP/TEST`, `INC/DEC/NEG/NOT`, `SHL/SHR/SAR` (immediate or `CL` count), `IMUL` in all three forms, one-operand `MUL`, and `DIV`/`IDIV` when `rdx` only extends `rax` (`xor edx, edx` or `CQO`/`CDQ` first). Any of them can take a memory operand: the lifter loads it, and a memory destination is stored back. A 64-bit one-operand `MUL`/`IMUL` writes `rax = a * b` and `rdx = UMulHi(a, b)` (`SMulHi` for `IMUL`), the high half of the 128-bit product, which the emitter prints as `((a as u128 * b as u128) >> 64) as u64`; the 32-bit form multiplies the zero- or sign-extended halves in 64 bits;
- flags: `Jcc`, `CMOVcc` (`Select`) and `SETcc` (`Cmp` zero-extended to a byte) all read the same lazy flags, including carry and overflow after `ADD`/`SUB`/`CMP`, and after `MUL`/`IMUL`, where CF = OF = "the product doesn't fit" (`seto` after an overflow-checked multiply);
- 16-byte copies: `MOVUPS`/`MOVAPS`/`MOVDQU`/`MOVDQA` between memory and xmm registers, `MOVUPD`/`MOVAPD`/`LDDQU`, `XORPS`/`XORPD`/`PXOR` (zeroing, or xor of known values) and the other bitwise ops (`ANDPS`, `ORPS`, `PAND`, `POR`, ...), `PCMPEQ x, x` (all ones), `MOVQ`/`MOVD`/`MOVSD` between xmm registers, general registers and memory, and `PUNPCKLQDQ`/`MOVLHPS`. An xmm register is a pair of 64-bit values (low, high), so a copy is two loads and two stores. Those values are tracked within a block only: an xmm register read before the block writes it is unsupported (SSE arithmetic and floating point aren't lifted yet);
- bit instructions, `adc`/`sbb`, double shifts, rotates, atomics and `rep movs` (below);
- jump tables, as a `Terminator::Switch` (below);
- the stack: `PUSH`, `POP` and `LEAVE` move `rsp` with a `PtrOffset` and a store or load;
- calls: `CALL` (direct, register or memory) becomes `InstKind::Call` with the System V argument registers; an indirect `JMP` that isn't a jump table (`jmp [rip+x]` through the GOT, `jmp rax`, `jmp [rax+8]` through a vtable) is a tail call through that pointer;
- `JMP`, `RET`, `NOP`/`ENDBR64`, and `UD2`/`INT3`/`HLT`, which end the block as `Unreachable`.

Anything else returns `LiftError::Unsupported` so the caller can decide what to do. It never guesses. Pass 1 already checks each mnemonic (`handled`), so a function with an instruction the lifter has no case for fails before any IR is built; pass 2 still rejects unsupported operand forms.

## Jump tables

`switch` statements and Rust `match`es compile to an indirect jump through a table of case addresses. Pass 1 recognizes the two shapes gcc and clang (and rustc, through LLVM) emit:

```asm
    cmp    edi, 7                 ; bounds check: case count 8
    ja     default
    jmp    [table + rdi*8]        ; non-PIC: absolute addresses

    cmp    edi, 7
    ja     default
    mov    eax, edi
    lea    rdx, [rip + table]
    movsxd rax, dword [rdx + rax*4]
    add    rax, rdx               ; PIC: offsets from the table
    jmp    rax
```

It keeps the last 12 decoded instructions in a ring (no allocation), matches the table load, and reads the table from the binary's sections (`lift_with_data`; `lift` alone looks in the code bytes). The case count comes from the bounds check, a `cmp`/`sub` with an immediate followed by `ja`/`jae`/`jbe`/`jb`, when it tests the index register or a register moved into it. A `match` on an enum has no bounds check, since the discriminant can't be out of range, so the table is read until an entry leaves the function and cut before the first one that isn't the start of an instruction. Every case target becomes a block leader.

In pass 2 the index is the index register's value at the table load, and the jump becomes `Switch { v, table, default }`: case `k` goes to `table[k]`, anything else to `default`, which is the target the most cases share. Switch edges carry no block arguments, so each distinct target is reached through a new block that has no parameters and jumps on to it, passing the registers the target needs. The structurer prints a switch as `if v == 0 { .. } else if matches!(v, 1 | 3..=5) { .. } else { .. }`, and the state machine as a `match`.

A jump that looks like a table but whose table can't be read (a relocatable object, whose tables are relocations) is unsupported rather than taken for a tail call.

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

## Bit instructions, atomics and the rest

- **`adc`/`sbb`** add or subtract the carry the previous instruction left (`cmp; sbb eax, eax` and 128-bit `add; adc` chains). The carry out of them isn't modelled.
- **`shld`/`shrd`** are `d << n | s >> (w - n)` (the other way round for `shrd`), with a count of 0 selecting `d` unchanged. **8/16-bit shifts** by `cl` or by 8 or more shift a 32-bit copy, since x86 masks the count to 5 bits, not to the operand width.
- **`bt`** (CF only, `Flags::Carry`), **`bsf`/`bsr`/`tzcnt`/`lzcnt`/`popcnt`**, **`bswap`**, **`rol`/`ror`**, **`xchg`**, **`xadd`** and **`cmpxchg`**. The atomics are lifted as plain loads and stores, which is right for one thread.
- **`rep movs`** is a `MemCopy` of `rcx` elements, with the direction flag assumed clear.
- **The stack protector's canary** `fs:[0x28]` reads as a constant, so the check at the end of the function always passes. Other `fs:`/`gs:` accesses (thread-locals) stay unsupported.
- **Calls that don't return** (`abort`, `exit`, `__stack_chk_fail`, `__cxa_throw`, ...; `discover::noreturn`) end their block, so a caller doesn't merge their undefined `rax` into its return value. `lift_full` takes a callback that `program.rs` answers from the relocation or PLT entry at the call.

## Not handled yet

- SSE beyond 16-byte copies and bitwise ops: vector compares and shuffles (`PCMPEQB` other than the all-ones idiom, `PSHUFD`, `PUNPCKLBW`, `PMOVMSKB`, ...), scalar floating point (`UCOMISD`, `CVTSI2SS`), xmm values that cross blocks or calls, and AVX.
- `rep stos` (there is no memset instruction in the IR yet).
- `DIV` with a real 128-bit dividend, and 8/16-bit `MUL`/`DIV`.
- Flags that cross blocks (`LiftError::FlagsNotInBlock`), parity, the carry out of `adc`/`sbb`, and the signed conditions after `ADD`.
- Thread-locals (`fs:`/`gs:` other than the canary).
- Conditional branches into another function (gcc's `.cold` parts).
