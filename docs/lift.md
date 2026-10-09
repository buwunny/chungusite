# Lifting x86_64 with iced-x86

[`src/lift.rs`](../src/lift.rs) turns decoded `iced_x86::Instruction`s into the IR from [ir.md](ir.md). It covers:

- data movement: `MOV` in every register, immediate and memory form, including 8 and 16-bit writes (merged into the old value) and `AH`-style high bytes; `MOVZX`, `MOVSX`, `MOVSXD`, `LEA`, `CDQE`/`CWDE`;
- arithmetic: `ADD/SUB/AND/OR/XOR/CMP/TEST`, `INC/DEC/NEG/NOT`, `SHL/SHR/SAR` (immediate or `CL` count), `IMUL` in all three forms, one-operand `MUL`, and `DIV`/`IDIV` when `rdx` only extends `rax` (`xor edx, edx` or `CQO`/`CDQ` first). The 8 and 16-bit one-operand `MUL`/`IMUL`/`DIV`/`IDIV` (`ax = al * src`, `al, ah = ax / src`, `dx:ax`) compute at twice the operand's width, where nothing can overflow. Any of them can take a memory operand: the lifter loads it, and a memory destination is stored back. A 64-bit one-operand `MUL`/`IMUL` writes `rax = a * b` and `rdx = UMulHi(a, b)` (`SMulHi` for `IMUL`), the high half of the 128-bit product, which the emitter prints as `((a as u128 * b as u128) >> 64) as u64`; the 32-bit form multiplies the zero- or sign-extended halves in 64 bits;
- flags: `Jcc`, `CMOVcc` (`Select`) and `SETcc` (`Cmp` zero-extended to a byte) all read the same lazy flags, including carry and overflow after `ADD`/`SUB`/`CMP`, after `MUL`/`IMUL`, where CF = OF = "the product doesn't fit" (`seto` after an overflow-checked multiply), and the signed conditions after `INC`/`DEC` (which are an add or subtract of 1 that leaves CF alone) and after an `ADD` of a constant. Flags set in one block and read in another become block parameters (below);
- SSE2 ([`src/lift/sse.rs`](../src/lift/sse.rs)): 16-byte moves (`MOVUPS`/`MOVAPS`/`MOVDQU`/`MOVDQA` and the `PD`/VEX forms), `MOVQ`/`MOVD`/`MOVSD`/`MOVSS`, half moves (`MOVHPS`, `MOVLPD`, ...), the bitwise ops, integer lane arithmetic (`PADD*`/`PSUB*` with and without saturation, `PMIN*`/`PMAX*`, `PAVG*`, `PMULLW`, `PMULUDQ`, `PSADBW`), compares (`PCMPEQ*`, `PCMPGT*`), shifts by an immediate or an xmm count and the byte shifts `PSLLDQ`/`PSRLDQ`, unpacks, `PSHUFD`/`PSHUFLW`/`PSHUFHW`/`SHUFPS`/`SHUFPD`, `PINSRW`/`PEXTRW`, `PMOVMSKB`/`MOVMSKPS`/`MOVMSKPD`, the saturating packs (`PACKSSWB`, `PACKUSWB`, `PACKSSDW`, `PACKUSDW`), and scalar and packed floating point (`ADDSD`..`DIVPS`, `MINSD`/`MAXPS`, `SQRT*`, `CMPSD`/`CMPPS` with every predicate, `UCOMISD`/`COMISS` for `Jcc`/`SETcc`, and the `CVT*` conversions between integers, `f32` and `f64`). AVX2 on `ymm` registers: their upper halves are 16 more register pairs, tracked only in a function that uses a `ymm` register (undefined at the entry and after a call, zeroed by `VZEROUPPER` and by a VEX instruction that writes the `xmm` part). Moves, the bitwise ops, compares and integer lane arithmetic, which work on each 128-bit half alone, are lifted once per half; `VPMOVMSKB` joins the two masks, and `VPBROADCASTB/W/D/Q` fill every lane. Below;
- bit instructions, `adc`/`sbb`, double shifts, rotates, atomics and `rep movs` (below);
- jump tables, as a `Terminator::Switch` (below);
- the stack: `PUSH`, `POP` and `LEAVE` move `rsp` with a `PtrOffset` and a store or load;
- calls: `CALL` (direct, register or memory) becomes `InstKind::Call` with the System V argument registers; an indirect `JMP` that isn't a jump table (`jmp [rip+x]` through the GOT, `jmp rax`, `jmp [rax+8]` through a vtable) is a tail call through that pointer;
- `JMP`, `RET`, `NOP`/`ENDBR64`, and `UD2`/`INT3`/`HLT`, which end the block as `Unreachable`.

Anything else returns `LiftError::Unsupported` so the caller can decide what to do. It never guesses. Pass 1 already checks each mnemonic (`handled`), so a function with an instruction the lifter has no case for fails before any IR is built; pass 2 still rejects unsupported operand forms.

## Flags across blocks

A condition read before the block sets the flags (`cmp; je A; jl B`, where the `jl` starts a block of its own, or a loop head that tests the flags of whichever block jumped to it) can't come from the block's own lazy flags. The block starts with `Flags::Entry` instead, and the read becomes a `Bool` block parameter for that condition (`BlockParam(FLAG_PARAM)`), one per condition code the block reads. `finalize` then has each predecessor supply it, the same way it wires live-in registers: a predecessor that set the flags computes the condition from them at its end (its instructions are moved to the end of the pool so the new ones can follow them), and one that didn't touch the flags passes on a parameter of its own. With one predecessor the parameter is trivial and `opt::clean` replaces it with that predecessor's `Cmp`, so `cmp; je; jl` reads as `if a == b { .. } else if a < b { .. }`. It is still `LiftError::FlagsNotInBlock` when a predecessor leaves flags that don't give the condition (a call, a division, a shift by `cl`), or when the flags reach the entry.

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

In the PIC shape the `lea` can be far back: a `switch` in a loop loads the table's address once, before the loop. gcc -O0 also computes the offset first and sign-extends with `cdqe` (`lea rdx, [rax*4]; lea rax, [rip + table]; mov eax, [rdx + rax]; cdqe; lea rdx, [rip + table]; add rax, rdx; jmp rax`), which is recognized too.

It keeps the last 12 decoded instructions in a ring (no allocation), and for each register the address the last `lea r, [rip + x]` set it to (forgotten when anything else writes it, a call included, in address order), matches the table load, and reads the table from the binary's sections (`lift_with_data`; `lift` alone looks in the code bytes). The case count comes from the bounds check, a `cmp`/`sub` with an immediate followed by `ja`/`jae`/`jbe`/`jb`, when it tests the index register or a register moved into it. A `match` on an enum has no bounds check, since the discriminant can't be out of range, so the table is read until an entry leaves the function and cut before the first one that isn't the start of an instruction. Every case target becomes a block leader.

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
- `tests/robust.rs` covers bad branch targets: into the middle of an instruction, a fall-through off the end, and flags that nothing before the read set, at the entry or after a call. Each returns a `LiftError`; a conditional branch past the end is a tail call. It also lifts 20,000 random byte strings without a panic, and 2,000 random programs built from supported instructions with random jumps, all of which must pass the verifier.

After lifting, `opt::clean` removes trivial block params and dead code; see [ownership.md](ownership.md) for that pass and the safe-mode analyses built on it.

## SSE

An xmm register is a pair of 64-bit values (low, high), in slots of the register file after the 16 general registers, so it crosses blocks as two block parameters like any register (`BlockParam(XMM_PARAM + 2n)` and `+ 2n + 1`). A 16-byte copy is two loads and two stores, and a bitwise op is two `Bin`s.

Lane arithmetic is `BinOp::Lane(op, w)` or `UnOp::Lane(op, w)` on one 64-bit half at a time, with `w`-byte lanes: `paddd` is `Lane(Add, 4)` on each half. The emitter prints them as calls into a `simd` module ([`src/simd.rs`](../src/simd.rs)) that the output file defines once when some function uses it: `simd::b32::add(x, y)` adds the two `u32` lanes of `x` and `y`, `simd::f64::cmplt` compares like `CMPLTSD` (all ones or zero), `simd::cvt::i32_from_f64_trunc` converts like `CVTTSD2SI`, giving the integer minimum for NaN and out-of-range values as the hardware does. Floats stay bit patterns in `u64`s; only the `simd` functions see `f32`/`f64`.

`UCOMISD`/`COMISD` set `Flags::Float`. Each condition is one lane compare: `ja` is "greater, ordered", `jb` is "not greater-or-equal" (true when unordered, like CF), `je` is "neither less nor greater" (ZF is set when unordered), `jp` is "unordered".

xmm registers carry float arguments and results ([calls.md](calls.md)): one read at entry is an entry parameter, and one read after a call is a `CallOut` that signature recovery resolves to the call's result, the value from before the call (the callee preserves it), or `Undef`. The exception is the part of a register that a scalar conversion or square root leaves alone (`cvtsi2sd xmm0, edi` keeps xmm0's high half): compilers treat it as garbage, so when the block hasn't set it, it is `Undef`.

## Bit instructions, atomics and the rest

- **`adc`/`sbb`** add or subtract the carry the previous instruction left (`cmp; sbb eax, eax` and 128-bit `add; adc` chains). The carry out of them isn't modelled.
- **`shld`/`shrd`** are `d << n | s >> (w - n)` (the other way round for `shrd`), with a count of 0 selecting `d` unchanged. **8/16-bit shifts** by `cl` or by 8 or more shift a 32-bit copy, since x86 masks the count to 5 bits, not to the operand width.
- **`bt`** (CF only, `Flags::Carry`), **`bts`/`btr`/`btc`** with a register (which also set, clear or flip the bit), **`bsf`/`bsr`/`tzcnt`/`lzcnt`/`popcnt`**, **`bswap`**, **`rol`/`ror`**, **`xchg`**, **`xadd`** and **`cmpxchg`**. The atomics are lifted as plain loads and stores, which is right for one thread.
- **`cpuid`** and **`xgetbv`** ask the CPU the output runs on: each output register is a call into the prelude's `simd::cpu` (`simd::cpu::cpuid_ebx(eax, ecx)`), so a feature check picks a code path for that machine.
- **A conditional branch out of the function** (into gcc's `.cold` part of another function) gets a block of its own, after the blocks of the function's code, that tail-calls the target.
- **`rep movs`** is a `MemCopy` of `rcx` elements and **`rep stos`** a `MemFill` of `rcx` copies of `al`/`ax`/`eax`/`rax`, with the direction flag assumed clear. **`pause`** (a spin-loop hint) is a no-op.
- **The stack protector's canary** `fs:[0x28]` reads as a constant, so the check at the end of the function always passes.
- **Thread-locals** of an executable sit just below the thread pointer, which `fs:0` holds (x86-64 TLS variant II), so code reads them as `fs:[-k]`, or loads `fs:0` and indexes down from it. `load::tls` gives the thread-local block (`.tdata`, then `.tbss`) an address range of its own above every section (`.tbss` has none: it overlaps the sections after it), and `Lifter::thread_pointer` is its end. `fs:[k]` becomes the constant address `thread_pointer + k`, and `fs:0` that address itself, so globals turn them into the static `THREAD_LOCALS` like any other data ([cli.md](cli.md#globals)). Thread-locals reached through a register (`mov rax, [rip+x]; mov eax, fs:[rax]`, the initial-exec model in shared objects), `__tls_get_addr` and `gs:` stay unsupported.
- **Calls that don't return** (`abort`, `exit`, `__stack_chk_fail`, `__cxa_throw`, ...; `discover::noreturn`) end their block, so a caller doesn't merge their undefined `rax` into its return value. `lift_full` takes a callback that `program.rs` answers from the relocation or PLT entry at the call.

## Not handled yet

- AVX2 instructions that cross the 128-bit halves (`VPERMQ`, `VINSERTI128`, ...) or shift by an `xmm` count, AVX floats on `ymm` registers, and SSSE3 and later (`PSHUFB`, `PTEST`, ...).
- `DIV` with a real 128-bit dividend.
- Parity, the carry out of `adc`/`sbb`, the signed conditions after an `ADD` of two registers, and `CF|ZF` (`ja`/`jbe`) after `ADD`.
- Thread-locals through a register, and `gs:`.
