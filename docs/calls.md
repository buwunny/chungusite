# Calls, signatures and stack frames

The lifter sees one function at a time and doesn't know what its callees read or return. [`src/program.rs`](../src/program.rs) puts the whole program together: it lifts every function, works out every function's signature from its callers and callees, rewrites each function to that signature, and emits calls by name. [`src/abi.rs`](../src/abi.rs) holds the per-function analyses and the rewrite, and [`src/frame.rs`](../src/frame.rs) turns stack slots into values.

## Pipeline

1. **Lift** every function in the binary (in parallel, one `Lifter` per thread), whether or not it was selected for output: unselected functions are still lifted so their callers get accurate signatures. Each call lists all ten registers that might matter to the callee (`rdi, rsi, rdx, rcx, r8, r9, rsp, rax, r10, r11`), and every caller-saved register afterwards is a `CallOut` placeholder. With `track_exits` on, each return records the caller-saved registers in an `Exit` instruction.
2. **Resolve call targets**: a direct `call` to a known function; a relocation at the call site (relocatable objects); a PLT stub (`jmp [rip+slot]`) or a GOT slot (`call [rip+slot]`) named through the dynamic relocations. Anything else is `Indirect`.
3. **Infer signatures** to a fixpoint, in rounds that run every function in parallel against the previous round's callee signatures. A signature (`abi::Sig`) is:
   - `args`: the highest System V argument register that is live on entry, where a call only uses the arguments its callee takes;
   - `stack_args`: 8-byte words read above the return address;
   - `ret`: rax is defined on every path to a return;
   - `ret2`: rdx is too, and some caller reads it (16-byte results such as Rust's fat pointers and `(u64, u64)` pairs);
   - `preserves`: caller-saved registers the function leaves as it found them. gcc's interprocedural register allocation relies on this: if `f` never touches `rdi`, a caller may keep a value in `rdi` across `call f`.

   Imports come from a table of C library signatures ([`src/libc.rs`](../src/libc.rs)). An unknown import or a function that didn't lift takes the most arguments any call site sets up.
4. **Apply** each signature (`abi::apply`): registers after a call become the callee's result, the value from before the call (preserved), or `Undef`; calls pass exactly their callee's arguments, stack arguments loaded from the caller's stack; returns return `rax`, `rax:rdx` or nothing; a tail call to a callee that returns less than the caller becomes a call and a return; entry registers that aren't arguments become `Undef`. Then `frame::promote` and `opt::clean`.
5. **Emit** in parallel. Calls to selected functions use their names; everything else goes through `mod ffi { extern "C" { ... } }` at the top of the file, declared with its recovered signature (`-> Pair` for a two-register result). An `Indirect` call transmutes the target to an `extern "C"` function pointer.

## Stack frames

`frame::promote` tracks every value's offset from the entry `rsp`. A slot (a fixed offset accessed with one size) whose address never escapes becomes SSA values: loads read the last store on each path, through new block parameters where paths meet. Stack arguments are slots too, read from new entry parameters `arg6`, `arg7`, ...

A slot's address escapes when a stack pointer is stored, passed to a call or returned. A pointer to offset `x` taints `[x, 0)` (the callee may write anywhere up to our return address) for a local, or everything from `x` up for a stack argument. If `rsp` is used afterwards, for tainted slots or accesses at unknown offsets, the function gets a real frame, `let mut frame = [0u128; N]` (16-byte aligned like a real stack), and `rsp` becomes an address in it. A function that loses track of `rsp` entirely (`and rsp, -32`, `alloca`) gets a 64 KiB frame.

## Safe mode

A decompiled caller passes addresses, not slices, so in safe mode every function that some decompiled function calls takes all its arguments as `u64`. A function that calls an `unsafe` function is `unsafe` too.

## Limits

- Floating-point and vector arguments (xmm registers) aren't tracked; neither are variadic calls' float arguments (`al`).
- Signatures of indirect calls are guessed from the call site.
- Values returned in memory (structs larger than 16 bytes) appear as the hidden pointer argument in `rdi` and the same pointer returned in `rax`.
- A function with no return (every path ends in a call that doesn't return, or a loop) preserves nothing and returns nothing.
