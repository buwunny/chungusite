# Safe mode: inferring moves, `&T` and `&mut T`

`--mode safe` has to decide, for every pointer the binary handles, which Rust form to emit:

- a move (the value is owned and ownership is transferred),
- a shared borrow `&T`,
- a mutable borrow `&mut T`,
- or, when none of these can be proven, a raw pointer inside `unsafe`.

This document describes the whole pipeline. All of its stages are implemented:

| Stage | Code | What it does |
|---|---|---|
| 0. CFG views: predecessors, reverse postorder, dominators | [`src/cfg.rs`](../src/cfg.rs) | |
| 1. Clean SSA: trivial block params, dead code | [`src/opt.rs`](../src/opt.rs) | |
| 2. Origins: which object and offset each value points into | [`src/borrow.rs`](../src/borrow.rs) | arguments, the stack frame, globals, allocations; pointers stored in the frame or the heap are followed |
| 3. Access facts and classes | [`src/borrow.rs`](../src/borrow.rs) | which objects can be indexed as slices |
| 4. Stack frames | [`src/frame.rs`](../src/frame.rs), [`src/emit.rs`](../src/emit.rs) | unescaped slots become SSA values; the rest is a byte array that safe code indexes and lends to callees |
| 5. Call summaries and moves | [`src/program.rs`](../src/program.rs), [`src/libc.rs`](../src/libc.rs) | slices across calls, raw twins, `malloc`/`free` as `Box`, `memcpy`/`memset` as slice operations |
| 6. Loans and moves as Datalog | [`src/loans.rs`](../src/loans.rs) | Polonius-style rules on `datafrog`; conflicts downgrade to raw |
| 7. Check with rustc, downgrade on failure | [`src/check.rs`](../src/check.rs), `--check` | whatever rustc rejects is emitted in fast mode |

The guiding rule is that **every unproven step falls back to a raw pointer, never to a guess.** A raw pointer compiles and behaves like the binary, and a wrong `&mut` is undefined behaviour. The analysis may be incomplete, but it must stay sound.

## The model the output uses

Safe mode doesn't change how values are represented: every value is still a `u64`, and a pointer is an address ([cli.md](cli.md)). What changes is *how memory is reached*. A safe *root* (an object safe code can name) is a byte slice: an argument `rdi_ref: &mut [u8]`, the frame `frame.0`, a static `TABLE.b`, a heap allocation `heap12: Box<[u8]>`. An access through a pointer `p` derived from that root becomes

```rust
u64::from_le_bytes(rdi_ref[p.wrapping_sub(rdi_base) as usize..][..8].try_into().unwrap())
```

where `rdi_base` is the root's address. That computes exactly the address the binary used, so if the analysis is right the access is the same one, and if it is wrong about which object `p` points into, the slice index is out of range and the access panics. A wrong origin costs a panic, never a stray read or write. What the analysis *must* get right is aliasing: a root is only safe if nothing reaches its memory except through it (and through slices lent to callees for the length of a call).

## 0–1. A clean SSA to analyse

The lifter already gives SSA with block parameters instead of phi nodes ([lift.md](lift.md)). It creates a parameter for every register that is live into a block, so loops carry many parameters that never change. `opt::clean` removes them:

1. **Trivial parameters** (Braun et al. 2013). A parameter whose incoming arguments are all one value `v`, or the parameter itself, is replaced by `v` and dropped from the block and from every edge. Removing one can make another trivial, so this repeats to a fixpoint. Blocks with a single predecessor lose all their parameters this way. The entry block is skipped, because its parameters are the function's arguments.
2. **Dead code.** A mark phase starts from side effects (stores, calls, terminators) and from the entry parameters, and propagates through operands. A live parameter marks its incoming edge arguments. Unmarked pure instructions and parameters are removed.

Ids stay stable, and removed instructions just leave their block's list. On the loop from `lift.md`, this takes `bb1` from four parameters to two (the accumulator and the counter), and `bb2`/`bb3` to none. `tests/safe.rs` checks the exact output, and `tests/robust.rs` cleans and re-verifies 2,000 random programs.

`cfg::Cfg` gives predecessors, reverse postorder and immediate dominators (Cooper, Harvey and Kennedy) from flat arrays. Later stages use it in two ways. Dataflow iterates in RPO, which converges in a few passes. Loans (stage 6) need dominance to decide where a borrow is live.

## 2. Origins: what each value points into

The first question is not "is this a reference?" but "*which object and which offset* does this value point into?" That is a forward dataflow analysis over the SSA graph:

- **Roots.** A root is an object a pointer can point into:
  - each entry parameter (a possible pointer argument; this includes RSP in IR straight from the lifter, which makes the stack one more root);
  - the stack frame, `AddrOfLocal` (after `frame::promote`, rsp is replaced by the frame's address);
  - a global, `IntToPtr(Const(addr))`, one root per address;
  - an allocation, the result of a call whose summary says it allocates (`malloc`, `calloc`, `operator new`, `__rust_alloc`).
  A function has at most 63 roots; the rest share one root that is never safe.
- **Lattice.** An origin is a set of roots (a `u64` bitmask) plus an offset, either `Known(i64)` or `Unknown`. Join takes the union of the roots. The offset survives only if both sides agree. That is finite height, so it terminates.
- **Transfer.**
  - `PtrOffset base+disp` shifts by `disp`.
  - An index makes the offset `Unknown`, which is array or slice access.
  - `Add`/`Sub` by a constant shift the offset. `Add` with a non-constant gives `Unknown`.
  - In `a + b` (or `[a + b*1]`) where both are derived, either could be the pointer, so both are kept, except that when one side certainly points into an object (the frame, a global, an allocation) and the other only comes from arguments, the argument is the index.
  - `Select` joins its two inputs; casts and `IntToPtr`/`PtrToInt` pass them through.
  - Block parameters join their incoming edge arguments.
  - A call returns an allocation (an allocator), or what its summary says the result points into (`memcpy` returns its first argument), with an unknown offset.
  - **A load from the frame or an allocation** gets whatever was stored at that root and offset (plus whatever was stored there at an unknown offset). This is a flow-insensitive points-to map, keyed on `(root, offset)`, built in the same fixpoint. Debug builds keep almost everything in the frame, so without it every pointer that passes through a spilled local would be lost. Loads from argument memory or globals still have no origin: what the caller stored there is outside the function.
  - Everything else has no origin.
- **Fixpoint.** Iterate blocks in RPO until nothing changes. The transfer functions are monotone. One detail matters: "no origin" must have exactly one representation. Shifting it must not invent an offset, or a decrementing loop counter never converges. The random-program test found that bug.

## 3. Access facts, and choosing what each root becomes

One pass over the reachable blocks turns uses of derived values into facts about their roots. Each fact is `(root, offset, kind, point)`, where a point is an instruction or a terminator (`loans.rs` numbers them):

| IR | Fact |
|---|---|
| `load ptr` | `Read(root(ptr), off)` |
| `store ptr <- val` | `Write(root(ptr), off)`; for each root of `val`, `Stash(root(ptr))` if `ptr` is in the frame or an allocation, otherwise `Escape` |
| `memcpy dst, src` | `Write(dst)`, `Read(src)`; the pointers stashed in `src` are copied (`CopyTo`) or leave (`Spill`) |
| an access through a pointer into several roots | `Escape` for each (it can't be bounds-checked against one) |
| `call f(args)`, tail call | per argument, what `f`'s summary says (stage 5); `Escape` for every argument of an unknown callee |
| `ret v` | `Return(root(v))` |
| `cmp.eq/ne p, 0` at offset 0 | `NullCheck(root(p))` |
| any other operation on a derived value | `Escape` (it leaves what we can track) |
| `p - q` of two derived values | nothing (a length, like `offset_from`) |
| `p < end` and other comparisons | nothing |

Then a fixpoint decides which roots are **safe**: they start safe, and a root becomes raw (never the other way) when

- something lets a pointer to it escape: an `Escape` fact, or a `Stash` into a container whose contents escape (a container that is raw itself, lent to a callee, or spilled), or a call that can't lend it (below);
- it is accessed before its start (a negative constant offset), except a global, whose offset is from the address the code used;
- and per kind of root: an argument must be dereferenced, not demoted and not freed; the frame must not be returned or freed, and not be the 64 KiB fallback frame for stack use the frame pass can't follow; a global must be in a read-only static that is emitted as `Bytes` (no pointer slots), and only read; an allocation must not be returned, not be made inside a loop (one variable holds it), be freed only at offset 0, and not be used after it may have been freed (stage 6).

Each argument then gets a class:

```
no Read, Write or Borrow           -> integer      (no deref, so no evidence of a pointer)
not safe                           -> *const T / *mut T  (raw)
Write, or a mutable Borrow         -> &mut T
otherwise                          -> &T
NullCheck                          -> wrap in Option<..>
Return                             -> the return value borrows from this argument
```

**Pointers need dereference evidence.** Compilers use `lea` and `add` for integer arithmetic, so being offset or returned doesn't make a value a pointer. Only a dereference of something derived from it does.

The accesses also give the pointee's shape. Constant offsets become struct fields: `fields: [(0, written), (8, read)]`. An access at an unknown offset (`indexed`) is evidence for a slice or array. The ML type model (design/ml-refinement.md) proposes types, and these facts gate its proposals.

`--emit borrows` prints the verdict, with the reason for every raw root:

```
  rdi: &T, fields at +0
  rsi: raw pointer (escapes)
  frame: &mut [u8]
  allocation v23: Box<[u8]>
```

**What this stage does not prove yet:**

- **Null-check dominance.** `nullable` only says that a null check exists. A nullable argument that is lent to a callee is not lent (the call goes to the raw twin), since the analysis can't tell whether it is null at the call.
- **Reads before writes.** An argument written before it is ever read could be an out-parameter (`&mut MaybeUninit<T>`). The facts record the point, so the order is available. The classification doesn't use it yet.

## 4. Stack frames

`frame::promote` turns every stack slot whose address doesn't escape into SSA values ([calls.md](calls.md)), and the rest live in a frame. In safe mode a safe frame is a byte array, aligned like a real frame:

```rust
#[repr(C, align(16))]
struct Frame([u8; 32]);
let mut frame = Frame([0; 32]);
let frame_base: u64 = frame.0.as_ptr() as u64;
...
frame.0[v8.wrapping_sub(frame_base) as usize..][..8].copy_from_slice(&rdi.to_le_bytes());
```

Frames over 4 KiB are a `Vec<u8>` instead. A slot whose address is passed to a callee is lent as `&mut frame.0[off..]` (stage 5). A frame that isn't safe keeps the fast-mode `[0u128; N]` array and raw accesses.

**Splitting.** The frame is one byte array, but the analysis splits it into objects, so one escaping object doesn't take the rest of the frame with it. Every fact about the frame covers a range of offsets: a load or store its width, a constant-length `memcpy`/`memset` its length, and a slot lent to a callee as much of it as the callee's summary says it reaches (`Pass::Borrow { len }`, the callee's `ParamBorrow::extent`). An access whose offset is unknown, or a callee whose reach is, covers everything from the lowest offset the pointer can have (`Origin::lo`) to the end of the frame. Overlapping ranges are one object, and each object is its own root, `Root::Frame(start)`, classified on its own. In `--emit borrows` the objects past the first one show up as `frame from N`. All the safe ones are still reached through `frame.0`. Lending several of them to one call groups them like slots of one object, splitting them in order of offset. An access to a raw object goes through `frame_base` as a raw pointer.

A call that lends a slice evaluates its other arguments first, into temporaries (`f(&mut s[..], s[8])` is E0502 otherwise).

On chungusite's own debug build (29,724 functions), splitting took bounds-checked accesses from 101,687 to 158,072, functions with no raw pointer from 18,713 to 19,780, and raw twins from 16,734 to 18,385 (more callers now lend a slice), with `--check` finding nothing to send back to fast mode.

## 5. Calls: summaries, twins, moves

`Program::summaries` runs the borrow analysis over the whole program, each call seeing its callee's **summary**: per argument, one of

| `Pass` | Meaning | Fact at the call |
|---|---|---|
| `Ignore` | the callee uses it as an integer and doesn't keep it | none |
| `Borrow { mutbl, nullable }` | the callee takes a slice | `Borrow`: the caller lends its root |
| `Access { write }` | read or written during the call only (`memcpy`, `memset`) | `Read` / `Write` |
| `Free` | freed (`free`, `operator delete`, `__rust_dealloc`) | `Free`: a move |
| `Escape` | anything else | `Escape` |

plus whether the result is a new allocation and which arguments it points into. A decompiled function's summary comes from its own analysis (a `&T`/`&mut T` argument is a `Borrow`, an integer that doesn't escape is `Ignore`); C library functions come from `libc::summary`. Summaries start optimistic (`Ignore` everywhere) and are recomputed for the callers of every function whose summary changed, until none does. They only get worse, so this terminates; after 64 rounds whatever still changes takes integers.

**Lending.** At a call whose callee borrows an argument, the caller passes a slice of the root that argument points into, starting at the argument:

```rust
unsafe { load(&frame.0[v5.wrapping_sub(frame_base) as usize..]) }
```

The callee sees the rest of the object, which is all a C callee can reach too (more than the object itself would be, so its bounds checks are looser than the C object's, never tighter). Two arguments lent from one root, one of them mutably, start at different known offsets (stage 6 rejects the rest), and are split apart:

```rust
unsafe { let __s = &mut heap23[v23.wrapping_sub(heap23_base) as usize..];
         let (__s3_0, __s) = __s.split_at_mut(v39.wrapping_sub(v23) as usize);
         let __s3_1 = __s; add_into(&mut *__s3_0, &*__s3_1) }
```

**Raw twins.** A call that can't lend a slice (the pointer has no origin, comes from several objects, from one that isn't safe, from a nullable argument, or conflicts with another loan) calls the callee's *raw twin*: `f_raw`, the same function emitted in fast mode, taking integers. So one caller that can't lend a slice doesn't take the slice away from all the others. Twins are also made for functions that data points to (vtables, callbacks: whoever calls through the pointer passes integers, so the static holds `f_raw`), and for the slice-taking callees of every twin and every function emitted in fast mode, since fast-mode code passes integers. A twin is emitted right after its function.

The twin still only accesses its arguments during the call (its summary says so), so the caller's objects don't escape through it: an argument the twin borrows whose root is safe in the caller is a `RawLend` fact, a read (and write) of the object that the loan rules leave alone, and the emitter passes a pointer made from the object's slice at the call, `(frame.0.as_mut_ptr() as u64).wrapping_add(off)`, instead of the address kept from earlier. Only the arguments that made the call raw lose their slices.

**Moves: `malloc` and `free` as `Box`.** An allocation that is safe (stage 3) and passes the move check (stage 6) is a `Box<[u8]>`:

```rust
let mut heap22: Box<[u8]> = Box::default();
let mut heap22_base: u64 = 0;
let v22: u64 = { heap22 = vec![0u8; (v11 as usize)].into_boxed_slice(); heap22_base = heap22.as_ptr() as u64; heap22_base };
heap22[v31.wrapping_sub(heap22_base) as usize..][..8].copy_from_slice(&rdi.to_le_bytes());
heap22 = Box::default(); // free(p)
```

`free` is a move out of the box (dropping it). An allocation that is never freed and never escapes is dropped at the end of the function, which nothing can observe. A function that frees its argument doesn't consume a `Box` yet: that needs passing `Box<[u8]>` by value, so such an argument escapes.

**Builtins.** `memcpy`/`memmove` and `memset` whose pointer arguments all have safe roots become `copy_from_slice`, `copy_within` (same root) and `fill`; if any of them hasn't, the call stays an FFI call, and each pointer argument with a safe root is a pointer made from its slice at the call, as for raw twins (one into a read-only static can't be written, so that one escapes).

Calls through GOT slots that the loader fills with a function in the binary (`R_X86_64_RELATIVE`, how PIE code calls a local function through the GOT) are direct calls to that function; they used to be indirect, which made every pointer passed to them escape.

## 6. Loans and moves as constraint solving

Stages 2–5 decide what *kind* of reference each pointer is. To emit code that rustc accepts, the borrows also have to be consistent with each other. That is the borrow checker's own problem run in reverse, and it fits Polonius's Datalog formulation. `loans.rs` runs the rules on [`datafrog`](https://crates.io/crates/datafrog), the engine Polonius itself uses:

```
contains(O, L, P)  :- loan_issued_at(O, L, P).
contains(O2, L, P) :- contains(O1, L, P), subset(O1, O2, P).
contains(O, L, Q)  :- contains(O, L, P), cfg_edge(P, Q), !loan_killed_at(L, P), origin_live_at(O, Q).
loan_live_at(L, P) :- contains(O, L, P), origin_live_at(O, P).
error(L, P)        :- invalidates(P, L), loan_live_at(L, P).

maybe_moved(X, Q)  :- moved_at(X, P), cfg_edge(P, Q).
maybe_moved(X, Q)  :- maybe_moved(X, P), cfg_edge(P, Q), !assigned_at(X, P).
move_error(X, P)   :- maybe_moved(X, P), accessed_at(X, P).
```

`borrow.rs` generates the input facts. In the output's model the long-lived references are the roots themselves (a parameter slice, the frame, a box), which rustc already accepts; the loans that can conflict are the reborrows lent to callees, each issued at its call, with an origin live at that call:

- `invalidates(P, L)` for another loan of the same root at the same call that overlaps it (an unknown offset, or the same one) where either is mutable. Shared loans never conflict, and loans at different known offsets are split with `split_at_mut` (stage 5).
- `moved_at` is a `free` of an allocation, `assigned_at` its `malloc`, and `accessed_at` every fact on it.

**Relaxation:** everything starts at the most precise assignment, and each error downgrades one step: a mutable loan in an error goes to the callee's raw twin (making its root raw), and an allocation with a move error stays a raw pointer from `malloc`. Then the classification of stage 3 runs again, and so do the rules. Each step only downgrades, so this terminates. `tests/program.rs` has the cases: two locals lent at once are split; the same local lent twice is downgraded; a use after free keeps the allocation raw.

**Lifetimes for signatures** are still elided: arguments are slices, and the return value is a `u64` address, which a caller indexes through its own root (the summary says which argument it points into). Returning `&'a [u8]` would need the `subset` facts for values that hold references across statements; the engine has the rule, the facts don't use it yet.

## 7. rustc as the oracle

The inference is sound in intent but complex. `--check` closes the loop:

1. Compile every emitted function with rustc (`--emit=metadata`), in batches of 400, in parallel. A batch has the `ffi` declarations, the statics (same names and types, zero contents, which rustc checks in a fraction of the time the real initializers take), the batch's functions, and a stub (`{ loop {} }`) for every other function so calls still resolve.
2. Map each error back to its function through the line of its primary span, and emit those functions in fast mode. Their callers are analysed again, since they now call integer-taking code, so everything is emitted again and re-checked, for up to 4 rounds.

On chungusite's own debug build `--check` takes about two minutes on 4 cores. It finds one function, and that function doesn't compile in fast mode either (a stack argument assigned to a `u32` variable, an emitter bug outside safe mode).

The DWARF comparison (measuring the precision and recall of inferred signatures against a corpus compiled from Rust with debug info) is still to do. design/fine-tuning.md sets up that corpus.

## Results

On a debug build of chungusite itself (20,025 functions, 18,131 lift), safe mode before and after this pipeline:

| | before | after |
|---|---:|---:|
| memory accesses bounds-checked | 18,025 of 129,655 (14%) | 47,535 of 104,122 (46%) |
| raw accesses through the frame | 62,470 | 40,629 |
| raw accesses through globals | 27,593 | 714 |
| raw accesses through arguments | 17,401 | 11,112 |
| raw accesses through other pointers (loaded, returned) | 4,166 | 4,132 |
| functions with no raw pointer at all | 4,919 (27%) | 11,746 (65%) |
| safe `fn`s (no `unsafe` in them or their callees) | 3,718 | 5,970 |
| raw twins, and raw accesses in them | | 5,250, 21,483 |

"Raw" counts loads, stores and copies through a raw pointer; the source of each comes from `sources.rs`, which classifies pointers the same way before and after. A function with no raw pointer has no raw access and no FFI or indirect call. There are fewer accesses in all because calls through GOT slots no longer load the slot (they are direct calls now). Counting the twins' raw accesses too, the output has 78,070 raw accesses instead of 111,630. The differential test (`tests/differential.rs`, 504 cases) passes as before, with new cases for lending two locals at once, the same local twice, a `Box` lent to a callee and freed, and `memset`/`memcpy` into a local.

Before, only arguments could be safe, and a function that some decompiled function called took every argument as an integer. Some of the old output was also unsound: functions that data points to (vtables, callbacks) took slices while being called through `extern "C" fn(u64, ..)` pointers, and a pointer into two objects was accessed raw while one of them was a `&mut` argument. Both now stay raw.

## Costs

- **Linear passes:** origins and facts each take one pass after the fixpoint, which needs about loop-depth + 2 RPO sweeps. Cleanup is linear per round.
- **Small state:** state is one small `Origin` per value, a points-to map per stored-to slot, and a fact list.
- **Datalog** is the only super-linear part, and it only runs on functions with loans or moves to check, on facts for those roots alone.
- **Whole program:** summaries iterate to a fixpoint, re-analysing only the callers of what changed, in parallel within each round. On the sample binary safe mode takes about 2.3 s instead of 1.4 s.
