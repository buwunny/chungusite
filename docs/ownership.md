# Safe mode: inferring moves, `&T` and `&mut T`

`--mode safe` has to decide, for every pointer the binary handles, which Rust form to emit:

- a move (the value is owned and ownership is transferred),
- a shared borrow `&T`,
- a mutable borrow `&mut T`,
- or, when none of these can be proven, a raw pointer inside `unsafe`.

This document is the plan for the whole pipeline. The first stages are implemented and tested on this branch:

| Stage | Code | Status |
|---|---|---|
| 0. CFG views: predecessors, reverse postorder, dominators | [`src/cfg.rs`](../src/cfg.rs) | done |
| 1. Clean SSA: trivial block params, dead code | [`src/opt.rs`](../src/opt.rs) | done |
| 2. Origins: which argument and offset each value points into | [`src/borrow.rs`](../src/borrow.rs) | done, for arguments and RSP |
| 3. Access facts and per-argument classes (`&`, `&mut`, raw, nullable) | [`src/borrow.rs`](../src/borrow.rs) | done |
| 4. Stack slots: promote to SSA, or keep as borrowed locals | `borrow::Analysis::stack_slots`, `frame::promote` | unescaped slots promoted to SSA; the rest live in a `frame` array, not yet borrowed |
| 5. Call summaries and moves | — | needs `CALL` in the lifter |
| 6. Loans and lifetimes (Polonius-style constraint solving) | — | design below |
| 7. Check with rustc, downgrade on failure | — | design below |

The guiding rule is that **every unproven step falls back to a raw pointer, never to a guess.** A raw pointer compiles and behaves like the binary, and a wrong `&mut` is undefined behaviour. The analysis may be incomplete, but it must stay sound.

## 0–1. A clean SSA to analyse

The lifter already gives SSA with block parameters instead of phi nodes ([lift.md](lift.md)). It creates a parameter for every register that is live into a block, so loops carry many parameters that never change. `opt::clean` removes them:

1. **Trivial parameters** (Braun et al. 2013). A parameter whose incoming arguments are all one value `v`, or the parameter itself, is replaced by `v` and dropped from the block and from every edge. Removing one can make another trivial, so this repeats to a fixpoint. Blocks with a single predecessor lose all their parameters this way. The entry block is skipped, because its parameters are the function's arguments.
2. **Dead code.** A mark phase starts from side effects (stores, calls, terminators) and from the entry parameters, and propagates through operands. A live parameter marks its incoming edge arguments. Unmarked pure instructions and parameters are removed.

Ids stay stable, and removed instructions just leave their block's list. On the loop from `lift.md`, this takes `bb1` from four parameters to two (the accumulator and the counter), and `bb2`/`bb3` to none. `tests/safe.rs` checks the exact output, and `tests/robust.rs` cleans and re-verifies 2,000 random programs.

`cfg::Cfg` gives predecessors, reverse postorder and immediate dominators (Cooper, Harvey and Kennedy) from flat arrays. Later stages use it in two ways. Dataflow iterates in RPO, which converges in a few passes. Loans (stage 6) need dominance to decide where a borrow is live.

## 2. Origins: what each value points into

The first question is not "is this a reference?" but "*which object and which offset* does this value point into?" That is a forward dataflow analysis over the SSA graph:

- **Roots.** Each entry parameter `k` is a possible pointer root with origin `{roots: 1<<k, off: 0}`. That includes RSP, which makes the stack frame just another root. Later, `malloc`-style calls and globals become roots too (stage 5).
- **Lattice.** An origin is a set of roots (a bitmask) plus an offset, either `Known(i64)` or `Unknown`. Join takes the union of the roots. The offset survives only if both sides agree. That is finite height, so it terminates.
- **Transfer.**
  - `PtrOffset base+disp` shifts by `disp`.
  - An index makes the offset `Unknown`, which is array or slice access.
  - `Add`/`Sub` by a constant shift the offset. `Add` with a non-constant gives `Unknown`.
  - `Select` joins its two inputs.
  - Block parameters join their incoming edge arguments.
  - Everything else (loads, casts, bit operations) has no origin.
- **Fixpoint.** Iterate blocks in RPO until nothing changes. The transfer functions are monotone. One detail matters: "no origin" must have exactly one representation. Shifting it must not invent an offset, or a decrementing loop counter never converges. The random-program test found that bug.

**Loads have no origin yet.** Following pointers stored in memory (`p->next`) needs a points-to model of the heap. Stage 6 adds one per field, keyed on `(root, offset)`.

## 3. Access facts, and choosing `&T` or `&mut T`

One pass over the reachable blocks turns uses of derived values into facts about their roots. Each fact is a tuple `(root, offset, kind, at)`:

| IR | Fact |
|---|---|
| `load ptr` | `Read(root(ptr), off)` |
| `store ptr <- val` | `Write(root(ptr), off)`; `Escape(root(val))` if `val` is itself derived |
| `memcpy dst, src` | `Write(dst)`, `Read(src)` |
| `call f(args)`, tail call | `Escape` for every derived argument, until summaries exist (stage 5) |
| `ret v` | `Return(root(v))` |
| `cmp.eq/ne p, 0` at offset 0 | `NullCheck(root(p))` |
| any other operation on a derived value | `Escape` (it leaves what we can track) |
| `p - q` of two derived values | nothing (a length, like `offset_from`) |
| `p < end` and other comparisons | nothing |

Each argument then gets a class:

```
no Read and no Write               -> integer      (no deref, so no evidence of a pointer)
Escape                             -> *const T / *mut T  (raw: lifetime unknown)
Write                              -> &mut T
otherwise                          -> &T
NullCheck                          -> wrap in Option<..>
Return                             -> the return value borrows from this argument
```

**Pointers need dereference evidence.** Compilers use `lea` and `add` for integer arithmetic, so being offset or returned doesn't make a value a pointer. Only a dereference of something derived from it does.

The accesses also give the pointee's shape. Constant offsets become struct fields: `fields: [(0, written), (8, read)]`. An access at an unknown offset (`indexed`) is evidence for a slice or array. The ML type model (design/ml-refinement.md) proposes types, and these facts gate its proposals.

Run on the null-check sample (`chungusite --hex "48 85 ff 74 09 48 89 77 08 48 8b 47 10 c3 31 c0 c3" --emit borrows`):

```
  rsi: integer
  rdi: Option<&mut T>, fields at +8 (written), +16
```

i.e. `fn f(p: Option<&mut S>, v: u64) -> u64`, with `S` having fields at 8 (written) and 16 (read).

**What this stage does not prove yet:**

- **Aliasing.** Two `&mut` arguments are only valid if they never point to the same memory. The callee can't prove that; the call site can. Stage 6 records it as a requirement on callers.
- **Null-check dominance.** `nullable` only says that a null check exists. To emit `if let Some(p) = p`, every dereference must be dominated by the non-null edge of that branch. That check uses `Cfg::dominates`. Otherwise the argument stays raw.
- **Reads before writes.** An argument written before it is ever read could be an out-parameter (`&mut MaybeUninit<T>`). The facts record `at`, so the order is available. The classification doesn't use it yet.

## 4. Stack offsets

The stack is the RSP root. `sub rsp, 0x18` is a constant `Sub`, so every slot gets a known offset from the entry RSP, and `Analysis::stack_slots` lists them with read, write and address-taken flags.

- **Slot with its address never taken** (only accessed directly at a known offset): promote it to an SSA value (mem2reg). This is what turns `-O0` spill code into plain variables. Run promotion before stage 3, so that a spill like `mov [rsp+8], rdi` doesn't count as `rdi` escaping.
- **Slot with its address taken** (`lea rax, [rsp+0x10]` that then escapes or is passed on): it becomes a real Rust local, and the borrow taken is `&local` or `&mut local`. Which one depends on what the receiver does, which is the same question as for arguments: a callee's summary in stage 5, or uses in this function.
- **Any RSP access at an unknown offset** (a variable-length array, alloca): `stack_slots` returns `None` and the whole frame stays raw.

RBP-based frames need one more rule: `mov rbp, rsp` is just a copy, so RBP inherits RSP's origin, which needs no extra code. Push and pop need RSP tracking in the lifter first.

## 5. Moves need call summaries

Rust's move semantics are about ownership transfer. A binary shows ownership mostly at call boundaries:

- `p = malloc(n)` / `__rust_alloc` / `operator new` creates an **owned** root.
- `free(p)` / `__rust_dealloc` / a `drop_in_place` callee **consumes** it.

So moves are an *interprocedural* question. The plan:

1. **Lift `CALL`** with the SysV ABI: arguments in rdi, rsi, rdx, rcx, r8, r9; caller-saved registers clobbered; result in rax.
2. **Build a summary for each function**, bottom-up over the call graph's strongly connected components. Within a component, iterate to a fixpoint starting from the most optimistic summary. A summary holds, per argument:
   - the class from stage 3 (shared, mut, raw),
   - `consumes`, which holds when every path frees it or passes it to a consuming argument,
   - `returns_from`, which says which argument the return value borrows from (lifetime elision).
   Seed known imports by hand: `free` consumes, `memcpy` reads `src` and writes `dst`, `strlen` takes `&`.
3. **At a call site**, apply the callee's summary instead of the blanket `Escape`. An argument passed to a `&` parameter is a `Read` at the call, and to a `&mut` parameter it is a `Write`. To a consuming parameter it is a `Move` fact. That turns most of today's `Raw` results into real borrows.
4. **Classify owned roots.** A root that comes from an allocator and is consumed exactly once on every path is `Box<T>` (or `Vec<T>` when it's indexed and its size is recorded). Passing it to a consuming callee is a **move**. A read-only use before that is an `&*b` reborrow. If it is consumed twice on some path (double free) or never consumed (a leak, or it is stored somewhere), it stays raw.
   - "Exactly once on every path" is a forward must-analysis over the CFG. The state per owned root is `Owned | Moved | Maybe` and the join is `Owned ⊔ Moved = Maybe`. A use or consume in state `Maybe` or `Moved` downgrades the root to raw.

For an argument, a move means its summary says `consumes`. Emit it as `T` by value (or `Box<T>`) instead of a reference.

## 6. Loans and lifetimes as constraint solving

Stages 2–5 decide what *kind* of reference each pointer is. To emit code that rustc accepts, references inside a function also need lifetimes that don't conflict. That is the borrow checker's own problem run in reverse, and it fits a Polonius-style Datalog formulation:

**Input facts**, generated from the SSA, CFG and stage 3:

```
loan_issued_at(L, origin, point)      // a borrow of (root, offset) is created at this instruction
loan_kind(L, Shared | Mut)
loan_killed_at(L, point)              // the borrowed place is overwritten
origin_live_at(origin, point)         // some value holding the reference is live here
subset(o1, o2, point)                 // a reference flows from o1 to o2 (copy, block param, return)
invalidates(point, L)                 // a write or move conflicts with loan L
```

`origin_live_at` comes from SSA liveness, which is a backward dataflow over the CFG. `subset` edges come from the same def-use chains that stage 2 walks.

**Rules** are standard Polonius:

```
origin_contains_loan_at(O, L, P) :- loan_issued_at(L, O, P).
origin_contains_loan_at(O2, L, P) :- origin_contains_loan_at(O1, L, P), subset(O1, O2, P).
origin_contains_loan_at(O, L, Q) :- origin_contains_loan_at(O, L, P), cfg_edge(P, Q),
                                    !loan_killed_at(L, P), origin_live_at(O, Q).
loan_live_at(L, P) :- origin_contains_loan_at(O, L, P), origin_live_at(O, P).
error(L, P) :- invalidates(P, L), loan_live_at(L, P).
```

They run in the [`datafrog`](https://crates.io/crates/datafrog) crate, the engine Polonius itself uses. It needs no allocation per tuple and handles functions of this size quickly.

**Solving by relaxation:** start from the most precise assignment (every candidate `&mut` or `&`) and run the rules. For each `error(L, P)`, downgrade the weakest loan involved one step:

- `&mut` becomes a raw `*mut`, or a `&Cell<T>` / `&RefCell<T>` when there are only shared writes,
- `&` becomes `*const`.

Then re-run. Each step only downgrades, so this terminates. The result is the largest set of references that is consistent with Rust's rules.

**Lifetimes for signatures** come out of the same solution:

- A return derived from exactly one reference argument gets the elided form `fn(&'a T) -> &'a U`.
- One derived from several gets an explicit shared `'a`.
- One derived from none is owned, or `'static` if it comes from a global.

Inside a function, NLL-style regions need no annotation. Only signatures and structs that store references need names.

## 7. rustc as the oracle

The inference is sound in intent but complex. The last stage closes the loop:

1. Emit the function in safe form and type-check it in a scratch crate with `cargo check`. Batch many functions per crate, since rustc start-up dominates.
2. Map each borrowck error (E0499, E0502, E0505, E0506) back to its loan through the span. Downgrade that loan as in stage 6, and re-check.
3. Downgrade the whole function to `--mode fast` output if it still doesn't compile after a bounded number of rounds.

This also checks the inference itself. In a test corpus compiled from Rust with debug info, comparing inferred signatures with the DWARF ones measures precision (how many `&mut` were really `&mut`) and recall (how many references we failed to recover). design/fine-tuning.md sets up that same corpus pipeline.

## Costs

- **Linear passes:** origins and facts each take one pass after the fixpoint, which needs about loop-depth + 2 RPO sweeps. Cleanup is linear per round.
- **Small state:** state is one small `Origin` per value plus a fact list, with no hash maps.
- **Datalog** (stage 6) is the only super-linear part, and it runs per function on facts for pointers only, which are a small fraction of values.
- **Parallel:** everything except bottom-up summaries is per-function, so it parallelises across functions just like the lifter. Summaries parallelise within each level of the call-graph SCC DAG.
