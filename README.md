# chungusite

An x86_64 to Rust decompiler.

```
cargo install --path .
chungusite ./prog --list                  # which functions lift
chungusite ./prog --mode safe -o prog.rs  # decompile to Rust
chungusite ./prog --cargo prog-rs         # ... as a Cargo project: cd prog-rs && cargo run
```

`--mode fast` emits raw pointers in `unsafe`. `--mode safe` turns the objects it can prove things about (pointer arguments, the stack frame, read-only globals, `malloc`'d buffers) into `&[u8]` / `&mut [u8]` / `Box<[u8]>` with bounds-checked accesses, lends them across calls, and leaves the rest raw; `--check` compiles the result with rustc and falls back to fast mode for whatever it rejects.

- [docs/cli.md](docs/cli.md): the command and what its output looks like
- [tools/decbench](tools/decbench/README.md): scoring it in the DecBench decompiler benchmark
- [docs/roadmap.md](docs/roadmap.md): what's left before it handles real-world binaries
- [docs/ir.md](docs/ir.md), [docs/lift.md](docs/lift.md), [docs/calls.md](docs/calls.md), [docs/ownership.md](docs/ownership.md): the design
