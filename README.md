# chungusite

An x86_64 to Rust decompiler.

```
cargo install --path .
chungusite ./prog --list                  # which functions lift
chungusite ./prog --mode safe -o prog.rs  # decompile to Rust
```

`--mode fast` emits raw pointers in `unsafe`. `--mode safe` turns pointer arguments it can prove into `&[u8]` / `&mut [u8]` with bounds-checked accesses, and leaves the rest raw.

- [docs/cli.md](docs/cli.md): the command and what its output looks like
- [docs/roadmap.md](docs/roadmap.md): what's left before it handles real-world binaries
- [docs/ir.md](docs/ir.md), [docs/lift.md](docs/lift.md), [docs/ownership.md](docs/ownership.md): the design
