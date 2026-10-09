# Running chungusite in DecBench

[DecBench](https://github.com/Noelo-Lab/decbench) ([leaderboard](https://decbench.com)) compiles C projects with debug info, decompiles the binaries, and scores each function against its source. This directory adds chungusite to it as a backend named `chungusite`.

## What gets scored

Every DecBench metric reads C. GED parses the decompiled code with Joern's C frontend, which finds no functions in Rust. So `chungusite_raw.py` hands DecBench a C *control-flow skeleton* of each emitted Rust function: the same `if`/`while`/`loop`/`match`/`break`/`continue`/`return` structure, labeled blocks and loops as `goto`s, the signature and `let` types mapped to C, and every other statement replaced by `x = 0;`. The Rust is still written to the output directory.

| Metric | Meaning for chungusite |
|---|---|
| `ged` | Valid. It compares CFG topology only, and Joern's CFG is block-level (calls and `&&`/`\|\|` don't split blocks), so the skeleton scores the same as a full translation would. |
| `type_match` | Partial. It goes through DecBench's C-signature fallback, so argument types and `let` types count; statement-level evidence doesn't exist. |
| `byte_match` | Not meaningful: there is no C to recompile. Leave it out. |

## Setup

```sh
# 1. chungusite on PATH (or set CHUNGUSITE_BIN=/path/to/chungusite)
cargo install --path .

# 2. DecBench, in a venv (Python 3.10+, Java for Joern, gcc)
git clone https://github.com/Noelo-Lab/decbench ~/src/decbench
python3 -m venv ~/src/decbench/.venv && . ~/src/decbench/.venv/bin/activate
pip install -e ~/src/decbench          # the first GED run downloads Joern (~1.8 GB)

# 3. Register the backend (links this file into the checkout; safe to re-run)
python tools/decbench/install.py ~/src/decbench
decbench list-decompilers             # chungusite: Available = Y
```

## Run

```sh
cd ~/src/decbench
decbench run projects/sailr/bzip2.toml -O O2 -d chungusite -d angr -m ged -m type_match -o results/bzip2
decbench report results/bzip2/scoreboard.toml -o results/bzip2/report.html
```

`decbench run` builds the project, decompiles every binary with each `-d`, and prints the scoreboard. Results land in `results/bzip2/`: `scoreboard.toml` (per-metric scores per decompiler), `function_results.json`, the per-function TOMLs under `O2/bzip2/evaluated/`, and chungusite's Rust as `O2/bzip2/decompiled/.../chungusite_<binary>.rs`. angr is the one comparison backend that needs no install or license; add `-d ghidra` with `GHIDRA_INSTALL_DIR` set.

`decbench run` gives the decompiler the binary with its symbols and DWARF, which chungusite uses for names and types. For numbers comparable to the leaderboard, use the driver the leaderboard uses, which strips the binary first:

```sh
python scripts/compile_all.py results/run 8 bzip2 gzip      # out dir, workers, projects
DECBENCH_DECOMPILERS=chungusite,angr python scripts/run_benchmark.py results/run -- bzip2 gzip
```

Settings: `CHUNGUSITE_MODE=safe` (default `fast`; safe mode also runs `--check` unless `CHUNGUSITE_CHECK=0`), or per version in `~/.config/decbench/decompilers.toml`:

```toml
[chungusite.versions.default]
bin = "/path/to/chungusite"
mode = "fast"
```

## Tests

```sh
python3 tools/decbench/test_chungusite_raw.py
```

They check the Rust-to-C translation and need neither DecBench nor a build.
