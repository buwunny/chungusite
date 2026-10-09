# Training the type and name models

Everything here runs on your own machine. The decompiler only needs the result:
a directory with `model.onnx`, `tokenizer.json` and `labels.txt`, passed to
`chungusite <binary> --refine <dir>` (built with `--features ml`, see
[ml-runtime.md](../../docs/ml-runtime.md)).

| Script | What it does | Needs |
|---|---|---|
| `corpus.py` | compiles C projects with debug info and turns them into labeled rows | `gcc`/`clang`, `target/release/chungusite` |
| `train_types.py` | trains the type classifier and exports it to ONNX | PyTorch, ideally a GPU |
| `train_names.py` | trains a model that names arguments (no runtime hook yet) | PyTorch, a GPU |
| `evaluate.py` | scores an exported type model exactly as `--refine` runs it | `onnxruntime`, `tokenizers` |

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r tools/train/requirements.txt   # install torch for your GPU first
cargo build --release
```

## 1. Build a corpus

The labels come from debug info: `chungusite <lib> --emit dataset` writes one JSON
line per argument and return value of each function with a DWARF prototype. The
text is what `--refine` shows the model (the lifted function, then `var vN`) and
the label is the C type the source declared (`unsigned long`, `char *`,
`struct *`). `corpus.py` does this for whole projects, at every compiler and
optimisation level you ask for, deduplicates, and holds out whole projects for
testing so near-copies can't leak across the split.

```sh
python tools/train/corpus.py --out data -j 16 --cc gcc clang --opt O0 O1 O2 O3 Os \
    --project zlib-1.3.1.tar.gz --project lz4-1.10.0.tar.gz --project zstd-1.5.6.tar.gz \
    --projects-file more-projects.txt --binary /usr/lib/debug/.build-id/ab/cdef.debug
```

Each project is a source directory or an archive. Every `.c` file that compiles
on its own (with every directory holding a header on the include path) goes into
one shared library per build. Projects that need `./configure` or generated
headers first lose most of their files that way: build those yourself with
`CFLAGS="-g -O2"` and pass the binaries with `--binary`. Debian's `-dbgsym`
packages and Fedora's `-debuginfo` are another source of binaries with full
debug info.

More data matters more than anything else here. Our check run (lz4, xxhash,
brotli and the test corpus, gcc only, five levels) gives about 17,000 rows; a
useful model wants a few hundred thousand, which is on the order of a hundred
mid-sized C libraries at ten builds each. Good candidates build from a flat
source tree: zlib, lz4, zstd, xxhash, brotli, libpng, libjpeg-turbo, sqlite (the
amalgamation), lua, mbedtls, libyaml, cJSON, jansson, pcre2, libsodium, miniz,
stb, the redis and git sources.

`stats.json` lists the label histogram. The common labels are pointer and
integer types; `struct *` stands for any struct pointer, since the decompiler
builds struct layouts itself.

## 2. Train the type model

Fine-tune a pretrained code encoder (downloads from Hugging Face):

```sh
python tools/train/train_types.py --data data --out types_model \
    --model microsoft/unixcoder-base --epochs 3 --lr 3e-5 --batch 32 --int8
```

or train a small one from scratch with its own tokenizer:

```sh
python tools/train/train_types.py --data data --out types_small --from-scratch small --epochs 10 --int8
```

The script evaluates on the held-out projects after each epoch, keeps the best
checkpoint, and writes the `--refine` directory: `model.onnx` (int8 with
`--int8`, which is about four times smaller and faster on a CPU; the float model
stays as `model.fp32.onnx`), `tokenizer.json` with the sequence length,
`labels.txt` and `metrics.json`.

Rough sizes, as estimates rather than measurements: UniXcoder-base (125M
parameters) at `--max-len 256` takes on the order of an hour per few hundred
thousand rows on one 24 GB GPU, and a `small` model from scratch is a few times
faster. On a CPU only `--from-scratch tiny` is practical: 400 steps of batch 32
took about five minutes on 4 cores. A Mac uses `mps` automatically.

Options worth knowing: `--max-len` (tokens of context; the start of a long
function is cut, the `var vN` marker always survives), `--labels` and
`--min-count` (which types get their own class; the rest are `other`, which is
never proposed), `--grad-accum` for small GPUs, and `--max-steps` for a quick
run.

## 3. Check it

```sh
python tools/train/evaluate.py types_model data/types_test.jsonl --threshold 0.5 0.7 0.9
```

It runs the ONNX model with the same tokenization and truncation as the
decompiler and prints, per threshold, the share of variables that get a
proposal and how many of those are right, plus the most common mistakes. Pick
`--refine-threshold` from it: the decompiler turns down proposals that
contradict how the code uses a value, so its precision is at least this.

Then try it on a real binary without its debug info and compare with a normal
run, which uses the DWARF types:

```sh
cargo build --release --features ml
export ORT_DYLIB_PATH=$(ls $(python -c "import onnxruntime, os; print(os.path.dirname(onnxruntime.__file__))")/capi/libonnxruntime.so.*)
target/release/chungusite libfoo.so --no-dwarf --refine types_model --refine-threshold 0.7 -o refined.rs
```

stderr counts the proposals used and turned down.

## 4. The name model

```sh
python tools/train/train_names.py --data data --out names_model \
    --model Salesforce/codet5p-220m --epochs 5 --lr 5e-5 --batch 16
```

It learns to write `v3: buf; v5: len` from a function's text, and reports
exact-name accuracy on held-out projects. It saves a Hugging Face checkpoint;
the decompiler doesn't call a name model yet, so this is for experiments until
that hook exists.

## The pipeline check

Every script was run end to end on CPU with tiny models: the corpus above, a
`--from-scratch tiny` type model exported to int8 ONNX and loaded by
`--refine`, `evaluate.py` reproducing the training script's own numbers, and a
tiny name model. Those models are only a check that the pieces fit; their
predictions mean nothing.
