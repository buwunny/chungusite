# ML runtime: from IR to an ONNX model

`src/refine/` turns a lifted `Function` into model input and runs it through ONNX Runtime. The model only *proposes* names and types. Nothing here changes the IR. Type proposals reach the decompiler through `types::TypeModel` ([types.md](types.md#the-model-hook)), which checks each one against what the code does before using it.

## Building

Default builds don't include any of this. The only dependency stays `iced-x86`. Turn it on with the `ml` feature:

```sh
ORT_DYLIB_PATH=/path/to/libonnxruntime.so cargo run --features ml --example refine -- tests/fixtures/tiny-types
```

`ort` (pinned to `=2.0.0-rc.13`, because its 2.0 release candidates change API) is built with `load-dynamic`. Building never downloads anything, and `libonnxruntime` is loaded at runtime from `ORT_DYLIB_PATH`. It needs ONNX Runtime 1.24 or newer (tested with 1.30), and any build works, including CPU, CUDA and DirectML. The `libonnxruntime.so` inside the `onnxruntime` Python wheel works too.

## Session setup

`refine::infer::Refiner::new(model_path, threads, pad_id, max_len)` wraps `Session::builder()` with full graph optimisation and a fixed intra-op thread count, then commits the `.onnx` file. `classify(&[&[u32]])` pads a batch to the next power of two (16–`max_len`) in reused buffers. It passes them to ONNX Runtime as borrowed tensors (`input_ids`, `attention_mask`) and returns the `logits` output. Use one `Refiner` per worker thread with `threads = 1`, or one shared `Refiner` with more threads, but not both.

## What the model sees

`refine::to_text(&f)` produces compact C-like text. For the null-check sample in `examples/refine.rs`:

```
bb0(v11, v0):
 v1 = 0;
 v2 = v0 == v1;
 if v2 goto bb2(); else goto bb1(v11, v0);
bb1(v3, v4):
 v5 = v4 + 8;
 *v5 = v3;
 v7 = v4 + 16;
 v8 = *v7;
 return v8;
bb2:
 v9 = 0;
 v10 = zext(v9);
 return v10;
```

Variables are SSA values (`vN`), not registers. `rax` holds unrelated things at different points in a function, while each `vN` is one value, and that value is what gets a name. The same `vN` appears at every use, which is how the model sees data flow.

`refine::to_ids(&f, &mut cache, &mut ids)` produces the same thing directly as token ids, without building the string. The serializer writes text in exactly the pieces the tokenizer's pre-tokenizer splits on (` v`, `12`, ` =`, `);` …). That makes the token ids of a whole function the concatenation of each piece's ids, and `TokenCache` computes those once from the model's own `tokenizer.json`. Two details matter:
- A run of punctuation is a single piece, so `);` is never `)` followed by `;`.
- `TokenCache::from_tokenizer_file` switches off any truncation or padding saved in `tokenizer.json`, because a saved limit would otherwise be applied to each piece. The fixture's tokenizer has one: left truncation at 192 tokens.

## `--refine`

`chungusite <binary> --refine <dir>` (built with `--features ml`) asks a trained type classifier about every argument and return value of each function without a usable prototype, through `types::TypeModel` ([types.md](types.md#the-model-hook)). `refine::model::TypeClassifier` is the implementation. `<dir>` holds what `train_types.py` saves, exported to ONNX:

- `model.onnx`: inputs `input_ids` and `attention_mask`, output `logits` (`[batch, classes]`).
- `tokenizer.json`: the model's own tokenizer. Its saved truncation length is the sequence length (256 if it has none), and its post-processor's special tokens (`<s>`, `</s>`) go around each row.
- The labels, class 0 first: `labels.txt` (one per line), or else `config.json`'s `id2label`. Labels are C type names as DWARF spells them (`int`, `unsigned char`, `char *`). A label `parse_label` doesn't know (`other`, a struct name) is never proposed, so set `id2label` when training: the default `LABEL_0` names nothing.

Each question is one row: the function's text, then `var vN` for the value asked about (`refine::to_var_text`; the text already ends in a newline). That is the format `train_types.py` documents for its JSONL: `{"text": "<function>\nvar v7", "label": 12}`. Like training, a row longer than the model takes is cut from the left, so the marker always survives. The highest-probability class is proposed if its softmax probability is at least `--refine-threshold` (default 0.5), and the gate then accepts or rejects it against the facts. The summary on stderr counts both.

Loading checks that the library loads (a clear error naming `ORT_DYLIB_PATH` rather than a panic inside `ort`) and that the model has one class per label. Questions are asked in parallel: each worker thread makes its own session (one intra-op thread) on first use, so memory grows with `-j`. If inference fails partway, a warning is printed once and decompilation continues without the model.

The cost is one row per question. On chungusite's own debug build with `--no-dwarf` that is 89,258 rows; with the 550 KB fixture model the run takes 74 s on 4 shared cores instead of 6 s, and a real model will cost more per row. With debug info, only functions without a prototype are asked.

## Tests

- `tests/refine_exact.rs` (with `--features ml`) checks that cached ids equal `tokenizer.encode(full_text)` for 1,500 random lifted programs. Run the same check whenever you change the serializer or swap tokenizers.
- `tests/refine_model.rs` checks the `var vN` row format, and (ignored, needs `ORT_DYLIB_PATH`) that `TypeClassifier`'s proposals and probabilities match the same rows run through `tokenizers` and `onnxruntime` in Python, including one cut from 1,162 tokens to the fixture's 192, and that the threshold holds back proposals below it. The fixture's `labels.txt` names its two classes `int` and `char *`.
- `tests/refine_ort.rs` is marked `#[ignore]` because it needs `ORT_DYLIB_PATH`. It runs the fixture model through `ort` and compares ids and logits against values computed independently in Python (`tokenizers` and `onnxruntime` 1.30). Run it with `cargo test --features ml -- --include-ignored`.

`tests/fixtures/tiny-types` is a 550 KB stand-in: a 2-layer random-initialised RoBERTa classifier briefly trained on synthetic labels, with a 2,000-token byte-level BPE. It exists to test the plumbing, and its predictions mean nothing. Real models are trained with the scripts and recipe described in the project's fine-tuning notes.
