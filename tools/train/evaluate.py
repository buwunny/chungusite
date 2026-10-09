#!/usr/bin/env python3
"""Score an exported type model the way `chungusite --refine` runs it.

Loads model.onnx, tokenizer.json and labels.txt from a model directory with
onnxruntime and tokenizers only (no PyTorch), builds each row as the decompiler
does (the text's ids cut from the left to the saved length, between the
tokenizer's special tokens), and reports, for each threshold, how many
variables would get a proposal and how many of those are right.

  python tools/train/evaluate.py types_model data/types_test.jsonl --threshold 0.5 0.7 0.9

The decompiler then checks each proposal against what the code does and turns
some down, so its precision is at least what this reports. For that, run
`chungusite <binary> --no-dwarf --refine types_model` on a binary with debug info
and compare with a normal run.
"""
import argparse, collections, json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("model", type=Path)
p.add_argument("rows", type=Path, help="types_test.jsonl from corpus.py")
p.add_argument("--threshold", type=float, nargs="+", default=[0.5, 0.7, 0.9])
p.add_argument("--batch", type=int, default=32)
p.add_argument("--limit", type=int, default=0, help="score only this many rows")
a = p.parse_args()

tok = Tokenizer.from_file(str(a.model / "tokenizer.json"))
max_len = tok.truncation["max_length"] if tok.truncation else 256
tok.no_truncation()
labels_txt = a.model / "labels.txt"
labels = (labels_txt.read_text().splitlines() if labels_txt.exists()
          else [l for _, l in sorted(json.loads((a.model / "config.json").read_text())["id2label"].items(), key=lambda x: int(x[0]))])
full, plain = tok.encode("x").ids, tok.encode("x", add_special_tokens=False).ids
at = next(i for i in range(len(full)) if full[i:i + len(plain)] == plain)
prefix, suffix = full[:at], full[at + len(plain):]
room = max_len - len(prefix) - len(suffix)
pad = tok.token_to_id("<pad>") or 0

opts = ort.SessionOptions()
sess = ort.InferenceSession(str(a.model / "model.onnx"), opts, providers=["CPUExecutionProvider"])
rows = [json.loads(l) for l in open(a.rows)]
if a.limit:
    rows = rows[:a.limit]
print(f"{len(rows)} rows, {len(labels)} classes, {max_len} tokens")

pred, conf = [], []
for s in range(0, len(rows), a.batch):
    part = rows[s:s + a.batch]
    seqs = [prefix + e.ids[-room:] + suffix for e in tok.encode_batch([r["text"][-max_len * 16:] for r in part], add_special_tokens=False)]
    n = max(map(len, seqs))
    ids = np.full((len(seqs), n), pad, dtype=np.int64)
    mask = np.zeros((len(seqs), n), dtype=np.int64)
    for i, q in enumerate(seqs):
        ids[i, :len(q)], mask[i, :len(q)] = q, 1
    logits = sess.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]
    e = np.exp(logits - logits.max(-1, keepdims=True))
    prob = e / e.sum(-1, keepdims=True)
    pred += list(prob.argmax(-1))
    conf += list(prob.max(-1))

gold = [r["label"] for r in rows]
got = [labels[i] for i in pred]
known = set(labels[1:])
out = {"rows": len(rows), "accuracy": sum(g == x for g, x in zip(gold, got)) / len(rows),
       "rows_with_a_known_label": sum(g in known for g in gold) / len(rows)}
for th in a.threshold:
    kept = [(g, x) for g, x, c in zip(gold, got, conf) if c >= th and x != labels[0]]
    out[f"proposed@{th}"] = round(len(kept) / len(rows), 4)
    out[f"precision@{th}"] = round(sum(g == x for g, x in kept) / max(1, len(kept)), 4)
wrong = collections.Counter((g, x) for g, x, c in zip(gold, got, conf) if c >= a.threshold[0] and x != labels[0] and g != x)
out["most_common_mistakes"] = [f"{g} proposed as {x}: {n}" for (g, x), n in wrong.most_common(10)]
print(json.dumps(out, indent=1))
