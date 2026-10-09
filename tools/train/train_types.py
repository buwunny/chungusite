#!/usr/bin/env python3
"""Train the type classifier that `chungusite --refine` runs, and export it.

Reads corpus.py's types_train.jsonl / types_test.jsonl ({"text": "<function>\\nvar v7",
"label": "unsigned int", ...}) and writes a directory `--refine` loads as is:

  model.onnx       inputs input_ids, attention_mask; output logits [batch, classes]
  tokenizer.json   with the sequence length saved as its (left) truncation
  labels.txt       one C type per class, class 0 first ("other" is never proposed)
  config.json, metrics.json, and the PyTorch checkpoint for further training

Fine-tune a pretrained encoder (the recipe: UniXcoder, about an hour on one 24 GB GPU
for a few hundred thousand rows):

  python tools/train/train_types.py --data data --out types_model \\
      --model microsoft/unixcoder-base --epochs 3 --lr 3e-5 --batch 32

or train a small one from scratch, with its own tokenizer (no download; minutes on a
CPU for a smoke test, but expect much less accuracy than a pretrained encoder):

  python tools/train/train_types.py --data data --out types_tiny --from-scratch tiny
"""
import argparse, collections, json, math, os, random, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--data", required=True, type=Path, help="corpus.py's --out directory")
p.add_argument("--out", required=True, type=Path)
g = p.add_mutually_exclusive_group(required=True)
g.add_argument("--model", help="pretrained encoder to fine-tune (Hugging Face id or local path)")
g.add_argument("--from-scratch", choices=["tiny", "small", "base"], help="random-initialised RoBERTa of this size")
p.add_argument("--max-len", type=int, default=256)
p.add_argument("--labels", type=int, default=200, help="most frequent labels kept; the rest are `other`")
p.add_argument("--min-count", type=int, default=20, help="labels seen fewer times are `other`")
p.add_argument("--epochs", type=float, default=3)
p.add_argument("--lr", type=float)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--grad-accum", type=int, default=1)
p.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (0: no limit)")
p.add_argument("--vocab", type=int, default=8000, help="--from-scratch: BPE vocabulary size")
p.add_argument("--int8", action="store_true", help="quantize: model.onnx becomes int8 (about 4x smaller and faster on a CPU), the float one model.fp32.onnx")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
torch.manual_seed(a.seed)
random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
a.out.mkdir(parents=True, exist_ok=True)

read = lambda f: [json.loads(l) for l in open(a.data / f)]
train, test = read("types_train.jsonl"), read("types_test.jsonl")
counts = collections.Counter(r["label"] for r in train)
labels = ["other"] + [l for l, n in counts.most_common(a.labels) if n >= a.min_count and l != "other"]
index = {l: i for i, l in enumerate(labels)}
y = lambda r: index.get(r["label"], 0)
# Only the end of a text reaches the model (the tokenizer cuts from the left), and a
# token is rarely more than a dozen characters: cut long functions before tokenizing.
tail = lambda t: t[-a.max_len * 16:]
print(f"{len(train)} train / {len(test)} test rows, {len(labels)} classes on {dev}")

# ---- tokenizer and model ------------------------------------------------------------
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerFast, RobertaConfig, RobertaForSequenceClassification

if a.model:
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=index, ignore_mismatched_sizes=True)
    lr = a.lr or 3e-5
else:
    from tokenizers import ByteLevelBPETokenizer, processors
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator((tail(r["text"]) for r in train), vocab_size=a.vocab, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    bpe.post_processor = processors.RobertaProcessing(("</s>", bpe.token_to_id("</s>")), ("<s>", bpe.token_to_id("<s>")))
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe._tokenizer, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>", mask_token="<mask>")
    size = {
        "tiny": dict(hidden_size=128, num_hidden_layers=2, num_attention_heads=4, intermediate_size=512),
        "small": dict(hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024),
        "base": dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072),
    }[a.from_scratch]
    cfg = RobertaConfig(vocab_size=len(tok), max_position_embeddings=a.max_len + 2, pad_token_id=tok.pad_token_id,
                        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, type_vocab_size=1,
                        num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=index, **size)
    model = RobertaForSequenceClassification(cfg)
    lr = a.lr or 5e-4
model.to(dev)

# What `--refine` does with a row: the text's ids, cut from the left so the
# `var vN` marker survives, between the tokenizer's own special tokens.
specials = tok("x")["input_ids"]
plain = tok("x", add_special_tokens=False)["input_ids"]
at = next(i for i in range(len(specials)) if specials[i:i + len(plain)] == plain)
prefix, suffix = specials[:at], specials[at + len(plain):]
room = a.max_len - len(prefix) - len(suffix)


def encode(rows):
    ids = tok([tail(r["text"]) for r in rows], add_special_tokens=False)["input_ids"]
    return [prefix + x[-room:] + suffix for x in ids]


def batches(rows, shuffle):
    enc = encode(rows)
    ys = [y(r) for r in rows]
    order = list(range(len(rows)))
    if shuffle:
        random.shuffle(order)
    else:  # similar lengths together: less padding
        order.sort(key=lambda i: len(enc[i]))
    for s in range(0, len(order), a.batch):
        idx = order[s:s + a.batch]
        n = max(len(enc[i]) for i in idx)
        ids = torch.full((len(idx), n), tok.pad_token_id)
        mask = torch.zeros((len(idx), n), dtype=torch.long)
        for k, i in enumerate(idx):
            ids[k, :len(enc[i])] = torch.tensor(enc[i])
            mask[k, :len(enc[i])] = 1
        yield idx, ids.to(dev), mask.to(dev), torch.tensor([ys[i] for i in idx]).to(dev)


@torch.no_grad()
def evaluate(rows):
    model.eval()
    probs = [None] * len(rows)
    for idx, ids, mask, _ in batches(rows, False):
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
        for i, row in zip(idx, logits.softmax(-1).cpu()):
            probs[i] = row
    model.train()
    return probs


def metrics(rows, probs):
    gold = [y(r) for r in rows]
    pred = [int(pr.argmax()) for pr in probs]
    conf = [float(pr.max()) for pr in probs]
    major = collections.Counter(y(r) for r in train).most_common(1)[0][0]
    out = {"rows": len(rows), "accuracy": sum(p == t for p, t in zip(pred, gold)) / max(1, len(rows)),
           "majority_baseline": sum(t == major for t in gold) / max(1, len(rows)), "majority_label": labels[major]}
    for th in (0.5, 0.7, 0.9):
        kept = [(p, t) for p, t, c in zip(pred, gold, conf) if c >= th and p != 0]
        out[f"proposed@{th}"] = len(kept) / max(1, len(rows))
        out[f"precision@{th}"] = sum(p == t for p, t in kept) / max(1, len(kept))
    per = collections.defaultdict(lambda: [0, 0])
    for p, t in zip(pred, gold):
        per[labels[t]][0] += p == t
        per[labels[t]][1] += 1
    out["per_label"] = {l: {"rows": n, "accuracy": round(c / n, 3)} for l, (c, n) in sorted(per.items(), key=lambda x: -x[1][1])[:30]}
    return out


# ---- training ---------------------------------------------------------------------------
steps_per_epoch = math.ceil(len(train) / a.batch / a.grad_accum)
total = a.max_steps or int(steps_per_epoch * a.epochs)
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
warm = max(1, int(total * 0.06))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
step, best, t0 = 0, None, time.time()
model.train()
while step < total:
    for k, (_, ids, mask, ys) in enumerate(batches(train, True)):
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            loss = torch.nn.functional.cross_entropy(model(input_ids=ids, attention_mask=mask).logits.float(), ys) / a.grad_accum
        loss.backward()
        if (k + 1) % a.grad_accum:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        step += 1
        if step % 50 == 0:
            print(f"step {step}/{total} loss {loss.item() * a.grad_accum:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if step % steps_per_epoch == 0 or step == total:
            m = metrics(test, evaluate(test)) if test else {"accuracy": 0}
            print(f"step {step}: test accuracy {m['accuracy']:.3f}", flush=True)
            if best is None or m["accuracy"] >= best[0]:
                best = (m["accuracy"], {k: v.detach().clone() for k, v in model.state_dict().items()})
        if step >= total:
            break
if best:
    model.load_state_dict(best[1])

# ---- save what --refine loads -------------------------------------------------------
model.to("cpu").float().eval()
dev = "cpu"
model.save_pretrained(a.out)
tok.save_pretrained(a.out)
# `--refine` reads the sequence length from tokenizer.json's truncation.
tj = json.loads((a.out / "tokenizer.json").read_text())
tj["truncation"] = {"direction": "Left", "max_length": a.max_len, "strategy": "LongestFirst", "stride": 0}
(a.out / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False))
(a.out / "labels.txt").write_text("\n".join(labels) + "\n")


class Logits(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


ids = torch.full((2, 16), tok.pad_token_id, dtype=torch.long)
mask = torch.ones((2, 16), dtype=torch.long)
torch.onnx.export(Logits(model), (ids, mask), str(a.out / "model.onnx"), input_names=["input_ids", "attention_mask"],
                  output_names=["logits"], dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                                         "logits": {0: "batch"}}, opset_version=17, dynamo=False)
if a.int8:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    (a.out / "model.onnx").replace(a.out / "model.fp32.onnx")
    quantize_dynamic(str(a.out / "model.fp32.onnx"), str(a.out / "model.onnx"), weight_type=QuantType.QInt8)

m = metrics(test, evaluate(test)) if test else {}
m.update(labels=len(labels), train_rows=len(train), steps=step, seconds=round(time.time() - t0))
(a.out / "metrics.json").write_text(json.dumps(m, indent=1) + "\n")
print(json.dumps({k: v for k, v in m.items() if k != "per_label"}, indent=1))
print(f"wrote {a.out}: run `chungusite <binary> --refine {a.out}` (built with --features ml)")
