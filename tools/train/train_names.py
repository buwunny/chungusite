#!/usr/bin/env python3
"""Train a model that names a function's arguments from its lifted text.

Reads corpus.py's names_train.jsonl / names_test.jsonl ({"source": "<function>",
"target": "v3: buf; v5: len"}) and trains a sequence-to-sequence model to write
the target from the source. The decompiler doesn't call a name model yet
(`--refine` only proposes types), so this saves a Hugging Face checkpoint for
offline use, evaluation, and the runtime hook when it lands.

Fine-tune a pretrained code model (the recipe: CodeT5+ 220M, a few hours on one
24 GB GPU for a few hundred thousand functions):

  python tools/train/train_names.py --data data --out names_model \\
      --model Salesforce/codet5p-220m --epochs 5 --lr 5e-5 --batch 16

or a small T5 from scratch with its own tokenizer (no download; a smoke test):

  python tools/train/train_names.py --data data --out names_tiny --from-scratch tiny
"""
import argparse, json, math, random, re, time
from pathlib import Path

import torch

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--data", required=True, type=Path, help="corpus.py's --out directory")
p.add_argument("--out", required=True, type=Path)
g = p.add_mutually_exclusive_group(required=True)
g.add_argument("--model", help="pretrained encoder-decoder to fine-tune (Hugging Face id or local path)")
g.add_argument("--from-scratch", choices=["tiny", "small"], help="random-initialised T5 of this size")
p.add_argument("--max-len", type=int, default=512, help="source tokens (the start of the function is kept)")
p.add_argument("--max-target", type=int, default=64)
p.add_argument("--epochs", type=float, default=5)
p.add_argument("--lr", type=float)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--grad-accum", type=int, default=1)
p.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (0: no limit)")
p.add_argument("--eval-rows", type=int, default=500, help="test functions decoded at each evaluation")
p.add_argument("--vocab", type=int, default=8000, help="--from-scratch: tokenizer vocabulary size")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
torch.manual_seed(a.seed)
random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
a.out.mkdir(parents=True, exist_ok=True)

read = lambda f: [json.loads(l) for l in open(a.data / f)]
train, test = read("names_train.jsonl"), read("names_test.jsonl")
print(f"{len(train)} train / {len(test)} test functions on {dev}")
head = lambda t: t[:a.max_len * 16]  # only the start reaches the model

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

if a.model:
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model)
    lr = a.lr or 5e-5
else:
    from tokenizers import ByteLevelBPETokenizer, processors
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator((x for r in train for x in (head(r["source"]), r["target"])), vocab_size=a.vocab,
                            special_tokens=["<pad>", "</s>", "<unk>"])
    bpe.post_processor = processors.TemplateProcessing(single="$A </s>", special_tokens=[("</s>", bpe.token_to_id("</s>"))])
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe._tokenizer, eos_token="</s>", pad_token="<pad>", unk_token="<unk>")
    size = {
        "tiny": dict(d_model=128, d_ff=512, d_kv=32, num_layers=2, num_decoder_layers=2, num_heads=4),
        "small": dict(d_model=384, d_ff=1536, d_kv=64, num_layers=4, num_decoder_layers=4, num_heads=6),
    }[a.from_scratch]
    cfg = T5Config(vocab_size=len(tok), pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                   decoder_start_token_id=tok.pad_token_id, **size)
    model = T5ForConditionalGeneration(cfg)
    lr = a.lr or 1e-3
model.to(dev)


def batches(rows, shuffle):
    order = list(range(len(rows)))
    random.shuffle(order) if shuffle else None
    for s in range(0, len(order), a.batch):
        part = [rows[i] for i in order[s:s + a.batch]]
        x = tok([head(r["source"]) for r in part], max_length=a.max_len, truncation=True, padding=True, return_tensors="pt")
        y = tok([r["target"] for r in part], max_length=a.max_target, truncation=True, padding=True, return_tensors="pt").input_ids
        y[y == tok.pad_token_id] = -100
        yield part, x.input_ids.to(dev), x.attention_mask.to(dev), y.to(dev)


def parse(t):
    """'v3: buf; v5: len' -> {'v3': 'buf', 'v5': 'len'}"""
    return dict(m.groups() for m in re.finditer(r"(v\d+):\s*([A-Za-z_][A-Za-z0-9_]*)", t))


@torch.no_grad()
def evaluate(rows):
    """Exact-name accuracy over every argument, and how many functions are entirely right."""
    model.eval()
    hit = total = whole = 0
    samples = []
    for part, ids, mask, _ in batches(rows, False):
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=a.max_target, num_beams=1)
        for r, o in zip(part, tok.batch_decode(out, skip_special_tokens=True)):
            gold, got = parse(r["target"]), parse(o)
            right = sum(got.get(v) == n for v, n in gold.items())
            hit, total, whole = hit + right, total + len(gold), whole + (right == len(gold))
            if len(samples) < 5:
                samples.append({"target": r["target"], "predicted": o})
    model.train()
    return {"functions": len(rows), "name_accuracy": hit / max(1, total), "functions_all_right": whole / max(1, len(rows)),
            "samples": samples}


held = random.Random(0).sample(test, min(a.eval_rows, len(test)))
steps_per_epoch = math.ceil(len(train) / a.batch / a.grad_accum)
total = a.max_steps or int(steps_per_epoch * a.epochs)
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
warm = max(1, int(total * 0.06))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
step, best, t0 = 0, None, time.time()
model.train()
while step < total:
    for k, (_, ids, mask, y) in enumerate(batches(train, True)):
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            loss = model(input_ids=ids, attention_mask=mask, labels=y).loss / a.grad_accum
        loss.backward()
        if (k + 1) % a.grad_accum:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        step += 1
        if step % 50 == 0:
            print(f"step {step}/{total} loss {loss.item() * a.grad_accum:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if step % steps_per_epoch == 0 or step == total:
            m = evaluate(held) if held else {"name_accuracy": 0}
            print(f"step {step}: name accuracy {m['name_accuracy']:.3f}", flush=True)
            if best is None or m["name_accuracy"] >= best[0]:
                best = (m["name_accuracy"], {k: v.detach().clone() for k, v in model.state_dict().items()})
        if step >= total:
            break
if best:
    model.load_state_dict(best[1])

model.to("cpu").eval()
dev = "cpu"
model.save_pretrained(a.out)
tok.save_pretrained(a.out)
m = evaluate(held) if held else {}
m.update(train_functions=len(train), steps=step, seconds=round(time.time() - t0))
(a.out / "metrics.json").write_text(json.dumps(m, indent=1) + "\n")
print(json.dumps(m, indent=1))
print(f"wrote {a.out}")
