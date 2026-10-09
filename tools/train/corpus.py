#!/usr/bin/env python3
"""Build training data for the type and name models from C source with debug info.

Each project (a source directory or a .tar.gz/.tgz/.zip of one) is compiled with
every compiler and optimisation level asked for, one shared library per build
(`-g`, every .c file that compiles on its own, unresolved symbols allowed).
`chungusite --emit dataset` reads each library's prototypes and writes one row
per argument and return value, in the text `--refine` will show the model.
Prebuilt binaries with debug info can be added with --binary.

Rows are deduplicated (the same function often lifts to the same text at two
levels), and whole projects go to train or test, so near-duplicates can't leak
across the split. Outputs, in --out:

  types_train.jsonl, types_test.jsonl   {"text", "label", "project", ...} per variable
  names_train.jsonl, names_test.jsonl   {"source", "target", "project"} per function
  stats.json                            row counts and the label histogram

  python tools/train/corpus.py --out data \\
      --project src/zlib-1.3.1 --project downloads/lz4-1.10.0.tar.gz \\
      --cc gcc clang --opt O0 O1 O2 O3 Os -j 16

Only the standard library is needed.
"""
import argparse, collections, concurrent.futures as cf, hashlib, json, os, random, shutil, subprocess, sys, tarfile, tempfile, zipfile
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--out", required=True, type=Path)
p.add_argument("--project", action="append", default=[], help="source directory or archive (repeatable); name=PATH to name it")
p.add_argument("--projects-file", type=Path, help="one --project per line")
p.add_argument("--binary", action="append", default=[], help="a prebuilt binary with debug info, its own project (repeatable)")
p.add_argument("--chungusite", default=str(Path(__file__).resolve().parents[2] / "target/release/chungusite"))
p.add_argument("--cc", nargs="+", default=["gcc", "clang"])
p.add_argument("--opt", nargs="+", default=["O0", "O1", "O2", "O3", "Os"])
p.add_argument("--cflags", default="", help="extra flags for every compile")
p.add_argument("--test-fraction", type=float, default=0.1, help="share of projects held out")
p.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
p.add_argument("--max-chars", type=int, default=16384, help="cut longer texts (types: keep the end, which the model sees; names: keep the start)")
p.add_argument("--keep", action="store_true", help="keep the build directory (printed)")
a = p.parse_args()

if not Path(a.chungusite).exists():
    sys.exit(f"{a.chungusite} not found: build it with `cargo build --release` or pass --chungusite")
projects = list(a.project)
if a.projects_file:
    projects += [l.strip() for l in a.projects_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
if not projects and not a.binary:
    sys.exit("nothing to do: pass --project or --binary")
a.out.mkdir(parents=True, exist_ok=True)
work = Path(tempfile.mkdtemp(prefix="chungusite-corpus-"))


def unpack(spec):
    name, _, path = spec.rpartition("=") if "=" in spec else ("", "", spec)
    path = Path(path)
    for suffix in (".tar.gz", ".tgz", ".tar.xz", ".tar.bz2", ".tar", ".zip"):
        if path.name.endswith(suffix):
            name = name or path.name[: -len(suffix)]
            dest = work / "src" / name
            dest.mkdir(parents=True, exist_ok=True)
            if suffix == ".zip":
                zipfile.ZipFile(path).extractall(dest)
            else:
                with tarfile.open(path) as t:
                    t.extractall(dest, filter="data")
            return name, dest
    return name or path.name, path


def run(cmd, timeout=300):
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def build(name, src, cc, opt):
    """One shared library of everything in `src` that compiles; None if nothing does."""
    out = work / "build" / name / f"{cc}-{opt}"
    out.mkdir(parents=True, exist_ok=True)
    cfiles = sorted(src.rglob("*.c"))
    incs = sorted({str(h.parent) for h in src.rglob("*.h")} | {str(src)})
    flags = [f"-{opt}", "-g", "-fPIC", "-w"] + a.cflags.split()
    objs = []
    for i, c in enumerate(cfiles):
        o = out / f"{i}.o"
        if run([cc, "-c", str(c), "-o", str(o), *flags, *[f"-I{d}" for d in incs]], timeout=120):
            objs.append(str(o))
    if not objs:
        return None
    lib = out / f"lib{name}.so"
    link = [cc, "-shared", "-o", str(lib), *objs, "-Wl,--unresolved-symbols=ignore-all"]
    if run(link) or run(link + ["-Wl,--allow-multiple-definition"]):
        return lib
    return None


def dataset(binary):
    rows = work / "rows" / (hashlib.sha1(str(binary).encode()).hexdigest() + ".jsonl")
    rows.parent.mkdir(parents=True, exist_ok=True)
    if not run([a.chungusite, str(binary), "--emit", "dataset", "-o", str(rows)], timeout=3600):
        return []
    return [json.loads(l) for l in rows.read_text().splitlines()]


jobs = []  # (project, build, binary or None, future)
with cf.ThreadPoolExecutor(a.jobs) as pool:
    for spec in projects:
        name, src = unpack(spec)
        for cc in a.cc:
            if not shutil.which(cc):
                print(f"skipping {cc}: not installed", file=sys.stderr)
                continue
            for opt in a.opt:
                jobs.append((name, f"{cc}-{opt}", pool.submit(build, name, src, cc, opt)))
    for b in a.binary:
        jobs.append((Path(b).name, "prebuilt", pool.submit(lambda b=b: Path(b))))
    built = [(n, k, f.result()) for n, k, f in jobs]
    for n, k, lib in built:
        if lib is None:
            print(f"{n} {k}: nothing compiled", file=sys.stderr)
    futures = [(n, k, pool.submit(dataset, lib)) for n, k, lib in built if lib is not None]
    rows = []
    for n, k, f in futures:
        got = f.result()
        print(f"{n} {k}: {len(got)} rows", file=sys.stderr)
        for r in got:
            r["project"], r["build"] = n, k
            rows.append(r)

# Deduplicate: the same text asking about the same variable is one example.
seen, uniq = set(), []
for r in rows:
    key = hashlib.sha1((r["text"] + "\0" + r["label"]).encode()).digest()
    if key not in seen:
        seen.add(key)
        uniq.append(r)

# Split by project: the held-out projects are a fixed share, chosen by a hash of
# their names so adding projects doesn't reshuffle the others.
names = sorted({r["project"] for r in uniq}, key=lambda n: hashlib.sha1(n.encode()).hexdigest())
n_test = max(1, round(len(names) * a.test_fraction)) if len(names) > 1 else 0
test = set(names[:n_test])


def write(path, items):
    with open(path, "w") as f:
        for x in items:
            f.write(json.dumps(x) + "\n")


for split, keep in (("train", lambda r: r["project"] not in test), ("test", lambda r: r["project"] in test)):
    part = [r for r in uniq if keep(r)]
    random.Random(0).shuffle(part)
    # the type model sees the end of a text (cut from the left), so keep that
    write(a.out / f"types_{split}.jsonl", [
        {"text": r["text"][-a.max_chars:], **{k: r[k] for k in ("label", "project", "build", "func", "var", "name")}} for r in part])
    # names: one row per function, its text without the `var vN` line, and the
    # source names of its arguments
    funcs = collections.OrderedDict()
    for r in part:
        if r["name"]:
            key = (r["project"], r["build"], r["func"])
            src = r["text"].rsplit("\nvar ", 1)[0][:a.max_chars] + "\n"  # the start: arguments come first
            funcs.setdefault(key, (src, {}))[1][r["text"].rsplit("\nvar ", 1)[1]] = r["name"]
    write(a.out / f"names_{split}.jsonl", [
        {"source": src, "target": "; ".join(f"{v}: {n}" for v, n in sorted(m.items(), key=lambda x: int(x[0][1:]))), "project": key[0]}
        for key, (src, m) in funcs.items()
    ])

labels = collections.Counter(r["label"] for r in uniq)
stats = {
    "rows": len(rows), "unique": len(uniq), "projects": len(names), "test_projects": sorted(test),
    "train": sum(r["project"] not in test for r in uniq), "test": sum(r["project"] in test for r in uniq),
    "labels": dict(labels.most_common()),
}
(a.out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
print(f"{stats['unique']} unique rows ({stats['rows']} before dedup) from {len(names)} projects; "
      f"train {stats['train']}, test {stats['test']} ({', '.join(sorted(test)) or 'none'})", file=sys.stderr)
if a.keep:
    print(f"build directory: {work}", file=sys.stderr)
else:
    shutil.rmtree(work, ignore_errors=True)
