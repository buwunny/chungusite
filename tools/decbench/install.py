"""Register the chungusite backend in a DecBench checkout.

    python tools/decbench/install.py ~/src/decbench

Links chungusite_raw.py into <checkout>/decbench/decompilers/raw/ and imports it
from that package's __init__.py, the same way DecBench's own backends register.
Linking (not copying) means edits here take effect without reinstalling, and
every DecBench entry point sees the backend, including the worker processes
scripts/run_benchmark.py spawns. Running it again changes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

IMPORT = "from decbench.decompilers.raw import chungusite_raw  # noqa: F401,E402  (chungusite)\n"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    raw = Path(sys.argv[1]).expanduser() / "decbench" / "decompilers" / "raw"
    init = raw / "__init__.py"
    if not init.is_file():
        print(f"{raw} is not a DecBench checkout's raw-backend package", file=sys.stderr)
        return 2

    src = Path(__file__).resolve().with_name("chungusite_raw.py")
    link = raw / "chungusite_raw.py"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(src)

    text = init.read_text()
    if IMPORT not in text:
        init.write_text(text.rstrip("\n") + "\n\n" + IMPORT)
    print(f"linked {link} -> {src}")
    print("check with: decbench list-decompilers   (chungusite should be Available = Y)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
