"""DecBench backend for chungusite, an x86_64 to Rust decompiler.

DecBench scores C: GED parses the decompiled code with Joern's C frontend,
byte_match recompiles it with gcc, and type_match's fallback reads a C
signature. Joern finds no functions in Rust, so chungusite's raw output would
score nothing. This backend therefore hands DecBench a *control-flow skeleton*
of each Rust function in C syntax (`to_c_skeleton`):

- `if`/`else`, `while`, `loop`, `break`, `continue`, `return` and `match` keep
  their shape; labeled blocks and labeled loops become `goto`s;
- every other statement becomes one opaque `x = 0;`, and every condition `c`;
- the signature and `let` bindings keep their types, mapped to C.

GED reads only the CFG's topology and entry/exit roles, and Joern's CFG is
block-level (calls and `&&`/`||` don't split blocks), so the skeleton has the
same GED as a faithful C translation would. type_match sees the recovered
argument and local types through its C-signature fallback. byte_match has no
code to recompile, so leave it out (`-m ged -m type_match`).

The Rust itself is written to `<output_dir>/chungusite_<binary>.rs` and kept
per function in `metadata["rust"]`.

Install into a DecBench checkout with `python tools/decbench/install.py
<decbench-checkout>`; see README.md next to this file.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Splitting chungusite's output into functions
# ---------------------------------------------------------------------------

# Every emitted function is preceded by `// <name> @ 0x<addr>, <n> bytes...`.
_HEADER = re.compile(r"^// (\S+) @ 0x([0-9a-fA-F]+), \d+ bytes", re.M)
_FN = re.compile(r"^pub (?:unsafe )?fn ", re.M)


def split_functions(rust: str) -> list[tuple[str, int, str]]:
    """(name, address, source) for each function in a chungusite output file.

    The source runs from the `pub fn` line to its closing `}` at column 0.
    """
    out = []
    for m in _HEADER.finditer(rust):
        f = _FN.search(rust, m.end())
        if f is None:
            continue
        end = rust.find("\n}\n", f.start())
        end = len(rust) if end < 0 else end + 2
        out.append((m.group(1), int(m.group(2), 16), rust[f.start() : end]))
    return out


# ---------------------------------------------------------------------------
# Rust -> C control-flow skeleton
# ---------------------------------------------------------------------------

_TOKEN = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<comment>//[^\n]*)
  | (?P<str>b?"(?:\\.|[^"\\])*")
  | (?P<char>b?'(?:\\.|\\u\{[0-9a-fA-F]+\}|[^'\\])')
  | (?P<label>'[A-Za-z_][A-Za-z0-9_]*)
  | (?P<ident>(?:r\#)?[A-Za-z_][A-Za-z0-9_]*)
  | (?P<num>[0-9][0-9a-zA-Z_.]*)
  | (?P<punct>=>|->|::|&&|\|\||==|!=|<=|>=|\.\.=|\.\.|[{}()\[\];:,<>=!&|.*+\-/%^?#@$~])
    """,
    re.X,
)

_INT_TYPES = {
    "u8": "unsigned char",
    "u16": "unsigned short",
    "u32": "unsigned int",
    "u64": "unsigned long",
    "usize": "unsigned long",
    "u128": "unsigned __int128",
    "i8": "signed char",
    "i16": "short",
    "i32": "int",
    "i64": "long",
    "isize": "long",
    "i128": "__int128",
    "bool": "_Bool",
    "f32": "float",
    "f64": "double",
}


def tokenize(src: str) -> list[str]:
    toks = []
    pos = 0
    while pos < len(src):
        m = _TOKEN.match(src, pos)
        if m is None:
            raise ValueError(f"cannot tokenize at {src[pos:pos + 20]!r}")
        pos = m.end()
        if m.lastgroup not in ("ws", "comment"):
            toks.append(m.group())
    return toks


def c_type(toks: list[str]) -> str:
    """A C spelling of a Rust type: integers map, everything pointer-like is a pointer."""
    t = [x for x in toks if x not in ("mut", "const", "dyn")]
    if not t:
        return "long"
    if len(t) == 1 and t[0] in _INT_TYPES:
        return _INT_TYPES[t[0]]
    if t[0] == "*":
        inner = t[1:]
        if len(inner) == 1 and inner[0] in _INT_TYPES:
            return _INT_TYPES[inner[0]] + " *"
        if len(inner) == 1 and re.fullmatch(r"[A-Za-z_]\w*", inner[0]):
            return f"struct {inner[0]} *"
        return "void *"
    if t[0] in ("&", "Option", "Box") or t[0] == "[":
        # &[u8], &mut [u8], Option<&[u8]>, Box<[u8]>: a byte pointer; &S: a struct pointer.
        if t[0] == "&" and len(t) == 2 and re.fullmatch(r"[A-Za-z_]\w*", t[1]) and t[1] not in _INT_TYPES:
            return f"struct {t[1]} *"
        return "unsigned char *"
    if t[0] == "(":
        return "unsigned __int128"  # (u64, u64): rax:rdx
    return "long"


def _ident(name: str) -> str:
    return name[2:] if name.startswith("r#") else name


class _Skeleton:
    """Recursive descent over one function's tokens, writing C lines."""

    def __init__(self, toks: list[str]):
        self.t = toks
        self.i = 0
        self.out: list[str] = []
        self.depth = 1
        self.ret_void = True
        # Enclosing loops, innermost last: (label or None, break target, continue target).
        self.loops: list[tuple[str | None, str, str]] = []
        self.blocks: dict[str, str] = {}  # labeled block -> its end label
        self.in_switch = 0  # switch nesting inside the innermost loop
        self.used: set[str] = set()  # goto targets something jumps to
        self.declared: set[str] = set()  # Rust shadows a name; C may not redeclare it
        self.n = 0

    # -- token helpers ----------------------------------------------------
    def peek(self, k: int = 0) -> str | None:
        j = self.i + k
        return self.t[j] if j < len(self.t) else None

    def take(self, want: str | None = None) -> str:
        tok = self.t[self.i]
        if want is not None and tok != want:
            raise ValueError(f"expected {want!r}, got {tok!r} at token {self.i}")
        self.i += 1
        return tok

    def fresh(self, base: str) -> str:
        self.n += 1
        return f"{base.lstrip(chr(39))}_{self.n}"

    def emit(self, line: str) -> None:
        self.out.append("    " * self.depth + line)

    def skip_balanced(self) -> None:
        """Skip one bracketed group starting at the current token."""
        pairs = {"(": ")", "[": "]", "{": "}"}
        close = pairs[self.take()]
        stack = [close]
        while stack:
            tok = self.take()
            if tok in pairs:
                stack.append(pairs[tok])
            elif tok == stack[-1]:
                stack.pop()

    def skip_expr(self, stops: tuple[str, ...]) -> list[str]:
        """Skip an expression up to (not including) a stop token at depth 0.

        `{` at depth 0 ends a condition unless it opens an `unsafe { .. }` block
        or follows `=` / `(` (an expression-level block).
        """
        start = self.i
        while self.i < len(self.t):
            tok = self.t[self.i]
            if tok in stops:
                break
            if tok == "{":
                prev = self.t[self.i - 1] if self.i > start else None
                if "{" in stops and prev not in ("unsafe", "=", "(", ",", "=>", "else"):
                    break
            if tok in ("(", "[", "{"):
                self.skip_balanced()
                continue
            self.i += 1
        return self.t[start : self.i]

    # -- function -----------------------------------------------------------
    def function(self) -> str:
        while self.peek() in ("pub", "unsafe", "extern", "\"C\""):
            self.take()
        self.take("fn")
        name = _ident(self.take())
        self.take("(")
        params = []
        while self.peek() != ")":
            if self.peek() == "mut":
                self.take()
            pname = _ident(self.take())
            self.take(":")
            ty = self.type_until((",", ")"))
            params.append(f"{c_type(ty)} {pname}")
            self.declared.add(pname)
            if self.peek() == ",":
                self.take()
        self.take(")")
        ret = "void"
        if self.peek() == "->":
            self.take()
            rt = self.type_until(("{",))
            ret = "void" if rt in (["(", ")"], ["!"]) else c_type(rt)
        self.ret_void = ret == "void"
        self.take("{")
        self.block_body()
        sig = f"{ret} {name}({', '.join(params) or 'void'})"
        return sig + " {\n" + "\n".join(self.out) + "\n}\n"

    def type_until(self, stops: tuple[str, ...]) -> list[str]:
        start = self.i
        angle = 0
        while True:
            tok = self.peek()
            if tok is None:
                break
            if angle == 0 and tok in stops:
                break
            if tok in ("(", "["):
                self.skip_balanced()
                continue
            if tok == "<":
                angle += 1
            elif tok == ">":
                angle -= 1
            self.i += 1
        return self.t[start : self.i]

    # -- statements -----------------------------------------------------------
    def block_body(self) -> None:
        """Statements up to and including the closing `}`."""
        while self.peek() != "}":
            self.statement()
        self.take("}")

    def nested_block(self) -> None:
        self.take("{")
        self.depth += 1
        self.block_body()
        self.depth -= 1

    def statement(self) -> None:
        tok = self.peek()
        if tok == ";":
            self.take()
            return
        if tok == "#":  # attribute
            self.take()
            self.skip_balanced()
            return
        if tok is not None and tok.startswith("'") and self.peek(1) == ":":
            label = self.take()
            self.take(":")
            self.labeled(label)
            return
        if tok == "let":
            self.let()
            return
        if tok == "if":
            self.if_chain()
            return
        if tok == "while":
            self.take()
            self.skip_expr(("{",))
            self.loop(None, "while (c)")
            return
        if tok == "loop":
            self.take()
            self.loop(None, "while (1)")
            return
        if tok == "match":
            self.match()
            return
        if tok in ("break", "continue"):
            self.jump()
            return
        if tok == "return":
            self.take()
            self.skip_expr((";", "}"))
            self.emit("return;" if self.ret_void else "return 0;")
            if self.peek() == ";":
                self.take()
            return
        if tok == "{":
            self.emit("{")
            self.nested_block()
            self.emit("}")
            return
        expr = self.skip_expr((";", "}"))
        if self.peek() == ";":
            self.take()
        if expr and expr[0] in ("panic", "todo", "unreachable", "unimplemented") and len(expr) > 1 and expr[1] == "!":
            self.emit("abort();")
        elif expr:
            self.emit("x = 0;")

    def let(self) -> None:
        self.take("let")
        if self.peek() == "mut":
            self.take()
        if self.peek() == "(":  # tuple pattern
            self.skip_balanced()
            self.skip_expr((";",))
            self.take(";")
            self.emit("x = 0;")
            return
        name = _ident(self.take())
        ty: list[str] = []
        if self.peek() == ":":
            self.take()
            ty = self.type_until(("=", ";"))
        init = False
        if self.peek() == "=":
            self.take()
            self.skip_expr((";",))
            init = True
        self.take(";")
        if name in self.declared or name == "_":
            # `let mut rdi: u64 = rdi as u64;` re-binds an argument: an assignment in C.
            if init:
                self.emit(f"{name} = 0;" if name != "_" else "x = 0;")
            return
        self.declared.add(name)
        self.emit(f"{c_type(ty)} {name}" + (" = 0;" if init else ";"))

    def if_chain(self) -> None:
        self.take("if")
        self.skip_expr(("{",))
        self.emit("if (c) {")
        self.nested_block()
        while self.peek() == "else":
            self.take()
            if self.peek() == "if":
                self.take()
                self.skip_expr(("{",))
                self.emit("} else if (c) {")
            else:
                self.emit("} else {")
            self.nested_block()
        self.emit("}")

    def loop(self, label: str | None, head: str) -> None:
        brk = self.fresh((label or "loop") + "_break")
        cont = self.fresh((label or "loop") + "_continue")
        self.loops.append((label, brk, cont))
        saved, self.in_switch = self.in_switch, 0
        self.emit(head + " {")
        self.nested_block()
        if cont in self.used:
            self.emit(f"    {cont}: ;")
        self.emit("}")
        if brk in self.used:
            self.emit(f"{brk}: ;")
        self.in_switch = saved
        self.loops.pop()

    def labeled(self, label: str) -> None:
        if self.peek() == "loop":
            self.take()
            self.loop(label, "while (1)")
        elif self.peek() == "while":
            self.take()
            self.skip_expr(("{",))
            self.loop(label, "while (c)")
        else:  # labeled block
            end = self.fresh(label + "_end")
            self.blocks[label] = end
            self.emit("{")
            self.nested_block()
            self.emit("}")
            self.emit(f"{end}: ;")

    def jump(self) -> None:
        kind = self.take()
        label = self.take() if self.peek() is not None and self.peek().startswith("'") else None
        self.skip_expr((";", "}", ","))
        if self.peek() == ";":
            self.take()
        if label is not None and label in self.blocks:
            self.emit(f"goto {self.blocks[label]};")
            return
        if label is None:
            if not self.loops:
                raise ValueError(f"{kind} outside a loop")
            target = self.loops[-1]
            innermost = True
        else:
            match = [lp for lp in self.loops if lp[0] == label]
            if not match:
                raise ValueError(f"{kind} to unknown label {label}")
            target = match[-1]
            innermost = target is self.loops[-1]
        if innermost and not (kind == "break" and self.in_switch):
            self.emit(f"{kind};")
            return
        dest = target[1] if kind == "break" else target[2]
        self.used.add(dest)
        self.emit(f"goto {dest};")

    def match(self) -> None:
        self.take("match")
        self.skip_expr(("{",))
        self.take("{")
        self.emit("switch (c) {")
        self.in_switch += 1
        case = 0
        while self.peek() != "}":
            pats = self.skip_expr(("=>",))
            self.take("=>")
            if pats == ["_"]:
                self.emit("default:")
            else:
                case += 1
                self.emit(f"case {case}:")
            before = len(self.out)
            self.depth += 1
            if self.peek() == "{":
                self.take("{")
                self.block_body()
            else:
                self.statement_until_comma()
            last = self.out[-1].strip() if len(self.out) > before else ""
            if not (last.startswith(("return", "goto", "break;", "continue;", "abort();"))):
                self.emit("break;")
            self.depth -= 1
            if self.peek() == ",":
                self.take()
        self.take("}")
        self.in_switch -= 1
        self.emit("}")

    def statement_until_comma(self) -> None:
        tok = self.peek()
        if tok in ("break", "continue"):
            self.jump()
        elif tok == "return":
            self.take()
            self.skip_expr((",", "}"))
            self.emit("return;" if self.ret_void else "return 0;")
        else:
            expr = self.skip_expr((",", "}"))
            if expr and expr[0] in ("panic", "todo", "unreachable") and len(expr) > 1 and expr[1] == "!":
                self.emit("abort();")
            elif expr and expr not in (["(", ")"], ["{", "}"]):
                self.emit("x = 0;")


def to_c_skeleton(rust_fn: str) -> str:
    """C with the same control flow and signature types as one emitted Rust function."""
    return _Skeleton(tokenize(rust_fn)).function()


# ---------------------------------------------------------------------------
# The DecBench backend
# ---------------------------------------------------------------------------

try:
    from decbench.decompilers.base import Decompiler
    from decbench.decompilers.registry import register_decompiler
    from decbench.decompilers.spec import version_settings
except ImportError:  # imported for its translator alone (tests)
    Decompiler = None


def _chungusite_bin(settings: dict) -> str | None:
    path = os.environ.get("CHUNGUSITE_BIN") or settings.get("bin")
    if path:
        return path if Path(path).is_file() else None
    return shutil.which("chungusite")


if Decompiler is not None:

    @register_decompiler("chungusite")
    class ChungusiteDecompiler(Decompiler):
        """chungusite's Rust, as a C control-flow skeleton for DecBench's metrics."""

        name = "chungusite"
        display_name = "chungusite"

        def _settings(self) -> dict:
            try:
                return version_settings(self.name, self.requested_version) or {}
            except Exception:  # noqa: BLE001 - no config file is fine
                return {}

        def _mode(self) -> str:
            return os.environ.get("CHUNGUSITE_MODE") or self._settings().get("mode", "fast")

        def is_available(self) -> bool:
            return _chungusite_bin(self._settings()) is not None

        def get_version(self) -> str | None:
            exe = _chungusite_bin(self._settings())
            if exe is None:
                return None
            try:
                out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                return None
            return out.stdout.strip().split()[-1] if out.stdout.strip() else None

        def decompile_binary(
            self,
            binary_path: Path,
            functions: list[tuple[str, int]] | None = None,
            output_dir: Path | None = None,
            function_names: set[int] | None = None,
            progress_path: Path | None = None,
        ):
            from decbench.decompilers.raw import common
            from decbench.models.decompilation import (
                DecompilationResult,
                DecompilerMetadata,
                FunctionDecompilation,
            )

            binary_path = Path(binary_path)
            exe = _chungusite_bin(self._settings())
            if exe is None:
                raise RuntimeError("chungusite not found: put it on PATH or set CHUNGUSITE_BIN")
            mode = self._mode()
            start = time.time()

            cmd = [exe, str(binary_path), "--mode", mode, "--skip-failed"]
            if mode == "safe" and os.environ.get("CHUNGUSITE_CHECK", "1") != "0":
                cmd.append("--check")
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.binary_timeout_seconds,
            )
            if proc.returncode == 2:
                raise RuntimeError(f"chungusite failed on {binary_path}: {proc.stderr.strip()}")
            rust = proc.stdout
            if output_dir is not None:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / f"chungusite_{binary_path.stem}.rs").write_text(rust)

            elf_base = common.elf_min_vaddr(binary_path)
            text_range = common.elf_text_ranges(binary_path)
            addr_targets = common.addr_targets_of(function_names)
            emitted = {}
            listed: list[tuple[str, int]] = []
            for fname, addr, src in split_functions(rust):
                # chungusite prints the binary's own virtual addresses; for a PE
                # without the ImageBase folded in, add it.
                file_addr = addr if addr >= elf_base else addr + elf_base
                if common.should_skip_function(fname, file_addr, text_range, addr_targets):
                    continue
                emitted[file_addr] = (fname, src)
                listed.append((fname, file_addr))
            if functions:
                wanted = {a for _, a in functions}
                listed = [(n, a) for n, a in listed if a in wanted]
            listed = common.narrow_to_source(
                listed, addr_targets, backend=self.name, binary_name=binary_path.stem
            )

            funcs = {}
            failed = []
            for fname, file_addr in listed:
                _, src = emitted[file_addr]
                try:
                    code = to_c_skeleton(src)
                except (ValueError, IndexError):
                    failed.append(fname)
                    continue
                funcs[fname] = FunctionDecompilation(
                    name=fname,
                    address=file_addr,
                    decompiled_code=code,
                    line_count=code.count("\n") + 1,
                    variables=[],
                    metadata={
                        "rust": src,
                        "rust_lines": src.count("\n") + 1,
                        **common.extract_metrics(code),
                    },
                )
            # Targets chungusite didn't lift (left out by --skip-failed).
            for addr in sorted(addr_targets - set(emitted)):
                failed.append(f"sub_{addr:x}")

            result = DecompilationResult(
                binary_path=binary_path,
                binary_name=binary_path.stem,
                decompiler=DecompilerMetadata(
                    decompiler_name=self.id,
                    decompiler_version=self.get_version(),
                    total_time_seconds=time.time() - start,
                    failed_functions=failed,
                    extra={"backend": "chungusite", "mode": mode, "output": "c-skeleton"},
                ),
                functions=funcs,
                output_dir=output_dir,
            )
            common.dump_progress(progress_path, result)
            return result
