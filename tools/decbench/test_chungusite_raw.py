"""Tests for the Rust -> C skeleton. Needs neither DecBench nor chungusite:

    python3 -m pytest tools/decbench      # or: python3 tools/decbench/test_chungusite_raw.py
"""

import importlib.util
import shutil
import subprocess
import tempfile
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "chungusite_raw", Path(__file__).with_name("chungusite_raw.py")
)
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)


def skel(rust: str) -> list[str]:
    return [line.strip() for line in cr.to_c_skeleton(rust).splitlines()]


def test_signature_and_lets():
    c = skel(
        """pub unsafe fn get(p: *mut Node, mut rdi: u64, s: &[u8]) -> i32 {
    let mut rdi: u64 = rdi as u32 as u64;
    let v3: u64 = unsafe { ((*p).f8 as *const u64).read_unaligned() }; // 0x10
    return v3 as i32;
}"""
    )
    assert c[0] == "int get(struct Node * p, unsigned long rdi, unsigned char * s) {"
    # A re-bound argument is an assignment, not a second declaration.
    assert c[1:4] == ["rdi = 0;", "unsigned long v3 = 0;", "return 0;"]


def test_if_while_and_unsafe_blocks_in_conditions():
    c = skel(
        """pub unsafe fn find(mut rdi: u64, rsi: u64) -> u64 {
    while (unsafe { (rdi as *const u64).read_unaligned() }) != rsi {
        if rdi == 0 || rsi == 1 {
            return rdi;
        } else if rdi == 2 {
            rdi = 3;
        } else {
            break;
        }
    }
    return 0;
}"""
    )
    assert c[1:] == [
        "while (c) {",
        "if (c) {",
        "return 0;",
        "} else if (c) {",
        "x = 0;",
        "} else {",
        "break;",
        "}",
        "}",
        "return 0;",
        "}",
    ]


def test_labeled_block_and_loop_become_gotos():
    c = skel(
        """pub fn f(rdi: u64) {
    'b3: {
        if rdi == 0 {
            break 'b3;
        }
        g();
    }
    'l: loop {
        loop {
            if rdi == 1 { break 'l; }
            if rdi == 2 { continue 'l; }
            break;
        }
    }
}"""
    )
    text = "\n".join(c)
    assert "goto b3_end_1;" in text and "b3_end_1: ;" in text
    assert "goto l_break_2;" in text and "l_break_2: ;" in text
    assert "goto l_continue_3;" in text and "l_continue_3: ;" in text
    assert c.count("break;") == 1


def test_match_arms_do_not_fall_through_and_break_leaves_the_loop():
    c = skel(
        """pub fn f(mut bb: u32) -> u64 {
    loop {
        match bb {
            1 => { bb = 2; }
            2 => break,
            3 | 4 => return 5,
            _ => unreachable!(),
        }
    }
    return 0;
}"""
    )
    text = "\n".join(c)
    assert "switch (c) {" in text
    # `break` in a match arm leaves the loop, which a C `break` in a switch would not.
    assert "goto loop_break_1;" in text and "loop_break_1: ;" in text
    assert "default:\nabort();" in text
    assert text.count("break;") == 1  # only after the first arm


def test_split_functions():
    out = """// Decompiled by chungusite
// f @ 0x1139, 10 bytes
pub fn f() {
    return;
}

// g @ 0x1150, 3 bytes; 1 of 1 memory accesses safe
/// doc
pub unsafe fn g() -> u64 {
    return 1;
}
"""
    fns = cr.split_functions(out)
    assert [(n, a) for n, a, _ in fns] == [("f", 0x1139), ("g", 0x1150)]
    assert fns[1][2].startswith("pub unsafe fn g()") and fns[1][2].endswith("\n}")


def test_skeleton_is_c():
    if shutil.which("gcc") is None:
        return
    rust = """pub fn h(rdi: u64) -> u64 {
    'b1: { if rdi == 0 { break 'b1; } }
    let mut bb: u32 = 0;
    loop { match bb { 0 => { bb = 1; } _ => break, } }
    return rdi;
}"""
    with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as f:
        f.write("long x; int c; void abort(void);\n" + cr.to_c_skeleton(rust))
    r = subprocess.run(["gcc", "-fsyntax-only", "-w", f.name], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
