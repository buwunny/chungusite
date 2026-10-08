//! Type recovery (`types.rs`, `dwarf.rs`): prototypes and structs from debug
//! info, inference from how values are used, and the gate for model proposals.
use chungusite::ir::{Mutbl, TyId, TyTable};
use chungusite::types::{accept, parse_label, Label, VarFacts};
use std::process::Command;

const SRC: &str = "
struct node { int key; long count; struct node *next; unsigned char flag; };
int sum(int *a, int n) { int s = 0; for (int i = 0; i < n; i++) s += a[i]; return s; }
long bump(struct node *p, int d) { p->count += d; if (p->key < 0) p->flag = 1; return p->count; }
int sgn(int x) { return x < 0 ? -1 : x > 0; }
unsigned div3(unsigned x) { return x / 3; }
int length(struct node *p) { int n = 0; while (p) { n++; p = p->next; } return n; }
";

/// Compiles `SRC` with `gcc -O2 -g` and decompiles it with `args`, or `None`
/// when there is no gcc.
fn decompile(args: &[&str]) -> Option<String> {
    let dir = std::env::temp_dir().join(format!("chungusite-types-{}-{}", std::process::id(), args.join("")));
    std::fs::create_dir_all(&dir).unwrap();
    let (c, o) = (dir.join("a.c"), dir.join("a.o"));
    std::fs::write(&c, SRC).unwrap();
    let ok = Command::new("gcc").args(["-O2", "-g", "-c"]).arg(&c).arg("-o").arg(&o).status().map(|s| s.success());
    if !matches!(ok, Ok(true)) {
        eprintln!("skipping: no gcc");
        return None;
    }
    let out = Command::new(env!("CARGO_BIN_EXE_chungusite")).args(args).arg(&o).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    Some(String::from_utf8(out.stdout).unwrap())
}

fn has(out: &str, line: &str) {
    assert!(out.contains(line), "missing {line:?} in:\n{out}");
}

#[test]
fn prototypes_and_structs_come_from_debug_info() {
    let Some(out) = decompile(&[]) else { return };
    has(&out, "pub struct Node {\n    pub key: i32,\n    pub _pad4: [u8; 4],\n    pub count: i64,\n    pub next: *mut Node,\n    pub flag: u8,\n");
    has(&out, "pub unsafe fn sum(a: *mut i32, n: i32) -> i32 {");
    has(&out, "pub unsafe fn bump(p: *mut Node, d: i32) -> i64 {");
    has(&out, "(*p).count");
    has(&out, "pub fn sgn(x: i32) -> i32 {");
    has(&out, "pub fn div3(x: u32) -> u32 {");
    has(&out, "pub unsafe fn length(p: *mut Node) -> i32 {");
}

#[test]
fn safe_mode_borrows_struct_arguments() {
    let Some(out) = decompile(&["--mode", "safe"]) else { return };
    has(&out, "pub fn bump(p: &mut Node, d: i32) -> i64 {");
    has(&out, "p.count = ");
}

#[test]
fn without_debug_info_types_are_inferred() {
    let Some(out) = decompile(&["--no-dwarf"]) else { return };
    // Field offsets 0, 8 and 24 of `bump`'s argument; a signed compare of the first.
    has(&out, "pub struct S1 {\n    pub f0: i32,\n    pub _pad4: [u8; 4],\n    pub f8: i64,\n");
    has(&out, "pub unsafe fn bump(rdi_p: *mut S1, rsi: i32) -> i64 {");
    // `length` follows a pointer at offset 16 to the same kind of struct.
    has(&out, "pub struct S2 {\n    pub _pad0: [u8; 16],\n    pub f16: *mut S2,\n");
    // Signedness from how the argument is compared and divided.
    has(&out, "pub fn sgn(rdi: i32) -> ");
    has(&out, "pub fn div3(rdi: u32) -> ");
}

#[test]
fn labels_parse_like_dwarf_names() {
    assert_eq!(parse_label("unsigned char"), Some(Label::Int { bytes: 1, signed: false }));
    assert_eq!(parse_label("long int"), Some(Label::Int { bytes: 8, signed: true }));
    assert_eq!(parse_label("const short"), Some(Label::Int { bytes: 2, signed: true }));
    assert_eq!(parse_label("size_t"), Some(Label::Int { bytes: 8, signed: false }));
    assert_eq!(parse_label("_Bool"), Some(Label::Bool));
    assert_eq!(parse_label("void *"), Some(Label::Ptr(None)));
    assert_eq!(parse_label("int *"), Some(Label::Ptr(Some(Box::new(Label::Int { bytes: 4, signed: true })))));
    assert_eq!(parse_label("struct foo"), None);
}

#[test]
fn the_gate_turns_down_proposals_the_code_contradicts() {
    let mut t = TyTable::new();
    let int = |bytes, evidence| VarFacts { pointer: false, bytes, evidence, boolish: false, pointee: None };
    let i32_ = Label::Int { bytes: 4, signed: true };
    // Wider than the code needs is fine; narrower, the wrong sign, or not a pointer is not.
    assert_eq!(accept(&i32_, &int(1, 0), &mut t), Some(t.int(4, true)));
    assert_eq!(accept(&i32_, &int(8, 0), &mut t), None);
    assert_eq!(accept(&i32_, &int(4, -2), &mut t), None);
    assert_eq!(accept(&Label::Int { bytes: 4, signed: false }, &int(4, 1), &mut t), None);
    assert_eq!(accept(&Label::Bool, &int(1, 0), &mut t), None);
    assert_eq!(accept(&Label::Bool, &VarFacts { boolish: true, ..int(1, 0) }, &mut t), Some(TyId::BOOL));
    let ptr = VarFacts { pointer: true, ..int(8, 0) };
    assert_eq!(accept(&i32_, &ptr, &mut t), None);
    let p = parse_label("int *").unwrap();
    let want = t.int(4, true);
    assert_eq!(accept(&p, &ptr, &mut t), Some(t.ptr(want, Mutbl::Mut)));
}
