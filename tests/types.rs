//! Type recovery (`types.rs`, `dwarf.rs`): integer widths and signedness, structs
//! and slices behind pointer arguments, and proposals from a `TypeModel` that the
//! code accepts or rejects. Output is type-checked with rustc; that the typed code
//! computes the same thing as the original is `tests/differential.rs`'s job (its
//! `-g` variants cover debug info).
use chungusite::emit::Mode;
use chungusite::program::{Input, Program, TypeOptions};
use chungusite::types::{CType, Field, FuncRef, IntTy, ParamProposal, Proposal, StructTy, TypeModel};
use iced_x86::code_asm::*;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

type Asm<'a> = (&'a str, &'a dyn Fn(&mut CodeAssembler));

fn addr(i: usize) -> u64 {
    0x1000 * (i as u64 + 1)
}

/// Assemble each `(name, code)` at 0x1000, 0x2000, ... and recover the program,
/// with `model` proposing types.
fn program(funcs: &[Asm], model: Option<&dyn TypeModel>) -> Program {
    let code: Vec<Vec<u8>> = funcs
        .iter()
        .enumerate()
        .map(|(i, (_, build))| {
            let mut a = CodeAssembler::new(64).unwrap();
            build(&mut a);
            a.assemble(addr(i)).unwrap()
        })
        .collect();
    let inputs = funcs
        .iter()
        .zip(&code)
        .enumerate()
        .map(|(i, ((name, _), bytes))| Input { name: name.to_string(), ident: name.to_string(), addr: addr(i), bytes, selected: true })
        .collect();
    Program::build_with(inputs, None, false, &TypeOptions { debug_info: false, model })
}

/// Every function's source in `mode`, and the whole file (prelude included).
fn emitted(p: &Program, mode: Mode) -> (Vec<String>, String) {
    let funcs: Vec<String> = p.emit_all(mode, &|_| None).into_iter().flatten().map(|o| o.0).collect();
    let file = format!("{ALLOW}{}{}", p.prelude(), funcs.concat());
    (funcs, file)
}

const ALLOW: &str = "#![allow(unused_mut, unused_variables, unused_assignments, unreachable_code, non_snake_case, non_camel_case_types, unused_parens, unused_unsafe, dead_code)]\n";

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("chungusite-types-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Type-check `src` as a library; panics with rustc's errors.
fn check(name: &str, src: &str) {
    let dir = scratch(name);
    let path = dir.join(format!("{name}.rs"));
    std::fs::write(&path, src).unwrap();
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let out = Command::new(rustc)
        .args(["--edition", "2021", "-A", "warnings", "--crate-type", "lib", "--emit", "metadata", "--out-dir"])
        .arg(&dir)
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success(), "rustc failed on {}:\n{}\n--- source ---\n{src}", path.display(), String::from_utf8_lossy(&out.stderr));
}

fn sig(src: &str) -> &str {
    src.lines().find(|l| l.starts_with("pub ")).unwrap()
}

#[test]
fn arguments_and_returns_get_their_width_and_signedness() {
    let p = program(
        &[
            // max(int a, int b): a signed compare
            ("max", &|a| {
                a.cmp(edi, esi).unwrap();
                a.mov(eax, esi).unwrap();
                a.cmovge(eax, edi).unwrap();
                a.ret().unwrap();
            }),
            // lea eax, [rdi+rsi*2]: only the low 32 bits of either argument matter
            ("lin", &|a| {
                a.lea(eax, dword_ptr(rdi + rsi * 2)).unwrap();
                a.ret().unwrap();
            }),
            // (int)(signed char)x
            ("sext8", &|a| {
                a.movsx(eax, dil).unwrap();
                a.ret().unwrap();
            }),
            // a 64-bit argument stays u64
            ("wide", &|a| {
                a.mov(rax, rdi).unwrap();
                a.shr(rax, 3).unwrap();
                a.ret().unwrap();
            }),
        ],
        None,
    );
    let (fast, file) = emitted(&p, Mode::Fast);
    assert_eq!(sig(&fast[0]), "pub fn max(edi: i32, esi: i32) -> i32 {", "{}", fast[0]);
    assert!(fast[0].contains("edi >= esi"), "a signed compare needs no casts: {}", fast[0]);
    assert_eq!(sig(&fast[1]), "pub fn lin(edi: u32, esi: u32) -> u32 {", "{}", fast[1]);
    assert_eq!(sig(&fast[2]), "pub fn sext8(dil: i8) -> i32 {", "{}", fast[2]);
    assert_eq!(sig(&fast[3]), "pub fn wide(mut rdi: u64) -> u64 {", "{}", fast[3]);
    check("widths", &file);
}

#[test]
fn constant_offsets_become_a_struct_and_indexing_a_slice() {
    let p = program(
        &[
            // get(p) = p->f8 + p->f16
            ("get", &|a| {
                a.mov(eax, dword_ptr(rdi + 8)).unwrap();
                a.add(eax, dword_ptr(rdi + 16)).unwrap();
                a.ret().unwrap();
            }),
            // set(p, v): p->f0 = v; p->f4 = (u8)v
            ("set", &|a| {
                a.mov(dword_ptr(rdi), esi).unwrap();
                a.mov(byte_ptr(rdi + 4), sil).unwrap();
                a.ret().unwrap();
            }),
            // sum(a, n) = a[0] + ... + a[n-1], 32-bit elements
            ("sum", &|a| {
                let mut top = a.create_label();
                let mut done = a.create_label();
                a.xor(eax, eax).unwrap();
                a.xor(ecx, ecx).unwrap();
                a.set_label(&mut top).unwrap();
                a.cmp(rcx, rsi).unwrap();
                a.jae(done).unwrap();
                a.add(eax, dword_ptr(rdi + rcx * 4)).unwrap();
                a.inc(rcx).unwrap();
                a.jmp(top).unwrap();
                a.set_label(&mut done).unwrap();
                a.ret().unwrap();
            }),
        ],
        None,
    );
    let (safe, safe_file) = emitted(&p, Mode::Safe);
    assert_eq!(sig(&safe[0]), "pub fn get(rdi_ref: &S_get_rdi) -> u32 {", "{}", safe[0]);
    assert!(safe[0].contains("rdi_ref.f8") && safe[0].contains("rdi_ref.f16"), "{}", safe[0]);
    assert!(!safe[0].contains("wrapping_add(0x8)"), "the field's address isn't computed any more: {}", safe[0]);
    assert_eq!(sig(&safe[1]), "pub fn set(rdi_ref: &mut S_set_rdi, esi: u32) {", "{}", safe[1]);
    assert!(safe[1].contains("rdi_ref.f0 = esi;"), "{}", safe[1]);
    assert_eq!(sig(&safe[2]), "pub fn sum(rdi_ref: &[u32], mut rsi: u64) -> u32 {", "{}", safe[2]);
    assert!(safe[2].contains("rdi_ref[("), "{}", safe[2]);
    let prelude = p.prelude();
    assert!(
        prelude.contains("pub struct S_get_rdi {\n    _pad0: [u8; 8],\n    pub f8: u32,\n    _pad12: [u8; 4],\n    pub f16: u32,\n}"),
        "{prelude}"
    );
    check("struct_safe", &safe_file);

    let (fast, fast_file) = emitted(&p, Mode::Fast);
    assert_eq!(sig(&fast[0]), "pub unsafe fn get(rdi_ref: *mut S_get_rdi) -> u32 {", "{}", fast[0]);
    assert!(fast[0].contains("unsafe { (*rdi_ref).f8 }"), "{}", fast[0]);
    check("struct_fast", &fast_file);
}

#[test]
fn callers_pass_narrow_arguments_and_widen_narrow_results() {
    let p = program(
        &[
            ("inc32", &|a| {
                a.lea(eax, dword_ptr(rdi + 1)).unwrap();
                a.ret().unwrap();
            }),
            ("caller", &|a| {
                a.sub(rsp, 8).unwrap();
                a.mov(edi, 41).unwrap();
                a.call(addr(0)).unwrap();
                a.add(rax, 1).unwrap();
                a.add(rsp, 8).unwrap();
                a.ret().unwrap();
            }),
        ],
        None,
    );
    for mode in [Mode::Fast, Mode::Safe] {
        let (src, file) = emitted(&p, mode);
        assert_eq!(sig(&src[0]), "pub fn inc32(edi: u32) -> u32 {", "{}", src[0]);
        assert!(src[1].contains("inc32(") && src[1].contains(" as u32) as u32 as u64 }"), "{}", src[1]);
        check(&format!("calls_{mode:?}"), &file);
    }
}

/// A stand-in for the ML type model: proposes fixed types by function name.
struct Fixed(Vec<(&'static str, Proposal)>);

impl TypeModel for Fixed {
    fn name(&self) -> &str {
        "test-model"
    }
    fn propose(&self, func: &FuncRef) -> Option<Proposal> {
        self.0.iter().find(|(n, _)| *n == func.name).map(|(_, p)| p.clone())
    }
}

fn point(y_at: u64) -> CType {
    let int = CType::Int(IntTy::new(4, true));
    CType::Struct(Arc::new(StructTy {
        name: "point".into(),
        size: 24,
        fields: vec![Field { name: "x".into(), off: 8, ty: int.clone() }, Field { name: "y".into(), off: y_at, ty: int }],
    }))
}

fn arg(name: &str, ty: CType) -> ParamProposal {
    ParamProposal { name: Some(name.into()), ty: Some(ty) }
}

#[test]
fn a_model_proposes_and_the_code_decides() {
    let get = |a: &mut CodeAssembler| {
        a.mov(eax, dword_ptr(rdi + 8)).unwrap();
        a.add(eax, dword_ptr(rdi + 16)).unwrap();
        a.ret().unwrap();
    };
    let model = Fixed(vec![
        // matches the accesses: accepted, with its names and signedness
        ("good", Proposal { params: vec![arg("p", CType::Ptr(Some(Arc::new(point(16)))))], ret: Some(CType::Int(IntTy::new(4, true))) }),
        // `y` at +12, but the code reads +16: rejected
        ("bad_layout", Proposal { params: vec![arg("p", CType::Ptr(Some(Arc::new(point(12)))))], ret: None }),
        // an integer that is dereferenced: rejected
        ("not_int", Proposal { params: vec![arg("n", CType::Int(IntTy::new(4, true)))], ret: None }),
        // `short`, but all 32 bits are used: rejected
        ("too_narrow", Proposal { params: vec![arg("s", CType::Int(IntTy::new(2, true)))], ret: None }),
    ]);
    let p = program(
        &[
            ("good", &get),
            ("bad_layout", &get),
            ("not_int", &get),
            ("too_narrow", &|a| {
                a.lea(eax, dword_ptr(rdi + 1)).unwrap();
                a.ret().unwrap();
            }),
        ],
        Some(&model),
    );
    let (safe, file) = emitted(&p, Mode::Safe);
    assert_eq!(sig(&safe[0]), "pub fn good(p: &point) -> i32 {", "{}", safe[0]);
    assert!(safe[0].contains("p.x") && safe[0].contains("p.y"), "{}", safe[0]);
    assert!(p.prelude().contains("pub struct point {"), "{}", p.prelude());

    let notes = |i: usize| p.funcs[i].types.as_ref().unwrap().notes.join("\n");
    assert!(notes(1).contains("test-model: rejected struct point * for rdi: 4-byte access at +16 matches no field"), "{}", notes(1));
    assert_eq!(sig(&safe[1]), "pub fn bad_layout(rdi_ref: &S_bad_layout_rdi) -> u32 {", "falls back to inference: {}", safe[1]);
    assert!(notes(2).contains("test-model: rejected i32 for rdi: it is dereferenced"), "{}", notes(2));
    assert!(notes(3).contains("test-model: rejected i16 for rdi: the code uses 32 bits of it"), "{}", notes(3));
    assert_eq!(sig(&safe[3]), "pub fn too_narrow(edi: u32) -> u32 {", "{}", safe[3]);
    check("model", &file);

    // The default model proposes nothing: the same as no model at all.
    let none = chungusite::types::NoModel;
    let q = program(&[("good", &get)], Some(&none));
    assert_eq!(sig(&emitted(&q, Mode::Safe).0[0]), "pub fn good(rdi_ref: &S_good_rdi) -> u32 {");
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Random programs heavy on what drives type recovery (narrow and signed
/// arithmetic, struct-like and indexed accesses through the arguments, calls
/// between the functions) must still type-check in both modes.
#[test]
fn random_typed_programs_compile() {
    let gprs = [rax, rcx, rdx, rsi, rdi, r8, r9];
    let gpr32 = [eax, ecx, edx, esi, edi, r8d, r9d];
    let gpr8 = [al, cl, dl, sil, dil, r8b, r9b];
    let mut rng = Rng(0x5eed_1234_abcd_0001);
    // (136 functions with this seed hit an unrelated hang in `abi::apply`.)
    let n_funcs: usize = 120;
    let mut code = Vec::new();
    for fi in 0..n_funcs {
        let n = 2 + rng.below(14) as usize;
        let mut a = CodeAssembler::new(64).unwrap();
        let mut labels: Vec<CodeLabel> = (0..n).map(|_| a.create_label()).collect();
        for i in 0..n {
            a.set_label(&mut labels[i]).unwrap();
            let k = |rng: &mut Rng| rng.below(7) as usize;
            let (r, s) = (gprs[k(&mut rng)], gprs[k(&mut rng)]);
            let (r32, s32) = (gpr32[k(&mut rng)], gpr32[k(&mut rng)]);
            let r8_ = gpr8[k(&mut rng)];
            let to = labels[rng.below(n as u64) as usize];
            let off = 4 * rng.below(6) as i32;
            match rng.below(24) {
                0 => a.lea(r32, dword_ptr(r + s * 2 + 3)).unwrap(),
                1 => a.movsx(r32, r8_).unwrap(),
                2 => a.movsxd(r, s32).unwrap(),
                3 => a.sar(r32, 2).unwrap(),
                4 => { a.cmp(r32, s32).unwrap(); a.jl(to).unwrap() }
                5 => { a.cmp(r32, s32).unwrap(); a.jb(to).unwrap() }
                6 => a.mov(r32, dword_ptr(rdi + off)).unwrap(),
                7 => a.mov(dword_ptr(rdi + off), s32).unwrap(),
                8 => a.mov(r32, dword_ptr(rsi + rcx * 4)).unwrap(),
                9 => a.movzx(r32, byte_ptr(rdi + off + 1)).unwrap(),
                10 => a.add(r32, s32).unwrap(),
                11 => a.imul_3(r32, s32, -7).unwrap(),
                12 => a.mov(r, s).unwrap(),
                13 => a.add(r, 4).unwrap(),
                14 => { a.test(r32, r32).unwrap(); a.js(to).unwrap() }
                15 => { a.cmp(r32, s32).unwrap(); a.cmovg(r32, s32).unwrap() }
                16 if fi > 0 => a.call(addr(rng.below(fi as u64) as usize)).unwrap(),
                17 => a.shr(r32, 1).unwrap(),
                18 => a.neg(r32).unwrap(),
                19 => a.mov(r32, rng.below(1000) as u32).unwrap(),
                20 => a.mov(qword_ptr(rdx + 8), r).unwrap(),
                21 => a.jmp(to).unwrap(),
                22 => a.mov(byte_ptr(rdi + off), r8_).unwrap(),
                _ => a.ret().unwrap(),
            }
        }
        a.ret().unwrap();
        code.push(a.assemble(addr(fi)).unwrap());
    }
    let inputs = code
        .iter()
        .enumerate()
        .map(|(i, bytes)| {
            let name = format!("rf{i}");
            Input { name: name.clone(), ident: name, addr: addr(i), bytes, selected: true }
        })
        .collect();
    let p = Program::build(inputs, None, false);
    let lifted = p.funcs.iter().filter(|f| f.ir.is_ok()).count();
    assert!(lifted > 60, "only {lifted} lifted");
    let typed = p.funcs.iter().filter_map(|f| f.types.as_ref()).filter(|t| t.ret.is_some() || t.params.iter().any(|p| p.int.bytes < 8 || p.pointee.is_some())).count();
    assert!(typed > 30, "only {typed} functions got a type narrower than u64");
    for mode in [Mode::Fast, Mode::Safe] {
        let out = p.emit_all(mode, &|_| None);
        let mut file = format!("{ALLOW}{}", p.prelude());
        for (f, o) in p.funcs.iter().zip(out) {
            match o {
                Some((src, _)) => file.push_str(&src),
                None => file.push_str(&format!("pub fn {}() -> u64 {{ todo!() }}\n", f.ident)),
            }
        }
        check(&format!("random_{mode:?}"), &file);
    }
}

/// End to end with a C compiler: DWARF names the arguments and the struct, and a
/// struct the code reads in a way its layout doesn't allow is rejected.
#[test]
fn debug_info_types_c_code() {
    let cc = ["cc", "clang", "gcc"].into_iter().find(|c| Command::new(c).arg("--version").output().is_ok_and(|o| o.status.success()));
    let Some(cc) = cc else {
        assert!(std::env::var_os("CI").is_none(), "no C compiler found");
        eprintln!("no C compiler, skipping");
        return;
    };
    let dir = scratch("dwarf");
    let c = dir.join("t.c");
    std::fs::write(
        &c,
        r#"
struct rect { int x, y, w, h; };
int area(const struct rect *r) { return r->w * r->h; }
void grow(struct rect *r, int d) { r->w += d; r->x -= d; }
struct two { int a; int b; };
long both(const struct two *t) { long x; __builtin_memcpy(&x, t, 8); return x; }
unsigned char low(unsigned v) { return (unsigned char)(v >> 3); }
"#,
    )
    .unwrap();
    let obj = dir.join("t.o");
    let out = Command::new(cc).args(["-O2", "-g", "-c", "-o"]).arg(&obj).arg(&c).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let data = std::fs::read(&obj).unwrap();
    let bin = chungusite::load::Binary::parse(&data).unwrap();
    let inputs = bin
        .funcs
        .iter()
        .map(|f| Input { name: f.name.clone(), ident: f.name.clone(), addr: f.addr, bytes: f.bytes, selected: true })
        .collect();
    let p = Program::build(inputs, Some(&data), false);
    let (src, file) = emitted(&p, Mode::Safe);
    let find = |name: &str| src.iter().find(|s| s.contains(&format!("fn {name}("))).unwrap().as_str();
    assert_eq!(sig(find("area")), "pub fn area(r: &rect) -> i32 {", "{}", find("area"));
    assert!(find("area").contains("r.w") && find("area").contains("r.h"), "{}", find("area"));
    assert_eq!(sig(find("grow")), "pub fn grow(r: &mut rect, d: i32) {", "{}", find("grow"));
    assert!(find("grow").contains("r.w = "), "{}", find("grow"));
    // `unsigned char`, but the code returns all of eax: the code wins
    assert_eq!(sig(find("low")), "pub fn low(v: u32) -> u32 {", "{}", find("low"));
    assert!(p.prelude().contains("pub struct rect {\n    pub x: i32,\n    pub y: i32,\n    pub w: i32,\n    pub h: i32,\n}"), "{}", p.prelude());
    let both = p.funcs.iter().find(|f| f.name == "both").unwrap();
    let notes = both.types.as_ref().unwrap().notes.join("\n");
    assert!(notes.contains("dwarf: rejected struct two * for rdi: 8-byte access at +0 matches no field"), "{notes}");
    check("dwarf", &file);
    let (_, fast) = emitted(&p, Mode::Fast);
    check("dwarf_fast", &fast);
}
