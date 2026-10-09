//! Whole-program recovery (`program.rs`, `abi.rs`, `frame.rs`): signatures from
//! callers and callees together, real calls, and stack slots as values.
use chungusite::emit::Mode;
use chungusite::ir::Function;
use chungusite::borrow::Root;
use chungusite::program::{BuildOptions, Input, Options, Program};
use chungusite::types::{Proposal, TypeModel, Var};
use iced_x86::code_asm::*;

type Asm<'a> = (&'a str, &'a dyn Fn(&mut CodeAssembler));

/// Assemble each `(name, code)` at its own address (0x1000, 0x2000, ...), with
/// `call`s between them by address, and recover the program.
fn program(funcs: &[Asm]) -> (Program, Vec<Vec<u8>>) {
    program_with(funcs, BuildOptions::default())
}

fn program_with(funcs: &[Asm], opts: BuildOptions) -> (Program, Vec<Vec<u8>>) {
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
    let p = Program::build_with(inputs, None, false, opts);
    (p, code)
}

fn addr(i: usize) -> u64 {
    0x1000 * (i as u64 + 1)
}

fn emitted(p: &Program, mode: Mode) -> Vec<String> {
    p.emit_all(mode, &|_| None).into_iter().map(|o| o.expect("lifted").0).collect()
}

#[test]
fn arguments_come_from_what_callees_read() {
    // inc(x) = x + 1; caller(a, b) = inc(a) + b, keeping b in rbx across the call
    let (p, _) = program(&[
        ("inc", &|a| {
            a.lea(rax, qword_ptr(rdi + 1)).unwrap();
            a.ret().unwrap();
        }),
        ("caller", &|a| {
            a.push(rbx).unwrap();
            a.mov(rbx, rsi).unwrap();
            a.call(addr(0)).unwrap();
            a.add(rax, rbx).unwrap();
            a.pop(rbx).unwrap();
            a.ret().unwrap();
        }),
    ]);
    assert_eq!((p.funcs[0].sig.args, p.funcs[0].sig.ret), (1, true));
    assert_eq!((p.funcs[1].sig.args, p.funcs[1].sig.ret), (2, true));
    let src = emitted(&p, Mode::Fast);
    assert!(src[1].contains("inc(rdi)"), "{}", src[1]);
    assert!(!src[1].contains("todo!"), "{}", src[1]);
    // nothing is saved to the stack any more
    assert!(!src[1].contains("frame"), "{}", src[1]);
}

#[test]
fn a_register_the_callee_preserves_survives_the_call() {
    // gcc's IPA-RA: the caller relies on `dbl` leaving rdi alone
    let (p, _) = program(&[
        ("dbl", &|a| {
            a.lea(rax, qword_ptr(rdi + rdi)).unwrap();
            a.ret().unwrap();
        }),
        ("caller", &|a| {
            a.call(addr(0)).unwrap();
            a.add(rax, rdi).unwrap();
            a.ret().unwrap();
        }),
    ]);
    assert!(p.funcs[0].sig.keeps(7), "dbl keeps rdi");
    assert_eq!(p.funcs[1].sig.args, 1);
    let src = emitted(&p, Mode::Fast);
    assert!(src[1].contains("wrapping_add(rdi)"), "{}", src[1]);
}

#[test]
fn a_callee_can_return_rax_and_rdx() {
    let (p, _) = program(&[
        ("both", &|a| {
            a.mov(rax, rdi).unwrap();
            a.mov(rdx, rsi).unwrap();
            a.ret().unwrap();
        }),
        ("caller", &|a| {
            a.call(addr(0)).unwrap();
            a.add(rax, rdx).unwrap();
            a.ret().unwrap();
        }),
    ]);
    assert!(p.funcs[0].sig.ret2);
    assert!(!p.funcs[1].sig.ret2, "nobody reads the caller's rdx");
    let src = emitted(&p, Mode::Fast);
    assert!(src[0].contains("-> (u64, u64)"), "{}", src[0]);
    assert!(src[1].contains("_pair.1"), "{}", src[1]);
}

#[test]
fn float_arguments_and_results_are_f64() {
    // half(x) = x * 0.5 in xmm0; caller(x, n) = half(x) + n as f64
    let (p, _) = program(&[
        ("half", &|a| {
            a.mov(rax, 0x3fe0_0000_0000_0000u64).unwrap();
            a.movq(xmm1, rax).unwrap();
            a.mulsd(xmm0, xmm1).unwrap();
            a.ret().unwrap();
        }),
        ("caller", &|a| {
            a.push(rbx).unwrap();
            a.mov(rbx, rdi).unwrap();
            a.call(addr(0)).unwrap();
            a.cvtsi2sd(xmm1, rbx).unwrap();
            a.addsd(xmm0, xmm1).unwrap();
            a.pop(rbx).unwrap();
            a.ret().unwrap();
        }),
    ]);
    let s = p.funcs[0].sig;
    assert_eq!((s.args, s.fargs, s.fret), (0, 1, true));
    let s = p.funcs[1].sig;
    assert_eq!((s.args, s.fargs, s.fret), (1, 1, true));
    let src = emitted(&p, Mode::Fast);
    assert!(src[0].contains("fn half(xmm0: f64) -> f64"), "{}", src[0]);
    assert!(src[1].contains("half(f64::from_bits(xmm0"), "{}", src[1]);
}

#[test]
fn an_integer_result_is_not_a_float_one() {
    // trunc(x) = x as i64 leaves x in xmm0, but returns rax
    let (p, _) = program(&[("trunc", &|a| {
        a.cvttsd2si(rax, xmm0).unwrap();
        a.ret().unwrap();
    })]);
    let s = p.funcs[0].sig;
    assert_eq!((s.fargs, s.ret, s.fret), (1, true, false));
}

#[test]
fn an_xmm_register_the_callee_preserves_survives_the_call() {
    // gcc's IPA-RA again: `sq` only touches xmm0, so xmm2 lives across it
    let (p, _) = program(&[
        ("sq", &|a| {
            a.mulsd(xmm0, xmm0).unwrap();
            a.ret().unwrap();
        }),
        ("caller", &|a| {
            a.movapd(xmm2, xmm1).unwrap();
            a.call(addr(0)).unwrap();
            a.addsd(xmm0, xmm2).unwrap();
            a.ret().unwrap();
        }),
    ]);
    assert!(p.funcs[0].sig.keeps(chungusite::lift::XMM0 + 4), "sq keeps xmm2");
    assert_eq!(p.funcs[1].sig.fargs, 2);
}

#[test]
fn stack_slots_become_values() {
    // gcc -O0: the argument goes through [rbp-8]
    let (p, _) = program(&[("o0", &|a| {
        a.push(rbp).unwrap();
        a.mov(rbp, rsp).unwrap();
        a.mov(qword_ptr(rbp - 8), rdi).unwrap();
        a.mov(rax, qword_ptr(rbp - 8)).unwrap();
        a.add(rax, 1).unwrap();
        a.pop(rbp).unwrap();
        a.ret().unwrap();
    })]);
    assert_eq!(p.funcs[0].sig.args, 1);
    let src = emitted(&p, Mode::Fast);
    assert!(!src[0].contains("frame") && !src[0].contains("unsafe"), "{}", src[0]);
}

#[test]
fn a_local_whose_address_escapes_stays_in_a_frame() {
    let (p, _) = program(&[
        ("load", &|a| {
            a.mov(rax, qword_ptr(rdi)).unwrap();
            a.ret().unwrap();
        }),
        ("caller", &|a| {
            a.sub(rsp, 24).unwrap();
            a.mov(qword_ptr(rsp + 8), rdi).unwrap();
            a.lea(rdi, qword_ptr(rsp + 8)).unwrap();
            a.call(addr(0)).unwrap();
            a.add(rsp, 24).unwrap();
            a.ret().unwrap();
        }),
    ]);
    assert_eq!(p.funcs[1].sig.args, 1);
    let src = emitted(&p, Mode::Fast);
    assert!(src[1].contains("let mut frame = [0u128; 2];"), "{}", src[1]);
}

#[test]
fn stack_arguments_past_the_sixth() {
    // seventh(a0..a6) = a6, read from [rsp+8]
    let (p, _) = program(&[("seventh", &|a| {
        a.mov(rax, qword_ptr(rsp + 8)).unwrap();
        a.ret().unwrap();
    })]);
    assert_eq!(p.funcs[0].sig.stack_args, 1);
    let src = emitted(&p, Mode::Fast);
    assert!(src[0].contains("arg6: u64") && !src[0].contains("frame"), "{}", src[0]);
}

// ---- safe mode across calls (docs/ownership.md, stages 5-6) ----

#[test]
fn a_local_lent_to_a_callee_is_a_borrow_of_the_frame() {
    let (p, _) = program(&[
        ("load", &|a| {
            a.mov(rax, qword_ptr(rdi)).unwrap();
            a.ret().unwrap();
        }),
        ("caller", &|a| {
            a.sub(rsp, 24).unwrap();
            a.mov(qword_ptr(rsp + 8), rdi).unwrap();
            a.lea(rdi, qword_ptr(rsp + 8)).unwrap();
            a.call(addr(0)).unwrap();
            a.add(rsp, 24).unwrap();
            a.ret().unwrap();
        }),
    ]);
    let src = emitted(&p, Mode::Safe);
    assert!(src[0].starts_with("pub fn load(rdi_ref: &[u8])"), "{}", src[0]);
    assert!(src[1].starts_with("pub fn caller("), "{}", src[1]);
    assert!(src[1].contains("struct Frame([u8; 32]);"), "{}", src[1]);
    assert!(src[1].contains("load(&frame.0["), "{}", src[1]);
}

/// `add(d, s)`: `*d += *s`, and a caller lending it two stack slots.
fn add_into(a: &mut CodeAssembler) {
    a.mov(rax, qword_ptr(rsi)).unwrap();
    a.add(qword_ptr(rdi), rax).unwrap();
    a.ret().unwrap();
}

fn lend_two(second: i32) -> impl Fn(&mut CodeAssembler) {
    move |a| {
        a.sub(rsp, 24).unwrap();
        a.mov(qword_ptr(rsp), rdi).unwrap();
        a.mov(qword_ptr(rsp + 8), rsi).unwrap();
        a.mov(rdi, rsp).unwrap();
        a.lea(rsi, qword_ptr(rsp + second)).unwrap();
        a.call(addr(0)).unwrap();
        a.mov(rax, qword_ptr(rsp)).unwrap();
        a.add(rsp, 24).unwrap();
        a.ret().unwrap();
    }
}

#[test]
fn two_locals_lent_at_once_are_split() {
    let caller = lend_two(8);
    let (p, _) = program(&[("add", &add_into), ("caller", &caller)]);
    let src = emitted(&p, Mode::Safe);
    assert!(src[0].starts_with("pub fn add(rdi_ref: &mut [u8], rsi_ref: &[u8])"), "{}", src[0]);
    assert!(src[1].contains("split_at_mut"), "{}", src[1]);
    assert!(src[1].starts_with("pub fn caller("), "{}", src[1]);
}

#[test]
fn the_same_local_lent_twice_calls_the_raw_twin() {
    // add(&x, &x): a mutable and a shared borrow of the same bytes. The callee
    // keeps its slices for other callers; this one calls `add_raw`.
    let caller = lend_two(0);
    let (p, _) = program(&[("add", &add_into), ("caller", &caller)]);
    let src = emitted(&p, Mode::Safe);
    assert!(src[0].starts_with("pub fn add(rdi_ref: &mut [u8], rsi_ref: &[u8])"), "{}", src[0]);
    // fast mode, with the pointee types recovered from the accesses
    assert!(src[0].contains("pub unsafe fn add_raw(rdi_p: *mut u64, rsi_p: *const u64)"), "{}", src[0]);
    assert!(src[1].contains("add_raw(") && !src[1].contains("&mut"), "{}", src[1]);
}

#[test]
fn an_escaping_local_leaves_the_rest_of_the_frame_safe() {
    // a[2] and b[2], each filled by `set2`; b's address is stored through the
    // argument, so b is raw, and a is still a slice of the frame
    let (p, _) = program(&[
        ("set2", &|a| {
            a.mov(qword_ptr(rdi), 1).unwrap();
            a.mov(qword_ptr(rdi + 8), 2).unwrap();
            a.ret().unwrap();
        }),
        ("caller", &|a| {
            a.push(rbx).unwrap();
            a.sub(rsp, 32).unwrap();
            a.mov(rbx, rdi).unwrap();
            a.mov(rdi, rsp).unwrap();
            a.call(addr(0)).unwrap();
            a.lea(rdi, qword_ptr(rsp + 16)).unwrap();
            a.call(addr(0)).unwrap();
            a.lea(rax, qword_ptr(rsp + 16)).unwrap();
            a.mov(qword_ptr(rbx), rax).unwrap();
            a.mov(rax, qword_ptr(rsp)).unwrap();
            a.add(rax, qword_ptr(rsp + 8)).unwrap();
            a.add(rsp, 32).unwrap();
            a.pop(rbx).unwrap();
            a.ret().unwrap();
        }),
    ]);
    let src = emitted(&p, Mode::Safe);
    assert!(src[1].contains("set2(&mut frame.0["), "{}", src[1]);
    assert!(src[1].contains("set2_raw("), "{}", src[1]);
    assert!(src[1].contains("u64::from_le_bytes(frame.0["), "{}", src[1]);
}

#[test]
fn a_local_lent_to_a_raw_twin_stays_safe() {
    // add(p, &x) where p escapes: the call goes to `add_raw`, which gets a
    // pointer made from the frame's slice, and x is still read through it
    let (p, _) = program(&[
        ("add", &add_into),
        ("caller", &|a| {
            a.sub(rsp, 24).unwrap();
            a.mov(qword_ptr(rsp + 8), rsi).unwrap();
            a.mov(qword_ptr(0x9000), rdi).unwrap();
            a.lea(rsi, qword_ptr(rsp + 8)).unwrap();
            a.call(addr(0)).unwrap();
            a.mov(rax, qword_ptr(rsp + 8)).unwrap();
            a.add(rsp, 24).unwrap();
            a.ret().unwrap();
        }),
    ]);
    let src = emitted(&p, Mode::Safe);
    assert!(src[1].contains("add_raw("), "{}", src[1]);
    assert!(src[1].contains("frame.0.as_ptr() as u64"), "{}", src[1]);
    assert!(src[1].contains("u64::from_le_bytes(frame.0["), "{}", src[1]);
}

/// Stand-ins for `malloc`, `free` and `memcpy`: safe mode goes by their names,
/// and the program by the arguments they read.
fn stub_malloc(a: &mut CodeAssembler) {
    a.mov(rax, rdi).unwrap();
    a.ret().unwrap();
}

fn stub_free(a: &mut CodeAssembler) {
    a.mov(rax, rdi).unwrap();
    a.ret().unwrap();
}

fn stub_memcpy(a: &mut CodeAssembler) {
    a.lea(rax, qword_ptr(rdi + rsi)).unwrap();
    a.add(rax, rdx).unwrap();
    a.ret().unwrap();
}

/// p = malloc(16); *p = rdi; r = *p; free(p); (optionally r = *p again); return r
fn use_heap(read_after_free: bool) -> impl Fn(&mut CodeAssembler) {
    move |a| {
        a.push(rbx).unwrap();
        a.push(r12).unwrap();
        a.push(rbx).unwrap();
        a.mov(rbx, rdi).unwrap();
        a.mov(edi, 16).unwrap();
        a.call(addr(0)).unwrap();
        a.mov(r12, rax).unwrap();
        a.mov(qword_ptr(r12), rbx).unwrap();
        a.mov(rbx, qword_ptr(r12)).unwrap();
        a.mov(rdi, r12).unwrap();
        a.call(addr(1)).unwrap();
        if read_after_free {
            a.add(rbx, qword_ptr(r12)).unwrap();
        }
        a.mov(rax, rbx).unwrap();
        a.pop(rbx).unwrap();
        a.pop(r12).unwrap();
        a.pop(rbx).unwrap();
        a.ret().unwrap();
    }
}

#[test]
fn an_allocation_used_then_freed_is_a_box() {
    let user = use_heap(false);
    let (p, _) = program(&[("malloc", &stub_malloc), ("free", &stub_free), ("user", &user)]);
    let src = emitted(&p, Mode::Safe);
    assert!(src[2].contains(": Box<[u8]> = Box::default();"), "{}", src[2]);
    assert!(src[2].contains("vec![0u8; "), "{}", src[2]);
    assert!(src[2].contains("= Box::default(); //"), "free drops the box: {}", src[2]);
    assert!(src[2].starts_with("pub fn user("), "{}", src[2]);
}

#[test]
fn a_use_after_free_keeps_the_allocation_raw() {
    let user = use_heap(true);
    let (p, _) = program(&[("malloc", &stub_malloc), ("free", &stub_free), ("user", &user)]);
    let src = emitted(&p, Mode::Safe);
    assert!(!src[2].contains("Box<[u8]>"), "{}", src[2]);
    assert!(src[2].contains("malloc("), "{}", src[2]);
    assert!(src[2].starts_with("pub unsafe fn user("), "{}", src[2]);
}

#[test]
fn memcpy_between_safe_roots_is_a_slice_copy() {
    // copy 16 bytes from the argument into a local, return the second word
    let user = |a: &mut CodeAssembler| {
        a.sub(rsp, 40).unwrap();
        a.mov(rsi, rdi).unwrap();
        a.mov(rdi, rsp).unwrap();
        a.mov(edx, 16).unwrap();
        a.call(addr(0)).unwrap();
        a.mov(rax, qword_ptr(rsp + 8)).unwrap();
        a.add(rsp, 40).unwrap();
        a.ret().unwrap();
    };
    let (p, _) = program(&[("memcpy", &stub_memcpy), ("user", &user)]);
    let src = emitted(&p, Mode::Safe);
    assert!(src[1].contains(".copy_from_slice(&rdi_ref["), "{}", src[1]);
    assert!(src[1].starts_with("pub fn user(rdi_ref: &[u8])"), "{}", src[1]);
}

/// Answers every argument with `arg` and the return value with `ret`.
struct Fixed {
    arg: &'static str,
    ret: &'static str,
}

impl TypeModel for Fixed {
    fn propose(&self, _: &Function, vars: &[Var]) -> Vec<Option<Proposal>> {
        let p = |label: &str| Some(Proposal { label: label.into(), score: 1.0 });
        vars.iter().map(|v| p(if matches!(v, Var::Ret { .. }) { self.ret } else { self.arg })).collect()
    }
}

#[test]
fn a_type_models_proposals_pass_the_gate_or_change_nothing() {
    // f(x) = (u32)x + 1: an argument read at 32 bits, a 32-bit result
    let f: Asm = ("f", &|a| {
        a.mov(eax, edi).unwrap();
        a.add(eax, 1).unwrap();
        a.ret().unwrap();
    });
    // `int` fits the argument; a `char` result says less than the code shows.
    let m = Fixed { arg: "int", ret: "char" };
    let (p, _) = program_with(&[f], BuildOptions { asm: true, dwarf: true, model: Some(&m), dataset: false });
    assert_eq!((p.type_stats.accepted, p.type_stats.rejected), (1, 1));
    let src = emitted(&p, Mode::Fast);
    assert!(src[0].contains("fn f(rdi: i32) -> u32 {"), "{}", src[0]);
}

// ---- what callees keep (docs/ownership.md, "Calls") ----

/// The frame objects of `p.funcs[i]` that aren't safe.
fn raw_frame(p: &Program, i: usize) -> Vec<Root> {
    let a = p.analyses(&Options::default())[i].clone().expect("safe mode analysis");
    a.roots.iter().enumerate().filter(|&(r, root)| matches!(root, Root::Frame(_)) && !a.safe[r]).map(|(_, &root)| root).collect()
}

/// `caller(x)`: a local holding `x`, a second local holding its address, and the
/// second lent to `callee` (index 0) before `x` is read back.
fn lend_a_reference(a: &mut CodeAssembler) {
    a.sub(rsp, 40).unwrap();
    a.mov(qword_ptr(rsp + 16), rdi).unwrap();
    a.lea(rax, qword_ptr(rsp + 16)).unwrap();
    a.mov(qword_ptr(rsp), rax).unwrap();
    a.mov(rdi, rsp).unwrap();
    a.call(addr(0)).unwrap();
    a.add(rax, qword_ptr(rsp + 16)).unwrap();
    a.add(rsp, 40).unwrap();
    a.ret().unwrap();
}

#[test]
fn a_reference_lent_inside_a_struct_stays_safe_if_the_callee_keeps_nothing() {
    // deref(pp) = **pp
    let (p, _) = program(&[
        ("deref", &|a| {
            a.mov(rax, qword_ptr(rdi)).unwrap();
            a.mov(rax, qword_ptr(rax)).unwrap();
            a.ret().unwrap();
        }),
        ("caller", &lend_a_reference),
    ]);
    let a = p.analyses(&Options::default())[0].clone().unwrap();
    assert!(!a.params[0].keeps, "{:?}", a.params[0]);
    assert_eq!(raw_frame(&p, 1), []);
}

#[test]
fn a_callee_that_keeps_a_lent_reference_makes_it_escape() {
    // stash(pp): a global = *pp
    let (p, _) = program(&[
        ("stash", &|a| {
            a.mov(rax, qword_ptr(rdi)).unwrap();
            a.mov(qword_ptr(0x9000), rax).unwrap();
            a.ret().unwrap();
        }),
        ("caller", &lend_a_reference),
    ]);
    let a = p.analyses(&Options::default())[0].clone().unwrap();
    assert!(a.params[0].keeps, "{:?}", a.params[0]);
    assert_ne!(raw_frame(&p, 1), []);
}

#[test]
fn a_returned_pointer_loaded_from_an_argument_still_borrows() {
    // first(v) = v.ptr, like `Vec::as_ptr`; caller reads x through it
    let (p, _) = program(&[
        ("first", &|a| {
            a.mov(rax, qword_ptr(rdi)).unwrap();
            a.ret().unwrap();
        }),
        ("caller", &|a| {
            a.sub(rsp, 40).unwrap();
            a.mov(qword_ptr(rsp + 16), rdi).unwrap();
            a.lea(rax, qword_ptr(rsp + 16)).unwrap();
            a.mov(qword_ptr(rsp), rax).unwrap();
            a.mov(rdi, rsp).unwrap();
            a.call(addr(0)).unwrap();
            a.mov(rax, qword_ptr(rax)).unwrap();
            a.add(rsp, 40).unwrap();
            a.ret().unwrap();
        }),
    ]);
    let a = p.analyses(&Options::default())[0].clone().unwrap();
    assert_eq!(a.params[0].returns_contents, (true, false), "{:?}", a.params[0]);
    assert!(!a.params[0].keeps, "{:?}", a.params[0]);
    assert_eq!(raw_frame(&p, 1), []);
}

#[test]
fn a_pointer_returned_in_rdx_still_borrows() {
    // pair(a, b) = (a, b); caller reads both locals back through the results
    let (p, _) = program(&[
        ("pair", &|a| {
            a.mov(rax, rdi).unwrap();
            a.mov(rdx, rsi).unwrap();
            a.ret().unwrap();
        }),
        ("caller", &|a| {
            a.sub(rsp, 24).unwrap();
            a.mov(qword_ptr(rsp), rdi).unwrap();
            a.mov(qword_ptr(rsp + 8), rsi).unwrap();
            a.mov(rdi, rsp).unwrap();
            a.lea(rsi, qword_ptr(rsp + 8)).unwrap();
            a.call(addr(0)).unwrap();
            a.mov(rax, qword_ptr(rax)).unwrap();
            a.add(rax, qword_ptr(rdx)).unwrap();
            a.add(rsp, 24).unwrap();
            a.ret().unwrap();
        }),
    ]);
    let a = p.analyses(&Options::default())[0].clone().unwrap();
    let reg = |r: u8| a.params.iter().find(|x| x.reg == r).unwrap();
    assert!(reg(7).returned && reg(6).returned2, "{:?}", a.params);
    assert_eq!(raw_frame(&p, 1), []);
}

#[test]
fn an_alignment_check_on_a_local_is_not_an_escape() {
    // the low bits of &x tested, then x read: `(&x as usize) & 7` reveals nothing
    let (p, _) = program(&[("f", &|a| {
        a.sub(rsp, 24).unwrap();
        a.mov(qword_ptr(rsp + 8), rdi).unwrap();
        a.lea(rcx, qword_ptr(rsp + 8)).unwrap();
        a.and(rcx, 7).unwrap();
        a.mov(rax, qword_ptr(rsp + 8)).unwrap();
        a.add(rax, rcx).unwrap();
        a.add(rsp, 24).unwrap();
        a.ret().unwrap();
    })]);
    assert_eq!(raw_frame(&p, 0), []);
}

#[test]
fn many_globals_do_not_crowd_out_the_frame() {
    // 140 globals read, more than there are roots, and a local lent to `load`
    let (p, _) = program(&[
        ("load", &|a| {
            a.mov(rax, qword_ptr(rdi)).unwrap();
            a.ret().unwrap();
        }),
        ("caller", &|a| {
            a.push(rbx).unwrap();
            a.sub(rsp, 16).unwrap();
            a.mov(qword_ptr(rsp + 8), rdi).unwrap();
            a.xor(ebx, ebx).unwrap();
            for g in 0..140u64 {
                a.add(rbx, qword_ptr(0x10_0000 + 0x100 * g)).unwrap();
            }
            a.lea(rdi, qword_ptr(rsp + 8)).unwrap();
            a.call(addr(0)).unwrap();
            a.add(rax, rbx).unwrap();
            a.add(rsp, 16).unwrap();
            a.pop(rbx).unwrap();
            a.ret().unwrap();
        }),
    ]);
    let a = p.analyses(&Options::default())[1].clone().unwrap();
    assert!(a.roots.len() <= 128, "{}", a.roots.len());
    assert!(a.roots.iter().any(|r| matches!(r, Root::Frame(_))), "{:?}", a.roots);
    assert_eq!(raw_frame(&p, 1), []);
}

// ---- readability (roadmap step 4) ----

#[test]
fn each_path_returns_on_its_own() {
    // f(p) = if p == 0 { 0 } else if *p != 5 { *p } else { 7 }: three paths into
    // one `ret`, two of them from inside the `else`
    let (p, _) = program(&[("f", &|a| {
        let mut zero = a.create_label();
        let mut done = a.create_label();
        a.test(rdi, rdi).unwrap();
        a.je(zero).unwrap();
        a.mov(rax, qword_ptr(rdi)).unwrap();
        a.cmp(rax, 5).unwrap();
        a.jne(done).unwrap();
        a.mov(eax, 7).unwrap();
        a.jmp(done).unwrap();
        a.set_label(&mut zero).unwrap();
        a.xor(eax, eax).unwrap();
        a.set_label(&mut done).unwrap();
        a.ret().unwrap();
    })]);
    let src = emitted(&p, Mode::Fast);
    assert!(!src[0].contains("'b"), "{}", src[0]);
    assert_eq!(src[0].matches("return ").count(), 3, "{}", src[0]);
    // the loaded value is declared where it is loaded, not up front
    assert!(!src[0].contains("let mut v"), "{}", src[0]);
}
