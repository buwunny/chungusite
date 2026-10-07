//! Whole-program recovery (`program.rs`, `abi.rs`, `frame.rs`): signatures from
//! callers and callees together, real calls, and stack slots as values.
use chungusite::emit::Mode;
use chungusite::program::{Input, Program};
use iced_x86::code_asm::*;

type Asm<'a> = (&'a str, &'a dyn Fn(&mut CodeAssembler));

/// Assemble each `(name, code)` at its own address (0x1000, 0x2000, ...), with
/// `call`s between them by address, and recover the program.
fn program(funcs: &[Asm]) -> (Program, Vec<Vec<u8>>) {
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
    let p = Program::build(inputs, None, false);
    (p, code)
}

fn addr(i: usize) -> u64 {
    0x1000 * (i as u64 + 1)
}

fn emitted(p: &Program, mode: Mode) -> Vec<String> {
    p.emit_all(mode).into_iter().map(|o| o.expect("lifted").0).collect()
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
    assert!(src[1].contains("inc(rdi as u64)"), "{}", src[1]);
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
