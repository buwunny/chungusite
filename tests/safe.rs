//! CFG utilities, SSA cleanup and the safe-mode borrow classifier.
mod common;
use chungusite::{
    borrow::{analyze, Class, ParamBorrow, StackSlot},
    cfg::Cfg,
    dump::dump,
    ir::*,
    lift::Lifter,
    opt::clean,
    verify::verify,
};
use iced_x86::code_asm::*;

fn lift_clean(code: &[u8]) -> Function {
    let mut f = Function::with_capacity(64, 8);
    Lifter::new().lift(code, common::BASE, &mut f).unwrap();
    clean(&mut f);
    verify(&f).unwrap();
    f
}

fn asm(build: impl FnOnce(&mut CodeAssembler)) -> Vec<u8> {
    let mut a = CodeAssembler::new(64).unwrap();
    build(&mut a);
    a.assemble(common::BASE).unwrap()
}

/// The classification of the argument arriving in `reg`.
fn param(f: &Function, reg: AsmRegister64) -> ParamBorrow {
    let n = iced_x86::Register::from(reg).number() as u8;
    analyze(f).params.into_iter().find(|p| p.reg == n).unwrap()
}

#[test]
fn dominators_of_the_loop() {
    let mut f = Function::with_capacity(64, 8);
    Lifter::new().lift(&common::sum_loop(), common::BASE, &mut f).unwrap();
    let cfg = Cfg::new(&f);
    let b = |i| BlockId::new(i);
    assert_eq!(cfg.rpo[0], b(0));
    assert_eq!(cfg.idom[0], None);
    assert_eq!(cfg.idom[1], Some(b(0)));
    assert_eq!(cfg.idom[2], Some(b(1)));
    assert_eq!(cfg.idom[3], Some(b(1)));
    assert_eq!(cfg.preds(b(1)), &[b(0), b(2)]);
    assert!(cfg.dominates(b(1), b(2)) && !cfg.dominates(b(2), b(3)));
}

#[test]
fn clean_removes_loop_invariant_params() {
    let mut f = Function::with_capacity(64, 8);
    Lifter::new().lift(&common::sum_loop(), common::BASE, &mut f).unwrap();
    let stats = clean(&mut f);
    verify(&f).unwrap();
    // rsi and rdi never change in the loop, so bb1 no longer carries them; bb2 and
    // bb3 have a single predecessor, so their params are just bb1's values.
    let expected = "\
bb0(v1, v18, v0):
  v2 = ptr v0 + 8
  store v2 <- v1
  v4 = const 0x0
  v5 = ZExt v4
  jump bb1(v0, v5)
bb1(v19, v6):
  v8 = cmp.Uge v6, v18
  br v8 bb3() bb2()
bb2():
  v11 = ptr v0 + v6*8 + 16
  v12 = load v11
  v14 = Add v19, v12
  v15 = const 0x1
  v16 = Add v6, v15
  jump bb1(v14, v16)
bb3():
  ret v19
";
    assert_eq!(dump(&f), expected, "{stats:?}");
    assert_eq!(stats.trivial_params, 7, "{stats:?}");
}

#[test]
fn sum_loop_needs_mut_slice_like_borrow() {
    let f = lift_clean(&common::sum_loop());
    let p = param(&f, rdi);
    assert_eq!(p.class, Class::Mut);
    assert_eq!(p.fields, vec![(8, true)]);
    assert!(p.indexed); // p[rdx*8 + 16]
    assert!(p.returned); // rax starts as rdi
    assert_eq!(param(&f, rsi).class, Class::NotPointer);
    assert_eq!(param(&f, rcx).class, Class::NotPointer); // stored as a value only
}

#[test]
fn null_checked_argument_is_optional() {
    // if p.is_null() { 0 } else { p[1] = v; p[2] }
    let f = lift_clean(&asm(|a| {
        let mut null = a.create_label();
        a.test(rdi, rdi).unwrap();
        a.je(null).unwrap();
        a.mov(qword_ptr(rdi + 8), rsi).unwrap();
        a.mov(rax, qword_ptr(rdi + 16)).unwrap();
        a.ret().unwrap();
        a.set_label(&mut null).unwrap();
        a.xor(eax, eax).unwrap();
        a.ret().unwrap();
    }));
    let p = param(&f, rdi);
    assert_eq!((p.class, p.nullable), (Class::Mut, true));
    assert_eq!(p.fields, vec![(8, true), (16, false)]);
    assert!(!param(&f, rsi).nullable);
}

#[test]
fn read_only_argument_is_shared() {
    let f = lift_clean(&asm(|a| {
        a.mov(rax, qword_ptr(rdi + 8)).unwrap();
        a.ret().unwrap();
    }));
    let p = param(&f, rdi);
    assert_eq!((p.class, p.fields, p.indexed, p.returned, p.nullable), (Class::Shared, vec![(8, false)], false, false, false));
}

#[test]
fn written_argument_is_mut_and_stored_value_is_not_a_pointer() {
    let f = lift_clean(&asm(|a| {
        a.mov(qword_ptr(rdi), rsi).unwrap();
        a.ret().unwrap();
    }));
    assert_eq!(param(&f, rdi).class, Class::Mut);
    assert_eq!(param(&f, rsi).class, Class::NotPointer);
}

#[test]
fn dereferenced_pointer_stored_to_memory_stays_raw() {
    let f = lift_clean(&asm(|a| {
        a.mov(rax, qword_ptr(rsi)).unwrap();
        a.mov(qword_ptr(rdi), rsi).unwrap(); // *rdi = rsi: rsi outlives this call
        a.ret().unwrap();
    }));
    assert_eq!(param(&f, rsi).class, Class::Raw);
    assert_eq!(param(&f, rdi).class, Class::Mut);
}

#[test]
fn returning_a_field_address_is_a_reborrow() {
    let f = lift_clean(&asm(|a| {
        a.mov(rcx, qword_ptr(rdi + 8)).unwrap();
        a.mov(qword_ptr(rdi), rcx).unwrap();
        a.lea(rax, qword_ptr(rdi + 16)).unwrap();
        a.ret().unwrap();
    }));
    let p = param(&f, rdi);
    assert_eq!(p.class, Class::Mut);
    assert!(p.returned);
    assert_eq!(p.fields, vec![(0, true), (8, false)]);
}

#[test]
fn pointer_advanced_in_a_loop_has_unknown_offset() {
    let f = lift_clean(&asm(|a| {
        let mut top = a.create_label();
        a.xor(eax, eax).unwrap();
        a.set_label(&mut top).unwrap();
        a.mov(rcx, qword_ptr(rdi)).unwrap();
        a.add(rax, rcx).unwrap();
        a.add(rdi, 8).unwrap();
        a.sub(rsi, 1).unwrap();
        a.jne(top).unwrap();
        a.ret().unwrap();
    }));
    let p = param(&f, rdi);
    assert_eq!(p.class, Class::Shared);
    assert!(p.indexed && p.fields.is_empty());
}

#[test]
fn pointer_difference_is_not_an_escape() {
    let f = lift_clean(&asm(|a| {
        a.mov(rcx, qword_ptr(rdi)).unwrap();
        a.mov(qword_ptr(rsi), rcx).unwrap();
        a.mov(rax, rsi).unwrap();
        a.sub(rax, rdi).unwrap(); // a length, like end - begin
        a.ret().unwrap();
    }));
    assert_eq!(param(&f, rdi).class, Class::Shared);
    assert_eq!(param(&f, rsi).class, Class::Mut);
}

#[test]
fn stack_slots_and_address_taken_locals() {
    let f = lift_clean(&asm(|a| {
        a.sub(rsp, 0x18).unwrap();
        a.mov(qword_ptr(rsp + 8), rsi).unwrap(); // spill: promotable
        a.lea(rax, qword_ptr(rsp + 0x10)).unwrap();
        a.mov(qword_ptr(rdi), rax).unwrap(); // address of a local escapes
        a.mov(rax, qword_ptr(rsp + 8)).unwrap();
        a.add(rsp, 0x18).unwrap();
        a.ret().unwrap();
    }));
    let slots = analyze(&f).stack_slots().unwrap();
    assert_eq!(
        slots,
        vec![
            StackSlot { off: -16, read: true, write: true, address_taken: false },
            StackSlot { off: -8, read: false, write: false, address_taken: true },
        ]
    );
}
