//! `--refine` end to end: the fixture classifier as a `TypeModel`, compared with
//! probabilities computed in Python (`tokenizers`, `onnxruntime` 1.29) from the
//! same text, tokenized with the fixture's saved left truncation (192 tokens) as
//! `train_types.py` does. Needs a libonnxruntime, so it is ignored by default:
//!
//!     ORT_DYLIB_PATH=/path/to/libonnxruntime.so cargo test --features ml -- --ignored
mod common;
use chungusite::{
    ir::{Function, Idx, ValueId},
    lift::Lifter,
    refine,
    types::{Proposal, TypeModel, Var},
};

const DIR: &str = "tests/fixtures/tiny-types";

/// `if p == null { 0 } else { p[1] = v; p[2] }`
const CFG: &[u8] = &[
    0x48, 0x85, 0xFF, 0x74, 0x09, 0x48, 0x89, 0x77, 0x08, 0x48, 0x8B, 0x47, 0x10, 0xC3, 0x31, 0xC0, 0xC3,
];

fn close(got: &Option<Proposal>, label: &str, score: f32) {
    let p = got.as_ref().unwrap_or_else(|| panic!("no proposal, want {label} {score}"));
    assert_eq!(p.label, label);
    assert!((p.score - score).abs() < 1e-4, "got {p:?}, want {score}");
}

#[test]
fn the_model_reads_the_function_then_the_variable() {
    let mut f = Function::with_capacity(64, 8);
    Lifter::new().lift(CFG, 0x1000, &mut f).unwrap();
    let text = refine::to_var_text(&f, ValueId::new(0));
    assert!(text.starts_with(&refine::to_text(&f)) && text.ends_with(" return v10;\nvar v0"), "{text}");
}

#[test]
#[ignore = "needs ORT_DYLIB_PATH pointing at libonnxruntime"]
fn proposals_match_python_reference() {
    let m = refine::model::TypeClassifier::load(DIR, 0.0, 1).unwrap();
    let v = ValueId::new;
    let mut f = Function::with_capacity(64, 8);
    Lifter::new().lift(CFG, 0x1000, &mut f).unwrap();
    // An argument never read gets no row, and no answer.
    let vars = [Var::Arg { j: 0, value: Some(v(0)) }, Var::Arg { j: 1, value: None }, Var::Ret { value: v(8) }];
    let got = m.propose(&f, &vars);
    close(&got[0], "char *", 0.508303);
    assert_eq!(got[1], None);
    close(&got[2], "char *", 0.5053133);

    // 1,162 tokens, so the start of the function is cut and `var v48` is kept.
    let code = &common::random_programs(15, 0x1234_5678_9ABC_DEF1)[14];
    Lifter::new().lift(code, common::BASE, &mut f).unwrap();
    close(&m.propose(&f, &[Var::Ret { value: v(48) }])[0], "int", 0.67512286);

    // Below the threshold: no proposal.
    let strict = refine::model::TypeClassifier::load(DIR, 0.6, 1).unwrap();
    close(&strict.propose(&f, &[Var::Ret { value: v(48) }])[0], "int", 0.67512286);
    Lifter::new().lift(CFG, 0x1000, &mut f).unwrap();
    assert_eq!(strict.propose(&f, &vars), vec![None, None, None]);
}
