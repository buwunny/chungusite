//! End to end through ONNX Runtime: lift, serialize, tokenize from the cache, run the
//! fixture classifier, and compare against logits computed independently in Python
//! (onnxruntime 1.30, same model, same ids). Needs a libonnxruntime, so it is ignored
//! by default:
//!
//!     ORT_DYLIB_PATH=/path/to/libonnxruntime.so cargo test --features ml -- --ignored
use chungusite::{ir::Function, lift::Lifter, refine};

const CFG: &[u8] = &[
    0x48, 0x85, 0xFF, 0x74, 0x09, 0x48, 0x89, 0x77, 0x08, 0x48, 0x8B, 0x47, 0x10, 0xC3, 0x31, 0xC0, 0xC3,
];

#[test]
#[ignore = "needs ORT_DYLIB_PATH pointing at libonnxruntime"]
fn classifier_matches_python_reference() {
    let dir = "tests/fixtures/tiny-types";
    let mut f = Function::with_capacity(64, 8);
    Lifter::new().lift(CFG, 0x1000, &mut f).unwrap();
    let mut cache = refine::cache::TokenCache::from_tokenizer_file(format!("{dir}/tokenizer.json")).unwrap();
    let mut ids = Vec::new();
    refine::to_ids(&f, &mut cache, &mut ids);
    assert_eq!(ids, EXPECTED_IDS);

    let mut model = refine::infer::Refiner::new(format!("{dir}/model.onnx"), 1, 0, 256).unwrap();
    // Batch of two, the second shorter, to exercise padding and the attention mask.
    let logits = model.classify(&[&ids, &ids[..20]]).unwrap().to_vec();
    assert_eq!(logits.len(), 4);
    for (got, want) in logits.iter().zip(EXPECTED_LOGITS) {
        assert!((got - want).abs() < 1e-4, "got {logits:?}, want {EXPECTED_LOGITS:?}");
    }
}

/// From the fixture tokenizer in Python (`tokenizers`), matching the Rust cache.
const EXPECTED_IDS: &[u32] = &[
    275, 20, 12, 90, 270, 16, 261, 20, 274, 203, 261, 21, 262, 225, 20, 31, 203, 261, 22, 262,
    261, 20, 262, 33, 261, 21, 31, 203, 225, 77, 74, 261, 22, 225, 75, 83, 88, 83, 225, 275, 22,
    12, 13, 31, 225, 73, 80, 87, 73, 225, 75, 83, 88, 83, 225, 275, 21, 12, 90, 270, 16, 261,
    20, 13, 31, 203, 275, 21, 12, 90, 23, 16, 261, 24, 274, 203, 261, 25, 262, 261, 24, 263,
    267, 31, 203, 264, 90, 25, 262, 261, 23, 31, 203, 261, 27, 262, 261, 24, 263, 266, 31, 203,
    261, 28, 262, 264, 90, 27, 31, 203, 281, 261, 28, 31, 203, 275, 22, 30, 203, 261, 29, 262,
    225, 20, 31, 203, 261, 269, 262, 225, 94, 73, 92, 88, 12, 90, 29, 13, 31, 203, 281, 261,
    269, 31, 203,
];
/// From `onnxruntime` 1.30 in Python on the same ids, padded to 256 with mask.
const EXPECTED_LOGITS: [f32; 4] = [-0.026603315, -0.034845207, 0.59149444, -0.5656633];
