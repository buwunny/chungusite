//! Lift a function, serialize it for the model, and run an ONNX classifier on it.
//!
//!     ORT_DYLIB_PATH=/path/to/libonnxruntime.so \
//!       cargo run --features ml --example refine -- tests/fixtures/tiny-types
//!
//! The directory needs `model.onnx` (inputs `input_ids`, `attention_mask`; output
//! `logits`) and the model's `tokenizer.json`.
use chungusite::{ir::Function, lift::Lifter, refine};

/// `if p == null { 0 } else { p[1] = v; p[2] }`
const CFG: &[u8] = &[
    0x48, 0x85, 0xFF, 0x74, 0x09, 0x48, 0x89, 0x77, 0x08, 0x48, 0x8B, 0x47, 0x10, 0xC3, 0x31, 0xC0, 0xC3,
];

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "tests/fixtures/tiny-types".into());

    let mut f = Function::with_capacity(64, 8);
    Lifter::new().lift(CFG, 0x1000, &mut f).map_err(|e| format!("{e:?}"))?;

    // What the model reads.
    let text = refine::to_text(&f);
    println!("--- text ---\n{text}");

    // The same thing as token ids, straight from the cache.
    let mut cache = refine::cache::TokenCache::from_tokenizer_file(format!("{dir}/tokenizer.json"))?;
    let mut ids = Vec::new();
    refine::to_ids(&f, &mut cache, &mut ids);
    println!("--- {} token ids ---\n{ids:?}", ids.len());

    // Load the ONNX model and run it.
    let mut model = refine::infer::Refiner::new(format!("{dir}/model.onnx"), 1, 0, 256)?;
    let logits = model.classify(&[&ids])?;
    println!("--- logits ---\n{logits:?}");
    Ok(())
}
