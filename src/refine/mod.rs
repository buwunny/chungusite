//! Bridge from the IR to a small ONNX model that proposes names and types.
//! See docs/ml-runtime.md.
//!
//! - `text`: serializes a `Function` as compact C-like text, one tokenizer pre-token
//!   at a time. The same walk produces the string or the token ids.
//! - `cache`: pre-token -> token ids, filled once, so the hot path does no BPE work.
//! - `infer` (feature `ml`): the ONNX Runtime session and batched inference.
pub mod cache;
pub mod text;
#[cfg(feature = "ml")]
pub mod infer;

use crate::ir::Function;

/// The text handed to the model for one function.
pub fn to_text(f: &Function) -> String {
    let mut t = text::Text::default();
    text::serialize(f, &mut t);
    t.0
}

/// Token ids for one function, appended to `out` (reuse it across functions).
pub fn to_ids(f: &Function, cache: &mut cache::TokenCache, out: &mut Vec<u32>) {
    text::serialize(f, &mut cache.sink(out));
}
