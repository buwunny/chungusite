//! Bridge from the IR to a small ONNX model that proposes names and types.
//! See docs/ml-runtime.md.
//!
//! - `text`: serializes a `Function` as compact C-like text, one tokenizer pre-token
//!   at a time. The same walk produces the string or the token ids.
//! - `cache`: pre-token -> token ids, filled once, so the hot path does no BPE work.
//! - `infer` (feature `ml`): the ONNX Runtime session and batched inference.
//! - `model` (feature `ml`): a trained type classifier as a `types::TypeModel`.
pub mod cache;
pub mod text;
#[cfg(feature = "ml")]
pub mod infer;
#[cfg(feature = "ml")]
pub mod model;

use crate::ir::{Function, ValueId};

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

/// The text a type classifier reads for value `v` of `f`: the function, then
/// `var vN` (`train_types.py`'s input).
pub fn to_var_text(f: &Function, v: ValueId) -> String {
    let mut t = text::Text::default();
    text::serialize(f, &mut t);
    text::var_marker(v, &mut t);
    t.0
}
