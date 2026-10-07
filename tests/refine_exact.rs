//! The cached ids must equal what the real tokenizer produces for the whole text.
//! Checked here with the fixture model's tokenizer.json (a byte-level BPE like
//! RoBERTa/CodeT5+/Qwen use). Run the same check against your production tokenizer.
mod common;
use chungusite::{ir::Function, lift::Lifter, refine};

#[test]
fn cached_ids_match_full_tokenization() {
    let path = "tests/fixtures/tiny-types/tokenizer.json";
    let mut tok = tokenizers::Tokenizer::from_file(path).unwrap();
    // The fixture's tokenizer.json was saved with left truncation at 192 tokens;
    // compare against untruncated output.
    tok.with_truncation(None).unwrap();
    let mut cache = refine::cache::TokenCache::from_tokenizer_file(path).unwrap();
    let mut lifter = Lifter::new();
    let mut f = Function::with_capacity(256, 32);
    let mut ids = Vec::new();
    for (i, code) in common::random_programs(1500, 0x1234_5678_9ABC_DEF1).iter().enumerate() {
        lifter.lift(code, common::BASE, &mut f).unwrap();
        let text = refine::to_text(&f);
        ids.clear();
        refine::to_ids(&f, &mut cache, &mut ids);
        assert_eq!(ids, tok.encode(text.as_str(), false).unwrap().get_ids(), "program {i}:\n{text}");
    }
}
