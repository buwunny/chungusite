//! Pre-token -> token ids, computed once with the model's real tokenizer.
use super::text::Emit;
use std::collections::HashMap;

const SMALL: usize = 4096;

type EncodeFn = Box<dyn Fn(&str) -> Vec<u32> + Send + Sync>;

pub struct TokenCache {
    encode: EncodeFn,
    lits: HashMap<&'static str, Box<[u32]>>,
    nums: Vec<Box<[u32]>>,    // "0".."4095"
    nums_sp: Vec<Box<[u32]>>, // " 0".." 4095"
}

impl TokenCache {
    /// `encode` must tokenize a string with no special tokens added, e.g.
    /// `|s| tok.encode(s, false).unwrap().get_ids().to_vec()`.
    pub fn new(encode: impl Fn(&str) -> Vec<u32> + Send + Sync + 'static) -> Self {
        let nums = (0..SMALL).map(|n| encode(&n.to_string()).into()).collect();
        let nums_sp = (0..SMALL).map(|n| encode(&format!(" {n}")).into()).collect();
        TokenCache { encode: Box::new(encode), lits: HashMap::with_capacity(128), nums, nums_sp }
    }

    /// Builds the cache from a Hugging Face `tokenizer.json` (the model's own tokenizer).
    #[cfg(feature = "ml")]
    pub fn from_tokenizer_file(path: impl AsRef<std::path::Path>) -> tokenizers::Result<Self> {
        let mut tok = tokenizers::Tokenizer::from_file(path)?;
        // A saved tokenizer.json can carry the truncation/padding used in training.
        // Pieces must be tokenized whole; length limits are applied to the full sequence.
        tok.with_truncation(None)?;
        tok.with_padding(None);
        Ok(Self::new(move |s| tok.encode(s, false).expect("tokenize").get_ids().to_vec()))
    }

    /// A sink that appends ids to `out`. Reuse `out` across functions.
    pub fn sink<'a>(&'a mut self, out: &'a mut Vec<u32>) -> Ids<'a> {
        Ids { cache: self, out }
    }
}

pub struct Ids<'a> {
    cache: &'a mut TokenCache,
    out: &'a mut Vec<u32>,
}

impl Emit for Ids<'_> {
    #[inline]
    fn lit(&mut self, s: &'static str) {
        // Only the first sight of a literal allocates; there are a few dozen in total.
        let c = &mut *self.cache;
        let ids = c.lits.entry(s).or_insert_with(|| (c.encode)(s).into());
        self.out.extend_from_slice(ids);
    }
    #[inline]
    fn num(&mut self, n: u64) {
        match self.cache.nums.get(n as usize) {
            Some(ids) => self.out.extend_from_slice(ids),
            None => self.out.extend((self.cache.encode)(&n.to_string())), // rare: big constants
        }
    }
    #[inline]
    fn num_sp(&mut self, n: u64) {
        match self.cache.nums_sp.get(n as usize) {
            Some(ids) => self.out.extend_from_slice(ids),
            None => self.out.extend((self.cache.encode)(&format!(" {n}"))),
        }
    }
}
