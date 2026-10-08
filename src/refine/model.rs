//! A trained type classifier as a `types::TypeModel` (`--refine`).
//!
//! The model directory holds what `train_types.py` saves, exported to ONNX:
//! `model.onnx` (inputs `input_ids`, `attention_mask`; output `logits`),
//! `tokenizer.json`, and the label of each class, either `labels.txt` (one per
//! line, class 0 first) or `config.json`'s `id2label`. Each argument and return
//! value is one row: the function's text, then `var vN` (`refine::to_var_text`),
//! cut from the left to the model's length as in training, with the tokenizer's
//! special tokens around it.
//!
//! Functions are asked about in parallel, each worker thread with its own ONNX
//! Runtime session (one intra-op thread each), so memory grows with `-j`.
use super::{cache::TokenCache, infer::Refiner, text};
use crate::ir::Function;
use crate::types::{parse_label, Proposal, TypeModel, Var};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

type Error = Box<dyn std::error::Error + Send + Sync>;

/// Sequence length when `tokenizer.json` doesn't save one (`train_types.py --max-len`).
const DEFAULT_MAX_LEN: usize = 256;

pub struct TypeClassifier {
    dir: PathBuf,
    /// One per worker thread, made on first use.
    slots: Vec<Mutex<Option<State>>>,
    /// Set after the first error, which is reported once.
    failed: AtomicBool,
    pad: u32,
    labels: Vec<String>,
    /// Special tokens the tokenizer puts before and after a sequence (`<s>`, `</s>`).
    prefix: Vec<u32>,
    suffix: Vec<u32>,
    max_len: usize,
    /// Proposals whose softmax probability is below this are not made.
    threshold: f32,
}

struct State {
    refiner: Refiner,
    cache: TokenCache,
    ids: Vec<u32>,
    seqs: Vec<Vec<u32>>,
}

impl TypeClassifier {
    /// Loads the model in `dir`, for up to `sessions` threads asking at once.
    pub fn load(dir: impl AsRef<Path>, threshold: f32, sessions: usize) -> Result<Self, Error> {
        let dir = dir.as_ref();
        super::infer::load_runtime()?;
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))?;
        let max_len = tok.get_truncation().map_or(DEFAULT_MAX_LEN, |t| t.max_length);
        let pad = tok.get_padding().map(|p| p.pad_id).or_else(|| tok.token_to_id("<pad>")).unwrap_or(0);
        let (prefix, word, suffix) = specials(&tok)?;
        if prefix.len() + suffix.len() >= max_len {
            return Err(format!("{}: special tokens fill the whole {max_len}-token sequence", dir.display()).into());
        }
        let labels = labels(dir)?;
        let mut state = State::new(dir, pad, max_len)?;
        // One row through the model, to check it has a class per label.
        let probe: Vec<u32> = [&prefix[..], &word, &suffix].concat();
        let classes = state.refiner.classify(&[&probe])?.len();
        if classes != labels.len() {
            return Err(format!("{}: the model has {classes} classes but there are {} labels", dir.display(), labels.len()).into());
        }
        let slots: Vec<_> = (0..sessions.max(1)).map(|_| Mutex::new(None)).collect();
        *slots[0].lock().unwrap() = Some(state);
        Ok(TypeClassifier {
            dir: dir.to_path_buf(),
            slots,
            failed: AtomicBool::new(false),
            pad,
            labels,
            prefix,
            suffix,
            max_len,
            threshold,
        })
    }

    /// Reports the first error and stops asking.
    fn fail(&self, e: impl std::fmt::Display) {
        if !self.failed.swap(true, Ordering::Relaxed) {
            eprintln!("warning: type model failed, continuing without it: {e}");
        }
    }
}

impl State {
    fn new(dir: &Path, pad: u32, max_len: usize) -> Result<Self, Error> {
        let refiner = Refiner::new(dir.join("model.onnx"), 1, pad, max_len)?;
        let cache = TokenCache::from_tokenizer_file(dir.join("tokenizer.json"))?;
        Ok(State { refiner, cache, ids: Vec::new(), seqs: Vec::new() })
    }
}

impl TypeModel for TypeClassifier {
    fn propose(&self, f: &Function, vars: &[Var]) -> Vec<Option<Proposal>> {
        let mut out = vec![None; vars.len()];
        if self.failed.load(Ordering::Relaxed) {
            return out;
        }
        let slot = &self.slots[rayon::current_thread_index().unwrap_or(0) % self.slots.len()];
        let mut st = slot.lock().unwrap_or_else(|e| e.into_inner());
        if st.is_none() {
            match State::new(&self.dir, self.pad, self.max_len) {
                Ok(s) => *st = Some(s),
                Err(e) => {
                    self.fail(e);
                    return out;
                }
            }
        }
        let State { refiner, cache, ids, seqs } = st.as_mut().unwrap();
        ids.clear();
        text::serialize(f, &mut cache.sink(ids));
        let body = ids.len();
        let room = self.max_len - self.prefix.len() - self.suffix.len();
        // Which entry of `vars` each row answers.
        let mut rows = Vec::with_capacity(vars.len());
        for (i, var) in vars.iter().enumerate() {
            let v = match *var {
                Var::Arg { value: Some(v), .. } | Var::Ret { value: v } => v,
                Var::Arg { value: None, .. } => continue,
            };
            ids.truncate(body);
            text::var_marker(v, &mut cache.sink(ids));
            if seqs.len() == rows.len() {
                seqs.push(Vec::new());
            }
            let s = &mut seqs[rows.len()];
            s.clear();
            s.extend_from_slice(&self.prefix);
            s.extend_from_slice(&ids[ids.len().saturating_sub(room)..]);
            s.extend_from_slice(&self.suffix);
            rows.push(i);
        }
        if rows.is_empty() {
            return out;
        }
        let batch: Vec<&[u32]> = seqs[..rows.len()].iter().map(Vec::as_slice).collect();
        let logits = match refiner.classify(&batch) {
            Ok(l) => l,
            Err(e) => {
                self.fail(e);
                return out;
            }
        };
        for (row, &i) in logits.chunks_exact(self.labels.len()).zip(&rows) {
            let (best, p) = softmax_max(row);
            let label = &self.labels[best];
            if p >= self.threshold && parse_label(label).is_some() {
                out[i] = Some(Proposal { label: label.clone(), score: p });
            }
        }
        out
    }
}

/// The class with the highest logit and its probability.
fn softmax_max(row: &[f32]) -> (usize, f32) {
    let (best, &m) = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
    let sum: f32 = row.iter().map(|&x| (x - m).exp()).sum();
    (best, 1.0 / sum)
}

/// The ids the tokenizer adds around a sequence: encode a word with and without
/// special tokens and see what surrounds it. Returns (before, the word, after).
fn specials(tok: &tokenizers::Tokenizer) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>), Error> {
    let plain = tok.encode("x", false)?.get_ids().to_vec();
    let full = tok.encode("x", true)?.get_ids().to_vec();
    let at = full.windows(plain.len()).position(|w| w == plain).ok_or("tokenizer: can't locate its special tokens")?;
    let after = full[at + plain.len()..].to_vec();
    Ok((full[..at].to_vec(), plain, after))
}

/// Class labels: `labels.txt`, else `config.json`'s `id2label`.
fn labels(dir: &Path) -> Result<Vec<String>, Error> {
    if let Ok(s) = std::fs::read_to_string(dir.join("labels.txt")) {
        return Ok(s.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect());
    }
    let path = dir.join("config.json");
    let s = std::fs::read_to_string(&path).map_err(|e| format!("{}: no labels.txt, and {e}", path.display()))?;
    let config: serde_json::Value = serde_json::from_str(&s)?;
    let map = config["id2label"].as_object().ok_or_else(|| format!("{}: no id2label", path.display()))?;
    let mut labels = vec![String::new(); map.len()];
    for (k, v) in map {
        let i: usize = k.parse()?;
        let slot = labels.get_mut(i).ok_or_else(|| format!("{}: id2label skips ids", path.display()))?;
        *slot = v.as_str().unwrap_or_default().to_string();
    }
    Ok(labels)
}
