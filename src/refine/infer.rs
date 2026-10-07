//! ONNX Runtime session setup and batched inference with reused input buffers.
//!
//! Built with `ort`'s `load-dynamic` feature: `libonnxruntime` is loaded at runtime
//! from `ORT_DYLIB_PATH` (or the system library path), so building chungusite never
//! downloads anything and users pick the ONNX Runtime build they want (CPU, CUDA ...).
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;

pub struct Refiner {
    session: Session,
    ids: Vec<i64>,
    mask: Vec<i64>,
    out: Vec<f32>,
    pad: i64,
    max_len: usize,
}

impl Refiner {
    /// `threads`: intra-op threads for this session. With one session per worker
    /// thread, use 1 and let the workers provide the parallelism.
    pub fn new(model: impl AsRef<std::path::Path>, threads: usize, pad: u32, max_len: usize) -> ort::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(model.as_ref())?;
        Ok(Refiner { session, ids: Vec::new(), mask: Vec::new(), out: Vec::new(), pad: pad as i64, max_len })
    }

    /// Runs one batch through an encoder with a classification head (e.g. one row
    /// per variable to type). Returns logits, row-major `[batch, classes]`.
    ///
    /// Sequences are padded to the next power of two (capped at `max_len`), so the
    /// runtime sees a handful of shapes instead of one per call. Callers should
    /// group sequences of similar length into the same batch.
    pub fn classify(&mut self, seqs: &[&[u32]]) -> ort::Result<&[f32]> {
        let longest = seqs.iter().map(|s| s.len()).max().unwrap_or(1);
        let t = longest.next_power_of_two().clamp(16, self.max_len);
        let b = seqs.len();
        self.ids.clear();
        self.mask.clear();
        for s in seqs {
            let s = &s[..s.len().min(t)];
            self.ids.extend(s.iter().map(|&x| x as i64));
            self.ids.resize(self.ids.len() + (t - s.len()), self.pad);
            self.mask.extend(std::iter::repeat_n(1, s.len()));
            self.mask.resize(self.mask.len() + (t - s.len()), 0);
        }
        // Borrow the buffers as tensors: no copy into ONNX Runtime-owned memory.
        let ids = TensorRef::from_array_view(([b, t], &self.ids[..]))?;
        let mask = TensorRef::from_array_view(([b, t], &self.mask[..]))?;
        let outputs = self.session.run(ort::inputs!["input_ids" => ids, "attention_mask" => mask])?;
        let (_, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
        self.out.clear();
        self.out.extend_from_slice(logits);
        Ok(&self.out)
    }
}
