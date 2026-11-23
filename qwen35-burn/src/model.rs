//! The Qwen3.5 text model: embedding → decoder layers → final RMSNorm, and the tied LM head. The
//! vision tower and the multi-token-prediction head of the HF checkpoint are not part of it.

use crate::blocks::{DecoderLayer, OffsetRmsNorm, rope_tables};
use crate::config::Qwen35Config;
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::{Device, Int, Tensor};

#[derive(Module, Debug)]
pub struct Qwen35TextModel {
    pub embed_tokens: Embedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: OffsetRmsNorm,
    pub rope_theta: f64,
    pub rot_dim: usize,
}

impl Qwen35TextModel {
    /// Fresh random weights. Use [`crate::loader::load_pretrained`] to fill in a checkpoint.
    pub fn new(c: &Qwen35Config, device: &Device) -> Self {
        Self {
            embed_tokens: EmbeddingConfig::new(c.vocab_size, c.hidden_size).init(device),
            layers: c
                .layer_kinds()
                .into_iter()
                .map(|k| DecoderLayer::new(c, k, device))
                .collect(),
            norm: OffsetRmsNorm::new(c.hidden_size, c.rms_norm_eps, device),
            rope_theta: c.rope_parameters.rope_theta,
            rot_dim: c.rot_dim(),
        }
    }

    /// Token ids `[B, S]` → final-normed hidden states `[B, S, hidden]`.
    pub fn forward(&self, ids: Tensor<2, Int>) -> Tensor<3> {
        let mut x = self.embed_tokens.forward(ids);
        let (cos, sin) = self.rope(x.dims()[1], &x.device());
        for layer in &self.layers {
            x = layer.forward(x, cos.clone(), sin.clone());
        }
        self.norm.forward(x)
    }

    /// The hidden state after every decoder layer (before the final norm), for parity tests.
    pub fn layer_outputs(&self, ids: Tensor<2, Int>) -> Vec<Tensor<3>> {
        let mut x = self.embed_tokens.forward(ids);
        let (cos, sin) = self.rope(x.dims()[1], &x.device());
        let mut out = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            x = layer.forward(x, cos.clone(), sin.clone());
            out.push(x.clone());
        }
        out
    }

    /// Language-model logits with the tied embedding: `[B, S, hidden]` → `[B, S, vocab]`.
    pub fn logits(&self, hidden: Tensor<3>) -> Tensor<3> {
        let [b, s, h] = hidden.dims();
        let e = self.embed_tokens.weight.val(); // [vocab, hidden], used as stored
        let v = e.dims()[0];
        // E h^T: transposes only the small [B*S, hidden] side, never the vocab table.
        e.matmul(hidden.reshape([b * s, h]).transpose())
            .transpose()
            .reshape([b, s, v])
    }

    fn rope(&self, len: usize, device: &Device) -> (Tensor<2>, Tensor<2>) {
        rope_tables(self.rope_theta, self.rot_dim, 0, len, device)
    }
}
