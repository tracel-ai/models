//! The text part of a Qwen3.5 `config.json` (HF field names).

use serde::Deserialize;

/// Qwen3.5 text config. Parse with [`Qwen35Config::from_json`], which accepts the whole HF
/// `config.json` (with `text_config` inside) or the text block alone.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "yes")]
    pub attn_output_gate: bool,
    #[serde(default = "four")]
    pub full_attention_interval: usize,
    /// Explicit layer types (`"linear_attention"` / `"full_attention"`); when absent, every
    /// `full_attention_interval`-th layer is full attention.
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default = "four")]
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub rope_parameters: RopeParameters,
    /// Filled from the top level of the HF file when the text block lacks it.
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(default = "theta")]
    pub rope_theta: f64,
    #[serde(default = "quarter")]
    pub partial_rotary_factor: f64,
}

impl Default for RopeParameters {
    fn default() -> Self {
        Self {
            rope_theta: theta(),
            partial_rotary_factor: quarter(),
        }
    }
}

fn yes() -> bool {
    true
}
fn four() -> usize {
    4
}
fn theta() -> f64 {
    10_000_000.0
}
fn quarter() -> f64 {
    0.25
}

/// Which mixer a decoder layer uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Gated DeltaNet (`linear_attn`).
    Linear,
    /// Gated softmax attention (`self_attn`).
    Full,
}

impl Qwen35Config {
    pub fn from_json(json: &str) -> Result<Self, String> {
        let v: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let tie_top = v.get("tie_word_embeddings").and_then(|t| t.as_bool());
        let text = v.get("text_config").cloned().unwrap_or(v);
        let mut c: Qwen35Config = serde_json::from_value(text).map_err(|e| e.to_string())?;
        if c.tie_word_embeddings.is_none() {
            c.tie_word_embeddings = tie_top;
        }
        Ok(c)
    }

    pub fn layer_kinds(&self) -> Vec<LayerKind> {
        match &self.layer_types {
            Some(t) => t
                .iter()
                .map(|s| {
                    if s == "full_attention" {
                        LayerKind::Full
                    } else {
                        LayerKind::Linear
                    }
                })
                .collect(),
            None => (0..self.num_hidden_layers)
                .map(|i| {
                    if (i + 1) % self.full_attention_interval == 0 {
                        LayerKind::Full
                    } else {
                        LayerKind::Linear
                    }
                })
                .collect(),
        }
    }

    /// Head dims that get rotary position (`head_dim * partial_rotary_factor`; 64 of 256).
    pub fn rot_dim(&self) -> usize {
        (self.head_dim as f64 * self.rope_parameters.partial_rotary_factor) as usize
    }

    pub fn gdn(&self) -> crate::gdn::GatedDeltaNetConfig {
        crate::gdn::GatedDeltaNetConfig::new()
            .with_hidden_size(self.hidden_size)
            .with_num_k_heads(self.linear_num_key_heads)
            .with_num_v_heads(self.linear_num_value_heads)
            .with_head_k_dim(self.linear_key_head_dim)
            .with_head_v_dim(self.linear_value_head_dim)
            .with_conv_kernel(self.linear_conv_kernel_dim)
            .with_eps(self.rms_norm_eps)
    }
}
