use crate::embedding::{MiniLmEmbeddings, MiniLmEmbeddingsConfig};
use burn::config::Config;
use burn::module::Module;
use burn::nn::Initializer::KaimingUniform;
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};
use std::path::Path;

/// MiniLM model configuration.
///
/// Load from HuggingFace's config.json using `load_from_hf`.
#[derive(Config, Debug)]
pub struct MiniLmConfig {
    /// Hidden size (embedding dimension).
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of transformer encoder layers.
    pub num_hidden_layers: usize,
    /// Feed-forward intermediate size.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum position embeddings.
    pub max_position_embeddings: usize,
    /// Token type vocabulary size.
    pub type_vocab_size: usize,
    /// Dropout probability.
    pub hidden_dropout_prob: f64,
    /// Layer normalization epsilon.
    pub layer_norm_eps: f64,
}

/// MiniLM encoder model.
///
/// A BERT-based encoder optimized for sentence embeddings.
#[derive(Module, Debug)]
pub struct MiniLmModel<B: Backend> {
    /// Token embeddings (word + position + token_type).
    pub embeddings: MiniLmEmbeddings<B>,
    /// Transformer encoder stack.
    pub encoder: TransformerEncoder<B>,
}

/// Output from the MiniLM model.
#[derive(Debug, Clone)]
pub struct MiniLmOutput<B: Backend> {
    /// Hidden states from the last encoder layer [batch_size, seq_len, hidden_size].
    pub hidden_states: Tensor<B, 3>,
}

impl MiniLmConfig {
    /// Initialize model with default (random) weights.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MiniLmModel<B> {
        let embeddings = self.embeddings_config().init(device);
        let encoder = self.encoder_config().init(device);

        MiniLmModel {
            embeddings,
            encoder,
        }
    }

    /// Load configuration from a HuggingFace config.json file.
    ///
    /// Extra fields in the config file are ignored.
    pub fn load_from_hf<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    fn embeddings_config(&self) -> MiniLmEmbeddingsConfig {
        MiniLmEmbeddingsConfig::new(
            self.vocab_size,
            self.max_position_embeddings,
            self.type_vocab_size,
            self.hidden_size,
            self.hidden_dropout_prob,
            self.layer_norm_eps,
        )
    }

    fn encoder_config(&self) -> TransformerEncoderConfig {
        TransformerEncoderConfig::new(
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.num_hidden_layers,
        )
        .with_dropout(self.hidden_dropout_prob)
        .with_norm_first(false) // BERT-style post-LayerNorm
        .with_quiet_softmax(false)
        .with_initializer(KaimingUniform {
            gain: 1.0 / libm::sqrt(3.0),
            fan_out_only: false,
        })
    }
}

impl<B: Backend> MiniLmModel<B> {
    /// Forward pass through the model.
    ///
    /// # Arguments
    /// - `input_ids`: Token IDs [batch_size, seq_len]
    /// - `attention_mask`: Attention mask where 1 = real token, 0 = padding [batch_size, seq_len]
    /// - `token_type_ids`: Optional segment IDs [batch_size, seq_len]
    ///
    /// # Returns
    /// Hidden states from the last encoder layer.
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2>,
        token_type_ids: Option<Tensor<B, 2, Int>>,
    ) -> MiniLmOutput<B> {
        // Get embeddings
        let embeddings = self.embeddings.forward(input_ids, token_type_ids);

        // Convert attention_mask to padding mask (bool tensor where true = padding)
        // attention_mask: 1 = real, 0 = padding
        // mask_pad: true = padding, false = real
        let mask_pad: Tensor<B, 2, Bool> = attention_mask.equal_elem(0);

        // Forward through encoder
        let encoder_input = TransformerEncoderInput::new(embeddings).mask_pad(mask_pad);
        let hidden_states = self.encoder.forward(encoder_input);

        MiniLmOutput { hidden_states }
    }
}
