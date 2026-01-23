use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

/// Configuration for MiniLM embeddings.
#[derive(Config, Debug)]
pub struct MiniLmEmbeddingsConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_position_embeddings: usize,
    /// Token type vocabulary size (usually 2 for sentence A/B).
    pub type_vocab_size: usize,
    /// Hidden dimension size.
    pub hidden_size: usize,
    /// Dropout probability.
    pub hidden_dropout_prob: f64,
    /// Layer normalization epsilon.
    pub layer_norm_eps: f64,
}

/// MiniLM embeddings module.
///
/// Combines word, position, and token type embeddings, followed by
/// layer normalization and dropout.
#[derive(Module, Debug)]
pub struct MiniLmEmbeddings<B: Backend> {
    word_embeddings: Embedding<B>,
    position_embeddings: Embedding<B>,
    token_type_embeddings: Embedding<B>,
    layer_norm: LayerNorm<B>,
    dropout: Dropout,
}

impl MiniLmEmbeddingsConfig {
    /// Initialize embeddings with default weights.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MiniLmEmbeddings<B> {
        let word_embeddings = EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device);
        let position_embeddings =
            EmbeddingConfig::new(self.max_position_embeddings, self.hidden_size).init(device);
        let token_type_embeddings =
            EmbeddingConfig::new(self.type_vocab_size, self.hidden_size).init(device);
        let layer_norm = LayerNormConfig::new(self.hidden_size)
            .with_epsilon(self.layer_norm_eps)
            .init(device);
        let dropout = DropoutConfig::new(self.hidden_dropout_prob).init();

        MiniLmEmbeddings {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
        }
    }
}

impl<B: Backend> MiniLmEmbeddings<B> {
    /// Forward pass through the embeddings layer.
    ///
    /// # Arguments
    /// - `input_ids`: Token IDs [batch_size, seq_len]
    /// - `token_type_ids`: Optional segment IDs [batch_size, seq_len]. Defaults to zeros.
    ///
    /// # Returns
    /// Embedded representation [batch_size, seq_len, hidden_size]
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        token_type_ids: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input_ids.dims();
        let device = &input_ids.device();

        // Word embeddings
        let word_embeds = self.word_embeddings.forward(input_ids);

        // Position embeddings (0 to seq_len-1)
        let position_ids = Tensor::arange(0..seq_len as i64, device)
            .reshape([1, seq_len])
            .expand([batch_size, seq_len]);
        let position_embeds = self.position_embeddings.forward(position_ids);

        // Token type embeddings (default to zeros if not provided)
        let token_type_ids =
            token_type_ids.unwrap_or_else(|| Tensor::zeros([batch_size, seq_len], device));
        let token_type_embeds = self.token_type_embeddings.forward(token_type_ids);

        // Combine: word + position + token_type
        let embeddings = word_embeds + position_embeds + token_type_embeds;

        // Layer norm and dropout
        let embeddings = self.layer_norm.forward(embeddings);
        self.dropout.forward(embeddings)
    }
}
