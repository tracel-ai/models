use burn::config::Config;
use burn::module::Module;
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
    LinearConfig,
};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

/// Configuration for ALBERT factorized embeddings.
#[derive(Config, Debug)]
pub(crate) struct AlbertEmbeddingsConfig {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub embedding_size: usize,
    pub hidden_size: usize,
    pub hidden_dropout_prob: f64,
    pub layer_norm_eps: f64,
}

/// ALBERT embeddings with factorized parameterization.
///
/// Uses a smaller embedding dimension (`embedding_size`) projected up to `hidden_size`
/// via a linear layer, reducing total embedding parameters.
#[derive(Module, Debug)]
pub struct AlbertEmbeddings<B: Backend> {
    word_embeddings: Embedding<B>,
    position_embeddings: Embedding<B>,
    token_type_embeddings: Embedding<B>,
    /// Projects from embedding_size to hidden_size.
    projection: Linear<B>,
    layer_norm: LayerNorm<B>,
    dropout: Dropout,
}

impl AlbertEmbeddingsConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AlbertEmbeddings<B> {
        let word_embeddings =
            EmbeddingConfig::new(self.vocab_size, self.embedding_size).init(device);
        let position_embeddings =
            EmbeddingConfig::new(self.max_position_embeddings, self.embedding_size).init(device);
        let token_type_embeddings =
            EmbeddingConfig::new(self.type_vocab_size, self.embedding_size).init(device);
        let projection = LinearConfig::new(self.embedding_size, self.hidden_size).init(device);
        let layer_norm = LayerNormConfig::new(self.embedding_size)
            .with_epsilon(self.layer_norm_eps)
            .init(device);
        let dropout = DropoutConfig::new(self.hidden_dropout_prob).init();

        AlbertEmbeddings {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            projection,
            layer_norm,
            dropout,
        }
    }
}

impl<B: Backend> AlbertEmbeddings<B> {
    /// Returns the word embeddings weight tensor `[vocab_size, embedding_size]`.
    pub fn word_embeddings_weight(&self) -> Tensor<B, 2> {
        self.word_embeddings.weight.val()
    }

    /// Forward pass: combine embeddings, layer norm, project to hidden_size, dropout.
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        token_type_ids: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input_ids.dims();
        let device = &input_ids.device();

        let word_embeds = self.word_embeddings.forward(input_ids);

        let position_ids = Tensor::arange(0..seq_len as i64, device)
            .reshape([1, seq_len])
            .expand([batch_size, seq_len]);
        let position_embeds = self.position_embeddings.forward(position_ids);

        let token_type_ids =
            token_type_ids.unwrap_or_else(|| Tensor::zeros([batch_size, seq_len], device));
        let token_type_embeds = self.token_type_embeddings.forward(token_type_ids);

        let embeddings = word_embeds + position_embeds + token_type_embeds;
        let embeddings = self.layer_norm.forward(embeddings);
        let embeddings = self.projection.forward(embeddings);
        self.dropout.forward(embeddings)
    }
}
