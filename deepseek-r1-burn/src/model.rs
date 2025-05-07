use burn::{
    module::Module,
    nn::{
        attention::{MhaConfig, MultiHeadAttention},
        dropout::Dropout,
        layer_norm::LayerNorm,
        linear::Linear,
        Embedding,
    },
    tensor::{backend::Backend, Tensor},
};
use serde::{Deserialize, Serialize};

#[derive(Module, Debug, Serialize, Deserialize)]
pub struct DeepSeekR1<B: Backend> {
    pub embedding: Embedding<B>,
    pub layers: Vec<TransformerBlock<B>>,
    pub ln_f: LayerNorm<B>,
    pub head: Linear<B>,
}

#[derive(Module, Debug, Serialize, Deserialize)]
pub struct TransformerBlock<B: Backend> {
    pub ln_1: LayerNorm<B>,
    pub attn: MultiHeadAttention<B>,
    pub ln_2: LayerNorm<B>,
    pub mlp: MLP<B>,
}

#[derive(Module, Debug, Serialize, Deserialize)]
pub struct MLP<B: Backend> {
    pub c_fc: Linear<B>,
    pub c_proj: Linear<B>,
    pub dropout: Dropout,
}

impl<B: Backend> DeepSeekR1<B> {
    pub fn new(config: &DeepSeekR1Config) -> Self {
        let embedding = Embedding::new(&config.embedding_config);
        let layers = (0..config.n_layer)
            .map(|_| TransformerBlock::new(config))
            .collect();
        let ln_f = LayerNorm::new(&config.layer_norm_config);
        let head = Linear::new(&config.head_config);

        Self {
            embedding,
            layers,
            ln_f,
            head,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut x = self.embedding.forward(input);
        
        for layer in &self.layers {
            x = layer.forward(x);
        }
        
        x = self.ln_f.forward(x);
        self.head.forward(x)
    }
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(config: &DeepSeekR1Config) -> Self {
        let ln_1 = LayerNorm::new(&config.layer_norm_config);
        let attn = MultiHeadAttention::new(&config.attention_config);
        let ln_2 = LayerNorm::new(&config.layer_norm_config);
        let mlp = MLP::new(config);

        Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = x + self.attn.forward(self.ln_1.forward(x.clone()));
        h + self.mlp.forward(self.ln_2.forward(h))
    }
}

impl<B: Backend> MLP<B> {
    pub fn new(config: &DeepSeekR1Config) -> Self {
        let c_fc = Linear::new(&config.mlp_config);
        let c_proj = Linear::new(&config.mlp_config);
        let dropout = Dropout::new(config.dropout);

        Self {
            c_fc,
            c_proj,
            dropout,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.c_fc.forward(x);
        let h = h.gelu();
        let h = self.dropout.forward(h);
        self.c_proj.forward(h)
    }
}

#[derive(Debug, Clone)]
pub struct DeepSeekR1Config {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub vocab_size: usize,
    pub dropout: f64,
    pub embedding_config: EmbeddingConfig,
    pub layer_norm_config: LayerNormConfig,
    pub attention_config: MhaConfig,
    pub mlp_config: LinearConfig,
    pub head_config: LinearConfig,
}

#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub vocab_size: usize,
    pub d_model: usize,
}

#[derive(Debug, Clone)]
pub struct LayerNormConfig {
    pub d_model: usize,
    pub eps: f64,
}

#[derive(Debug, Clone)]
pub struct LinearConfig {
    pub d_in: usize,
    pub d_out: usize,
    pub bias: bool,
} 