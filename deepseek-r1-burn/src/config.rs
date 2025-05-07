use crate::model::*;
use burn::nn::attention::MhaConfig;

pub fn deepseek_r1_config() -> DeepSeekR1Config {
    let n_layer = 32;
    let n_head = 32;
    let n_embd = 4096;
    let vocab_size = 100000;
    let dropout = 0.1;

    let embedding_config = EmbeddingConfig {
        vocab_size,
        d_model: n_embd,
    };

    let layer_norm_config = LayerNormConfig {
        d_model: n_embd,
        eps: 1e-5,
    };

    let attention_config = MhaConfig {
        d_model: n_embd,
        n_head,
        dropout,
        bias: true,
    };

    let mlp_config = LinearConfig {
        d_in: n_embd,
        d_out: n_embd * 4,
        bias: true,
    };

    let head_config = LinearConfig {
        d_in: n_embd,
        d_out: vocab_size,
        bias: true,
    };

    DeepSeekR1Config {
        n_layer,
        n_head,
        n_embd,
        vocab_size,
        dropout,
        embedding_config,
        layer_norm_config,
        attention_config,
        mlp_config,
        head_config,
    }
} 