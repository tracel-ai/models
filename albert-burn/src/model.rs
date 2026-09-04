use crate::embedding::{AlbertEmbeddings, AlbertEmbeddingsConfig};
use crate::encoder::AlbertEncoder;
use crate::loader::LoadError;
use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::{Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::Device;
use burn::tensor::{Bool, Int, Tensor, assert_shape};
use std::path::Path;

/// ALBERT model configuration.
///
/// Load from HuggingFace's config.json using `load_from_hf`.
#[derive(Config, Debug)]
pub struct AlbertConfig {
    pub hidden_size: usize,
    pub embedding_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_hidden_groups: usize,
    pub inner_group_num: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    pub layer_norm_eps: f64,
}

/// ALBERT base encoder model.
#[derive(Module, Debug)]
pub struct AlbertModel {
    pub embeddings: AlbertEmbeddings,
    pub encoder: AlbertEncoder,
}

/// ALBERT for masked language modeling.
///
/// The decoder weight is tied to word embeddings (not a separate parameter).
/// Only the decoder bias is stored separately.
#[derive(Module, Debug)]
pub struct AlbertMaskedLM {
    pub albert: AlbertModel,
    pub mlm_dense: Linear,
    pub mlm_layer_norm: LayerNorm,
    pub mlm_decoder_bias: Param<Tensor<1>>,
    gelu: Gelu,
}

/// Output from the ALBERT model.
#[derive(Debug, Clone)]
pub struct AlbertOutput {
    pub hidden_states: Tensor<3>,
}

impl AlbertConfig {
    pub fn init(&self, device: &Device) -> AlbertModel {
        let embeddings = self.embeddings_config().init(device);
        let encoder = AlbertEncoder::new(
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.embedding_size,
            self.num_hidden_layers,
            self.hidden_dropout_prob,
            self.layer_norm_eps,
            device,
        );

        AlbertModel {
            embeddings,
            encoder,
        }
    }

    pub fn init_masked_lm(&self, device: &Device) -> AlbertMaskedLM {
        let albert = self.init(device);
        let mlm_dense = LinearConfig::new(self.hidden_size, self.embedding_size).init(device);
        let mlm_layer_norm = LayerNormConfig::new(self.embedding_size)
            .with_epsilon(self.layer_norm_eps)
            .init(device);
        let vocab_size = self.vocab_size;
        let mlm_decoder_bias: Param<Tensor<1>> = Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<1>::zeros([vocab_size], device),
            device.clone(),
            false,
            [vocab_size].into(),
        );

        AlbertMaskedLM {
            albert,
            mlm_dense,
            mlm_layer_norm,
            mlm_decoder_bias,
            gelu: Gelu::new(),
        }
    }

    pub fn load_from_hf<P: AsRef<Path>>(path: P) -> Result<Self, LoadError> {
        let content =
            std::fs::read_to_string(path).map_err(|e| LoadError::Config(e.to_string()))?;
        serde_json::from_str(&content).map_err(|e| LoadError::Config(e.to_string()))
    }

    fn embeddings_config(&self) -> AlbertEmbeddingsConfig {
        AlbertEmbeddingsConfig::new(
            self.vocab_size,
            self.max_position_embeddings,
            self.type_vocab_size,
            self.embedding_size,
            self.hidden_dropout_prob,
            self.layer_norm_eps,
        )
    }
}

impl AlbertModel {
    pub fn forward(
        &self,
        input_ids: Tensor<2, Int>,
        attention_mask: Tensor<2>,
        token_type_ids: Option<Tensor<2, Int>>,
    ) -> AlbertOutput {
        let [batch_size, seq_len] = input_ids.dims();
        assert_shape!(attention_mask, [batch_size, seq_len]);

        let embeddings = self.embeddings.forward(input_ids, token_type_ids);

        let device = attention_mask.device();
        let zeros = Tensor::<2>::zeros(attention_mask.shape(), &device);
        let mask_pad: Tensor<2, Bool> = attention_mask.equal(zeros);

        let hidden_states = self.encoder.forward(embeddings, Some(mask_pad));

        AlbertOutput { hidden_states }
    }
}

impl AlbertMaskedLM {
    /// Forward pass returning logits over the vocabulary.
    ///
    /// Uses weight tying: the decoder matrix is the word embeddings weight.
    ///
    /// # Returns
    /// Logits tensor `[batch_size, seq_len, vocab_size]`.
    pub fn forward(
        &self,
        input_ids: Tensor<2, Int>,
        attention_mask: Tensor<2>,
        token_type_ids: Option<Tensor<2, Int>>,
    ) -> Tensor<3> {
        let output = self
            .albert
            .forward(input_ids, attention_mask, token_type_ids);
        let hidden = output.hidden_states;

        // MLM head: dense → gelu → layernorm
        let hidden = self.mlm_dense.forward(hidden);
        let hidden = self.gelu.forward(hidden);
        let hidden = self.mlm_layer_norm.forward(hidden);

        // Decoder: matmul with word_embeddings weight (weight tying) + bias
        // hidden: [batch, seq, embedding_size]
        // word_embeddings.weight: [vocab_size, embedding_size]
        // logits: [batch, seq, vocab_size]
        let word_weight: Tensor<3> = self
            .albert
            .embeddings
            .word_embeddings_weight()
            .transpose()
            .unsqueeze();
        let [_, embedding_size, vocab_size] = word_weight.dims();
        assert_shape!(hidden, [_, _, embedding_size]);

        let bias = self.mlm_decoder_bias.val();
        assert_shape!(bias, [vocab_size]);

        let logits = hidden.matmul(word_weight);

        logits + bias.unsqueeze()
    }
}
