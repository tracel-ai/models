use crate::data::BertInferenceBatch;
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::Initializer::KaimingUniform;
use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, Tensor},
};
use serde::Deserialize;
use crate::embedding::{BertEmbeddings, BertEmbeddingsConfig};

// Define the Bert model configuration
#[derive(Config)]
pub struct BertModelConfig {
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub layer_norm_eps: f64,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub hidden_dropout_prob: f64,
    pub model_type: String,
    pub pad_token_id: usize,
    pub max_seq_len: Option<usize>
}

// Define the Bert model structure
#[derive(Module, Debug)]
pub struct BertModel<B: Backend> {
    pub embeddings: BertEmbeddings<B>,
    pub encoder: TransformerEncoder<B>,
}

impl BertModelConfig {
    /// Initializes a Bert model with default weights
    pub fn init<B: Backend>(&self, device: &B::Device) -> BertModel<B> {
        let embeddings = BertEmbeddingsConfig {
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            type_vocab_size: self.type_vocab_size,
            hidden_size: self.hidden_size,
            hidden_dropout_prob: self.hidden_dropout_prob,
            layer_norm_eps: self.layer_norm_eps,
        }
        .init(device);

        let encoder = TransformerEncoderConfig {
            n_heads: self.num_attention_heads,
            n_layers: self.num_hidden_layers,
            d_model: self.hidden_size,
            d_ff: self.intermediate_size,
            dropout: self.hidden_dropout_prob,
            norm_first: true,
            quiet_softmax: false,
            initializer: KaimingUniform {
                gain: 1.0 / libm::sqrt(3.0),
                fan_out_only: false,
            },
        }
        .init(device);

        BertModel {
            embeddings,
            encoder,
        }
    }

    /// Initializes a Bert model with provided weights
    pub fn init_with<B: Backend>(&self, record: BertModelRecord<B>) -> BertModel<B> {
        let embeddings = BertEmbeddingsConfig {
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            type_vocab_size: self.type_vocab_size,
            hidden_size: self.hidden_size,
            hidden_dropout_prob: self.hidden_dropout_prob,
            layer_norm_eps: self.layer_norm_eps,
        }
        .init_with(record.embeddings);

        let encoder = TransformerEncoderConfig {
            n_heads: self.num_attention_heads,
            n_layers: self.num_hidden_layers,
            d_model: self.hidden_size,
            d_ff: self.intermediate_size,
            dropout: self.hidden_dropout_prob,
            norm_first: true,
            quiet_softmax: false,
            initializer: KaimingUniform {
                gain: 1.0 / libm::sqrt(3.0),
                fan_out_only: false,
            },
        }
        .init_with(record.encoder);

        BertModel {
            encoder,
            embeddings,
        }
    }
}

impl<B: Backend> BertModel<B> {
    /// Defines forward pass
    pub fn forward(&self, input: BertInferenceBatch<B>) -> Tensor<B, 3> {
        let embedding = self.embeddings.forward(input.clone());
        let device = &self.embeddings.devices()[0];

        let mask_pad = input.mask_pad.to_device(device);

        let encoder_input = TransformerEncoderInput::new(embedding).mask_pad(mask_pad);
        let output = self.encoder.forward(encoder_input);
        output
    }
}
