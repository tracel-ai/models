use crate::data::BertInferenceBatch;
use crate::embedding::{BertEmbeddings, BertEmbeddingsConfig};
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::Initializer::KaimingUniform;
use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, Tensor},
};

// Define the Bert model configuration
#[derive(Config)]
pub struct BertModelConfig {
    /// Number of attention heads in the multi-head attention
    pub num_attention_heads: usize,
    /// Number of transformer encoder layers/blocks
    pub num_hidden_layers: usize,
    /// Layer normalization epsilon
    pub layer_norm_eps: f64,
    /// Size of bert embedding (e.g., 768 for roberta-base)
    pub hidden_size: usize,
    /// Size of the intermediate position wise feedforward layer
    pub intermediate_size: usize,
    /// Size of the vocabulary
    pub vocab_size: usize,
    /// Max position embeddings, typically max_seq_len + 2 to account for [BOS] and [PAD] tokens
    pub max_position_embeddings: usize,
    /// Identifier for sentence type in input (e.g., 0 for single sentence, 1 for pair)
    pub type_vocab_size: usize,
    /// Dropout value across layers, typically 0.1
    pub hidden_dropout_prob: f64,
    /// BERT model name (roberta)
    pub model_type: String,
    /// Index of the padding token
    pub pad_token_id: usize,
    /// Maximum sequence length for the tokenizer
    pub max_seq_len: Option<usize>,
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
