use crate::data::BertInferenceBatch;
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::Initializer::KaimingUniform;
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig};
use burn::tensor::{Data, Float, Int, Shape};
use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, Tensor},
};

#[derive(Config)]
pub struct BertEmbeddingsConfig {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub hidden_size: usize,
    pub hidden_dropout_prob: f64,
    pub layer_norm_eps: f64,
}

#[derive(Module, Debug)]
pub struct BertEmbeddings<B: Backend> {
    word_embeddings: Embedding<B>,
    position_embeddings: Embedding<B>,
    token_type_embeddings: Embedding<B>,
    layer_norm: LayerNorm<B>,
    dropout: Dropout,
    max_position_embeddings: usize,
}

impl BertEmbeddingsConfig {
    /// Initializes BertEmbeddings with default weights
    pub fn init<B: Backend>(&self, device: &B::Device) -> BertEmbeddings<B> {
        let word_embeddings = EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device);
        let position_embeddings =
            EmbeddingConfig::new(self.max_position_embeddings, self.hidden_size).init(device);
        let token_type_embeddings =
            EmbeddingConfig::new(self.type_vocab_size, self.hidden_size).init(device);
        let layer_norm = LayerNormConfig::new(self.hidden_size)
            .with_epsilon(self.layer_norm_eps)
            .init(device);

        let dropout = DropoutConfig::new(self.hidden_dropout_prob).init();

        BertEmbeddings {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
            max_position_embeddings: self.max_position_embeddings,
        }
    }

    /// Initializes BertEmbeddings with provided weights
    pub fn init_with<B: Backend>(&self, record: BertEmbeddingsRecord<B>) -> BertEmbeddings<B> {
        let word_embeddings = EmbeddingConfig::new(self.vocab_size, self.hidden_size)
            .init_with(record.word_embeddings);
        let position_embeddings =
            EmbeddingConfig::new(self.max_position_embeddings, self.hidden_size)
                .init_with(record.position_embeddings);
        let token_type_embeddings = EmbeddingConfig::new(self.type_vocab_size, self.hidden_size)
            .init_with(record.token_type_embeddings);
        let layer_norm = LayerNormConfig::new(self.hidden_size)
            .with_epsilon(self.layer_norm_eps)
            .init_with(record.layer_norm);

        let dropout = DropoutConfig::new(self.hidden_dropout_prob).init();

        BertEmbeddings {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
            max_position_embeddings: self.max_position_embeddings,
        }
    }
}

impl<B: Backend> BertEmbeddings<B> {
    pub fn forward(&self, item: BertInferenceBatch<B>) -> Tensor<B, 3, Float> {
        // Extract tokens from the batch
        let input_shape = &item.tokens.shape();
        let input_ids = item.tokens;

        // Embed tokens
        let inputs_embeds = self.word_embeddings.forward(input_ids);
        let mut embeddings = inputs_embeds;

        let device = &self.position_embeddings.devices()[0];

        let token_type_ids = Tensor::<B, 2, Int>::zeros(input_shape.clone(), device);
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids);

        embeddings = embeddings + token_type_embeddings;

        // Position embeddings
        // Assuming position_ids is a range from 0 to seq_length
        let seq_length = input_shape.dims[1];
        let position_values: Vec<i32> = (0..self.max_position_embeddings)
            .map(|x| x as i32) // Convert each usize to Int
            .collect::<Vec<_>>()[0..seq_length]
            .to_vec();

        let shape = Shape::new([1, seq_length]);
        let data = Data::new(position_values, shape);
        let position_ids_tensor = Tensor::<B, 2, Int>::from_ints(data, device);

        let position_embeddings = self.position_embeddings.forward(position_ids_tensor);
        embeddings = embeddings + position_embeddings;

        // Layer normalization and dropout
        let embeddings = self.layer_norm.forward(embeddings);
        let embeddings = self.dropout.forward(embeddings);

        embeddings
    }
}

// Define the Bert model configuration
#[derive(Config)]
pub struct BertModelConfig {
    pub n_heads: usize,
    pub n_layers: usize,
    pub layer_norm_eps: f64,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub hidden_dropout_prob: f64,
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
            n_heads: self.n_heads,
            n_layers: self.n_layers,
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
            n_heads: self.n_heads,
            n_layers: self.n_layers,
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
