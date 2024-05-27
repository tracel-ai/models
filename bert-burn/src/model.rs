use crate::data::BertInferenceBatch;
use crate::embedding::{BertEmbeddings, BertEmbeddingsConfig};
use crate::loader::{
    load_decoder_from_safetensors, load_embeddings_from_safetensors, load_encoder_from_safetensors,
    load_layer_norm_safetensor, load_linear_safetensor, load_pooler_from_safetensors,
};
use crate::pooler::{Pooler, PoolerConfig};
use burn::config::Config;
use burn::module::Module;
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::Initializer::KaimingUniform;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use candle_core::{safetensors, Device, Tensor as CandleTensor};
use std::collections::HashMap;
use std::path::PathBuf;

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
    /// Max position embeddings, in RoBERTa equal to max_seq_len + 2 (514), for BERT equal to max_seq_len(512)
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
    /// Whether to add a pooling layer to the model
    pub with_pooling_layer: Option<bool>,
}

// Define the Bert model structure
#[derive(Module, Debug)]
pub struct BertModel<B: Backend> {
    pub embeddings: BertEmbeddings<B>,
    pub encoder: TransformerEncoder<B>,
    pub pooler: Option<Pooler<B>>,
}

#[derive(Debug, Clone)]
pub struct BertModelOutput<B: Backend> {
    pub hidden_states: Tensor<B, 3>,
    pub pooled_output: Option<Tensor<B, 3>>,
}

impl BertModelConfig {
    /// Initializes a Bert model with default weights
    pub fn init<B: Backend>(&self, device: &B::Device) -> BertModel<B> {
        let embeddings = self.get_embeddings_config().init(device);
        let encoder = self.get_encoder_config().init(device);

        let pooler = if self.with_pooling_layer.unwrap_or(false) {
            Some(
                PoolerConfig {
                    hidden_size: self.hidden_size,
                }
                .init(device),
            )
        } else {
            None
        };

        BertModel {
            embeddings,
            encoder,
            pooler,
        }
    }

    pub fn init_with_lm_head<B: Backend>(&self, device: &B::Device) -> BertMaskedLM<B> {
        let bert = self.init(device);
        let lm_head = BertLMHead {
            dense: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            layer_norm: LayerNormConfig::new(self.hidden_size)
                .with_epsilon(self.layer_norm_eps)
                .init(device),
            decoder: LinearConfig::new(self.hidden_size, self.vocab_size).init(device),
        };

        BertMaskedLM { bert, lm_head }
    }

    fn get_embeddings_config(&self) -> BertEmbeddingsConfig {
        BertEmbeddingsConfig {
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            type_vocab_size: self.type_vocab_size,
            hidden_size: self.hidden_size,
            hidden_dropout_prob: self.hidden_dropout_prob,
            layer_norm_eps: self.layer_norm_eps,
            pad_token_idx: self.pad_token_id,
        }
    }

    fn get_encoder_config(&self) -> TransformerEncoderConfig {
        TransformerEncoderConfig {
            n_heads: self.num_attention_heads,
            n_layers: self.num_hidden_layers,
            d_model: self.hidden_size,
            d_ff: self.intermediate_size,
            dropout: self.hidden_dropout_prob,
            norm_first: false,
            quiet_softmax: false,
            initializer: KaimingUniform {
                gain: 1.0 / libm::sqrt(3.0),
                fan_out_only: false,
            },
        }
    }
}

impl<B: Backend> BertModel<B> {
    /// Defines forward pass
    pub fn forward(&self, input: BertInferenceBatch<B>) -> BertModelOutput<B> {
        let embedding = self.embeddings.forward(input.clone());
        let device = &self.embeddings.devices()[0];

        let mask_pad = input.mask_pad.to_device(device);

        let encoder_input = TransformerEncoderInput::new(embedding).mask_pad(mask_pad);
        let hidden_states = self.encoder.forward(encoder_input);

        let pooled_output = self
            .pooler
            .as_ref()
            .map(|pooler| pooler.forward(hidden_states.clone()));

        BertModelOutput {
            hidden_states,
            pooled_output,
        }
    }

    pub fn from_safetensors(
        file_path: PathBuf,
        device: &B::Device,
        config: BertModelConfig,
    ) -> BertModelRecord<B> {
        let model_name = config.model_type.as_str();
        let weight_result = safetensors::load::<PathBuf>(file_path, &Device::Cpu);

        // Match on the result of loading the weights
        let weights = match weight_result {
            Ok(weights) => weights,
            Err(e) => panic!("Error loading weights: {:?}", e),
        };

        // Weights are stored in a HashMap<String, Tensor>
        // For each layer, it will either be prefixed with "encoder.layer." or "embeddings."
        // We need to extract both.
        let mut encoder_layers: HashMap<String, CandleTensor> = HashMap::new();
        let mut embeddings_layers: HashMap<String, CandleTensor> = HashMap::new();
        let mut pooler_layers: HashMap<String, CandleTensor> = HashMap::new();

        for (key, value) in weights.iter() {
            // If model name prefix present in keys, remove it to load keys consistently
            // across variants (bert-base, roberta-base etc.)

            let prefix = String::from(model_name) + ".";
            let key_without_prefix = key.replace(&prefix, "");

            if key_without_prefix.starts_with("encoder.layer.") {
                encoder_layers.insert(key_without_prefix, value.clone());
            } else if key_without_prefix.starts_with("embeddings.") {
                embeddings_layers.insert(key_without_prefix, value.clone());
            } else if key_without_prefix.starts_with("pooler.") {
                pooler_layers.insert(key_without_prefix, value.clone());
            }
        }

        let embeddings_record = load_embeddings_from_safetensors::<B>(embeddings_layers, device);
        let encoder_record = load_encoder_from_safetensors::<B>(encoder_layers, device);

        let pooler_record = if config.with_pooling_layer.unwrap_or(false) {
            Some(load_pooler_from_safetensors::<B>(pooler_layers, device))
        } else {
            None
        };

        let model_record = BertModelRecord {
            embeddings: embeddings_record,
            encoder: encoder_record,
            pooler: pooler_record,
        };
        model_record
    }
}

#[derive(Module, Debug)]
pub struct BertMaskedLM<B: Backend> {
    pub bert: BertModel<B>,
    pub lm_head: BertLMHead<B>,
}

#[derive(Module, Debug)]
pub struct BertLMHead<B: Backend> {
    pub dense: Linear<B>,
    pub layer_norm: LayerNorm<B>,
    pub decoder: Linear<B>,
}

impl<B: Backend> BertMaskedLM<B> {
    pub fn forward(&self, input: BertInferenceBatch<B>) -> Tensor<B, 3> {
        let output = self.bert.forward(BertInferenceBatch {
            tokens: input.tokens.clone(),
            mask_pad: input.mask_pad.clone(),
        });
        let output = self.lm_head.forward(output.hidden_states);
        output
    }

    pub fn from_safetensors(
        file_path: PathBuf,
        device: &B::Device,
        config: BertModelConfig,
    ) -> BertMaskedLMRecord<B> {
        let bert = BertModel::from_safetensors(file_path.clone(), device, config.clone());
        let lm_head = BertLMHead::from_safetensors(file_path, device, config);

        BertMaskedLMRecord { bert, lm_head }
    }
}

impl<B: Backend> BertLMHead<B> {
    pub fn forward(&self, features: Tensor<B, 3>) -> Tensor<B, 3> {
        let output = self.dense.forward(features);
        let output = gelu(output);
        let output = self.layer_norm.forward(output);

        let output = self.decoder.forward(output);
        output
    }

    pub fn from_safetensors(
        file_path: PathBuf,
        device: &B::Device,
        _config: BertModelConfig,
    ) -> BertLMHeadRecord<B> {
        let weight_result = safetensors::load::<PathBuf>(file_path, &Device::Cpu);

        // Match on the result of loading the weights
        let weights = match weight_result {
            Ok(weights) => weights,
            Err(e) => panic!("Error loading weights: {:?}", e),
        };

        let dense = load_linear_safetensor::<B>(
            &weights["lm_head.dense.bias"],
            &weights["lm_head.dense.weight"],
            device,
        );
        let layer_norm = load_layer_norm_safetensor::<B>(
            &weights["lm_head.layer_norm.bias"],
            &weights["lm_head.layer_norm.weight"],
            device,
        );
        let decoder = load_decoder_from_safetensors::<B>(
            &weights["lm_head.bias"],
            &weights
                .iter()
                .find(|(k, _)| k.contains("word_embeddings.weight"))
                .unwrap()
                .1,
            device,
        );

        BertLMHeadRecord {
            dense,
            layer_norm,
            decoder,
        }
    }
}
