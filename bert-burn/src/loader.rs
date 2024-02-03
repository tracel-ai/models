use crate::bert::{BertEmbeddingsRecord, BertModel, BertModelConfig, BertModelRecord};

use burn::nn::attention::MultiHeadAttentionRecord;
use burn::nn::transformer::{
    PositionWiseFeedForwardRecord, TransformerEncoderLayerRecord, TransformerEncoderRecord,
};
use burn::{
    module::ConstantRecord,
    nn::LayerNormRecord,
    nn::{EmbeddingRecord, LinearRecord},
    tensor::{backend::Backend, Data, Shape, Tensor},
};
use candle_core::{safetensors, Device, Tensor as CandleTensor};
use serde_json::Value;
use std::collections::HashMap;

fn load_1d_tensor_from_candle<B: Backend>(
    tensor: &CandleTensor,
    device: &B::Device,
) -> Tensor<B, 1> {
    let dims = tensor.dims();
    let data = tensor.to_vec1::<f32>().unwrap();
    let array: [usize; 1] = dims.try_into().expect("Unexpected size");
    let data = Data::new(data, Shape::new(array));
    let weight = Tensor::<B, 1>::from_floats(data, &device.clone());
    weight
}

fn load_2d_tensor_from_candle<B: Backend>(
    tensor: &CandleTensor,
    device: &B::Device,
) -> Tensor<B, 2> {
    let dims = tensor.dims();
    let data = tensor
        .to_vec2::<f32>()
        .unwrap()
        .into_iter()
        .flatten()
        .collect::<Vec<f32>>();
    let array: [usize; 2] = dims.try_into().expect("Unexpected size");
    let data = Data::new(data, Shape::new(array));
    let weight = Tensor::<B, 2>::from_floats(data, &device.clone());
    weight
}

fn load_layer_norm_safetensor<B: Backend>(
    bias: &CandleTensor,
    weight: &CandleTensor,
    device: &B::Device,
) -> LayerNormRecord<B> {
    let beta = load_1d_tensor_from_candle::<B>(bias, device);
    let gamma = load_1d_tensor_from_candle::<B>(weight, device);

    let layer_norm_record = LayerNormRecord {
        beta: beta.into(),
        gamma: gamma.into(),
        epsilon: ConstantRecord::new(),
    };
    layer_norm_record
}

fn load_linear_safetensor<B: Backend>(
    bias: &CandleTensor,
    weight: &CandleTensor,
    device: &B::Device,
) -> LinearRecord<B> {
    let bias = load_1d_tensor_from_candle::<B>(bias, device);
    let weight = load_2d_tensor_from_candle::<B>(weight, device);

    let weight = weight.transpose();

    let linear_record = LinearRecord {
        weight: weight.into(),
        bias: Some(bias.into()),
    };
    linear_record
}

fn load_intermediate_layer_safetensor<B: Backend>(
    linear_inner_weight: &CandleTensor,
    linear_inner_bias: &CandleTensor,
    linear_outer_weight: &CandleTensor,
    linear_outer_bias: &CandleTensor,
    device: &B::Device,
) -> PositionWiseFeedForwardRecord<B> {
    let linear_inner = load_linear_safetensor::<B>(linear_inner_bias, linear_inner_weight, device);
    let linear_outer = load_linear_safetensor::<B>(linear_outer_bias, linear_outer_weight, device);

    let pwff_record = PositionWiseFeedForwardRecord {
        linear_inner: linear_inner,
        linear_outer: linear_outer,
        dropout: ConstantRecord::new(),
        gelu: ConstantRecord::new(),
    };

    pwff_record
}

fn load_attention_layer_safetensor<B: Backend>(
    attention_tensors: HashMap<String, CandleTensor>,
    device: &B::Device,
) -> MultiHeadAttentionRecord<B> {
    let query = load_linear_safetensor::<B>(
        &attention_tensors["attention.self.query.bias"],
        &attention_tensors["attention.self.query.weight"],
        device,
    );

    let key = load_linear_safetensor::<B>(
        &attention_tensors["attention.self.key.bias"],
        &attention_tensors["attention.self.key.weight"],
        device,
    );

    let value = load_linear_safetensor::<B>(
        &attention_tensors["attention.self.value.bias"],
        &attention_tensors["attention.self.value.weight"],
        device,
    );

    let output = load_linear_safetensor::<B>(
        &attention_tensors["attention.output.dense.bias"],
        &attention_tensors["attention.output.dense.weight"],
        device,
    );

    let attention_record = MultiHeadAttentionRecord {
        query: query,
        key: key,
        value: value,
        output: output,
        dropout: ConstantRecord::new(),
        activation: ConstantRecord::new(),
        n_heads: ConstantRecord::new(),
        d_k: ConstantRecord::new(),
        min_float: ConstantRecord::new(),
        quiet_softmax: ConstantRecord::new(),
    };
    attention_record
}

fn load_encoder_from_safetensors<B: Backend>(
    encoder_tensors: HashMap<String, CandleTensor>,
    device: &B::Device,
) -> TransformerEncoderRecord<B> {
    // Each layer in encoder_tensors has a key like encoder.layer.0, encoder.layer.1, etc.
    // We need to extract the layers in order by iterating over the tensors and extracting the layer number
    let mut layers: HashMap<usize, HashMap<String, CandleTensor>> = HashMap::new();

    for (key, value) in encoder_tensors.iter() {
        let layer_number = key.split(".").collect::<Vec<&str>>()[3]
            .parse::<usize>()
            .unwrap();
        if !layers.contains_key(&layer_number) {
            layers.insert(layer_number, HashMap::new());
        }
        layers
            .get_mut(&layer_number)
            .unwrap()
            .insert(key.to_string(), value.clone());
    }

    // Sort layers.iter() by key
    let mut layers = layers
        .into_iter()
        .collect::<Vec<(usize, HashMap<String, CandleTensor>)>>();
    layers.sort_by(|a, b| a.0.cmp(&b.0));

    // Now, we can iterate over the layers and load each layer
    let mut bert_encoder_layers: Vec<TransformerEncoderLayerRecord<B>> = Vec::new();
    for (key, value) in layers.iter() {
        let layer_key = format!("roberta.encoder.layer.{}", key.to_string());
        let attention_tensors = value.clone();
        // Remove the layer number from the key
        let attention_tensors = attention_tensors
            .iter()
            .map(|(k, v)| (k.replace(&format!("{}.", layer_key), ""), v.clone()))
            .collect::<HashMap<String, CandleTensor>>();

        let attention_layer =
            load_attention_layer_safetensor::<B>(attention_tensors.clone(), device);

        let norm_1 = load_layer_norm_safetensor(
            &attention_tensors["attention.output.LayerNorm.bias"],
            &attention_tensors["attention.output.LayerNorm.weight"],
            device,
        );

        let pwff = load_intermediate_layer_safetensor::<B>(
            &value[&format!("{}.intermediate.dense.weight", layer_key)],
            &value[&format!("{}.intermediate.dense.bias", layer_key)],
            &value[&format!("{}.output.dense.weight", layer_key)],
            &value[&format!("{}.output.dense.bias", layer_key)],
            device,
        );

        let norm_2 = load_layer_norm_safetensor::<B>(
            &value[&format!("{}.output.LayerNorm.bias", layer_key)],
            &value[&format!("{}.output.LayerNorm.weight", layer_key)],
            device,
        );

        let layer_record = TransformerEncoderLayerRecord {
            mha: attention_layer,
            pwff: pwff,
            norm_1: norm_1,
            norm_2: norm_2,
            dropout: ConstantRecord::new(),
            norm_first: ConstantRecord::new(),
        };

        bert_encoder_layers.push(layer_record);
    }

    let encoder_record = TransformerEncoderRecord {
        layers: bert_encoder_layers,
    };
    encoder_record
}

fn load_embedding_safetensor<B: Backend>(
    weight: &CandleTensor,
    device: &B::Device,
) -> EmbeddingRecord<B> {
    let weight = load_2d_tensor_from_candle(weight, device);

    let embedding = EmbeddingRecord {
        weight: weight.into(),
    };

    embedding
}

fn load_embeddings_from_safetensors<B: Backend>(
    embedding_tensors: HashMap<String, CandleTensor>,
    device: &B::Device,
) -> BertEmbeddingsRecord<B> {
    let word_embeddings = load_embedding_safetensor(
        &embedding_tensors["roberta.embeddings.word_embeddings.weight"],
        device,
    );

    let position_embeddings = load_embedding_safetensor(
        &embedding_tensors["roberta.embeddings.position_embeddings.weight"],
        device,
    );

    let token_type_embeddings = load_embedding_safetensor(
        &embedding_tensors["roberta.embeddings.token_type_embeddings.weight"],
        device,
    );

    let layer_norm = load_layer_norm_safetensor::<B>(
        &embedding_tensors["roberta.embeddings.LayerNorm.bias"],
        &embedding_tensors["roberta.embeddings.LayerNorm.weight"],
        device,
    );

    let embeddings_record = BertEmbeddingsRecord {
        word_embeddings,
        position_embeddings,
        token_type_embeddings,
        layer_norm,
        dropout: ConstantRecord::new(),
        max_position_embeddings: ConstantRecord::new(),
    };

    embeddings_record
}

pub fn load_model_from_safetensors<B: Backend>(
    file_path: &str,
    device: &B::Device,
    config: BertModelConfig,
) -> BertModel<B> {
    let file_path = std::path::Path::new(file_path);
    let weight_result = safetensors::load::<&std::path::Path>(file_path, &Device::Cpu);

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

    for (key, value) in weights.iter() {
        if key.starts_with("roberta.encoder.layer.") {
            encoder_layers.insert(key.to_string(), value.clone());
        } else if key.starts_with("roberta.embeddings.") {
            embeddings_layers.insert(key.to_string(), value.clone());
        }
    }

    let embeddings_record = load_embeddings_from_safetensors::<B>(embeddings_layers, device);
    let encoder_record = load_encoder_from_safetensors::<B>(encoder_layers, device);
    let model_record = BertModelRecord {
        embeddings: embeddings_record,
        encoder: encoder_record,
    };

    let model = config.init_with::<B>(model_record);
    model
}

pub fn load_model_config(config: HashMap<String, Value>) -> BertModelConfig {
    let model_config = BertModelConfig {
        n_heads: config["num_attention_heads"].as_i64().unwrap() as usize,
        n_layers: config["num_hidden_layers"].as_i64().unwrap() as usize,
        layer_norm_eps: config["layer_norm_eps"].as_f64().unwrap(),
        hidden_size: config["hidden_size"].as_i64().unwrap() as usize,
        intermediate_size: config["intermediate_size"].as_i64().unwrap() as usize,
        vocab_size: config["vocab_size"].as_i64().unwrap() as usize,
        max_position_embeddings: config["max_position_embeddings"].as_i64().unwrap() as usize,
        type_vocab_size: config["type_vocab_size"].as_i64().unwrap() as usize,
        hidden_dropout_prob: config["hidden_dropout_prob"].as_f64().unwrap(),
    };

    model_config
}
