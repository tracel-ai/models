// Safetensors loading for the BERT family (BERT / RoBERTa) via `burn-store`.
//
// Key mappings translate HuggingFace tensor names to the Burn module structure:
//   encoder.layer.{i}.attention.self.{query,key,value} -> encoder.layers.{i}.mha.{query,key,value}
//   encoder.layer.{i}.attention.output.dense           -> encoder.layers.{i}.mha.output
//   encoder.layer.{i}.attention.output.LayerNorm       -> encoder.layers.{i}.norm_1
//   encoder.layer.{i}.intermediate.dense               -> encoder.layers.{i}.pwff.linear_inner
//   encoder.layer.{i}.output.dense                     -> encoder.layers.{i}.pwff.linear_outer
//   encoder.layer.{i}.output.LayerNorm                 -> encoder.layers.{i}.norm_2
//   embeddings.LayerNorm                               -> embeddings.layer_norm
//   pooler.dense                                       -> pooler.output
//
// `PyTorchToBurnAdapter` handles LayerNorm weight/bias -> gamma/beta and Linear weight transpose.

use crate::model::{BertMaskedLM, BertModel, BertModelConfig};
use burn::config::Config;
use burn::module::Param;
use burn::tensor::backend::Backend;
use burn_store::{KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use std::path::{Path, PathBuf};

/// Error type for model loading operations.
#[derive(Debug)]
pub enum LoadError {
    Store(String),
    Download(String),
    Config(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Store(msg) => write!(f, "Store error: {}", msg),
            LoadError::Download(msg) => write!(f, "Download error: {}", msg),
            LoadError::Config(msg) => write!(f, "Config error: {}", msg),
        }
    }
}

impl std::error::Error for LoadError {}

fn bert_structure_mappings() -> Vec<(&'static str, &'static str)> {
    vec![
        // encoder.layer.X -> encoder.layers.X
        (r"encoder\.layer\.([0-9]+)", "encoder.layers.$1"),
        // Self-attention
        (r"attention\.self\.query", "mha.query"),
        (r"attention\.self\.key", "mha.key"),
        (r"attention\.self\.value", "mha.value"),
        (r"attention\.output\.dense", "mha.output"),
        (r"attention\.output\.LayerNorm", "norm_1"),
        // Feed-forward. More specific patterns first (ffn output lives at layers.X.output.*,
        // but we must not match attention.output which was already rewritten above).
        (r"intermediate\.dense", "pwff.linear_inner"),
        (r"(layers\.[0-9]+)\.output\.dense", "$1.pwff.linear_outer"),
        (r"(layers\.[0-9]+)\.output\.LayerNorm", "$1.norm_2"),
        // Embedding LayerNorm
        (r"embeddings\.LayerNorm", "embeddings.layer_norm"),
        // Pooler dense -> output (field name in the Burn Pooler module)
        (r"pooler\.dense", "pooler.output"),
    ]
}

fn build_remapper(mappings: Vec<(&str, &str)>) -> Result<KeyRemapper, LoadError> {
    KeyRemapper::from_patterns(mappings).map_err(|e| LoadError::Store(e.to_string()))
}

/// Load pre-trained weights from a safetensors file into a `BertModel`.
///
/// Supports both BERT (`bert.*`) and RoBERTa (`roberta.*`) checkpoints; the prefix is stripped.
pub fn load_pretrained<B: Backend>(
    model: &mut BertModel<B>,
    checkpoint_path: impl AsRef<Path>,
) -> Result<(), LoadError> {
    let mut mappings = vec![(r"^(?:bert|roberta)\.(.+)", "$1")];
    mappings.extend(bert_structure_mappings());

    let remapper = build_remapper(mappings)?;
    let checkpoint_path: PathBuf = checkpoint_path.as_ref().to_path_buf();
    let mut store = SafetensorsStore::from_file(checkpoint_path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .remap(remapper)
        .allow_partial(true);

    model
        .load_from(&mut store)
        .map_err(|e| LoadError::Store(e.to_string()))?;

    Ok(())
}

/// Load pre-trained weights from a safetensors file into a `BertMaskedLM`.
///
/// The HF prefix (`bert.` or `roberta.`) is rewritten to `bert.` so it matches the Burn field.
/// The MLM decoder weight is tied to `word_embeddings.weight` after loading, mirroring HF's
/// runtime behavior (RoBERTa checkpoints store the decoder bias as `lm_head.bias`, and the
/// decoder weight is tied rather than stored).
pub fn load_pretrained_masked_lm<B: Backend>(
    model: &mut BertMaskedLM<B>,
    checkpoint_path: impl AsRef<Path>,
) -> Result<(), LoadError> {
    let mut mappings = vec![(r"^(?:bert|roberta)\.(.+)", "bert.$1")];
    mappings.extend(bert_structure_mappings());
    // RoBERTa MLM stores the decoder bias as `lm_head.bias`.
    mappings.push((r"^lm_head\.bias$", "lm_head.decoder.bias"));

    let remapper = build_remapper(mappings)?;
    let checkpoint_path: PathBuf = checkpoint_path.as_ref().to_path_buf();
    let mut store = SafetensorsStore::from_file(checkpoint_path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .remap(remapper)
        .allow_partial(true);

    model
        .load_from(&mut store)
        .map_err(|e| LoadError::Store(e.to_string()))?;

    // Tie: decoder.weight = word_embeddings.weight.T  (HF convention)
    let tied_weight = model.bert.embeddings.word_embeddings_weight().transpose();
    model.lm_head.decoder.weight = Param::from_tensor(tied_weight);

    Ok(())
}

/// Load the BERT model config from the JSON format available on Hugging Face Hub.
pub fn load_model_config(path: PathBuf) -> BertModelConfig {
    let mut model_config = BertModelConfig::load(path).expect("Config file present");
    model_config.max_seq_len = Some(512);
    model_config
}

/// Download model config and weights from Hugging Face Hub.
/// Cached files are reused.
pub fn download_hf_model(model_name: &str) -> (PathBuf, PathBuf) {
    let api = hf_hub::api::sync::Api::new().expect("Failed to create HF API client");
    let repo = api.model(model_name.to_string());

    let model_filepath = repo.get("model.safetensors").unwrap_or_else(|_| {
        panic!(
            "Failed to download: {} weights with name: model.safetensors from HuggingFace Hub",
            model_name
        )
    });

    let config_filepath = repo.get("config.json").unwrap_or_else(|_| {
        panic!(
            "Failed to download: {} config with name: config.json from HuggingFace Hub",
            model_name
        )
    });

    (config_filepath, model_filepath)
}
