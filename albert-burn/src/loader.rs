use crate::model::{AlbertConfig, AlbertMaskedLM};
use burn::config::Config;
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

/// Load pre-trained weights from a safetensors file into the masked LM model.
///
/// # Key Mappings (HuggingFace ALBERT → Burn)
///
/// Embedding keys:
/// - `albert.embeddings.*` → `albert.embeddings.*`
/// - `albert.encoder.embedding_hidden_mapping_in` → `albert.embeddings.projection`
/// - `albert.embeddings.LayerNorm` → `albert.embeddings.layer_norm`
///
/// Encoder keys (shared layer):
/// - `albert.encoder.albert_layer_groups.0.albert_layers.0.attention.query` → `albert.encoder.layer.mha.query`
/// - `albert.encoder.albert_layer_groups.0.albert_layers.0.attention.key` → `albert.encoder.layer.mha.key`
/// - `albert.encoder.albert_layer_groups.0.albert_layers.0.attention.value` → `albert.encoder.layer.mha.value`
/// - `albert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense` → `albert.encoder.layer.mha.output`
/// - `albert.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm` → `albert.encoder.layer.norm_1`
/// - `albert.encoder.albert_layer_groups.0.albert_layers.0.ffn` → `albert.encoder.layer.pwff.linear_inner`
/// - `albert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output` → `albert.encoder.layer.pwff.linear_outer`
/// - `albert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm` → `albert.encoder.layer.norm_2`
///
/// MLM head keys:
/// - `predictions.dense` → `mlm_dense`
/// - `predictions.LayerNorm` → `mlm_layer_norm`
/// - `predictions.decoder` → `mlm_decoder`
/// - `predictions.bias` → `mlm_bias`
pub fn load_pretrained<B: Backend>(
    model: &mut AlbertMaskedLM<B>,
    checkpoint_path: impl AsRef<Path>,
) -> Result<(), LoadError> {
    // HF keys start with "albert." for the base model, which matches our Burn field name.
    // We keep the "albert." prefix and only remap the inner structure.
    let key_mappings: Vec<(&str, &str)> = vec![
        // Embedding projection: HF and Burn both have it under encoder
        (
            "albert\\.encoder\\.embedding_hidden_mapping_in",
            "albert.encoder.projection",
        ),
        // Shared layer attention mappings
        (
            "albert\\.encoder\\.albert_layer_groups\\.0\\.albert_layers\\.0\\.attention\\.query",
            "albert.encoder.layer.mha.query",
        ),
        (
            "albert\\.encoder\\.albert_layer_groups\\.0\\.albert_layers\\.0\\.attention\\.key",
            "albert.encoder.layer.mha.key",
        ),
        (
            "albert\\.encoder\\.albert_layer_groups\\.0\\.albert_layers\\.0\\.attention\\.value",
            "albert.encoder.layer.mha.value",
        ),
        (
            "albert\\.encoder\\.albert_layer_groups\\.0\\.albert_layers\\.0\\.attention\\.dense",
            "albert.encoder.layer.mha.output",
        ),
        (
            "albert\\.encoder\\.albert_layer_groups\\.0\\.albert_layers\\.0\\.attention\\.LayerNorm",
            "albert.encoder.layer.norm_1",
        ),
        // Feed-forward mappings (ffn_output before ffn to avoid partial match)
        (
            "albert\\.encoder\\.albert_layer_groups\\.0\\.albert_layers\\.0\\.ffn_output",
            "albert.encoder.layer.pwff.linear_outer",
        ),
        (
            "albert\\.encoder\\.albert_layer_groups\\.0\\.albert_layers\\.0\\.ffn\\b",
            "albert.encoder.layer.pwff.linear_inner",
        ),
        (
            "albert\\.encoder\\.albert_layer_groups\\.0\\.albert_layers\\.0\\.full_layer_layer_norm",
            "albert.encoder.layer.norm_2",
        ),
        // Embedding LayerNorm
        (
            "albert\\.embeddings\\.LayerNorm",
            "albert.embeddings.layer_norm",
        ),
        // MLM head (HF uses "predictions.*", Burn uses "mlm_*")
        ("predictions\\.dense", "mlm_dense"),
        ("predictions\\.LayerNorm", "mlm_layer_norm"),
        // predictions.bias in safetensors has non-zero values, but in HF it's tied to
        // predictions.decoder.bias which ends up as zeros after from_pretrained(). We load
        // the decoder.bias (zeros) to match HF's runtime behavior.
        ("predictions\\.decoder\\.bias", "mlm_decoder_bias"),
    ];

    let remapper =
        KeyRemapper::from_patterns(key_mappings).map_err(|e| LoadError::Store(e.to_string()))?;

    let checkpoint_path: PathBuf = checkpoint_path.as_ref().to_path_buf();
    let mut store = SafetensorsStore::from_file(checkpoint_path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .remap(remapper);

    model
        .load_from(&mut store)
        .map_err(|e| LoadError::Store(e.to_string()))?;

    Ok(())
}

#[cfg(feature = "pretrained")]
pub struct HfModelFiles {
    pub config_path: PathBuf,
    pub weights_path: PathBuf,
    pub tokenizer_path: PathBuf,
}

#[cfg(feature = "pretrained")]
fn default_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("burn-models")
}

#[cfg(feature = "pretrained")]
pub fn download_hf_model(
    model_name: &str,
    cache_dir: Option<PathBuf>,
) -> Result<HfModelFiles, LoadError> {
    let cache_dir = cache_dir.unwrap_or_else(default_cache_dir);
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir)
        .build()
        .map_err(|e| LoadError::Download(format!("Failed to create HF API: {}", e)))?;
    let repo = api.model(model_name.to_string());

    let config_path = repo
        .get("config.json")
        .map_err(|e| LoadError::Download(format!("Failed to download config: {}", e)))?;

    let weights_path = repo
        .get("model.safetensors")
        .map_err(|e| LoadError::Download(format!("Failed to download weights: {}", e)))?;

    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| LoadError::Download(format!("Failed to download tokenizer: {}", e)))?;

    Ok(HfModelFiles {
        config_path,
        weights_path,
        tokenizer_path,
    })
}

pub fn load_config(path: impl AsRef<Path>) -> Result<AlbertConfig, LoadError> {
    AlbertConfig::load(path).map_err(|e| LoadError::Config(e.to_string()))
}

#[cfg(feature = "pretrained")]
impl<B: Backend> AlbertMaskedLM<B> {
    /// Load a pre-trained ALBERT masked LM model.
    ///
    /// Downloads from HuggingFace Hub (cached after first download).
    /// Returns the model and tokenizer.
    pub fn pretrained(
        device: &B::Device,
        cache_dir: Option<PathBuf>,
    ) -> Result<(Self, tokenizers::Tokenizer), LoadError> {
        let files = download_hf_model("albert/albert-base-v2", cache_dir)?;

        let config = AlbertConfig::load_from_hf(&files.config_path)?;
        let mut model = config.init_masked_lm(device);

        load_pretrained(&mut model, &files.weights_path)?;

        let tokenizer = tokenizers::Tokenizer::from_file(&files.tokenizer_path)
            .map_err(|e| LoadError::Config(format!("Failed to load tokenizer: {}", e)))?;

        Ok((model, tokenizer))
    }
}
