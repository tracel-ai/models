use crate::model::{MiniLmConfig, MiniLmModel};
use burn::config::Config;
use burn::tensor::backend::Backend;
use burn_store::{KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use std::path::{Path, PathBuf};

/// Error type for model loading operations.
#[derive(Debug)]
pub enum LoadError {
    /// Error during weight loading from store.
    Store(String),
    /// Error during model download.
    Download(String),
    /// Error loading configuration.
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

/// Load pre-trained weights from a safetensors file into the model.
///
/// # Arguments
/// - `model`: The initialized MiniLmModel to load weights into.
/// - `checkpoint_path`: Path to the safetensors file.
///
/// # Key Mappings
/// Maps HuggingFace BERT key naming to Burn's TransformerEncoder structure:
/// - `bert.encoder.layer.{i}.*` → `encoder.layers.{i}.*`
/// - `attention.self.query/key/value` → `mha.query/key/value`
/// - `attention.output.dense` → `mha.output`
/// - `intermediate.dense` → `pwff.linear_inner`
/// - `output.dense` → `pwff.linear_outer`
/// - `LayerNorm.weight/bias` → `gamma/beta`
pub fn load_pretrained<B: Backend>(
    model: &mut MiniLmModel<B>,
    checkpoint_path: impl AsRef<Path>,
) -> Result<(), LoadError> {
    // Key mappings: HuggingFace BERT -> Burn TransformerEncoder
    // Applied in order, so more specific patterns should come first
    let key_mappings: Vec<(&str, &str)> = vec![
        // Remove bert. prefix
        ("^bert\\.(.+)", "$1"),
        // encoder.layer.X -> encoder.layers.X
        ("encoder\\.layer\\.([0-9]+)", "encoder.layers.$1"),
        // Attention mappings
        ("attention\\.self\\.query", "mha.query"),
        ("attention\\.self\\.key", "mha.key"),
        ("attention\\.self\\.value", "mha.value"),
        ("attention\\.output\\.dense", "mha.output"),
        ("attention\\.output\\.LayerNorm", "norm_1"),
        // Feed-forward mappings
        ("intermediate\\.dense", "pwff.linear_inner"),
        // output.dense (not in attention) -> pwff.linear_outer
        // This needs to match "layers.X.output.dense" but not "attention.output.dense"
        ("(layers\\.[0-9]+)\\.output\\.dense", "$1.pwff.linear_outer"),
        ("(layers\\.[0-9]+)\\.output\\.LayerNorm", "$1.norm_2"),
        // Embedding LayerNorm -> layer_norm
        ("embeddings\\.LayerNorm", "embeddings.layer_norm"),
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

/// Download model files from HuggingFace Hub.
///
/// Downloads to `~/.cache/burn-models/` and returns paths.
/// Files are cached and won't be re-downloaded if they exist.
///
/// # Arguments
/// - `model_name`: HuggingFace model identifier (e.g., "sentence-transformers/all-MiniLM-L12-v2")
///
/// # Returns
/// [`HfModelFiles`] containing paths to config, weights, and tokenizer.
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

/// Load model configuration from a JSON file.
pub fn load_config(path: impl AsRef<Path>) -> Result<MiniLmConfig, LoadError> {
    MiniLmConfig::load(path).map_err(|e| LoadError::Config(e.to_string()))
}

/// Available pretrained MiniLM model variants.
#[cfg(feature = "pretrained")]
#[derive(Debug, Clone, Copy, Default)]
pub enum MiniLmVariant {
    /// all-MiniLM-L6-v2: 6 layers, faster inference.
    L6,
    /// all-MiniLM-L12-v2: 12 layers, better quality.
    #[default]
    L12,
}

#[cfg(feature = "pretrained")]
impl MiniLmVariant {
    fn model_id(&self) -> &'static str {
        match self {
            MiniLmVariant::L6 => "sentence-transformers/all-MiniLM-L6-v2",
            MiniLmVariant::L12 => "sentence-transformers/all-MiniLM-L12-v2",
        }
    }
}

#[cfg(feature = "pretrained")]
impl<B: Backend> MiniLmModel<B> {
    /// Load a pre-trained MiniLM model.
    ///
    /// Downloads from HuggingFace Hub (cached after first download).
    /// Returns the model and tokenizer.
    ///
    /// # Arguments
    /// - `device`: The device to load the model on.
    /// - `variant`: Model variant (L6 or L12). Defaults to L12.
    /// - `cache_dir`: Optional cache directory. Defaults to system cache dir.
    pub fn pretrained(
        device: &B::Device,
        variant: MiniLmVariant,
        cache_dir: Option<PathBuf>,
    ) -> Result<(Self, tokenizers::Tokenizer), LoadError> {
        let files = download_hf_model(variant.model_id(), cache_dir)?;

        let config = MiniLmConfig::load_from_hf(&files.config_path)
            .map_err(|e| LoadError::Config(e.to_string()))?;
        let mut model = config.init(device);

        load_pretrained(&mut model, &files.weights_path)?;

        let tokenizer = tokenizers::Tokenizer::from_file(&files.tokenizer_path)
            .map_err(|e| LoadError::Config(format!("Failed to load tokenizer: {}", e)))?;

        Ok((model, tokenizer))
    }
}
