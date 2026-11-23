//! HF safetensors → [`Qwen35TextModel`] through `burn-store`.
//!
//! `PyTorchToBurnAdapter` transposes the `[out, in]` linear weights, `FloatCastAdapter` turns
//! the BF16 checkpoint into F32 (Qwen3.5's recurrent state should stay F32). The remaps:
//!
//! ```text
//! model.language_model.X            -> X        (also language_model.X, model.X)
//! linear_attn.A_log                 -> linear_attn.a_log
//! linear_attn.conv1d.weight         -> linear_attn.conv1d
//! linear_attn.norm.weight           -> linear_attn.norm
//! ```
//!
//! Vision and multi-token-prediction tensors in the file are not used.

use crate::config::Qwen35Config;
use crate::model::Qwen35TextModel;
use burn::tensor::{DType, Device};
use burn_store::{
    FloatCastAdapter, KeyRemapper, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter,
    SafetensorsStore,
};
use std::path::Path;

#[derive(Debug)]
pub enum LoadError {
    Io(String),
    Config(String),
    Store(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(m) => write!(f, "io error: {m}"),
            LoadError::Config(m) => write!(f, "config error: {m}"),
            LoadError::Store(m) => write!(f, "store error: {m}"),
        }
    }
}

impl std::error::Error for LoadError {}

fn remapper() -> Result<KeyRemapper, LoadError> {
    KeyRemapper::from_patterns(vec![
        (
            r"^(?:model\.language_model\.|language_model\.|model\.)(.+)",
            "$1",
        ),
        (r"linear_attn\.A_log$", "linear_attn.a_log"),
        (r"linear_attn\.conv1d\.weight$", "linear_attn.conv1d"),
        (r"linear_attn\.norm\.weight$", "linear_attn.norm"),
    ])
    .map_err(|e| LoadError::Store(e.to_string()))
}

/// Loads checkpoint weights into `model`.
pub fn load_pretrained(
    model: &mut Qwen35TextModel,
    safetensors: impl AsRef<Path>,
) -> Result<(), LoadError> {
    let mut store = SafetensorsStore::from_file(safetensors.as_ref().to_path_buf())
        .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(DType::F32)))
        .remap(remapper()?);
    model
        .load_from(&mut store)
        .map_err(|e| LoadError::Store(e.to_string()))?;
    Ok(())
}

/// Reads `config.json` from an HF model directory.
pub fn load_config(dir: impl AsRef<Path>) -> Result<Qwen35Config, LoadError> {
    let path = dir.as_ref().join("config.json");
    let json = std::fs::read_to_string(&path)
        .map_err(|e| LoadError::Io(format!("{}: {e}", path.display())))?;
    Qwen35Config::from_json(&json).map_err(LoadError::Config)
}

/// Config and weights from an HF model directory (`config.json`, `model.safetensors`).
pub fn from_dir(dir: impl AsRef<Path>, device: &Device) -> Result<Qwen35TextModel, LoadError> {
    let dir = dir.as_ref();
    let config = load_config(dir)?;
    let mut model = Qwen35TextModel::new(&config, device);
    load_pretrained(&mut model, dir.join("model.safetensors"))?;
    Ok(model)
}
