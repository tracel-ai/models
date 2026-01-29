//! MiniLM sentence transformer model for Burn.
//!
//! This crate provides an implementation of the MiniLM models
//! for generating sentence embeddings.
//!
//! Supports two variants:
//! - `MiniLmVariant::L6` - 6 layers, faster inference
//! - `MiniLmVariant::L12` - 12 layers, better quality (default)
//!
//! # Example
//!
//! ```ignore
//! use minilm_burn::{MiniLmConfig, MiniLmModel};
//! use burn::backend::NdArray;
//!
//! type B = NdArray<f32>;
//!
//! let device = Default::default();
//! let (model, tokenizer) = MiniLmModel::<B>::pretrained(&device, Default::default(), None)?;
//! ```

mod embedding;
mod loader;
mod model;
mod pooling;
mod tokenize;

pub use embedding::*;
pub use loader::*;
pub use model::*;
pub use pooling::*;
pub use tokenize::*;
