//! MiniLM sentence transformer model for Burn.
//!
//! This crate provides an implementation of the MiniLM-L12-v2 model
//! for generating sentence embeddings.
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
//! let (model, tokenizer) = MiniLmModel::<B>::pretrained(&device)?;
//! ```

mod embedding;
mod loader;
mod model;
mod pooling;

pub use embedding::*;
pub use loader::*;
pub use model::*;
pub use pooling::*;
