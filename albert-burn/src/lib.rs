//! ALBERT masked language model for Burn.
//!
//! This crate provides an implementation of ALBERT (A Lite BERT) for
//! masked language modeling, using cross-layer parameter sharing and
//! factorized embedding parameterization.
//!
//! # Example
//!
//! ```ignore
//! use albert_burn::{AlbertMaskedLM, tokenize_batch};
//! use burn::backend::NdArray;
//!
//! type B = NdArray<f32>;
//!
//! let device = Default::default();
//! let (model, tokenizer) = AlbertMaskedLM::<B>::pretrained(&device, Default::default(), None)?;
//! ```

mod embedding;
mod encoder;
mod loader;
mod model;
mod tokenize;

pub use embedding::*;
pub use loader::*;
pub use model::*;
pub use tokenize::*;
