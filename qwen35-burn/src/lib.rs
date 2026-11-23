//! Qwen3.5's text model with Burn.
//!
//! Qwen3.5 is a hybrid: three of every four layers are Gated DeltaNet ([`gdn`]), the fourth is
//! gated softmax attention. Everything is plain Burn tensor ops, so the model runs on every Burn
//! backend and trains with autodiff.
//!
//! * [`Qwen35Config`]: parses the HF `config.json`.
//! * [`Qwen35TextModel`]: embedding, decoder layers, final norm, tied LM head.
//! * [`loader`]: HF safetensors through `burn-store`.
//! * [`Tokenizer`]: Qwen's byte-level BPE in pure Rust.

pub mod blocks;
pub mod config;
pub mod gdn;
pub mod loader;
pub mod model;
pub mod tokenizer;

pub use config::{LayerKind, Qwen35Config};
pub use model::Qwen35TextModel;
pub use tokenizer::Tokenizer;
