//! Gated DeltaNet (Yang et al., 2024, arXiv:2412.06464), the linear-attention layer used by 18 of
//! Qwen3.5-0.8B's 24 layers. Burn has no such layer yet, so it lives here for now.
//!
//! * [`gated_delta_rule`]: the recurrence, chunked so it runs well on CPU and GPU.
//! * [`GatedDeltaNet`]: the layer as Qwen3.5 uses it, with the checkpoint's weight names.

mod chunked;
mod layer;

pub use chunked::{DEFAULT_CHUNK, gated_delta_rule, unit_lower_inverse};
pub use layer::{GatedDeltaNet, GatedDeltaNetConfig, GdnState};
