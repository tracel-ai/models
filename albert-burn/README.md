# ALBERT-Burn

ALBERT (A Lite BERT) masked language model implementation in Rust using
[Burn](https://github.com/tracel-ai/burn).

Loads the pretrained [albert-base-v2](https://huggingface.co/albert/albert-base-v2) model from
HuggingFace.

ALBERT uses factorized embedding parameterization and cross-layer parameter sharing to reduce model
size while maintaining performance.

## Usage

```rust
use burn::backend::ndarray::NdArray;
use albert_burn::{AlbertMaskedLM, tokenize_batch};

type B = NdArray<f32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();

    // Load pretrained model and tokenizer (downloads from HuggingFace)
    let (model, tokenizer) = AlbertMaskedLM::<B>::pretrained(&device, None)?;

    // Tokenize input with [MASK] token
    let sentence = "The capital of France is [MASK].";
    let (input_ids, attention_mask) = tokenize_batch::<B>(&tokenizer, &[sentence], &device);

    // Forward pass returns logits over vocabulary
    let logits = model.forward(input_ids, attention_mask, None);

    Ok(())
}
```

## Features

- `pretrained` - Enables model download utilities (default)
- `ndarray` - NdArray backend

Backend features:

- `wgpu` - WebGPU backend
- `cuda` - CUDA backend
- `tch-cpu` - LibTorch CPU backend
- `tch-gpu` - LibTorch GPU backend

## Example

Run the fill-mask inference example:

```bash
cargo run --example inference --features ndarray --release
```

Output:

```
Input: "The capital of France is [MASK]."

Top 5 predictions for [MASK]:
  1: "reims" (logit: 16.3455)
  2: "toulouse" (logit: 16.1738)
  3: "paris" (logit: 15.8940)
  4: "amiens" (logit: 15.6570)
  5: "cannes" (logit: 15.6186)
```

## Testing

Integration tests (requires model download):

```bash
cargo test --features "pretrained,ndarray" -- --ignored
```

Tests verify logit values, top-5 predictions, statistics, and per-position L2 norms across 3
sentences against Python HuggingFace reference at 5e-4 relative tolerance.

## License

MIT OR Apache-2.0
