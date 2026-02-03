# ALBERT-Burn

ALBERT (A Lite BERT) masked language model implementation in Rust using
[Burn](https://github.com/tracel-ai/burn).

ALBERT uses factorized embedding parameterization and cross-layer parameter sharing to reduce model
size while maintaining performance.

Supports all v2 variants from HuggingFace:

| Variant          | Hidden Size | Parameters | HuggingFace                                                          |
| ---------------- | ----------- | ---------- | -------------------------------------------------------------------- |
| BaseV2 (default) | 768         | ~12M       | [albert-base-v2](https://huggingface.co/albert/albert-base-v2)       |
| LargeV2          | 1,024       | ~18M       | [albert-large-v2](https://huggingface.co/albert/albert-large-v2)     |
| XLargeV2         | 2,048       | ~60M       | [albert-xlarge-v2](https://huggingface.co/albert/albert-xlarge-v2)   |
| XXLargeV2        | 4,096       | ~235M      | [albert-xxlarge-v2](https://huggingface.co/albert/albert-xxlarge-v2) |

## Usage

```rust
use burn::backend::ndarray::NdArray;
use albert_burn::{AlbertMaskedLM, AlbertVariant, tokenize_batch};

type B = NdArray<f32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();

    // Load pretrained model and tokenizer (downloads from HuggingFace)
    let (model, tokenizer) = AlbertMaskedLM::<B>::pretrained(&device, AlbertVariant::BaseV2, None)?;

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
cargo run --example inference --features "pretrained,ndarray" --release
```

Specify a variant:

```bash
cargo run --example inference --features "pretrained,ndarray" --release -- xxlarge
```

### Results by variant

Prompt: `"The capital of France is [MASK]."`

**BaseV2** (12M params):

| Rank | Token    | Logit |
| ---- | -------- | ----- |
| 1    | reims    | 16.35 |
| 2    | toulouse | 16.17 |
| 3    | paris    | 15.89 |
| 4    | amiens   | 15.66 |
| 5    | cannes   | 15.62 |

**LargeV2** (18M params):

| Rank | Token      | Logit |
| ---- | ---------- | ----- |
| 1    | paris      | 14.41 |
| 2    | strasbourg | 12.26 |
| 3    | lyon       | 11.82 |
| 4    | brest      | 11.62 |
| 5    | cannes     | 11.58 |

**XLargeV2** (60M params):

| Rank | Token      | Logit |
| ---- | ---------- | ----- |
| 1    | paris      | 16.82 |
| 2    | lyon       | 16.06 |
| 3    | strasbourg | 15.86 |
| 4    | toulouse   | 15.02 |
| 5    | grenoble   | 13.91 |

**XXLargeV2** (235M params):

| Rank | Token      | Logit |
| ---- | ---------- | ----- |
| 1    | paris      | 20.15 |
| 2    | reims      | 17.17 |
| 3    | marseille  | 17.02 |
| 4    | versailles | 17.01 |
| 5    | nantes     | 16.96 |

## Testing

Integration tests (requires model download):

```bash
cargo test --features "pretrained,ndarray" -- --ignored
```

Tests verify logit values, top-5 predictions, statistics, and per-position L2 norms across 3
sentences against Python HuggingFace reference at 5e-4 relative tolerance.

## License

MIT OR Apache-2.0
