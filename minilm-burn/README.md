# MiniLM-Burn

MiniLM sentence transformer implementation in Rust using [Burn](https://github.com/tracel-ai/burn).

Supports two model variants from HuggingFace:
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - 6 layers, faster
- [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) - 12 layers, better quality (default)

## Usage

```rust
use burn::backend::ndarray::NdArray;
use minilm_burn::{mean_pooling, MiniLmModel};

type B = NdArray<f32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();

    // Load pretrained model and tokenizer (downloads from HuggingFace)
    // Use MiniLmVariant::L6 for faster inference, L12 for better quality
    let (model, tokenizer) = MiniLmModel::<B>::pretrained(&device, Default::default(), None)?;

    // Tokenize and run inference
    let output = model.forward(input_ids, attention_mask.clone(), None);
    let embeddings = mean_pooling(output.hidden_states, attention_mask);

    Ok(())
}
```

## Features

- `pretrained` - Enables model download utilities (default)
- `ndarray` - NdArray backend
- `wgpu` - WebGPU backend
- `cuda` - CUDA backend
- `tch-cpu` - LibTorch CPU backend
- `tch-gpu` - LibTorch GPU backend

## Example

Run the inference example:

```bash
cargo run --example inference --features ndarray --release
```

## Testing

Unit tests:

```bash
cargo test --features ndarray
```

Integration tests (requires model download):

```bash
cargo test --features ndarray -- --ignored
```

## Benchmarks

Run for each backend:

```bash
cargo bench --features ndarray
cargo bench --features wgpu
cargo bench --features tch-cpu
```

Results are saved to `target/criterion/` for comparison across backends.

### Results (Apple M3 Max)

**L6 vs L12 (single sentence):**

| Variant | ndarray | wgpu  | tch-cpu |
| ------- | ------- | ----- | ------- |
| L6      | 53 ms   | 18 ms | 14 ms   |
| L12     | 105 ms  | 35 ms | 27 ms   |

**L12 batch scaling:**

| Batch size | ndarray | wgpu  | tch-cpu |
| ---------- | ------- | ----- | ------- |
| 1          | 102 ms  | 35 ms | 26 ms   |
| 4          | 387 ms  | 39 ms | 49 ms   |
| 8          | 774 ms  | 44 ms | 77 ms   |
| 16         | 1.54 s  | 73 ms | 130 ms  |

## License

MIT OR Apache-2.0
