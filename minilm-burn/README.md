# MiniLM-Burn

MiniLM sentence transformer implementation in Rust using [Burn](https://github.com/tracel-ai/burn).

Supports loading pretrained weights from HuggingFace's
[sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2).

## Usage

```rust
use burn::backend::ndarray::NdArray;
use minilm_burn::{mean_pooling, MiniLmModel};

type B = NdArray<f32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();

    // Load pretrained model and tokenizer (downloads from HuggingFace)
    let (model, tokenizer) = MiniLmModel::<B>::pretrained(&device)?;

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

| Benchmark          | ndarray | wgpu  | tch-cpu |
| ------------------ | ------- | ----- | ------- |
| forward (batch=1)  | 102 ms  | 35 ms | 26 ms   |
| forward (batch=4)  | 387 ms  | 39 ms | 49 ms   |
| forward (batch=8)  | 774 ms  | 44 ms | 77 ms   |
| forward (batch=16) | 1.54 s  | 73 ms | 130 ms  |
| full_pipeline      | 101 ms  | 35 ms | 26 ms   |
| mean_pooling       | 41 µs   | 97 µs | 89 µs   |
| normalize_l2       | 1.1 µs  | 99 µs | 2.8 µs  |

## License

MIT OR Apache-2.0
