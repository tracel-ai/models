# MiniLM-Burn

MiniLM sentence transformer implementation in Rust using [Burn](https://github.com/tracel-ai/burn).

Supports two model variants from HuggingFace:
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - 6 layers, faster
- [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) - 12 layers, better quality (default)

## Usage

```rust
use minilm_burn::{mean_pooling, MiniLmModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();

    // Load pretrained model and tokenizer (downloads from HuggingFace)
    // Use MiniLmVariant::L6 for faster inference, L12 for better quality
    let (model, tokenizer) = MiniLmModel::pretrained(&device, Default::default(), None)?;

    // Tokenize and run inference
    let output = model.forward(input_ids, attention_mask.clone(), None);
    let embeddings = mean_pooling(output.hidden_states, attention_mask);

    Ok(())
}
```

## Example

Run the inference example:

```bash
cargo run --example inference --release
```

## Testing

Unit tests:

```bash
cargo test
```

Integration tests (requires model download):

```bash
cargo test -- --ignored
```

## Benchmarks

Run the benchmarks:

```bash
cargo bench
```

Results are saved to `target/criterion/`. View the HTML report:

```bash
open target/criterion/report/index.html
```

## License

MIT OR Apache-2.0
