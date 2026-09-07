# Bert-Burn Model

This project provides an example implementation for inference on the BERT family of models. The following compatible
bert-variants: `roberta-base`(**default**)/`roberta-large`, `bert-base-uncased`/`bert-large-uncased`/`bert-base-cased`/`bert-large-cased`
can be loaded as following. The pre-trained weights and config files are automatically downloaded
from: [HuggingFace Model hub](https://huggingface.co/FacebookAI/roberta-base/tree/main)

### To include the model in your project

Add this to your `Cargo.toml`:

```toml
[dependencies]
bert-burn = { git = "https://github.com/tracel-ai/models", package = "bert-burn", default-features = false }
```

## Example Usage

Example usage for getting sentence embedding from given input text. The example runs on Burn's
default device, which is the Flex CPU backend unless another backend feature is enabled. Set
`BURN_DEVICE` to pick among the backends that are enabled. Safetensors weights are loaded through
[`burn-store`](https://crates.io/crates/burn-store).

### Sentence embeddings

```bash
cd bert-burn/
# Get sentence embeddings from the RoBERTa encoder (default)
cargo run --example infer-embedding --release

# Using bert-base-uncased model
cargo run --example infer-embedding --release -- bert-base-uncased

# Using roberta-large model
cargo run --example infer-embedding --release -- roberta-large
```


