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

Example usage for getting sentence embedding from given input text. The model supports multiple backends from burn
(e.g. `ndarray`, `wgpu`, `tch-gpu`, `tch-cpu`, `cuda`) which can be selected using the `--features` flag. An example with `wgpu`
backend is shown below. The `fusion` flag is used to enable kernel fusion for the `wgpu` backend. It is not required
with other backends. The `safetensors` flag is used to support loading weights in `safetensors` format via `candle-core`
crate.

### WGPU backend

```bash
cd bert-burn/
# Get sentence embeddings from the RobBERTa encoder (default)
cargo run --example infer-embedding --release --features wgpu,fusion,safetensors

# Using bert-base-uncased model
cargo run --example infer-embedding --release --features wgpu,fusion,safetensors bert-base-uncased 

# Using roberta-large model
cargo run --example infer-embedding --release --features wgpu,fusion,safetensors roberta-large
```


