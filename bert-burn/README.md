# Bert-Burn Model

This project provides an example implementation for inference on the BERT family of models. Any compatible BERT model
can be used by providing the correct configuration and weights (in safetensor format) which are available to download
from Hugging Face's model hub. For eg: Roberta-base weights can be downloaded from: [robert-base](https://huggingface.co/FacebookAI/roberta-base/tree/main)

### To include the model in your project

Add this to your `Cargo.toml`:

```toml
[dependencies]
bert-burn = { git = "https://github.com/burn-rs/models", package = "bert-burn", default-features = false }
```


# Example Usage
Place the  `config.json` and `model.safetensors` files to be downloaded from the [HF model hub](https://huggingface.co/FacebookAI/roberta-base/tree/main)
in the `weights/` directory. Example usage for obtaining sentence embeddings from the RoBERTa Encoder using Burn is shown below:

### WGPU backend
```bash
cd bert-burn/

cargo run --example infer-embedding --release --features wgpu   # Get sentence embeddings from the BERT encoder on sample text strings

```
