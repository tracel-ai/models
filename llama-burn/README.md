# Llama Burn

Llama-3 implementation.

You can find the [Burn](https://github.com/tracel-ai/burn) implementation for the YOLOX variants in
[src/model/yolox.rs](src/model/yolox.rs).

The model is [no_std compatible](https://docs.rust-embedded.org/book/intro/no-std.html).

## Usage

### `Cargo.toml`

Add this to your `Cargo.toml`:

```toml
[dependencies]
llama-burn = { git = "https://github.com/tracel-ai/models", package = "llama-burn", default-features = false }
```

If you want to get the COCO pre-trained weights, enable the `pretrained` feature flag.

```toml
[dependencies]
llama-burn = { git = "https://github.com/tracel-ai/models", package = "llama-burn", features = ["pretrained"] }
```

**Important:** this feature requires `std`.

### Example Usage

The [text generation example](examples/generate.rs) initializes a Llama-3-8B from the provided
weights file with the `Wgpu` backend and generates a sequence of text based on the input prompt.

You can run the example with the following command:

```sh
cargo run --example generate --release -- --model Meta-Llama-3-8B/consolidated.00.pth --tokenizer Meta-Llama-3-8B/tokenizer.model
```
