# Llama Burn

Llama-3 implementation.

You can find the [Burn](https://github.com/tracel-ai/burn) implementation for the Llama variants in
[src/llama.rs](src/llama.rs).

## Usage

### `Cargo.toml`

Add this to your `Cargo.toml`:

```toml
[dependencies]
llama-burn = { git = "https://github.com/tracel-ai/models", package = "llama-burn", default-features = false }
```

If you want to use Llama 3 or TinyLlama (including pre-trained weights if default features are
active), enable the corresponding feature flag.

**Important:** these features require `std`.

#### Llama 3

```toml
[dependencies]
llama-burn = { git = "https://github.com/tracel-ai/models", package = "llama-burn", features = ["llama3"] }
```

#### TinyLlama

```toml
[dependencies]
llama-burn = { git = "https://github.com/tracel-ai/models", package = "llama-burn", features = ["tiny"] }
```

### Example Usage

The [text generation example](examples/generate.rs) initializes a Llama model from the provided
weights file and generates a sequence of text based on the input prompt.

You can run the example with the following command:

### LLama 3

```sh
cargo run --features llama3 --example generate --release
```

### TinyLlama

```sh
cargo run --features tiny --example generate --release
```
