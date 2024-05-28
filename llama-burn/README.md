# Llama Burn

The popular Llama LLM is here! This repository contains the
[Llama-3](https://github.com/meta-llama/llama3) and
[TinyLlama](https://github.com/jzhang38/TinyLlama) implementations with their corresponding
tokenizers.

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

The [chat completion example](examples/chat.rs) initializes a Llama model from the provided weights
file and generates a sequence of text based on the input prompt. The instruction-tuned model is
loaded for dialogue applications, so the prompt is automatically formatted for chat completion.

You can run the example with the following command:

### LLama 3

```sh
cargo run --release --features llama3 --example chat [-- --prompt "<your question/prompt here>"]
```

**Built with Meta Llama 3.** This example uses the
[Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
instruction-tuned model. Note that the [base pre-trained Llama-3 model](./src/pretrained.rs#L77) is
also available if you wish to use it in your application.

### TinyLlama

```sh
cargo run --release --features tiny --example chat [-- --prompt "<your question/prompt here>"]
```

This example uses the
[TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
instruction-tuned model based on the Llama2 architecture and tokenizer.
