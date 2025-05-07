# DeepSeek-R1-Burn

A Rust implementation of the DeepSeek-R1 language model using the Burn framework.

## Features

- Full implementation of the DeepSeek-R1 architecture
- Support for both CPU and GPU (CUDA/WebGPU) backends
- Tokenization support using HuggingFace's tokenizers
- Training utilities with Burn's training framework
- Model configuration and serialization
- Text generation with temperature sampling
- Fine-tuning support

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
deepseek-r1-burn = { version = "0.1.0", features = ["webgpu"] }  # or "cuda" for CUDA support
```

## Usage

### Basic Usage

```rust
use deepseek_r1_burn::{deepseek_r1_config, DeepSeekR1, DeepSeekTokenizer};
use std::path::Path;

fn main() {
    // Create model configuration
    let config = deepseek_r1_config();
    let device = burn::tensor::Device::default();
    let model: DeepSeekR1<_> = DeepSeekR1::new(&config);

    // Load tokenizer
    let tokenizer = DeepSeekTokenizer::new(Path::new("path/to/tokenizer.json"))
        .expect("Failed to load tokenizer");

    // Tokenize input
    let input = "Hello, world!";
    let tokens = tokenizer.encode(input).expect("Failed to encode input");

    // Forward pass
    let input_tensor = burn::tensor::Tensor::<_, 2>::from_data(
        burn::tensor::TensorData::new(tokens, [1, tokens.len()]),
        &device,
    );
    let output = model.forward(input_tensor);

    // Decode output
    let output_tokens = output.argmax(2).into_data().value;
    let text = tokenizer.decode(&output_tokens).expect("Failed to decode output");
    println!("Generated text: {}", text);
}
```

### Model Serialization

```rust
use deepseek_r1_burn::{deepseek_r1_config, DeepSeekR1};
use std::path::Path;

fn main() {
    let config = deepseek_r1_config();
    let device = burn::tensor::Device::default();
    let model: DeepSeekR1<_> = DeepSeekR1::new(&config);

    // Save model
    model.save_file(Path::new("model.pt")).expect("Failed to save model");

    // Load model
    let loaded_model: DeepSeekR1<_> = DeepSeekR1::load_file(Path::new("model.pt"), &device)
        .expect("Failed to load model");
}
```

### Text Generation

```rust
use deepseek_r1_burn::{deepseek_r1_config, DeepSeekR1, DeepSeekTokenizer};
use std::path::Path;

fn main() {
    let config = deepseek_r1_config();
    let device = burn::tensor::Device::default();
    let model: DeepSeekR1<_> = DeepSeekR1::new(&config);
    let tokenizer = DeepSeekTokenizer::new(Path::new("path/to/tokenizer.json"))
        .expect("Failed to load tokenizer");

    // Generate text with temperature sampling
    let prompt = "Once upon a time";
    let generated = generate_text(&model, &tokenizer, prompt, 100, 0.8);
    println!("Generated text: {}", generated);
}
```

### Fine-tuning

```rust
use deepseek_r1_burn::{deepseek_r1_config, DeepSeekR1, TrainingConfig};

fn main() {
    let config = deepseek_r1_config();
    let device = burn::tensor::Device::default();
    let model: DeepSeekR1<_> = DeepSeekR1::new(&config);

    // Create training configuration
    let mut training_config = TrainingConfig::default();
    training_config.learning_rate = 5e-5; // Lower learning rate for fine-tuning
    training_config.epochs = 3;
    training_config.batch_size = 4;

    // Fine-tune the model
    let fine_tuned_model = train(model, dataset, training_config);
}
```

## Examples

The repository includes several examples:

- `examples/generate.rs`: Text generation with temperature sampling
- `examples/save_load.rs`: Model serialization and deserialization
- `examples/finetune.rs`: Fine-tuning the model on custom data

Run an example with:

```bash
cargo run --example generate
```

## Features

- `std`: Enable standard library support (default)
- `webgpu`: Enable WebGPU backend support
- `cuda`: Enable CUDA backend support

## License

This project is licensed under both the Apache License 2.0 and MIT License. See the [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) files for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 