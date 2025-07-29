# TTS-Burn

A Text-to-Speech model implementation using the Burn framework. This implementation is based on the Tacotron 2 architecture, which combines a sequence-to-sequence model with a neural vocoder to generate high-quality speech from text.

## Features

- Text-to-speech synthesis using Tacotron 2 architecture
- Support for multiple languages
- High-quality speech output
- Easy integration with existing Burn-based projects

## Usage

```rust
use tts_burn::model::Tacotron2;
use burn::tensor::backend::Backend;

// Initialize the model
let model = Tacotron2::<Backend>::new();

// Generate speech from text
let text = "Hello, this is a test.";
let audio = model.synthesize(text);
```

## Model Architecture

The model consists of two main components:

1. **Encoder-Decoder with Attention**: Converts input text into a mel spectrogram
2. **Neural Vocoder**: Converts the mel spectrogram into raw audio

## Training

To train the model:

```bash
cargo run --release --bin train
```

## License

This project is licensed under both MIT and Apache 2.0 licenses, following the main repository's licensing scheme. 