use burn::tensor::{backend::Backend, Device};
use llama_burn::{llama::LlamaConfig, sampling::Sampler};

fn model_path() -> String {
    let home = std::env::var("HOME").expect("HOME not set");
    format!("{}/.cache/tinyllama-pytorch/model.safetensors", home)
}

fn tokenizer_path() -> String {
    let home = std::env::var("HOME").expect("HOME not set");
    format!("{}/.cache/tinyllama-pytorch/tokenizer.json", home)
}

pub fn test<B: Backend>(device: Device<B>) {
    let max_seq_len = 128;
    let model_path = model_path();
    let tokenizer_path = tokenizer_path();

    println!("Loading TinyLlama from safetensors...");
    println!("Model: {}", model_path);
    println!("Tokenizer: {}", tokenizer_path);

    let mut llama = LlamaConfig::tiny_llama(&tokenizer_path)
        .with_max_seq_len(max_seq_len)
        .load_pretrained::<B, llama_burn::tokenizer::SentiencePieceTokenizer>(&model_path, &device)
        .expect("Failed to load model");

    println!("Model loaded successfully!");

    // Test generation
    let prompt = "<|system|>\nYou are a friendly assistant.</s>\n<|user|>\nWhat is 2+2?</s>\n<|assistant|>\n";

    let mut sampler = Sampler::Argmax;

    println!("Generating response...");
    let output = llama.generate(prompt, 20, 0.0, &mut sampler);

    println!("Generated: {}", output.text);
    println!("Tokens: {}, Time: {:.2}s", output.tokens, output.time);
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use super::*;
    use burn::backend::{libtorch::LibTorchDevice, LibTorch};

    pub fn run() {
        let device = LibTorchDevice::Cpu;
        test::<LibTorch>(device);
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_backend {
    use super::*;
    use burn::backend::{ndarray::NdArrayDevice, NdArray};

    pub fn run() {
        let device = NdArrayDevice::Cpu;
        test::<NdArray>(device);
    }
}

pub fn main() {
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();

    #[cfg(feature = "ndarray")]
    ndarray_backend::run();

    #[cfg(not(any(feature = "tch-cpu", feature = "ndarray")))]
    {
        eprintln!("Please enable either 'tch-cpu' or 'ndarray' feature to run this test");
        std::process::exit(1);
    }
}
