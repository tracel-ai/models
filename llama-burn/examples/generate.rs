use std::time::Instant;

use burn::{
    backend::{libtorch::LibTorchDevice, LibTorch},
    record::{HalfPrecisionSettings, NamedMpkFileRecorder},
};
use clap::Parser;
use llama_burn::{
    llama::{Llama, LlamaConfig},
    tokenizer::Tiktoken,
};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Config {
    // TODO: add prompt with default text?
    /// Model checkpoint path.
    #[arg(short, long)]
    model: String,

    /// Tokenizer path.
    #[arg(short, long)]
    tokenizer: String,

    /// Top-p probability threshold.
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,

    /// Temperature value for controlling randomness in sampling.
    #[arg(long, default_value_t = 0.6)]
    temperature: f64,

    /// Maximum sequence length for input text.
    #[arg(long, default_value_t = 512)]
    max_seq_len: usize,

    /// The number of new tokens to generate (i.e., the number of generation steps to take).
    #[arg(long, short = 'n', default_value_t = 50)]
    sample_len: usize,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

pub fn main() {
    // Parse arguments
    let args = Config::parse();

    let device = LibTorchDevice::Cuda(0);
    println!("Loading Llama...");
    let llama: Llama<LibTorch, Tiktoken> = LlamaConfig::llama3_8b(&args.tokenizer)
        // .load_pretrained(&args.model, &device) // takes too long, let's load the pre-saved mpk record
        .init(&device)
        .map_err(|err| format!("Failed to load pre-trained Llama model.\nError: {err}"))
        .unwrap();

    // Load model record
    let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
    // llama
    //     .save("llama_model", &recorder)
    //     .map_err(|err| format!("Failed to save weights to file {file_path}.\nError: {err}"))
    //     .unwrap();
    let mut llama = llama
        .load(&args.model, &recorder)
        .map_err(|err| {
            format!(
                "Failed to load weights to file {}.\nError: {err}",
                &args.model
            )
        })
        .unwrap();

    let prompt = "I believe the meaning of life is";

    println!("Processing prompt: {}", prompt);
    let now = Instant::now();
    let generated = llama.generate(
        prompt,
        args.sample_len,
        args.temperature,
        args.top_p,
        args.seed,
    );
    let elapsed = now.elapsed().as_secs();

    println!("> {}\n", generated);

    println!(
        "Generation completed in {}m{}s",
        (elapsed / 60),
        elapsed % 60
    );
}
