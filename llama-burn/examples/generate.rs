use std::time::Instant;

use burn::{
    backend::{libtorch::LibTorchDevice, LibTorch},
    tensor::backend::Backend,
};
use clap::Parser;
use llama_burn::{
    llama::{Llama, LlamaConfig},
    sampling::{Sampler, TopP},
    tokenizer::Tokenizer,
};

const DEFAULT_PROMPT: &str = "I believe the meaning of life is";

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Config {
    // TODO: download checkpoint from HF hub.
    /// Model checkpoint path (automatically downloaded from the web if not present).
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

    /// The input prompt.
    #[arg(short, long, default_value_t = String::from(DEFAULT_PROMPT))]
    prompt: String,
}

pub fn generate<B: Backend, T: Tokenizer>(
    llama: &mut Llama<B, T>,
    prompt: &str,
    sample_len: usize,
    temperature: f64,
    sampler: &mut Sampler,
) {
    println!("Processing prompt: {}", prompt);
    let now = Instant::now();
    let generated = llama.generate(prompt, sample_len, temperature, sampler);
    let elapsed = now.elapsed().as_secs();

    println!("> {}\n", generated.text);
    println!(
        "{} tokens generated ({:.4} tokens/s)\n",
        generated.tokens,
        generated.tokens as f64 / generated.time
    );

    println!(
        "Generation completed in {}m{}s",
        (elapsed / 60),
        elapsed % 60
    );
}

pub fn main() {
    type B = LibTorch;

    // Parse arguments
    let args = Config::parse();

    let device = LibTorchDevice::Cuda(0);

    // Sampling strategy
    let mut sampler = if args.temperature > 0.0 {
        Sampler::TopP(TopP::new(args.top_p, args.seed))
    } else {
        Sampler::Argmax
    };

    #[cfg(feature = "tiny")]
    {
        let mut llama =
            LlamaConfig::load_tiny_llama::<B>(&args.model, &args.tokenizer, &device).unwrap();
        generate(
            &mut llama,
            &args.prompt,
            args.sample_len,
            args.temperature,
            &mut sampler,
        );
    }

    #[cfg(feature = "llama3")]
    {
        let mut llama =
            LlamaConfig::load_llama3_8b::<B>(&args.model, &args.tokenizer, &device).unwrap();
        generate(
            &mut llama,
            &args.prompt,
            args.sample_len,
            args.temperature,
            &mut sampler,
        );
    }
}
