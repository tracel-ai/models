use std::time::Instant;

use burn::{
    backend::{libtorch::LibTorchDevice, LibTorch},
    tensor::{backend::Backend, f16},
};
use clap::Parser;
use llama_burn::{
    llama::{Llama, LlamaConfig},
    sampling::{Sampler, TopP},
    tokenizer::Tokenizer,
};

const DEFAULT_PROMPT: &str = "How many helicopters can a human eat in one sitting?";

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Config {
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

    /// Chat assistant mode.
    #[arg(short, long, default_value_t = true)]
    chat: bool,
}

pub fn generate<B: Backend, T: Tokenizer>(
    llama: &mut Llama<B, T>,
    prompt: &str,
    sample_len: usize,
    temperature: f64,
    sampler: &mut Sampler,
) {
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
    type B = LibTorch<f16>;

    // Parse arguments
    let args = Config::parse();

    let device = LibTorchDevice::Cuda(0);
    let mut prompt = args.prompt;

    // Sampling strategy
    let mut sampler = if args.temperature > 0.0 {
        Sampler::TopP(TopP::new(args.top_p, args.seed))
    } else {
        Sampler::Argmax
    };

    #[cfg(feature = "tiny")]
    {
        let mut llama = LlamaConfig::tiny_llama_pretrained::<B>(&device).unwrap();
        println!("Processing prompt: {}", prompt);

        if args.chat {
            // Prompt formatting for chat model
            prompt = format!(
                "<|system|>\nYou are a friendly chatbot who always responds in the style of a pirate</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
            )
        }

        generate(
            &mut llama,
            &prompt,
            args.sample_len,
            args.temperature,
            &mut sampler,
        );
    }

    #[cfg(feature = "llama3")]
    {
        let mut llama = LlamaConfig::llama3_8b_pretrained::<B>(args.chat, &device).unwrap();
        println!("Processing prompt: {}", prompt);

        if args.chat {
            // Prompt formatting for chat model
            prompt = format!(
                "<|start_header_id|>system<|end_header_id|>\n\nYou are a friendly chatbot who always responds in the style of a pirate<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        }

        generate(
            &mut llama,
            &prompt,
            args.sample_len,
            args.temperature,
            &mut sampler,
        );
    }
}
