use std::time::Instant;

use burn::backend::{libtorch::LibTorchDevice, LibTorch};
// use burn::backend::{wgpu::WgpuDevice, Wgpu};
use clap::Parser;
use llama_burn::llama::{Llama, LlamaConfig};

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
    // let device = WgpuDevice::default();
    // let mut llama: Llama<Wgpu> = LlamaConfig::llama3_8b(&args.tokenizer)
    println!("Loading Llama...");
    let mut llama: Llama<LibTorch> = LlamaConfig::llama3_8b(&args.tokenizer)
        .load_pretrained(&args.model, &device)
        .map_err(|err| format!("Failed to load pre-trained Llama model.\nError: {err}"))
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

// fn main() {
//     let tokenizer =
//         Tiktoken::new("/home/laggui/workspace/llama3/Meta-Llama-3-8B/tokenizer.model").unwrap();

//     let prompts = [
//         "I believe the meaning of life is",
//         // [128000, 40, 4510, 279, 7438, 315, 2324, 374]
//         "Simply put, the theory of relativity states that ",
//         // [128000, 61346, 2231, 11, 279, 10334, 315, 1375, 44515, 5415, 430, 220]
//         "A brief message congratulating the team on the launch:

//         Hi everyone,

//         I just ",
//         // [128000, 32, 10015, 1984, 40588, 15853, 279, 2128, 389, 279, 7195, 1473, 286, 21694, 5127, 3638, 286, 358, 1120, 220]
//         "Translate English to French:

//         sea otter => loutre de mer
//         peppermint => menthe poivrée
//         plush girafe => girafe peluche
//         cheese =>",
//         // [128000, 28573, 6498, 311, 8753, 1473, 286, 9581, 14479, 466, 591, 326, 412, 265, 409, 4809, 198, 286, 83804, 94932, 591, 11540, 383, 3273, 58866, 8047, 198, 286, 72779, 41389, 5763, 591, 41389, 5763, 12077, 34927, 198, 286, 17604, 591]
//     ];

//     for prompt in prompts {
//         println!("Prompt:\n{}", prompt);
//         let tokens = tokenizer.encode(prompt, true, false);
//         println!("Tokens:\n{:?}", tokens);
//     }
// }
