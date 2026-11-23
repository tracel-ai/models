//! Greedy next tokens for a prompt.
//!
//! ```sh
//! cargo run -p qwen35-burn --release --example next-token -- <model dir> "The capital of Sweden is"
//! ```
//!
//! `<model dir>` holds `config.json`, `model.safetensors` and `tokenizer.json` of a Qwen3.5
//! checkpoint. Runs on Burn's default device.

use burn::tensor::{Device, Int, Tensor, TensorData};
use qwen35_burn::{Tokenizer, loader};
use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("usage: next-token <model dir> <prompt>"));
    let prompt = args
        .next()
        .unwrap_or_else(|| "The capital of Sweden is".to_string());
    let device = Device::default();

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let model = loader::from_dir(&dir, &device).unwrap();

    let mut ids: Vec<u32> = tok.encode(&prompt);
    for _ in 0..8 {
        let len = ids.len();
        let input: Vec<i64> = ids.iter().map(|&i| i as i64).collect();
        let input = Tensor::<2, Int>::from_data(TensorData::new(input, [1, len]), &device);
        let last = model.forward(input).narrow(1, len - 1, 1);
        let next = model.logits(last).argmax(2).into_scalar::<i64>() as u32;
        ids.push(next);
    }
    println!("{}", tok.decode(&ids));
}
