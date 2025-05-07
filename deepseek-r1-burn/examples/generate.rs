use burn::tensor::backend::Backend;
use deepseek_r1_burn::{deepseek_r1_config, DeepSeekR1, DeepSeekTokenizer};
use std::path::Path;

fn generate_text<B: Backend>(
    model: &DeepSeekR1<B>,
    tokenizer: &DeepSeekTokenizer,
    prompt: &str,
    max_length: usize,
    temperature: f32,
) -> String {
    let device = B::Device::default();
    let mut tokens = tokenizer.encode(prompt).expect("Failed to encode prompt");
    let mut generated = String::new();

    for _ in 0..max_length {
        // Convert tokens to tensor
        let input = burn::tensor::Tensor::<_, 2>::from_data(
            burn::tensor::TensorData::new(tokens.clone(), [1, tokens.len()]),
            &device,
        );

        // Get model predictions
        let output = model.forward(input);
        let logits = output.slice([0..1, -1..]);

        // Apply temperature
        let logits = logits / temperature;

        // Sample next token
        let probs = logits.softmax(1);
        let next_token = probs.multinomial(1).into_data().value[0];

        // Add token to sequence
        tokens.push(next_token);

        // Decode new token
        let new_text = tokenizer
            .decode(&[next_token])
            .expect("Failed to decode token");
        generated.push_str(&new_text);

        // Stop if we generate an EOS token
        if next_token == tokenizer.vocab_size() as u32 - 1 {
            break;
        }
    }

    generated
}

fn main() {
    // Create model configuration
    let config = deepseek_r1_config();
    let device = burn::tensor::Device::default();
    let model: DeepSeekR1<_> = DeepSeekR1::new(&config);

    // Load tokenizer
    let tokenizer = DeepSeekTokenizer::new(Path::new("path/to/tokenizer.json"))
        .expect("Failed to load tokenizer");

    // Generate text
    let prompt = "Once upon a time";
    let generated = generate_text(&model, &tokenizer, prompt, 100, 0.8);
    println!("Generated text: {}", generated);
} 