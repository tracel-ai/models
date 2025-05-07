use burn::tensor::backend::Backend;
use deepseek_r1_burn::{deepseek_r1_config, DeepSeekR1};
use std::path::Path;

fn main() {
    // Create model configuration
    let config = deepseek_r1_config();
    let device = burn::tensor::Device::default();
    let model: DeepSeekR1<_> = DeepSeekR1::new(&config);

    // Save model
    let save_path = Path::new("model.pt");
    model.save_file(save_path).expect("Failed to save model");

    // Load model
    let loaded_model: DeepSeekR1<_> = DeepSeekR1::load_file(save_path, &device)
        .expect("Failed to load model");

    // Verify the models are the same
    let input = burn::tensor::Tensor::<_, 2>::zeros([1, 10], &device);
    let output1 = model.forward(input.clone());
    let output2 = loaded_model.forward(input);
    assert!(output1.equal(output2));
    println!("Model successfully saved and loaded!");
} 