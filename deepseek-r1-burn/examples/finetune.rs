use burn::{
    data::dataset::Dataset,
    tensor::backend::Backend,
};
use deepseek_r1_burn::{
    deepseek_r1_config, DeepSeekR1, DeepSeekTokenizer, TrainingConfig, train,
};
use std::path::Path;

struct TextDataset {
    texts: Vec<String>,
    tokenizer: DeepSeekTokenizer,
    max_length: usize,
}

impl<B: Backend> Dataset<burn::tensor::Tensor<B, 2>> for TextDataset {
    fn get(&self, index: usize) -> Option<burn::tensor::Tensor<B, 2>> {
        let text = self.texts.get(index)?;
        let tokens = self.tokenizer.encode(text).ok()?;
        
        // Truncate or pad to max_length
        let mut padded = vec![0; self.max_length];
        let len = tokens.len().min(self.max_length);
        padded[..len].copy_from_slice(&tokens[..len]);
        
        Some(burn::tensor::Tensor::<_, 2>::from_data(
            burn::tensor::TensorData::new(padded, [1, self.max_length]),
            &B::Device::default(),
        ))
    }

    fn len(&self) -> usize {
        self.texts.len()
    }
}

fn main() {
    // Create model configuration
    let config = deepseek_r1_config();
    let device = burn::tensor::Device::default();
    let model: DeepSeekR1<_> = DeepSeekR1::new(&config);

    // Load tokenizer
    let tokenizer = DeepSeekTokenizer::new(Path::new("path/to/tokenizer.json"))
        .expect("Failed to load tokenizer");

    // Create dataset
    let texts = vec![
        "This is a sample text for fine-tuning.".to_string(),
        "Another example text for the model to learn from.".to_string(),
        // Add more training examples here
    ];
    let dataset = TextDataset {
        texts,
        tokenizer,
        max_length: 512,
    };

    // Configure training
    let mut training_config = TrainingConfig::default();
    training_config.learning_rate = 5e-5; // Lower learning rate for fine-tuning
    training_config.epochs = 3;
    training_config.batch_size = 4;

    // Fine-tune the model
    let fine_tuned_model = train(model, dataset, training_config);

    // Save the fine-tuned model
    fine_tuned_model
        .save_file(Path::new("fine_tuned_model.pt"))
        .expect("Failed to save fine-tuned model");
    println!("Model fine-tuning completed and saved!");
} 