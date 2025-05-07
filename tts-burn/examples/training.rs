use burn::optim::Adam;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_ndarray::NdArrayBackend;
use std::error::Error;
use tts_burn::preprocess::TextPreprocessor;
use tts_burn::train::{TrainingConfig, TrainingLoop};
use tts_burn::Tacotron2;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the model and device
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let model: Tacotron2<NdArrayBackend> = Tacotron2::new(&device);
    
    // Create optimizer
    let learning_rate = 0.001;
    let optimizer = Adam::new(&device, learning_rate);

    // Create text preprocessor
    let preprocessor = TextPreprocessor::new();

    // Training configuration
    let config = TrainingConfig {
        batch_size: 32,
        num_epochs: 100,
        learning_rate,
        save_interval: 10,
        checkpoint_dir: "checkpoints".into(),
    };

    // Create training loop
    let mut training_loop = TrainingLoop::new(model, optimizer, config);

    // Example training data
    // In a real scenario, you would load this from a dataset
    let training_data = vec![
        ("Hello, this is a training example.", vec![0.0; 16000]), // 1 second of silence
        ("Another example for training.", vec![0.0; 16000]),
    ];

    // Training loop
    for epoch in 0..config.num_epochs {
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;

        for (text, audio) in training_data.iter() {
            // Preprocess text
            let processed_text = preprocessor.process(text)?;
            let text_tensor = Tensor::from_data(processed_text, &device);
            let audio_tensor = Tensor::from_data(audio.clone(), &device);

            // Forward pass and loss computation
            let loss = training_loop.step(text_tensor, audio_tensor)?;
            epoch_loss += loss;
            num_batches += 1;
        }

        // Print epoch statistics
        let avg_loss = epoch_loss / num_batches as f32;
        println!("Epoch {}: Average Loss = {}", epoch + 1, avg_loss);

        // Save checkpoint if needed
        if (epoch + 1) % config.save_interval == 0 {
            training_loop.save_checkpoint(epoch + 1)?;
        }
    }

    Ok(())
} 