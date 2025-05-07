use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_ndarray::NdArrayBackend;
use std::error::Error;
use tts_burn::preprocess::TextPreprocessor;
use tts_burn::Tacotron2;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the model and device
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let mut model: Tacotron2<NdArrayBackend> = Tacotron2::new(&device);

    // Load pre-trained weights
    let checkpoint_path = "checkpoints/model_epoch_100.pt";
    model.load_checkpoint(checkpoint_path)?;
    println!("Loaded checkpoint from {}", checkpoint_path);

    // Create text preprocessor
    let preprocessor = TextPreprocessor::new();

    // Example texts to synthesize
    let texts = vec![
        "Welcome to the text to speech system.",
        "This is a demonstration of the pre-trained model.",
        "The model can generate natural sounding speech.",
    ];

    // Process each text
    for (i, text) in texts.iter().enumerate() {
        println!("Processing text {}: {}", i + 1, text);

        // Preprocess text
        let processed_text = preprocessor.process(text)?;
        let text_tensor = Tensor::from_data(processed_text, &device);

        // Generate mel spectrogram
        let mel_spec = model.forward(text_tensor)?;
        println!("Generated mel spectrogram with shape: {:?}", mel_spec.shape());

        // Synthesize audio
        let audio = model.synthesize(text)?;
        println!("Generated audio with {} samples", audio.len());

        // Save the audio to a file
        // Note: You'll need to implement audio saving functionality
        // This is a placeholder for the actual implementation
        let output_path = format!("output_{}.wav", i + 1);
        println!("Would save audio to {}", output_path);
    }

    Ok(())
} 