use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_ndarray::NdArrayBackend;
use std::error::Error;
use tts_burn::preprocess::TextPreprocessor;
use tts_burn::Tacotron2;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the model with default parameters
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let model: Tacotron2<NdArrayBackend> = Tacotron2::new(&device);

    // Create a text preprocessor
    let preprocessor = TextPreprocessor::new();

    // Text to synthesize
    let text = "Hello, this is a test of the text to speech system.";

    // Preprocess the text
    let processed_text = preprocessor.process(text)?;

    // Convert text to tensor
    let text_tensor = Tensor::from_data(processed_text, &device);

    // Generate mel spectrogram
    let mel_spec = model.forward(text_tensor)?;

    // Synthesize audio
    let audio = model.synthesize(text)?;

    // Save the audio to a file
    // Note: You'll need to implement audio saving functionality
    // This is a placeholder for the actual implementation
    println!("Generated audio with {} samples", audio.len());

    Ok(())
} 