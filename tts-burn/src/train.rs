use burn::{
    module::Module,
    nn::loss::Loss,
    optim::{Adam, Optimizer},
    tensor::{backend::Backend, Tensor},
    train::{Metric, TrainOutput, TrainStep, ValidStep},
};

use crate::{Tacotron2, TextPreprocessor, AudioPreprocessor};

#[derive(Debug)]
pub struct Tacotron2Loss<B: Backend> {
    mel_loss: Tensor<B, 1>,
    gate_loss: Tensor<B, 1>,
}

impl<B: Backend> Tacotron2Loss<B> {
    pub fn new() -> Self {
        Self {
            mel_loss: Tensor::zeros([1]),
            gate_loss: Tensor::zeros([1]),
        }
    }

    pub fn forward(
        &self,
        pred_mel: Tensor<B, 3>,
        target_mel: Tensor<B, 3>,
        pred_gate: Tensor<B, 2>,
        target_gate: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        // Mel spectrogram loss (L1)
        let mel_loss = (pred_mel - target_mel).abs().mean();
        
        // Gate loss (Binary Cross Entropy)
        let gate_loss = pred_gate.binary_cross_entropy(target_gate);
        
        mel_loss + gate_loss
    }
}

#[derive(Debug)]
pub struct Tacotron2Trainer<B: Backend> {
    model: Tacotron2<B>,
    optimizer: Adam<B>,
    loss_fn: Tacotron2Loss<B>,
    text_processor: TextPreprocessor,
    audio_processor: AudioPreprocessor,
}

impl<B: Backend> Tacotron2Trainer<B> {
    pub fn new(
        model: Tacotron2<B>,
        learning_rate: f64,
        text_processor: TextPreprocessor,
        audio_processor: AudioPreprocessor,
    ) -> Self {
        let optimizer = Adam::new(learning_rate);
        let loss_fn = Tacotron2Loss::new();

        Self {
            model,
            optimizer,
            loss_fn,
            text_processor,
            audio_processor,
        }
    }

    pub fn train_step(
        &mut self,
        text: &str,
        audio: &[f32],
    ) -> TrainOutput<B> {
        // Convert inputs to tensors
        let text_tensor = self.text_processor.text_to_sequence(text);
        let target_mel = self.audio_processor.mel_spectrogram(audio);
        
        // Forward pass
        let pred_mel = self.model.forward(text_tensor);
        
        // Compute loss
        let loss = self.loss_fn.forward(
            pred_mel,
            target_mel,
            Tensor::zeros([1, 1]), // Placeholder for gate predictions
            Tensor::zeros([1, 1]), // Placeholder for gate targets
        );
        
        // Backward pass
        let grads = loss.backward();
        self.optimizer.step(grads);
        
        TrainOutput::new(loss)
    }
}

#[derive(Debug)]
pub struct Tacotron2Metrics<B: Backend> {
    mel_loss: Tensor<B, 1>,
    gate_loss: Tensor<B, 1>,
}

impl<B: Backend> Metric<B> for Tacotron2Metrics<B> {
    fn update(&mut self, loss: &Tensor<B, 1>) {
        self.mel_loss = loss.clone();
    }

    fn compute(&self) -> f64 {
        self.mel_loss.into_scalar() as f64
    }
} 