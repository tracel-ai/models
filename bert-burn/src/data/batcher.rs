use super::tokenizer::Tokenizer;
use burn::{
    data::dataloader::batcher::Batcher,
    nn::attention::generate_padding_mask,
    tensor::{backend::Backend, Bool, Int, Tensor},
};
use std::sync::Arc;

#[derive(new)]
pub struct BertInputBatcher<B: Backend> {
    tokenizer: Arc<dyn Tokenizer>, // Tokenizer for converting text to token IDs
    device: B::Device, // Device on which to perform computation (e.g., CPU or CUDA device)
    max_seq_length: usize, // Maximum sequence length for tokenized text
}

#[derive(Debug, Clone, new)]
pub struct BertInferenceBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,    // Tokenized text
    pub mask_pad: Tensor<B, 2, Bool>, // Padding mask for the tokenized text
}

impl<B: Backend> Batcher<String, BertInferenceBatch<B>> for BertInputBatcher<B> {
    /// Batches a vector of strings into an inference batch
    fn batch(&self, items: Vec<String>) -> BertInferenceBatch<B> {
        let mut tokens_list = Vec::with_capacity(items.len());

        // Tokenize each string
        for item in items {
            tokens_list.push(self.tokenizer.encode(&item));
        }

        // Generate padding mask for tokenized text
        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens_list,
            Some(self.max_seq_length),
            &B::Device::default(),
        );

        // Create and return inference batch
        BertInferenceBatch {
            tokens: mask.tensor.to_device(&self.device),
            mask_pad: mask.mask.to_device(&self.device),
        }
    }
}
