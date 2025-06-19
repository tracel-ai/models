use super::tokenizer::Tokenizer;
use burn::data::dataloader::batcher::Batcher;
use burn::nn::attention::generate_padding_mask;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};
use std::sync::Arc;

#[derive(new)]
pub struct BertInputBatcher {
    /// Tokenizer for converting input text string to token IDs
    tokenizer: Arc<dyn Tokenizer>,
    /// Maximum sequence length for tokenized text
    max_seq_length: usize,
}

#[derive(Debug, Clone, new)]
pub struct BertInferenceBatch<B: Backend> {
    /// Tokenized text as 2D tensor: [batch_size, max_seq_length]
    pub tokens: Tensor<B, 2, Int>,
    /// Padding mask for the tokenized text containing booleans for padding locations
    pub mask_pad: Tensor<B, 2, Bool>,
}

impl<B: Backend> Batcher<B, String, BertInferenceBatch<B>> for BertInputBatcher {
    /// Batches a vector of strings into an inference batch
    fn batch(&self, items: Vec<String>, device: &B::Device) -> BertInferenceBatch<B> {
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
            device,
        );

        // Create and return inference batch
        BertInferenceBatch {
            tokens: mask.tensor,
            mask_pad: mask.mask,
        }
    }
}
