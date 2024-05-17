use std::sync::Arc;

use crate::{
    data::Tokenizer,
    data::{BertInferenceBatch, BertTokenizer},
    model::BertMaskedLM,
    model::BertModelConfig,
};
use burn::tensor::{activation::softmax, backend::Backend, Data, Element};

type TokenType = usize;
const MASK_TOKEN_ID: TokenType = 50264;

fn find_masks<T: Element>(tokens: &Data<T, 1>, mask_token_id: TokenType) -> Vec<usize> {
    let mut masks = Vec::new();
    for (i, token) in tokens.value.iter().enumerate() {
        if token.to_usize() == Some(mask_token_id) {
            masks.push(i);
        }
    }
    masks
}

fn data_to_vec_f32<T: Element>(data: &Data<T, 1>) -> Vec<f32> {
    data.value.iter().map(|x| x.to_f32().unwrap()).collect()
}

fn top_k(k: usize, probabilities: Vec<f32>) -> Vec<(usize, f32)> {
    let mut probabilities = probabilities.iter().enumerate().collect::<Vec<_>>();

    probabilities.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    probabilities.truncate(k);

    probabilities.into_iter().map(|(i, &p)| (i, p)).collect()
}

#[derive(Debug, Clone)]
pub struct FillMaskResult {
    pub mask_idx: usize,
    pub top_k: Vec<(f32, String)>,
}

pub fn fill_mask<B: Backend>(
    model: &BertMaskedLM<B>,
    model_config: &BertModelConfig,
    tokenizer: &Arc<BertTokenizer>,
    input: BertInferenceBatch<B>,
) -> Vec<Vec<FillMaskResult>> {
    let [batch_size, seq_len] = input.tokens.dims();
    let output = model.forward(input.clone());

    let mut results = vec![];

    // Embedding size
    let d_model = model_config.vocab_size.clone();
    for i in 0..batch_size {
        let mut batch_results = vec![];
        let input_tokens = input
            .tokens
            .clone()
            .slice([i..i + 1, 0..seq_len])
            .squeeze(0)
            .to_data();
        // Find the mask tokens in the input, as a list of indices
        let masks = find_masks(&input_tokens, MASK_TOKEN_ID);
        for mask in masks {
            let logits = output
                .clone()
                .slice([i..i + 1, mask..(mask + 1), 0..d_model])
                .squeeze::<2>(0)
                .squeeze(0);
            // Find the top k tokens with the highest probabilities
            let probs = data_to_vec_f32(&softmax(logits, 0).to_data());
            let top_k = top_k(5, probs);
            batch_results.push(FillMaskResult {
                mask_idx: mask,
                top_k: top_k
                    .iter()
                    .map(|(k, prob)| (*prob, tokenizer.decode(&[*k])))
                    .collect(),
            });
        }
        results.push(batch_results);
    }

    results
}
