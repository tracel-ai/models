use crate::{
    data::Tokenizer,
    data::{BertInferenceBatch, BertTokenizer},
    model::BertMaskedLM,
    model::BertModelConfig,
};
use burn::tensor::{activation::softmax, Element, Tensor};

type TokenType = usize;
const MASK_TOKEN_ID: TokenType = 50264;

#[derive(Debug, Clone)]
pub struct FillMaskResult {
    pub mask_idx: usize,
    pub top_k: Vec<(f32, String)>,
}

pub fn fill_mask(
    model: &BertMaskedLM,
    model_config: &BertModelConfig,
    tokenizer: &BertTokenizer,
    input: BertInferenceBatch,
) -> Vec<Vec<FillMaskResult>> {
    let [batch_size, seq_len] = input.tokens.dims();
    let output = model.forward(input.clone());

    let mut results = vec![];

    // Embedding size
    let d_model = model_config.vocab_size;
    for i in 0..batch_size {
        let mut batch_results = vec![];
        let input_tokens = input
            .tokens
            .clone()
            .slice([i..i + 1, 0..seq_len])
            .squeeze_dim::<1>(0)
            .into_data();
        // Find the mask tokens in the input, as a list of indices
        let masks = find_masks(
            &input_tokens.iter::<i64>().collect::<Vec<_>>(),
            MASK_TOKEN_ID,
        );
        for mask in masks {
            let logits = output
                .clone()
                .slice([i..i + 1, mask..(mask + 1), 0..d_model])
                .squeeze_dim::<2>(0)
                .squeeze_dim(0);
            // Find the top k tokens with the highest probabilities
            let top_k = top_k(5, logits);
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

fn find_masks<T: Element>(tokens: &[T], mask_token_id: TokenType) -> Vec<usize> {
    let mut masks = Vec::new();
    for (i, token) in tokens.iter().enumerate() {
        if token.to_usize() == mask_token_id {
            masks.push(i);
        }
    }
    masks
}

fn top_k(k: usize, logits: Tensor<1>) -> Vec<(usize, f32)> {
    let (pre_soft_probs, indices) = logits.sort_with_indices(0);
    let (probabilities, indices) = (
        softmax(pre_soft_probs, 0)
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>(),
        indices
            .into_data()
            .iter::<i64>()
            .map(|i| i as usize)
            .collect::<Vec<_>>(),
    );
    probabilities
        .iter()
        .enumerate()
        .rev()
        .take(k)
        .map(|(i, &p)| (indices[i], p))
        .collect()
}
