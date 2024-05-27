use crate::{
    data::Tokenizer,
    data::{BertInferenceBatch, BertTokenizer},
    model::BertMaskedLM,
    model::BertModelConfig,
};
use burn::tensor::{activation::softmax, backend::Backend, Data, Element, Tensor};

type TokenType = usize;
const MASK_TOKEN_ID: TokenType = 50264;

#[derive(Debug, Clone)]
pub struct FillMaskResult {
    pub mask_idx: usize,
    pub top_k: Vec<(f32, String)>,
}

pub fn fill_mask<B: Backend>(
    model: &BertMaskedLM<B>,
    model_config: &BertModelConfig,
    tokenizer: &BertTokenizer,
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

fn data_to_vec_usize<T: Element>(data: &Data<T, 1>) -> Vec<usize> {
    data.value.iter().map(|x| x.to_usize().unwrap()).collect()
}

fn top_k<B: Backend>(k: usize, logits: Tensor<B, 1>) -> Vec<(usize, f32)> {
    let (pre_soft_probs, indices) = logits.sort_with_indices(0);
    let (probabilities, indices) = (
        data_to_vec_f32(&softmax(pre_soft_probs, 0).to_data()),
        data_to_vec_usize(&indices.to_data()),
    );
    probabilities
        .iter()
        .enumerate()
        .rev()
        .take(k)
        .map(|(i, &p)| (indices[i], p))
        .collect()
}
