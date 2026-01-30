//! Integration tests comparing Rust outputs with Python HuggingFace reference.
//!
//! Reference values generated with:
//! ```
//! uv run --with transformers --with torch --with safetensors --with sentencepiece scripts/generate_reference.py
//! ```
//!
//! Run with: `cargo test --features "pretrained,ndarray" -- --ignored`

#![cfg(feature = "ndarray")]

use burn::backend::ndarray::NdArray;
use burn::tensor::Tensor;

use albert_burn::{AlbertMaskedLM, tokenize_batch};

type B = NdArray<f32>;

const SENTENCE: &str = "The capital of France is [MASK].";

// Reference: first 10 logits at [MASK] position from Python (HF transformers)
const EXPECTED_MASK_LOGITS_FIRST_10: [f32; 10] = [
    -1.843555, 1.679254, -4.783491, 7.542968, -1.355777, 1.560815, 2.360741, -3.474569, 2.421065,
    1.421136,
];

// Top-5 predicted token IDs
const EXPECTED_TOP5_IDS: [i64; 5] = [29847, 20220, 1162, 29872, 16586];

// Top-5 logit values
const EXPECTED_TOP5_LOGITS: [f32; 5] = [16.343622, 16.171743, 15.892544, 15.654947, 15.617084];

fn load_and_predict() -> (Tensor<B, 1>, usize) {
    let device = Default::default();
    let (model, tokenizer) =
        AlbertMaskedLM::<B>::pretrained(&device, None).expect("Failed to load model");

    let (input_ids, attention_mask) = tokenize_batch::<B>(&tokenizer, &[SENTENCE], &device);

    // Find [MASK] position
    let input_ids_data = input_ids.to_data();
    let ids: &[i64] = input_ids_data.as_slice().unwrap();
    let mask_token_id = tokenizer
        .token_to_id("[MASK]")
        .expect("[MASK] token not found");
    let mask_pos = ids
        .iter()
        .position(|&id| id == mask_token_id as i64)
        .expect("[MASK] position not found");

    let logits = model.forward(input_ids, attention_mask, None);

    let [_, _, vocab_size] = logits.dims();
    let mask_logits: Tensor<B, 1> = logits
        .slice([0..1, mask_pos..mask_pos + 1, 0..vocab_size])
        .reshape([vocab_size]);

    (mask_logits, vocab_size)
}

#[test]
#[ignore] // Requires model download; run with: cargo test --features "pretrained,ndarray" -- --ignored
fn test_logits_match_python() {
    let (mask_logits, _) = load_and_predict();

    let data = mask_logits.to_data();
    let logits: &[f32] = data.as_slice().unwrap();

    let tolerance = 0.05;
    for i in 0..10 {
        assert!(
            (logits[i] - EXPECTED_MASK_LOGITS_FIRST_10[i]).abs() < tolerance,
            "Logit {}: rust={}, python={}",
            i,
            logits[i],
            EXPECTED_MASK_LOGITS_FIRST_10[i]
        );
    }
}

#[test]
#[ignore] // Requires model download; run with: cargo test --features "pretrained,ndarray" -- --ignored
fn test_top5_predictions_match_python() {
    let (mask_logits, _) = load_and_predict();

    let top_k = mask_logits.sort_descending_with_indices(0);
    let top_indices_data = top_k.1.to_data();
    let top_values_data = top_k.0.to_data();
    let indices: &[i64] = top_indices_data.as_slice().unwrap();
    let scores: &[f32] = top_values_data.as_slice().unwrap();

    // Check top-5 token IDs match
    for i in 0..5 {
        assert_eq!(
            indices[i],
            EXPECTED_TOP5_IDS[i],
            "Top-{} token ID: rust={}, python={}",
            i + 1,
            indices[i],
            EXPECTED_TOP5_IDS[i]
        );
    }

    // Check top-5 logit values are close
    let tolerance = 0.05;
    for i in 0..5 {
        assert!(
            (scores[i] - EXPECTED_TOP5_LOGITS[i]).abs() < tolerance,
            "Top-{} logit: rust={}, python={}",
            i + 1,
            scores[i],
            EXPECTED_TOP5_LOGITS[i]
        );
    }
}
