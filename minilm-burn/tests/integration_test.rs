//! Integration tests comparing Rust outputs with Python sentence-transformers reference.
//!
//! Reference values generated with:
//! ```
//! uv run --with sentence-transformers scripts/generate_reference.py
//! ```
//!
//! Run with: `cargo test --features ndarray -- --ignored`

#![cfg(feature = "ndarray")]

use burn::backend::ndarray::NdArray;
use burn::tensor::linalg::cosine_similarity;
use burn::tensor::{Int, Tensor};
use minilm_burn::{MiniLmModel, mean_pooling, normalize_l2};

type B = NdArray<f32>;

// Test sentences (must match Python script)
const SENTENCES: [&str; 3] = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "Rust is a systems programming language",
];

// Reference: first 10 dimensions of each embedding from Python
const EXPECTED_0_FIRST_10: [f32; 10] = [
    -0.07597314,
    -0.00526199,
    0.01145625,
    -0.06798457,
    -0.00306876,
    -0.18362328,
    0.06599247,
    0.02946932,
    -0.05323604,
    0.08215267,
];

const EXPECTED_1_FIRST_10: [f32; 10] = [
    0.00727466,
    0.03941640,
    -0.03635290,
    0.03405705,
    0.07380778,
    -0.00372924,
    0.02434149,
    -0.01693923,
    -0.01971077,
    -0.04784514,
];

const EXPECTED_2_FIRST_10: [f32; 10] = [
    -0.11221215,
    0.01692677,
    -0.03655620,
    -0.01834833,
    0.00683290,
    -0.06806342,
    -0.01151933,
    -0.00640351,
    0.01122445,
    0.02174890,
];

// Reference cosine similarities from Python
const SIM_0_1: f32 = 0.04713578;
const SIM_0_2: f32 = 0.21687716;
const SIM_1_2: f32 = 0.14440186;

#[test]
#[ignore] // Requires model download; run with: cargo test --features ndarray -- --ignored
fn test_embeddings_match_python() {
    let device = Default::default();

    // Load model
    let (model, tokenizer) =
        MiniLmModel::<B>::pretrained(&device, None).expect("Failed to load model");

    // Tokenize
    let encodings = tokenizer
        .encode_batch(SENTENCES.to_vec(), true)
        .expect("Failed to tokenize");

    let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap();
    let batch_size = SENTENCES.len();

    // Prepare tensors
    let mut input_ids_data = vec![0i64; batch_size * max_len];
    let mut attention_mask_data = vec![0.0f32; batch_size * max_len];

    for (i, encoding) in encodings.iter().enumerate() {
        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        for (j, &id) in ids.iter().enumerate() {
            input_ids_data[i * max_len + j] = id as i64;
            attention_mask_data[i * max_len + j] = mask[j] as f32;
        }
    }

    let input_ids: Tensor<B, 2, Int> =
        Tensor::<B, 1, Int>::from_data(input_ids_data.as_slice(), &device)
            .reshape([batch_size, max_len]);
    let attention_mask: Tensor<B, 2> =
        Tensor::<B, 1>::from_data(attention_mask_data.as_slice(), &device)
            .reshape([batch_size, max_len]);

    // Run inference
    let output = model.forward(input_ids, attention_mask.clone(), None);
    let embeddings = mean_pooling(output.hidden_states, attention_mask);
    let embeddings = normalize_l2(embeddings); // Match sentence-transformers default

    // Extract first 10 dims for each sentence
    let emb_data = embeddings.to_data();
    let emb_slice = emb_data.as_slice::<f32>().unwrap();

    let rust_0: Vec<f32> = emb_slice[0..10].to_vec();
    let rust_1: Vec<f32> = emb_slice[384..394].to_vec();
    let rust_2: Vec<f32> = emb_slice[768..778].to_vec();

    // Compare embeddings (allow small tolerance for floating point differences)
    let tolerance = 1e-4;

    for i in 0..10 {
        assert!(
            (rust_0[i] - EXPECTED_0_FIRST_10[i]).abs() < tolerance,
            "Sentence 0, dim {}: rust={}, python={}",
            i,
            rust_0[i],
            EXPECTED_0_FIRST_10[i]
        );
        assert!(
            (rust_1[i] - EXPECTED_1_FIRST_10[i]).abs() < tolerance,
            "Sentence 1, dim {}: rust={}, python={}",
            i,
            rust_1[i],
            EXPECTED_1_FIRST_10[i]
        );
        assert!(
            (rust_2[i] - EXPECTED_2_FIRST_10[i]).abs() < tolerance,
            "Sentence 2, dim {}: rust={}, python={}",
            i,
            rust_2[i],
            EXPECTED_2_FIRST_10[i]
        );
    }
}

#[test]
#[ignore] // Requires model download; run with: cargo test --features ndarray -- --ignored
fn test_cosine_similarities_match_python() {
    let device = Default::default();

    let (model, tokenizer) =
        MiniLmModel::<B>::pretrained(&device, None).expect("Failed to load model");

    let encodings = tokenizer
        .encode_batch(SENTENCES.to_vec(), true)
        .expect("Failed to tokenize");

    let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap();
    let batch_size = SENTENCES.len();

    let mut input_ids_data = vec![0i64; batch_size * max_len];
    let mut attention_mask_data = vec![0.0f32; batch_size * max_len];

    for (i, encoding) in encodings.iter().enumerate() {
        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        for (j, &id) in ids.iter().enumerate() {
            input_ids_data[i * max_len + j] = id as i64;
            attention_mask_data[i * max_len + j] = mask[j] as f32;
        }
    }

    let input_ids: Tensor<B, 2, Int> =
        Tensor::<B, 1, Int>::from_data(input_ids_data.as_slice(), &device)
            .reshape([batch_size, max_len]);
    let attention_mask: Tensor<B, 2> =
        Tensor::<B, 1>::from_data(attention_mask_data.as_slice(), &device)
            .reshape([batch_size, max_len]);

    let output = model.forward(input_ids, attention_mask.clone(), None);
    let embeddings = mean_pooling(output.hidden_states, attention_mask);
    let embeddings = normalize_l2(embeddings);

    // Extract individual embeddings
    let emb0: Tensor<B, 1> = embeddings.clone().slice([0..1, 0..384]).squeeze();
    let emb1: Tensor<B, 1> = embeddings.clone().slice([1..2, 0..384]).squeeze();
    let emb2: Tensor<B, 1> = embeddings.clone().slice([2..3, 0..384]).squeeze();

    // Compute cosine similarities
    let sim_01: f32 = cosine_similarity(emb0.clone(), emb1.clone(), 0, None).into_scalar();
    let sim_02: f32 = cosine_similarity(emb0, emb2.clone(), 0, None).into_scalar();
    let sim_12: f32 = cosine_similarity(emb1, emb2, 0, None).into_scalar();

    // Allow slightly larger tolerance for accumulated floating point errors
    let tolerance = 1e-3;

    assert!(
        (sim_01 - SIM_0_1).abs() < tolerance,
        "sim_01: rust={}, python={}",
        sim_01,
        SIM_0_1
    );
    assert!(
        (sim_02 - SIM_0_2).abs() < tolerance,
        "sim_02: rust={}, python={}",
        sim_02,
        SIM_0_2
    );
    assert!(
        (sim_12 - SIM_1_2).abs() < tolerance,
        "sim_12: rust={}, python={}",
        sim_12,
        SIM_1_2
    );
}
