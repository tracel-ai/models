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
use burn::tensor::Tensor;
use burn::tensor::linalg::cosine_similarity;
use minilm_burn::{MiniLmModel, MiniLmVariant, mean_pooling, normalize_l2, tokenize_batch};

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

/// Load default model and encode test sentences, returning normalized embeddings.
fn encode_test_sentences() -> (Tensor<B, 2>, usize) {
    let device = Default::default();
    let (model, tokenizer) = MiniLmModel::<B>::pretrained(&device, Default::default(), None)
        .expect("Failed to load model");

    let (input_ids, attention_mask) = tokenize_batch::<B>(&tokenizer, &SENTENCES, &device);
    let output = model.forward(input_ids, attention_mask.clone(), None);
    let embeddings = mean_pooling(output.hidden_states, attention_mask);
    let embeddings = normalize_l2(embeddings);

    let [_, hidden_size] = embeddings.dims();
    (embeddings, hidden_size)
}

#[test]
#[ignore] // Requires model download; run with: cargo test --features ndarray -- --ignored
fn test_embeddings_match_python() {
    let (embeddings, hidden_size) = encode_test_sentences();

    let emb_data = embeddings.to_data();
    let emb_slice = emb_data.as_slice::<f32>().unwrap();

    let rust_0: Vec<f32> = emb_slice[0..10].to_vec();
    let rust_1: Vec<f32> = emb_slice[hidden_size..hidden_size + 10].to_vec();
    let rust_2: Vec<f32> = emb_slice[hidden_size * 2..hidden_size * 2 + 10].to_vec();

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
    let (embeddings, hidden_size) = encode_test_sentences();

    let emb0: Tensor<B, 1> = embeddings.clone().slice([0..1, 0..hidden_size]).squeeze();
    let emb1: Tensor<B, 1> = embeddings.clone().slice([1..2, 0..hidden_size]).squeeze();
    let emb2: Tensor<B, 1> = embeddings.clone().slice([2..3, 0..hidden_size]).squeeze();

    let sim_01: f32 = cosine_similarity(emb0.clone(), emb1.clone(), 0, None).into_scalar();
    let sim_02: f32 = cosine_similarity(emb0, emb2.clone(), 0, None).into_scalar();
    let sim_12: f32 = cosine_similarity(emb1, emb2, 0, None).into_scalar();

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

#[test]
#[ignore] // Requires model download; run with: cargo test --features ndarray -- --ignored
fn test_l6_variant_loads_and_runs() {
    let device = Default::default();

    let (model, tokenizer) = MiniLmModel::<B>::pretrained(&device, MiniLmVariant::L6, None)
        .expect("Failed to load L6 model");

    assert_eq!(
        model.encoder.layers.len(),
        6,
        "L6 should have 6 encoder layers"
    );

    let (input_ids, attention_mask) = tokenize_batch::<B>(&tokenizer, &SENTENCES, &device);

    let output = model.forward(input_ids, attention_mask.clone(), None);
    let embeddings = mean_pooling(output.hidden_states, attention_mask);
    let embeddings = normalize_l2(embeddings);

    let [b, hidden] = embeddings.dims();
    assert_eq!(b, 3, "Batch size should be 3");
    assert_eq!(hidden, 384, "Hidden size should be 384");

    let emb0: Tensor<B, 1> = embeddings.clone().slice([0..1, 0..hidden]).squeeze();
    let norm: f32 = emb0.clone().powf_scalar(2.0).sum().sqrt().into_scalar();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "L2 normalized embedding should have norm ≈ 1, got {}",
        norm
    );
}
