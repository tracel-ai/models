//! Integration tests comparing Rust outputs with Python HuggingFace reference.
//!
//! Reference values generated with:
//! ```
//! uv run --with transformers --with torch --with sentencepiece --with protobuf scripts/generate_reference.py
//! ```
//!
//! Run with: `cargo test --features "pretrained,ndarray" -- --ignored`

#![cfg(feature = "ndarray")]

use burn::backend::ndarray::NdArray;
use burn::prelude::ElementConversion;
use burn::tensor::Tensor;

use albert_burn::{AlbertMaskedLM, tokenize_batch};

type B = NdArray<f32>;

/// Relative tolerance for comparing logit values.
///
/// f32 matmul precision differs between ndarray (Rust) and PyTorch (MKL/Accelerate).
/// After 12 shared transformer layers, these accumulate to ~1e-4 relative error.
const REL_TOL: f32 = 5e-4;

fn rel_diff(actual: f32, expected: f32) -> f32 {
    let abs_diff = (actual - expected).abs();
    let scale = expected.abs().max(1.0);
    abs_diff / scale
}

fn assert_close(actual: f32, expected: f32, label: &str) {
    let rd = rel_diff(actual, expected);
    assert!(
        rd < REL_TOL,
        "{}: rust={}, python={}, rel_diff={:.2e} (limit={:.2e})",
        label,
        actual,
        expected,
        rd,
        REL_TOL
    );
}

fn load_model() -> (AlbertMaskedLM<B>, tokenizers::Tokenizer) {
    let device = Default::default();
    AlbertMaskedLM::<B>::pretrained(&device, Default::default(), None).expect("Failed to load model")
}

fn predict_at_mask(
    model: &AlbertMaskedLM<B>,
    tokenizer: &tokenizers::Tokenizer,
    sentence: &str,
) -> (Tensor<B, 1>, Tensor<B, 2>) {
    let device = Default::default();
    let (input_ids, attention_mask) = tokenize_batch::<B>(tokenizer, &[sentence], &device);

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

    let [_, seq_len, vocab_size] = logits.dims();
    let mask_logits: Tensor<B, 1> = logits
        .clone()
        .slice([0..1, mask_pos..mask_pos + 1, 0..vocab_size])
        .reshape([vocab_size]);

    // All positions: [seq_len, vocab_size]
    let all_logits: Tensor<B, 2> = logits.reshape([seq_len, vocab_size]);

    (mask_logits, all_logits)
}

fn check_first_10(logits: &[f32], expected: &[f32; 10], label: &str) {
    for i in 0..10 {
        assert_close(logits[i], expected[i], &format!("{label} logit[{i}]"));
    }
}

fn check_top5(
    logits: &Tensor<B, 1>,
    expected_ids: &[i64; 5],
    expected_scores: &[f32; 5],
    label: &str,
) {
    let top_k = logits.clone().sort_descending_with_indices(0);
    let indices_data = top_k.1.to_data();
    let values_data = top_k.0.to_data();
    let indices: &[i64] = indices_data.as_slice().unwrap();
    let scores: &[f32] = values_data.as_slice().unwrap();

    for i in 0..5 {
        assert_eq!(
            indices[i],
            expected_ids[i],
            "{label} top-{}: rust id={}, python id={}",
            i + 1,
            indices[i],
            expected_ids[i]
        );
        assert_close(
            scores[i],
            expected_scores[i],
            &format!("{label} top-{} logit", i + 1),
        );
    }
}

fn check_stats(
    logits: &[f32],
    expected_min: f32,
    expected_max: f32,
    expected_mean: f32,
    label: &str,
) {
    let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = logits.iter().sum::<f32>() / logits.len() as f32;

    assert_close(min, expected_min, &format!("{label} min"));
    assert_close(max, expected_max, &format!("{label} max"));
    assert_close(mean, expected_mean, &format!("{label} mean"));
}

fn check_seq_norms(all_logits: &Tensor<B, 2>, expected_norms: &[f32], label: &str) {
    let [seq_len, _vocab] = all_logits.dims();
    assert_eq!(seq_len, expected_norms.len(), "{label} seq_len mismatch");

    for pos in 0..seq_len {
        let row = all_logits.clone().slice([pos..pos + 1]);
        let norm_sq: f32 = row.clone().mul(row).sum().into_scalar().elem();
        let norm = norm_sq.sqrt();
        assert_close(norm, expected_norms[pos], &format!("{label} norm[{pos}]"));
    }
}

// === S1: "The capital of France is [MASK]." ===
const S1: &str = "The capital of France is [MASK].";
const S1_FIRST_10_LOGITS: [f32; 10] = [
    -1.8435553312,
    1.6792540550,
    -4.7834906578,
    7.5429682732,
    -1.3557765484,
    1.5608149767,
    2.3607411385,
    -3.4745693207,
    2.4210648537,
    1.4211364985,
];
const S1_TOP5_IDS: [i64; 5] = [29847, 20220, 1162, 29872, 16586];
const S1_TOP5_LOGITS: [f32; 5] = [
    16.3436222076,
    16.1717433929,
    15.8925437927,
    15.6549472809,
    15.6170835495,
];
const S1_LOGIT_MIN: f32 = -14.2863645554;
const S1_LOGIT_MAX: f32 = 16.3436222076;
const S1_LOGIT_MEAN: f32 = -0.3709427118;
const S1_SEQ_NORMS: [f32; 10] = [
    909.2635498047,
    977.9346923828,
    674.7073974609,
    838.0267944336,
    620.0793457031,
    810.7951049805,
    592.5648193359,
    458.9917907715,
    559.2899780273,
    415.0333251953,
];

// === S2: "The [MASK] sat on the mat." ===
const S2: &str = "The [MASK] sat on the mat.";
const S2_FIRST_10_LOGITS: [f32; 10] = [
    -9.9090566635,
    -5.0451688766,
    -7.2376966476,
    -6.5224504471,
    -9.5459375381,
    -6.8127403259,
    -5.9883685112,
    -5.1755027771,
    -8.0674257278,
    -3.1775133610,
];
const S2_TOP5_IDS: [i64; 5] = [1244, 1258, 695, 1626, 2717];
const S2_TOP5_LOGITS: [f32; 5] = [
    9.3129110336,
    9.0611057281,
    9.0517425537,
    8.8680877686,
    8.8596239090,
];
const S2_LOGIT_MIN: f32 = -13.2401380539;
const S2_LOGIT_MAX: f32 = 9.3129110336;
const S2_LOGIT_MEAN: f32 = -0.5874535441;
const S2_SEQ_NORMS: [f32; 9] = [
    988.3486328125,
    753.5031738281,
    573.7915649414,
    739.5702514648,
    1321.8070068359,
    824.2498168945,
    612.6429443359,
    454.0575256348,
    432.3054809570,
];

// === S3: "She studied [MASK] at the university." ===
const S3: &str = "She studied [MASK] at the university.";
const S3_FIRST_10_LOGITS: [f32; 10] = [
    -0.4670740664,
    -5.4004788399,
    -2.7991139889,
    0.3256093860,
    -0.2207909226,
    -2.9407899380,
    2.3930442333,
    -0.3755640090,
    -4.9276309013,
    -4.6748538017,
];
const S3_TOP5_IDS: [i64; 5] = [25534, 4264, 29368, 6182, 28988];
const S3_TOP5_LOGITS: [f32; 5] = [
    15.3436059952,
    13.8445081711,
    12.9867172241,
    12.9093313217,
    12.7724552155,
];
const S3_LOGIT_MIN: f32 = -15.2323818207;
const S3_LOGIT_MAX: f32 = 15.3436059952;
const S3_LOGIT_MEAN: f32 = -0.0736505240;
const S3_SEQ_NORMS: [f32; 9] = [
    842.8552856445,
    805.7095947266,
    862.0442504883,
    650.7914428711,
    1051.3500976562,
    580.3734130859,
    713.8647460938,
    427.9932861328,
    418.7301940918,
];

#[test]
#[ignore]
fn test_s1_logits() {
    let (model, tokenizer) = load_model();
    let (mask_logits, all_logits) = predict_at_mask(&model, &tokenizer, S1);
    let data = mask_logits.to_data();
    let logits: &[f32] = data.as_slice().unwrap();

    check_first_10(logits, &S1_FIRST_10_LOGITS, "S1");
    check_top5(&mask_logits, &S1_TOP5_IDS, &S1_TOP5_LOGITS, "S1");
    check_stats(logits, S1_LOGIT_MIN, S1_LOGIT_MAX, S1_LOGIT_MEAN, "S1");
    check_seq_norms(&all_logits, &S1_SEQ_NORMS, "S1");
}

#[test]
#[ignore]
fn test_s2_logits() {
    let (model, tokenizer) = load_model();
    let (mask_logits, all_logits) = predict_at_mask(&model, &tokenizer, S2);
    let data = mask_logits.to_data();
    let logits: &[f32] = data.as_slice().unwrap();

    check_first_10(logits, &S2_FIRST_10_LOGITS, "S2");
    check_top5(&mask_logits, &S2_TOP5_IDS, &S2_TOP5_LOGITS, "S2");
    check_stats(logits, S2_LOGIT_MIN, S2_LOGIT_MAX, S2_LOGIT_MEAN, "S2");
    check_seq_norms(&all_logits, &S2_SEQ_NORMS, "S2");
}

#[test]
#[ignore]
fn test_s3_logits() {
    let (model, tokenizer) = load_model();
    let (mask_logits, all_logits) = predict_at_mask(&model, &tokenizer, S3);
    let data = mask_logits.to_data();
    let logits: &[f32] = data.as_slice().unwrap();

    check_first_10(logits, &S3_FIRST_10_LOGITS, "S3");
    check_top5(&mask_logits, &S3_TOP5_IDS, &S3_TOP5_LOGITS, "S3");
    check_stats(logits, S3_LOGIT_MIN, S3_LOGIT_MAX, S3_LOGIT_MEAN, "S3");
    check_seq_norms(&all_logits, &S3_SEQ_NORMS, "S3");
}
