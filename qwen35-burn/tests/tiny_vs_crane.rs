//! The text model against Crane's `Qwen3_5TextModel` on a tiny random model: 4 layers (3 Gated
//! DeltaNet + 1 gated attention), 4 query heads over 2 KV heads, partial RoPE, tied embedding.
//! `fixtures/tiny/model.safetensors` holds its weights under HF names, so this also covers the
//! `burn-store` loading path. `reference.safetensors` holds Crane's hidden state after every
//! layer and the last token's logits, for 1, 5 and 150 tokens (150 crosses the 64-token chunk
//! edge twice).

use burn::tensor::{Device, Int, Tensor, TensorData};
use qwen35_burn::{Qwen35TextModel, loader};
use std::collections::HashMap;
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny")
}

/// The F32 tensors of a safetensors file, by name.
fn read_f32(path: PathBuf) -> HashMap<String, Vec<f32>> {
    let bytes = std::fs::read(path).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: HashMap<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let data = &bytes[8 + n..];
    header
        .into_iter()
        .map(|(name, e)| {
            let o = &e["data_offsets"];
            let (a, b) = (
                o[0].as_u64().unwrap() as usize,
                o[1].as_u64().unwrap() as usize,
            );
            let v = data[a..b]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            (name, v)
        })
        .collect()
}

fn model(dev: &Device) -> Qwen35TextModel {
    loader::from_dir(dir(), dev).unwrap()
}

fn ids(r: &HashMap<String, Vec<f32>>, len: usize, dev: &Device) -> Tensor<2, Int> {
    let ids: Vec<i64> = r[&format!("ids_{len}")].iter().map(|&x| x as i64).collect();
    Tensor::from_data(TensorData::new(ids, [1, len]), dev)
}

/// Max |got - want| relative to max |want| (at least 1).
fn rel_err(got: Tensor<3>, want: &[f32]) -> f32 {
    let got: Vec<f32> = got.into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(got.len(), want.len(), "length");
    let scale = want.iter().map(|x| x.abs()).fold(1.0, f32::max);
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
        / scale
}

#[test]
fn every_layer_and_the_logits_match_crane() {
    let dev = Device::default();
    let r = read_f32(dir().join("reference.safetensors"));
    let model = model(&dev);
    for len in [1, 5, 150] {
        for (l, h) in model
            .layer_outputs(ids(&r, len, &dev))
            .into_iter()
            .enumerate()
        {
            let err = rel_err(h, &r[&format!("h_{len}_{l}")]);
            assert!(err < 1e-4, "{len} tokens, layer {l}: {err:.2e}");
        }
        let hidden = model.forward(ids(&r, len, &dev)).narrow(1, len - 1, 1);
        let err = rel_err(model.logits(hidden), &r[&format!("logits_{len}")]);
        assert!(err < 1e-4, "{len} tokens, logits: {err:.2e}");
    }
}

#[test]
fn the_model_trains() {
    let dev = Device::default().autodiff();
    let r = read_f32(dir().join("reference.safetensors"));
    let model = model(&dev);
    let hidden = model.forward(ids(&r, 5, &dev));
    let grads = model.logits(hidden).powi_scalar(2).mean().backward();
    let attn = model.layers[3]
        .self_attn
        .as_ref()
        .expect("layer 3 is attention");
    let gdn = model.layers[0]
        .linear_attn
        .as_ref()
        .expect("layer 0 is Gated DeltaNet");
    for (n, g) in [
        ("q_proj", attn.q_proj.weight.val().grad(&grads)),
        ("in_proj_qkv", gdn.in_proj_qkv.weight.val().grad(&grads)),
        ("a_log", gdn.a_log.val().grad(&grads).map(|g| g.unsqueeze())),
        ("embed_tokens", model.embed_tokens.weight.val().grad(&grads)),
    ] {
        let g = g
            .expect("gradient")
            .powi_scalar(2)
            .sum()
            .into_scalar::<f32>();
        assert!(g.is_finite() && g > 0.0, "{n}: gradient norm {g}");
    }
}
