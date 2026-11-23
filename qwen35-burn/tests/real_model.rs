//! The real Qwen3.5-0.8B checkpoint against Crane's logits on the same weights. Needs the
//! weights and the reference logits, so it is ignored by default:
//!
//! ```sh
//! QWEN35_DIR=path/to/model cargo test -p qwen35-burn --release --test real_model -- --ignored
//! ```
//!
//! `QWEN35_DIR` holds `config.json`, `model.safetensors` and `crane_ref/{ids,logits}_<len>.f32`
//! (little-endian f32, written by running Crane on the same ids).

use burn::tensor::{Device, Int, Tensor, TensorData};
use qwen35_burn::loader;
use std::path::PathBuf;

fn read_f32(path: PathBuf) -> Vec<f32> {
    std::fs::read(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

#[test]
#[ignore = "needs the Qwen3.5 checkpoint and Crane reference logits"]
fn logits_match_crane() {
    let dir = PathBuf::from(std::env::var("QWEN35_DIR").expect("set QWEN35_DIR"));
    let dev = Device::default();
    let model = loader::from_dir(&dir, &dev).unwrap();
    for len in [7usize, 64, 300] {
        let ids: Vec<i64> = read_f32(dir.join(format!("crane_ref/ids_{len}.f32")))
            .iter()
            .map(|&x| x as i64)
            .collect();
        let want = read_f32(dir.join(format!("crane_ref/logits_{len}.f32")));
        let ids = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, len]), &dev);
        let hidden = model.forward(ids).narrow(1, len - 1, 1);
        let got: Vec<f32> = model
            .logits(hidden)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let dot: f64 = got
            .iter()
            .zip(&want)
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (norm(&got) * norm(&want));
        let top = |v: &[f32]| (0..v.len()).max_by(|&i, &j| v[i].total_cmp(&v[j])).unwrap();
        println!(
            "{len:>3} tokens: cosine {cos:.7}, top token {} / {}",
            top(&got),
            top(&want)
        );
        assert!(cos > 0.999, "{len} tokens: cosine {cos}");
        assert_eq!(top(&got), top(&want), "{len} tokens: different top token");
    }
}
