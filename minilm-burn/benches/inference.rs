//! Benchmarks for MiniLM inference across backends.
//!
//! Run for each backend:
//!   cargo bench --features ndarray
//!   cargo bench --features wgpu
//!   cargo bench --features tch-cpu
//!
//! Results are saved to target/criterion/ for comparison.

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use minilm_burn::{MiniLmModel, mean_pooling, normalize_l2};

// Backend selection via features
#[cfg(feature = "ndarray")]
mod backend {
    pub type B = burn::backend::ndarray::NdArray<f32>;
    pub const NAME: &str = "ndarray";
}

#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
mod backend {
    pub type B = burn::backend::wgpu::Wgpu;
    pub const NAME: &str = "wgpu";
}

#[cfg(all(feature = "tch-cpu", not(any(feature = "ndarray", feature = "wgpu"))))]
mod backend {
    pub type B = burn::backend::libtorch::LibTorch;
    pub const NAME: &str = "tch-cpu";
}

#[cfg(all(
    feature = "cuda",
    not(any(feature = "ndarray", feature = "wgpu", feature = "tch-cpu"))
))]
mod backend {
    pub type B = burn::backend::cuda::Cuda;
    pub const NAME: &str = "cuda";
}

use backend::{B, NAME};

fn prepare_inputs<BE: Backend>(
    tokenizer: &tokenizers::Tokenizer,
    sentences: &[&str],
    device: &BE::Device,
) -> (Tensor<BE, 2, Int>, Tensor<BE, 2>) {
    let encodings = tokenizer
        .encode_batch(sentences.to_vec(), true)
        .expect("Failed to tokenize");

    let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap();
    let batch_size = sentences.len();

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

    let input_ids: Tensor<BE, 2, Int> =
        Tensor::<BE, 1, Int>::from_data(input_ids_data.as_slice(), device)
            .reshape([batch_size, max_len]);
    let attention_mask: Tensor<BE, 2> =
        Tensor::<BE, 1>::from_data(attention_mask_data.as_slice(), device)
            .reshape([batch_size, max_len]);

    (input_ids, attention_mask)
}

fn bench_forward(c: &mut Criterion) {
    let device = Default::default();
    let (model, tokenizer) =
        MiniLmModel::<B>::pretrained(&device, Default::default(), None).expect("Failed to load model");

    let sentences = vec!["The quick brown fox jumps over the lazy dog"];
    let (input_ids, attention_mask) = prepare_inputs::<B>(&tokenizer, &sentences, &device);

    c.bench_function(&format!("{}/forward_single", NAME), |b| {
        b.iter(|| {
            let output = model.forward(
                black_box(input_ids.clone()),
                black_box(attention_mask.clone()),
                None,
            );
            black_box(output)
        })
    });
}

fn bench_forward_batch(c: &mut Criterion) {
    let device = Default::default();
    let (model, tokenizer) =
        MiniLmModel::<B>::pretrained(&device, Default::default(), None).expect("Failed to load model");

    let mut group = c.benchmark_group(format!("{}/forward_batch", NAME));
    group.sample_size(20);

    for batch_size in [1, 4, 8, 16].iter() {
        let sentences: Vec<&str> = (0..*batch_size)
            .map(|_| "The quick brown fox jumps over the lazy dog")
            .collect();
        let (input_ids, attention_mask) = prepare_inputs::<B>(&tokenizer, &sentences, &device);

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let output = model.forward(
                        black_box(input_ids.clone()),
                        black_box(attention_mask.clone()),
                        None,
                    );
                    black_box(output)
                })
            },
        );
    }
    group.finish();
}

fn bench_full_pipeline(c: &mut Criterion) {
    let device = Default::default();
    let (model, tokenizer) =
        MiniLmModel::<B>::pretrained(&device, Default::default(), None).expect("Failed to load model");

    let sentences = vec!["The quick brown fox jumps over the lazy dog"];

    c.bench_function(&format!("{}/full_pipeline", NAME), |b| {
        b.iter(|| {
            let (input_ids, attention_mask) =
                prepare_inputs::<B>(&tokenizer, black_box(&sentences), &device);
            let output = model.forward(input_ids, attention_mask.clone(), None);
            let embeddings = mean_pooling(output.hidden_states, attention_mask);
            let embeddings = normalize_l2(embeddings);
            black_box(embeddings)
        })
    });
}

fn bench_pooling(c: &mut Criterion) {
    let device: <B as Backend>::Device = Default::default();

    let hidden_states: Tensor<B, 3> = Tensor::zeros([1, 128, 384], &device);
    let attention_mask: Tensor<B, 2> = Tensor::ones([1, 128], &device);

    c.bench_function(&format!("{}/mean_pooling", NAME), |b| {
        b.iter(|| {
            let embeddings = mean_pooling(
                black_box(hidden_states.clone()),
                black_box(attention_mask.clone()),
            );
            black_box(embeddings)
        })
    });

    c.bench_function(&format!("{}/normalize_l2", NAME), |b| {
        let embeddings: Tensor<B, 2> = Tensor::zeros([1, 384], &device);
        b.iter(|| {
            let normalized = normalize_l2(black_box(embeddings.clone()));
            black_box(normalized)
        })
    });
}

criterion_group!(
    benches,
    bench_forward,
    bench_forward_batch,
    bench_full_pipeline,
    bench_pooling,
);
criterion_main!(benches);
