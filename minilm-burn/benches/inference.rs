//! Benchmarks for MiniLM inference.
//!
//! Run with: `cargo bench -p minilm-burn`
//! Results are saved to target/criterion/ for comparison.

use burn::tensor::Tensor;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use minilm_burn::{MiniLmModel, MiniLmVariant, mean_pooling, normalize_l2, tokenize_batch};

type B = burn_flex::Flex;
const NAME: &str = "flex";

fn bench_forward(c: &mut Criterion) {
    let device = Default::default();
    let (model, tokenizer) = MiniLmModel::<B>::pretrained(&device, Default::default(), None)
        .expect("Failed to load model");

    let sentences = vec!["The quick brown fox jumps over the lazy dog"];
    let (input_ids, attention_mask) = tokenize_batch::<B>(&tokenizer, &sentences, &device);

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
    let (model, tokenizer) = MiniLmModel::<B>::pretrained(&device, Default::default(), None)
        .expect("Failed to load model");

    let mut group = c.benchmark_group(format!("{}/forward_batch", NAME));
    group.sample_size(20);

    for batch_size in [1, 4, 8, 16].iter() {
        let sentences: Vec<&str> = (0..*batch_size)
            .map(|_| "The quick brown fox jumps over the lazy dog")
            .collect();
        let (input_ids, attention_mask) = tokenize_batch::<B>(&tokenizer, &sentences, &device);

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
    let (model, tokenizer) = MiniLmModel::<B>::pretrained(&device, Default::default(), None)
        .expect("Failed to load model");

    let sentences = vec!["The quick brown fox jumps over the lazy dog"];

    c.bench_function(&format!("{}/full_pipeline", NAME), |b| {
        b.iter(|| {
            let (input_ids, attention_mask) =
                tokenize_batch::<B>(&tokenizer, black_box(&sentences), &device);
            let output = model.forward(input_ids, attention_mask.clone(), None);
            let embeddings = mean_pooling(output.hidden_states, attention_mask);
            let embeddings = normalize_l2(embeddings);
            black_box(embeddings)
        })
    });
}

fn bench_pooling(c: &mut Criterion) {
    let device = Default::default();

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

fn bench_variants(c: &mut Criterion) {
    let device = Default::default();

    let (model_l6, tokenizer) =
        MiniLmModel::<B>::pretrained(&device, MiniLmVariant::L6, None).expect("Failed to load L6");
    let (model_l12, _) = MiniLmModel::<B>::pretrained(&device, MiniLmVariant::L12, None)
        .expect("Failed to load L12");

    let sentences = vec!["The quick brown fox jumps over the lazy dog"];
    let (input_ids, attention_mask) = tokenize_batch::<B>(&tokenizer, &sentences, &device);

    let mut group = c.benchmark_group(format!("{}/variants", NAME));

    group.bench_function("L6", |b| {
        b.iter(|| {
            let output = model_l6.forward(
                black_box(input_ids.clone()),
                black_box(attention_mask.clone()),
                None,
            );
            black_box(output)
        })
    });

    group.bench_function("L12", |b| {
        b.iter(|| {
            let output = model_l12.forward(
                black_box(input_ids.clone()),
                black_box(attention_mask.clone()),
                None,
            );
            black_box(output)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_forward,
    bench_forward_batch,
    bench_full_pipeline,
    bench_pooling,
    bench_variants,
);
criterion_main!(benches);
