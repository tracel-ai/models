//! Benchmark ALBERT BaseV2 inference (forward pass).
//!
//! Run:
//! ```bash
//! cargo bench --bench inference -p albert-burn
//! ```

use albert_burn::{AlbertMaskedLM, AlbertVariant, tokenize_batch};
use burn::prelude::*;
use burn_flex::Flex;
use divan::{AllocProfiler, Bencher};
use std::cell::RefCell;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

thread_local! {
    static STATE: RefCell<Option<BenchState<Flex>>> = const { RefCell::new(None) };
}

struct BenchState<B: Backend> {
    model: AlbertMaskedLM<B>,
    input_ids: Tensor<B, 2, Int>,
    attention_mask: Tensor<B, 2>,
}

fn init_state<B: Backend>(device: &B::Device) -> BenchState<B> {
    let (model, tokenizer) =
        AlbertMaskedLM::<B>::pretrained(device, AlbertVariant::BaseV2, None)
            .expect("Failed to load pretrained model");

    let sentence = "The capital of France is [MASK].";
    let (input_ids, attention_mask) = tokenize_batch::<B>(&tokenizer, &[sentence], device);

    BenchState {
        model,
        input_ids,
        attention_mask,
    }
}

fn main() {
    println!("Loading ALBERT BaseV2 for benchmarking...");
    STATE.with(|s| *s.borrow_mut() = Some(init_state(&Default::default())));
    println!("Model loaded. Running benchmarks...\n");

    divan::main();
}

#[divan::bench_group(name = "Flex (CPU)", sample_count = 10)]
mod flex_backend {
    use super::*;

    #[divan::bench]
    fn forward_pass(bencher: Bencher) {
        bencher.bench(|| {
            STATE.with(|s| {
                let s = s.borrow();
                let s = s.as_ref().expect("state not initialized");
                s.model
                    .forward(s.input_ids.clone(), s.attention_mask.clone(), None)
            })
        });
    }
}
