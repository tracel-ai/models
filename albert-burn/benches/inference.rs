//! Benchmark ALBERT BaseV2 inference (forward pass) across backends.
//!
//! Run:
//! ```bash
//! cargo bench --bench inference -p albert-burn
//! ```

use albert_burn::{AlbertMaskedLM, AlbertVariant, tokenize_batch};
use burn::prelude::*;
use divan::{AllocProfiler, Bencher};
use std::cell::RefCell;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

// Backend type aliases
type NdArrayBackend = burn::backend::ndarray::NdArray<f32>;

#[cfg(feature = "wgpu")]
type WgpuBackend = burn::backend::wgpu::Wgpu;

#[cfg(feature = "cuda")]
type CudaBackend = burn::backend::cuda::Cuda<f32, i32>;

#[cfg(feature = "tch-cpu")]
type TchBackend = burn::backend::libtorch::LibTorch<f32>;

// Shared model + inputs, initialized once in main()
thread_local! {
    static NDARRAY_STATE: RefCell<Option<BenchState<NdArrayBackend>>> = const { RefCell::new(None) };

    #[cfg(feature = "wgpu")]
    static WGPU_STATE: RefCell<Option<BenchState<WgpuBackend>>> = const { RefCell::new(None) };

    #[cfg(feature = "cuda")]
    static CUDA_STATE: RefCell<Option<BenchState<CudaBackend>>> = const { RefCell::new(None) };

    #[cfg(feature = "tch-cpu")]
    static TCH_STATE: RefCell<Option<BenchState<TchBackend>>> = const { RefCell::new(None) };
}

struct BenchState<B: Backend> {
    model: AlbertMaskedLM<B>,
    input_ids: Tensor<B, 2, Int>,
    attention_mask: Tensor<B, 2>,
}

/// Run a forward pass to warm up the backend (triggers shader compilation, etc.).
fn warmup<B: Backend>(state: &BenchState<B>) {
    let _ = state.model.forward(
        state.input_ids.clone(),
        state.attention_mask.clone(),
        None,
    );
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

    // Initialize all enabled backends
    NDARRAY_STATE.with(|s| *s.borrow_mut() = Some(init_state(&Default::default())));

    #[cfg(feature = "wgpu")]
    WGPU_STATE.with(|s| {
        let state = init_state(&Default::default());
        warmup(&state);
        *s.borrow_mut() = Some(state);
    });

    #[cfg(feature = "cuda")]
    CUDA_STATE.with(|s| {
        let state = init_state(&Default::default());
        warmup(&state);
        *s.borrow_mut() = Some(state);
    });

    #[cfg(feature = "tch-cpu")]
    TCH_STATE.with(|s| *s.borrow_mut() = Some(init_state(&Default::default())));

    println!("Model loaded. Running benchmarks...\n");

    divan::main();
}

macro_rules! bench_backend {
    ($backend:ty, $state:ident, $mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name, sample_count = 10)]
        mod $mod_name {
            use super::*;

            #[divan::bench]
            fn forward_pass(bencher: Bencher) {
                bencher.bench(|| {
                    $state.with(|s| {
                        let s = s.borrow();
                        let s = s.as_ref().expect("state not initialized");
                        // TODO: Implement inference call
                        // Consider: should we include argmax/top-k in the measurement,
                        // or only the raw forward pass? Pure forward is more comparable
                        // across backends since post-processing is CPU-bound.
                        s.model.forward(
                            s.input_ids.clone(),
                            s.attention_mask.clone(),
                            None,
                        )
                    })
                });
            }
        }
    };
}

bench_backend!(NdArrayBackend, NDARRAY_STATE, ndarray_backend, "NdArray (CPU)");

#[cfg(feature = "wgpu")]
bench_backend!(WgpuBackend, WGPU_STATE, wgpu_backend, "WGPU (GPU)");

#[cfg(feature = "cuda")]
bench_backend!(CudaBackend, CUDA_STATE, cuda_backend, "CUDA (NVIDIA GPU)");

#[cfg(feature = "tch-cpu")]
bench_backend!(TchBackend, TCH_STATE, tch_backend, "LibTorch (CPU)");
