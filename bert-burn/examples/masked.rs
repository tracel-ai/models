use bert_burn::data::{BertInputBatcher, BertTokenizer};
use bert_burn::fill_mask::fill_mask;
use bert_burn::loader::{download_hf_model, load_model_config};
use bert_burn::model::{BertMaskedLM, BertMaskedLMRecord};
use burn::data::dataloader::batcher::Batcher;
use burn::module::Module;
use burn::tensor::backend::Backend;
use std::env;
use std::sync::Arc;

#[cfg(not(feature = "f16"))]
#[allow(dead_code)]
type ElemType = f32;
#[cfg(feature = "f16")]
type ElemType = burn::tensor::f16;

pub fn launch<B: Backend>(device: B::Device) {
    let args: Vec<String> = env::args().collect();
    let default_model = "roberta-base".to_string();
    let model_variant = if args.len() > 1 {
        // Use the argument provided by the user
        // Possible values: "bert-base-uncased", "roberta-large" etc.
        &args[1]
    } else {
        // Use the default value if no argument is provided
        &default_model
    };

    println!("Model variant: {}", model_variant);

    let text_samples = vec![
        "Paris is the <mask> of France.".to_string(),
        "The goal of life is <mask>.".to_string(),
    ];

    let (config_file, model_file) = download_hf_model(model_variant);
    let model_config = load_model_config(config_file);

    let model_record: BertMaskedLMRecord<B> =
        BertMaskedLM::from_safetensors(model_file, &device, model_config.clone());

    let model = model_config
        .init_with_lm_head(&device)
        .load_record(model_record);

    let tokenizer = Arc::new(BertTokenizer::new(
        model_variant.to_string(),
        model_config.pad_token_id,
    ));

    // Batch the input samples to max sequence length with padding
    let batcher = Arc::new(BertInputBatcher::<B>::new(
        tokenizer.clone(),
        device.clone(),
        model_config.max_seq_len.unwrap(),
    ));

    // Batch input samples using the batcher Shape: [Batch size, Seq_len]
    let input = batcher.batch(text_samples.clone());
    let [batch_size, _seq_len] = input.tokens.dims();
    println!("Input: {:?} // (Batch Size, Seq_len)", input.tokens.shape());

    let output = fill_mask(&model, &model_config, tokenizer.as_ref(), input);

    for i in 0..batch_size {
        let input = &text_samples[i];
        let result = &output[i];
        println!("Input: {}", input);
        for fill_mask_result in result.iter() {
            let mask_idx = fill_mask_result.mask_idx;
            let top_k = &fill_mask_result.top_k;
            for (k, (score, token)) in top_k.iter().enumerate() {
                println!(
                    "Top {} Prediction for {}: {} (Score: {:.4})",
                    k + 1,
                    mask_idx,
                    token,
                    score
                );
            }
        }
    }
}

#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-blas-accelerate",
))]
mod ndarray {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    use crate::{launch, ElemType};

    pub fn run() {
        launch::<NdArray<ElemType>>(NdArrayDevice::Cpu);
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use crate::{launch, ElemType};
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        launch::<LibTorch<ElemType>>(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use crate::{launch, ElemType};
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};

    pub fn run() {
        launch::<LibTorch<ElemType>>(LibTorchDevice::Cpu);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use crate::{launch, ElemType};
    use burn::backend::wgpu::{AutoGraphicsApi, Wgpu, WgpuDevice};

    pub fn run() {
        launch::<Wgpu<AutoGraphicsApi, ElemType, i32>>(WgpuDevice::default());
    }
}

fn main() {
    #[cfg(any(
        feature = "ndarray",
        feature = "ndarray-blas-netlib",
        feature = "ndarray-blas-openblas",
        feature = "ndarray-blas-accelerate",
    ))]
    ndarray::run();
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
}
