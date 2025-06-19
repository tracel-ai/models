#![recursion_limit = "256"] // wgpu

use bert_burn::data::{BertInputBatcher, BertTokenizer};
use bert_burn::loader::{download_hf_model, load_model_config};
use bert_burn::model::{BertModel, BertModelRecord};
use burn::data::dataloader::batcher::Batcher;
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
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
        "Jays power up to take finale Contrary to popular belief, the power never really \
                 snapped back at SkyDome on Sunday. The lights came on after an hour delay, but it \
                 took some extra time for the batting orders to provide some extra wattage."
            .to_string(),
        "Yemen Sentences 15 Militants on Terror Charges A court in Yemen has sentenced one \
                 man to death and 14 others to prison terms for a series of attacks and terrorist \
                 plots in 2002, including the bombing of a French oil tanker."
            .to_string(),
        "IBM puts grids to work at U.S. Open IBM will put a collection of its On \
                 Demand-related products and technologies to this test next week at the U.S. Open \
                 tennis championships, implementing a grid-based infrastructure capable of running \
                 multiple workloads including two not associated with the tournament."
            .to_string(),
    ];

    let (config_file, model_file) = download_hf_model(model_variant);
    let model_config = load_model_config(config_file);

    let model_record: BertModelRecord<B> =
        BertModel::from_safetensors(model_file, &device, model_config.clone());

    let model = model_config.init(&device).load_record(model_record);

    let tokenizer = Arc::new(BertTokenizer::new(
        model_variant.to_string(),
        model_config.pad_token_id,
    ));

    // Batch the input samples to max sequence length with padding
    let batcher = Arc::new(BertInputBatcher::new(
        tokenizer.clone(),
        model_config.max_seq_len.unwrap(),
    ));

    // Batch input samples using the batcher Shape: [Batch size, Seq_len]
    let input = batcher.batch(text_samples.clone(), &device);
    let [batch_size, _seq_len] = input.tokens.dims();
    println!("Input: {}", input.tokens);

    let output = model.forward(input);

    // get sentence embedding from the first [CLS] token
    let cls_token_idx = 0;

    // Embedding size
    let d_model = model_config.hidden_size;
    let sentence_embedding = output.hidden_states.clone().slice([
        0..batch_size,
        cls_token_idx..cls_token_idx + 1,
        0..d_model,
    ]);

    let sentence_embedding: Tensor<B, 2> = sentence_embedding.squeeze(1);
    println!("Roberta Sentence embedding: {}", sentence_embedding);
}

#[cfg(feature = "ndarray")]
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
    use crate::launch;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    pub fn run() {
        launch::<Wgpu>(WgpuDevice::default());
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use crate::launch;
    use burn::backend::{cuda::CudaDevice, Cuda};

    pub fn run() {
        launch::<Cuda>(CudaDevice::default());
    }
}

fn main() {
    #[cfg(feature = "ndarray")]
    ndarray::run();
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
    #[cfg(feature = "cuda")]
    cuda::run();
}
