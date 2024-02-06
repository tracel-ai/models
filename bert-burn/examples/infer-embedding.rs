use bert_burn::bert::{BertModel, BertModelConfig, BertModelRecord};
use bert_burn::data::{BertInputBatcher, BertTokenizer};
use bert_burn::loader::{load_model_config, load_model_from_safetensors};
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use burn::record::{FullPrecisionSettings, Record, Recorder};

#[cfg(not(feature = "f16"))]
#[allow(dead_code)]
type ElemType = f32;
#[cfg(feature = "f16")]
type ElemType = burn::tensor::f16;

pub fn launch<B: Backend>(device: B::Device) {
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

    let file = File::open("weights/config.json").expect("Unable to open file");
    let config: HashMap<String, Value> =
        serde_json::from_reader(file).expect("Unable to parse JSON");

    // Change to "large" for large model variant"
    // model size is not explicitly defined in config hence we need to append it to the model name
    let model_size = "base";
    let model_name = format!("{}-{}", config["model_type"].as_str().unwrap(), model_size);

    let tokenizer = Arc::new(BertTokenizer::new(
        model_name,
        config["pad_token_id"].as_i64().unwrap() as usize,
    ));

    // Initialize batcher for batching samples
    let batcher = Arc::new(BertInputBatcher::<B>::new(
        tokenizer.clone(),
        device.clone(),
        512,
    ));

    // Batch input samples using the batcher Shape: [Batch size, Seq_len]
    let input = batcher.batch(text_samples.clone());
    println!(
        "Input shape: {:?} // (Batch Size, Seq_len)",
        input.tokens.shape()
    );

    let model_config: BertModelConfig = load_model_config(config);
    // let model = load_model_from_safetensors("weights/model.safetensors", &device, model_config);

    // let output = model.forward(input);

    // get sentence embedding from the first [CLS] token
    // let sentence_embedding = output.clone().slice([0..3, 0..1, 0..768]);
    // let sentence_embedding: Tensor<B, 2> = sentence_embedding.squeeze(1);
    // println!(
    //     "Roberta Sentence embedding shape {:?} // (Batch Size, Embedding_dim)",
    //     sentence_embedding.shape()
    // );

    // Not working
    let load_args = LoadArgs::new("weights/model.pt".into())
        .with_key_remap("roberta\\.embeddings\\.(?!LayerNorm\\.)\\S+", "embeddings.$0")
        .with_key_remap("roberta\\.embeddings\\.LayerNorm\\.weight", "embeddings.layer_norm.gamma")
        .with_key_remap("roberta\\.embeddings\\.LayerNorm\\.bias", "embeddings.layer_norm.beta")
        .with_key_remap("roberta\\.encoder\\.layer\\.(\\d+)\\.attention\\.self\\.(.*)", "encoder.layers.$1.mha.$2")
        .with_key_remap("roberta\\.encoder\\.layer\\.(\\d+)\\.attention\\.output\\.dense\\.(.*)", "encoder.layers.$1.mha.output.$2")
        .with_key_remap("roberta\\.encoder\\.layer.(\\d+)\\.attention\\.output\\.LayerNorm\\.weight", "encoder.layers.$1.norm_1.gamma")
        .with_key_remap("roberta\\.encoder\\.layer.(\\d+)\\.attention\\.output\\.LayerNorm\\.weight", "encoder.layers.$1.norm_1.beta")
        .with_key_remap("roberta\\.encoder\\.layer\\.(\\d+)\\.intermediate\\.dense\\.(.*)", "encoder.layers.$1.pwff.linear_inner.$2")
        .with_key_remap("roberta\\.encoder\\.layer\\.(\\d+)\\.output\\.dense\\.(.*)", "encoder.layers.$1.pwff.linear_outer.$2")
        .with_key_remap("roberta\\.encoder\\.layer.(\\d+)\\.output\\.LayerNorm\\.weight", "encoder.layers.$1.norm_2.gamma")
        .with_key_remap("roberta\\.encoder\\.layer.(\\d+)\\.output\\.LayerNorm\\.weight", "encoder.layers.$1.norm_2.beta");

    let record: BertModelRecord<B> = PyTorchFileRecorder::<FullPrecisionSettings>::default()
        .load(load_args, &device)
        .expect("Should load model successfully");


    println!("{:?}", record.embeddings.position_embeddings.weight.shape());

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
    use burn::backend::Fusion;

    pub fn run() {
        launch::<Fusion<Wgpu<AutoGraphicsApi, ElemType, i32>>>(WgpuDevice::default());
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
