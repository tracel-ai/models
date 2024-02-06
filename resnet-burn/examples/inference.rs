use resnet_burn::model::resnet::ResNet;

use burn::{
    backend::{ndarray::NdArrayDevice, NdArray},
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

const MODEL_PATH: &str = "resnet18-ImageNet1k";
const NUM_CLASSES: usize = 1000;

pub fn main() {
    // Create ResNet-18
    // let device = Default::default();
    let device = NdArrayDevice::default();
    let model: ResNet<NdArray, _> = ResNet::resnet18(NUM_CLASSES, &device);

    // Load weights from torch state_dict
    // NOTE: the remap chain below only works if you remove the `break` in `remap`
    // https://github.com/tracel-ai/burn/blob/main/burn-core/src/record/serde/data.rs#L180
    // Download URL: https://download.pytorch.org/models/resnet18-f37072fd.pth
    let load_args = LoadArgs::new("resnet18-f37072fd.pth".into())
        // Map *.downsample.0.* -> *.downsample.conv.*
        .with_key_remap("(.+)\\.downsample\\.0\\.(.+)", "$1.downsample.conv.$2")
        // Map *.downsample.1.* -> *.downsample.bn.*
        .with_key_remap("(.+)\\.downsample\\.1\\.(.+)", "$1.downsample.bn.$2")
        // Map layer[i].[j].* -> layer[i].blocks.[j].*
        .with_key_remap("layer[1-4]\\.([0-9])\\.(.+)", "layer$1.blocks.$2.$3");
    println!("Loading record w/ key remap");
    let record = PyTorchFileRecorder::<FullPrecisionSettings>::new()
        .load(load_args, &device)
        .map_err(|err| format!("Failed to load weights.\nError: {err}"))
        .unwrap();
    // .expect("Should load PyTorch model weights correctly");

    println!("Loading record into model");
    let model = model.load_record(record);

    // Save the model to a supported format and load it back
    println!("Saving the model to Named MessagePack");
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone() // `save_file` takes ownership but we want to load the file after
        .save_file(MODEL_PATH, &recorder)
        .expect(&format!(
            "Should be able to save weights to file {MODEL_PATH}"
        ));
    let _ = model
        .load_file(MODEL_PATH, &recorder, &device)
        .map_err(|err| format!("Failed to load weights from file {MODEL_PATH}.\nError: {err}"))
        .unwrap();
}
