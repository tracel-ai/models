use resnet_burn::model::{imagenet, resnet::ResNet};

use burn::{
    backend::NdArray,
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::{backend::Backend, Data, Device, Element, Shape, Tensor},
};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

const TORCH_WEIGHTS: &str = "resnet18-f37072fd.pth";
const MODEL_PATH: &str = "resnet18-ImageNet1k";
const NUM_CLASSES: usize = 1000;
const HEIGHT: usize = 224;
const WIDTH: usize = 224;

fn to_tensor<B: Backend, T: Element>(
    data: Vec<T>,
    shape: [usize; 3],
    device: &Device<B>,
) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(Data::new(data, Shape::new(shape)).convert(), device)
        // permute(2, 0, 1)
        .swap_dims(2, 1) // [H, C, W]
        .swap_dims(1, 0) // [C, H, W]
        / 255 // normalize between [0, 1]
}

pub fn main() {
    // Parse arguments
    let img_path = std::env::args().nth(1).expect("No image path provided");

    // Create ResNet-18
    let device = Default::default();
    let model: ResNet<NdArray, _> = ResNet::resnet18(NUM_CLASSES, &device);

    // Load weights from torch state_dict
    let load_args = LoadArgs::new(TORCH_WEIGHTS.into())
        // Map *.downsample.0.* -> *.downsample.conv.*
        .with_key_remap("(.+)\\.downsample\\.0\\.(.+)", "$1.downsample.conv.$2")
        // Map *.downsample.1.* -> *.downsample.bn.*
        .with_key_remap("(.+)\\.downsample\\.1\\.(.+)", "$1.downsample.bn.$2")
        // Map layer[i].[j].* -> layer[i].blocks.[j].*
        .with_key_remap("(layer[1-4])\\.([0-9])\\.(.+)", "$1.blocks.$2.$3");
    let record = PyTorchFileRecorder::<FullPrecisionSettings>::new()
        .load(load_args, &device)
        .map_err(|err| format!("Failed to load weights.\nError: {err}"))
        .unwrap();

    let model = model.load_record(record);

    // Save the model to a supported format and load it back
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone() // `save_file` takes ownership but we want to load the file after
        .save_file(MODEL_PATH, &recorder)
        .map_err(|err| format!("Failed to save weights to file {MODEL_PATH}.\nError: {err}"))
        .unwrap();
    let model = model
        .load_file(MODEL_PATH, &recorder, &device)
        .map_err(|err| format!("Failed to load weights from file {MODEL_PATH}.\nError: {err}"))
        .unwrap();

    // Load image
    let img = image::open(&img_path)
        .map_err(|err| format!("Failed to load image {img_path}.\nError: {err}"))
        .unwrap();

    // Resize to 224x224
    let resized_img = img.resize_exact(
        WIDTH as u32,
        HEIGHT as u32,
        image::imageops::FilterType::Triangle, // also known as bilinear in 2D
    );

    // Create tensor from image data
    let img_tensor = to_tensor(
        resized_img.into_rgb8().into_raw(),
        [HEIGHT, WIDTH, 3],
        &device,
    )
    .unsqueeze::<4>(); // [B, C, H, W]

    // Normalize the image
    let x = imagenet::Normalizer::new(&device).normalize(img_tensor);

    // Forward pass
    let out = model.forward(x);

    // Output class index w/ score (raw)
    let (score, idx) = out.max_dim_with_indices(1);
    let idx = idx.into_scalar() as usize;

    println!(
        "Predicted: {}\nCategory Id: {}\nScore: {:.4}",
        imagenet::CLASSES[idx],
        idx,
        score.into_scalar()
    );
}
