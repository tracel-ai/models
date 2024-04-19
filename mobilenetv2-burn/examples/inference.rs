use mobilenetv2_burn::model::{imagenet, mobilenetv2::MobileNetV2, weights};

use burn::{
    backend::NdArray,
    tensor::{backend::Backend, Data, Device, Element, Shape, Tensor},
};

const HEIGHT: usize = 224;
const WIDTH: usize = 224;

fn to_tensor<B: Backend, T: Element>(
    data: Vec<T>,
    shape: [usize; 3],
    device: &Device<B>,
) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(Data::new(data, Shape::new(shape)).convert(), device)
        // [H, W, C] -> [C, H, W]
        .permute([2, 0, 1])
        / 255 // normalize between [0, 1]
}

pub fn main() {
    // Parse arguments
    let img_path = std::env::args().nth(1).expect("No image path provided");

    // Create ResNet-18
    let device = Default::default();
    let model: MobileNetV2<NdArray> =
        MobileNetV2::pretrained(weights::MobileNetV2::ImageNet1kV2, &device)
            .map_err(|err| format!("Failed to load pre-trained weights.\nError: {err}"))
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
