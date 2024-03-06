use yolox_burn::model::{weights, yolox::Yolox};

use burn::{
    backend::NdArray,
    tensor::{backend::Backend, Data, Device, Element, Shape, Tensor},
};

const HEIGHT: usize = 640;
const WIDTH: usize = 640;

fn to_tensor<B: Backend, T: Element>(
    data: Vec<T>,
    shape: [usize; 3],
    device: &Device<B>,
) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(Data::new(data, Shape::new(shape)).convert(), device)
        // permute(2, 0, 1)
        .swap_dims(2, 1) // [H, C, W]
        .swap_dims(1, 0) // [C, H, W]
}

pub fn main() {
    // Parse arguments
    let img_path = std::env::args().nth(1).expect("No image path provided");

    // Create YOLOX-Nano
    let device = Default::default();
    let model: Yolox<NdArray> = Yolox::yolox_s_pretrained(weights::YoloxS::Coco, &device)
        .map_err(|err| format!("Failed to load pre-trained weights.\nError: {err}"))
        .unwrap();

    // Load image
    let img = image::open(&img_path)
        .map_err(|err| format!("Failed to load image {img_path}.\nError: {err}"))
        .unwrap();

    // Resize to 640x640
    let resized_img = img.resize_exact(
        WIDTH as u32,
        HEIGHT as u32,
        image::imageops::FilterType::Triangle, // also known as bilinear in 2D
    );

    // Create tensor from image data
    let x = to_tensor(
        resized_img.into_rgb8().into_raw(),
        [HEIGHT, WIDTH, 3],
        &device,
    )
    .unsqueeze::<4>(); // [B, C, H, W]

    // Forward pass
    let out = model.forward(x);

    // TODO: post-processing and display
    println!("Output shape: {:?}", out.shape());
}
