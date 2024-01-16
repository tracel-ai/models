use std::path::Path;

use resnet_burn::model::{imagenet, resnet::ResNet};

use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkGzFileRecorder},
    tensor::Tensor,
};
use burn_ndarray::NdArray;
use image::{self, GenericImageView, Pixel};

const HEIGHT: usize = 224;
const WIDTH: usize = 224;

type Backend = NdArray<f32>;

pub fn main() {
    // Load image
    let img_path = std::env::args().nth(1).expect("No image path provided");
    let img = image::open(&img_path).unwrap_or_else(|_| panic!("Failed to load image: {img_path}"));

    // Resize to 224x224
    let resized_img = img.resize_exact(
        WIDTH as u32,
        HEIGHT as u32,
        image::imageops::FilterType::Triangle, // also known as bilinear in 2D
    );

    // 3d array of 224x224x3 floats
    let mut img_array = [[[0.0; WIDTH]; HEIGHT]; 3];

    // Iterate over the pixels and populate the array
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let pixel = resized_img.get_pixel(x as u32, y as u32);
            let rgb = pixel.to_rgb();

            img_array[0][y][x] = rgb[0] as f32 / 255.0;
            img_array[1][y][x] = rgb[1] as f32 / 255.0;
            img_array[2][y][x] = rgb[2] as f32 / 255.0;
        }
    }

    // Create a tensor from the array
    let image_input = Tensor::<Backend, 3>::from_data(img_array).reshape([1, 3, HEIGHT, WIDTH]);
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();

    // Normalize the image
    let x = imagenet::Normalizer::new().normalize(image_input);

    // Load pre-trained ResNet-18
    let model_path = Path::new(file!())
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("model/resnet18-ImageNet1k");
    let model = ResNet::<Backend>::resnet18(1000)
        .load_file(model_path.clone(), &recorder)
        .unwrap_or_else(|_| panic!("Failed to load model file: {}", model_path.display()));

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
