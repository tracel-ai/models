use squeezenet_burn::model::{label::LABELS, normalizer::Normalizer, squeezenet1::Model};

use burn::tensor::Tensor;
use burn_ndarray::NdArrayBackend;

use image::{self, GenericImageView, Pixel};

const HEIGHT: usize = 224;
const WIDTH: usize = 224;

type Backend = NdArrayBackend<f32>;

fn main() {
    // Path to the image from the main args
    let img_path = std::env::args().nth(1).expect("No image path provided");

    // Load the image
    let img = image::open(&img_path).expect(format!("Failed to load image: {img_path}").as_str());

    // Resize it to 224x224
    let resized_img = img.resize_exact(
        WIDTH as u32,
        HEIGHT as u32,
        image::imageops::FilterType::Lanczos3,
    );

    // 3d array of 224x224x3 floats
    let mut img_array = [[[0.0; WIDTH]; HEIGHT]; 3];

    // Iterate over the pixels and populate the array
    for y in 0..224usize {
        for x in 0..224usize {
            let pixel = resized_img.get_pixel(x as u32, y as u32);
            let rgb = pixel.to_rgb();

            img_array[0][y][x] = rgb[0] as f32 / 255.0;
            img_array[1][y][x] = rgb[1] as f32 / 255.0;
            img_array[2][y][x] = rgb[2] as f32 / 255.0;
        }
    }

    // Create a tensor from the array
    let image_input = Tensor::<Backend, 3>::from_data(img_array).reshape([1, 3, HEIGHT, WIDTH]);

    // Normalize the image
    let normalizer = Normalizer::new();
    let normalized_image = normalizer.normalize(image_input);

    // Create the model
    let model = Model::<Backend>::default();

    // Run the model
    let output = model.forward(normalized_image);

    // Get the argmax of the output
    let arg_max = output.argmax(1).into_scalar() as usize;

    // Get the label from the argmax
    let label = LABELS[arg_max];

    println!("Predicted label: {}", label);
}
