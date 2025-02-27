use squeezenet_burn::model::{label::LABELS, normalizer::Normalizer, squeezenet1::Model};

#[cfg(feature = "weights_embedded")]
use burn::backend::ndarray::NdArrayDevice;

use burn::backend::NdArray;
use burn::tensor::Tensor;

use image::{self, GenericImageView, Pixel};

const HEIGHT: usize = 224;
const WIDTH: usize = 224;

#[cfg(feature = "weights_file")]
const RECORD_FILE: &str = "squeezenet1";

type Backend = NdArray<f32>;

fn main() {
    // Path to the image from the main args
    let img_path = std::env::args().nth(1).expect("No image path provided");

    // Load the image
    let img = image::open(&img_path).unwrap_or_else(|_| panic!("Failed to load image: {img_path}"));

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

    let device = Default::default();

    // Create a tensor from the array
    let image_input =
        Tensor::<Backend, 3>::from_data(img_array, &device).reshape([1, 3, HEIGHT, WIDTH]);
    // Normalize the image
    let normalizer = Normalizer::new(&device);
    let normalized_image = normalizer.normalize(image_input);

    // Create the model
    // Load the weights from the file next to the executable
    #[cfg(feature = "weights_file")]
    let weights_file = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .join(RECORD_FILE);

    #[cfg(feature = "weights_file")]
    let model = Model::<Backend>::from_file(weights_file.to_str().unwrap(), &device);

    #[cfg(feature = "weights_embedded")]
    // Load model from embedded weights
    let model = Model::<Backend>::from_embedded(&NdArrayDevice::Cpu);

    // Run the model
    let output = model.forward(normalized_image);

    // Get the argmax of the output
    let arg_max = output.argmax(1).into_scalar() as usize;

    // Get the label from the argmax
    let label = LABELS[arg_max];

    println!("Predicted label: {}", label);
}
