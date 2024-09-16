use std::path::Path;

use image::{DynamicImage, ImageBuffer};
use yolox_burn::model::{boxes::nms, weights, yolox::Yolox, BoundingBox};

use burn::{
    backend::NdArray,
    tensor::{backend::Backend, Device, Element, Tensor, TensorData},
};

const HEIGHT: usize = 640;
const WIDTH: usize = 640;

fn to_tensor<B: Backend, T: Element>(
    data: Vec<T>,
    shape: [usize; 3],
    device: &Device<B>,
) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(
        TensorData::new(data, shape).convert::<B::FloatElem>(),
        device,
    )
    // [H, W, C] -> [C, H, W]
    .permute([2, 0, 1])
}

/// Draws bounding boxes on the given image.
///
/// # Arguments
///
/// * `image`: Original input image.
/// * `boxes` - Bounding boxes, grouped per class.
/// * `color` - [R, G, B] color values to draw the boxes.
/// * `ratio` - [x, y] aspect ratio to scale the predicted boxes.
///
/// # Returns
///
/// The image annotated with bounding boxes.
fn draw_boxes(
    image: DynamicImage,
    boxes: &[Vec<BoundingBox>],
    color: &[u8; 3],
    ratio: &[f32; 2], // (x, y) ratio
) -> DynamicImage {
    // Assumes x1 <= x2 and y1 <= y2
    fn draw_rect(
        image: &mut ImageBuffer<image::Rgb<u8>, Vec<u8>>,
        x1: u32,
        x2: u32,
        y1: u32,
        y2: u32,
        color: &[u8; 3],
    ) {
        for x in x1..=x2 {
            let pixel = image.get_pixel_mut(x, y1);
            *pixel = image::Rgb(*color);
            let pixel = image.get_pixel_mut(x, y2);
            *pixel = image::Rgb(*color);
        }
        for y in y1..=y2 {
            let pixel = image.get_pixel_mut(x1, y);
            *pixel = image::Rgb(*color);
            let pixel = image.get_pixel_mut(x2, y);
            *pixel = image::Rgb(*color);
        }
    }

    // Annotate the original image and print boxes information.
    let (image_h, image_w) = (image.height(), image.width());
    let mut image = image.to_rgb8();
    for (class_index, bboxes_for_class) in boxes.iter().enumerate() {
        for b in bboxes_for_class.iter() {
            let xmin = (b.xmin * ratio[0]).clamp(0., image_w as f32 - 1.);
            let ymin = (b.ymin * ratio[1]).clamp(0., image_h as f32 - 1.);
            let xmax = (b.xmax * ratio[0]).clamp(0., image_w as f32 - 1.);
            let ymax = (b.ymax * ratio[1]).clamp(0., image_h as f32 - 1.);

            println!(
                "Predicted {} ({:.2}) at [{:.2}, {:.2}, {:.2}, {:.2}]",
                class_index, b.confidence, xmin, ymin, xmax, ymax,
            );

            draw_rect(
                &mut image,
                xmin as u32,
                xmax as u32,
                ymin as u32,
                ymax as u32,
                color,
            );
        }
    }
    DynamicImage::ImageRgb8(image)
}

pub fn main() {
    // Parse arguments
    let img_path = std::env::args().nth(1).expect("No image path provided");

    // Create YOLOX-Tiny
    let device = Default::default();
    let model: Yolox<NdArray> = Yolox::yolox_tiny_pretrained(weights::YoloxTiny::Coco, &device)
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

    // Post-processing
    let [_, num_boxes, num_outputs] = out.dims();
    let boxes = out.clone().slice([0..1, 0..num_boxes, 0..4]);
    let obj_scores = out.clone().slice([0..1, 0..num_boxes, 4..5]);
    let cls_scores = out.slice([0..1, 0..num_boxes, 5..num_outputs]);
    let scores = cls_scores * obj_scores;
    let boxes = nms(boxes, scores, 0.65, 0.5);

    // Draw outputs and save results
    let (h, w) = (img.height(), img.width());
    let img_out = draw_boxes(
        img,
        &boxes[0],
        &[239u8, 62u8, 5u8],
        &[w as f32 / WIDTH as f32, h as f32 / HEIGHT as f32],
    );

    let img_path = Path::new(&img_path);
    let _ = img_out.save(img_path.with_extension("output.png"));
}
