use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::vision::{Annotation, ImageDatasetItem, PixelDepth},
    },
    prelude::*,
};

use super::dataset::CLASSES;

// ImageNet mean and std values
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

// Planets patch size
const WIDTH: usize = 256;
const HEIGHT: usize = 256;

/// Create a multi-hot encoded tensor.
///
/// # Example
///
/// ```rust, ignore
/// let multi_hot = multi_hot::<B>(&[2, 5, 8], 10, &device);
/// println!("{}", multi_hot.to_data());
/// // [0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
/// ```
pub fn multi_hot<B: Backend>(
    indices: &[usize],
    num_classes: usize,
    device: &B::Device,
) -> Tensor<B, 1, Int> {
    Tensor::zeros(Shape::new([num_classes]), device).scatter(
        0,
        Tensor::from_ints(
            indices
                .iter()
                .map(|i| *i as i32)
                .collect::<Vec<_>>()
                .as_slice(),
            device,
        ),
        Tensor::ones(Shape::new([indices.len()]), device),
    )
}

/// Normalizer with ImageNet values as it helps accelerate training since we are fine-tuning from
/// ImageNet pre-trained weights and the model expects the data to be in this normalized range.
#[derive(Clone)]
pub struct Normalizer<B: Backend> {
    pub mean: Tensor<B, 4>,
    pub std: Tensor<B, 4>,
}

impl<B: Backend> Normalizer<B> {
    /// Creates a new normalizer.
    pub fn new(device: &Device<B>) -> Self {
        let mean = Tensor::<B, 1>::from_floats(MEAN, device).reshape([1, 3, 1, 1]);
        let std = Tensor::<B, 1>::from_floats(STD, device).reshape([1, 3, 1, 1]);
        Self { mean, std }
    }

    /// Normalizes the input image according to the ImageNet dataset.
    ///
    /// The input image should be in the range [0, 1].
    /// The output image will be in the range [-1, 1].
    ///
    /// The normalization is done according to the following formula:
    /// `input = (input - mean) / std`
    pub fn normalize(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        (input - self.mean.clone()) / self.std.clone()
    }

    /// Returns a new normalizer on the given device.
    pub fn to_device(&self, device: &B::Device) -> Self {
        Self {
            mean: self.mean.clone().to_device(device),
            std: self.std.clone().to_device(device),
        }
    }
}

#[derive(Clone)]
pub struct ClassificationBatcher<B: Backend> {
    normalizer: Normalizer<B>,
}

#[derive(Clone, Debug)]
pub struct ClassificationBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> ClassificationBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            normalizer: Normalizer::<B>::new(&device),
        }
    }
}

impl<B: Backend> Batcher<B, ImageDatasetItem, ClassificationBatch<B>> for ClassificationBatcher<B> {
    fn batch(&self, items: Vec<ImageDatasetItem>, device: &B::Device) -> ClassificationBatch<B> {
        fn image_as_vec_u8(item: ImageDatasetItem) -> Vec<u8> {
            // Convert Vec<PixelDepth> to Vec<u8> (Planet images are u8)
            item.image
                .into_iter()
                .map(|p: PixelDepth| -> u8 { p.try_into().unwrap() })
                .collect::<Vec<u8>>()
        }

        let targets = items
            .iter()
            .map(|item| {
                // Expect multi-hot encoded class labels as target (e.g., [0, 1, 0, 0, 1])
                if let Annotation::MultiLabel(y) = &item.annotation {
                    multi_hot(y, CLASSES.len(), device)
                } else {
                    panic!("Invalid target type")
                }
            })
            .collect();

        let images = items
            .into_iter()
            .map(|item| TensorData::new(image_as_vec_u8(item), Shape::new([HEIGHT, WIDTH, 3])))
            .map(|data| Tensor::<B, 3>::from_data(data.convert::<B::FloatElem>(), device))
            .map(|tensor| tensor.permute([2, 0, 1]) / 255) // normalize between [0, 1]
            .collect();

        let images = Tensor::stack(images, 0);
        let targets = Tensor::stack(targets, 0);

        let images = self.normalizer.to_device(device).normalize(images);

        ClassificationBatch { images, targets }
    }
}
