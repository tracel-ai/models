use burn::tensor::Device;
use burn::tensor::Tensor;

// Values are taken from the [ONNX SqueezeNet]
// (https://github.com/onnx/models/tree/main/vision/classification/squeezenet#preprocessing)
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Normalizer for the imagenet dataset.
pub struct Normalizer {
    pub mean: Tensor<4>,
    pub std: Tensor<4>,
}

impl Normalizer {
    /// Creates a new normalizer.
    pub fn new(device: &Device) -> Self {
        let mean = Tensor::<1>::from_floats(MEAN, device).reshape([1, 3, 1, 1]);
        let std = Tensor::<1>::from_floats(STD, device).reshape([1, 3, 1, 1]);
        Self { mean, std }
    }

    /// Normalizes the input image according to the imagenet dataset.
    ///
    /// The input image should be in the range [0, 1].
    /// The output image will be in the range [-1, 1].
    ///
    /// The normalization is done according to the following formula:
    /// `input = (input - mean) / std`
    pub fn normalize(&self, input: Tensor<4>) -> Tensor<4> {
        (input - self.mean.clone()) / self.std.clone()
    }
}
