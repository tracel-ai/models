use core::f64::consts::SQRT_2;

use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig, Initializer, Linear, LinearConfig, PaddingConfig2d, Relu,
    },
    tensor::{backend::Backend, Device, Tensor},
};

use super::block::{LayerBlock, LayerBlockConfig};

#[cfg(feature = "pretrained")]
use {
    super::weights::{self, WeightsMeta},
    burn::record::{FullPrecisionSettings, Recorder, RecorderError},
    burn_import::pytorch::{LoadArgs, PyTorchFileRecorder},
};

// ResNet residual layer block configs
const RESNET18_BLOCKS: [usize; 4] = [2, 2, 2, 2];
const RESNET34_BLOCKS: [usize; 4] = [3, 4, 6, 3];
const RESNET50_BLOCKS: [usize; 4] = [3, 4, 6, 3];
const RESNET101_BLOCKS: [usize; 4] = [3, 4, 23, 3];
const RESNET152_BLOCKS: [usize; 4] = [3, 8, 36, 3];

/// ResNet implementation.
/// Derived from [torchivision.models.resnet.ResNet](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
#[derive(Module, Debug)]
pub struct ResNet<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    relu: Relu,
    maxpool: MaxPool2d,
    layer1: LayerBlock<B>,
    layer2: LayerBlock<B>,
    layer3: LayerBlock<B>,
    layer4: LayerBlock<B>,
    avgpool: AdaptiveAvgPool2d,
    fc: Linear<B>,
}

impl<B: Backend> ResNet<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        // First block
        let out = self.conv1.forward(input);
        let out = self.bn1.forward(out);
        let out = self.relu.forward(out);
        let out = self.maxpool.forward(out);

        // Residual blocks
        let out = self.layer1.forward(out);
        let out = self.layer2.forward(out);
        let out = self.layer3.forward(out);
        let out = self.layer4.forward(out);

        let out = self.avgpool.forward(out);
        // Reshape [B, C, 1, 1] -> [B, C]
        let out = out.flatten(1, 3);

        self.fc.forward(out)
    }

    /// ResNet-18 from [`Deep Residual Learning for Image Recognition`](https://arxiv.org/abs/1512.03385).
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A ResNet-18 module.
    pub fn resnet18(num_classes: usize, device: &Device<B>) -> Self {
        ResNetConfig::new(RESNET18_BLOCKS, num_classes, 1).init(device)
    }

    /// ResNet-18 from [`Deep Residual Learning for Image Recognition`](https://arxiv.org/abs/1512.03385)
    /// with pre-trained weights.
    ///
    /// # Arguments
    ///
    /// * `weights`: Pre-trained weights to load.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A ResNet-18 module with pre-trained weights.
    #[cfg(feature = "pretrained")]
    pub fn resnet18_pretrained(
        weights: weights::ResNet18,
        device: &Device<B>,
    ) -> Result<Self, RecorderError> {
        let weights = weights.weights();
        let record = Self::load_weights_record(&weights, device)?;
        let model = ResNet::<B>::resnet18(weights.num_classes, device).load_record(record);

        Ok(model)
    }

    /// ResNet-34 from [`Deep Residual Learning for Image Recognition`](https://arxiv.org/abs/1512.03385).
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A ResNet-34 module.
    pub fn resnet34(num_classes: usize, device: &Device<B>) -> Self {
        ResNetConfig::new(RESNET34_BLOCKS, num_classes, 1).init(device)
    }

    /// ResNet-34 from [`Deep Residual Learning for Image Recognition`](https://arxiv.org/abs/1512.03385)
    /// with pre-trained weights.
    ///
    /// # Arguments
    ///
    /// * `weights`: Pre-trained weights to load.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A ResNet-34 module with pre-trained weights.
    #[cfg(feature = "pretrained")]
    pub fn resnet34_pretrained(
        weights: weights::ResNet34,
        device: &Device<B>,
    ) -> Result<Self, RecorderError> {
        let weights = weights.weights();
        let record = Self::load_weights_record(&weights, device)?;
        let model = ResNet::<B>::resnet34(weights.num_classes, device).load_record(record);

        Ok(model)
    }

    /// ResNet-50 from [`Deep Residual Learning for Image Recognition`](https://arxiv.org/abs/1512.03385).
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A ResNet-50 module.
    pub fn resnet50(num_classes: usize, device: &Device<B>) -> Self {
        ResNetConfig::new(RESNET50_BLOCKS, num_classes, 4).init(device)
    }

    /// ResNet-50 from [`Deep Residual Learning for Image Recognition`](https://arxiv.org/abs/1512.03385)
    /// with pre-trained weights.
    ///
    /// # Arguments
    ///
    /// * `weights`: Pre-trained weights to load.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A ResNet-50 module with pre-trained weights.
    #[cfg(feature = "pretrained")]
    pub fn resnet50_pretrained(
        weights: weights::ResNet50,
        device: &Device<B>,
    ) -> Result<Self, RecorderError> {
        let weights = weights.weights();
        let record = Self::load_weights_record(&weights, device)?;
        let model = ResNet::<B>::resnet50(weights.num_classes, device).load_record(record);

        Ok(model)
    }

    /// ResNet-101 from [`Deep Residual Learning for Image Recognition`](https://arxiv.org/abs/1512.03385).
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A ResNet-101 module.
    pub fn resnet101(num_classes: usize, device: &Device<B>) -> Self {
        ResNetConfig::new(RESNET101_BLOCKS, num_classes, 4).init(device)
    }

    /// ResNet-101 from [`Deep Residual Learning for Image Recognition`](https://arxiv.org/abs/1512.03385)
    /// with pre-trained weights.
    ///
    /// # Arguments
    ///
    /// * `weights`: Pre-trained weights to load.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A ResNet-101 module with pre-trained weights.
    #[cfg(feature = "pretrained")]
    pub fn resnet101_pretrained(
        weights: weights::ResNet101,
        device: &Device<B>,
    ) -> Result<Self, RecorderError> {
        let weights = weights.weights();
        let record = Self::load_weights_record(&weights, device)?;
        let model = ResNet::<B>::resnet101(weights.num_classes, device).load_record(record);

        Ok(model)
    }

    /// ResNet-152 from [`Deep Residual Learning for Image Recognition`](https://arxiv.org/abs/1512.03385).
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A ResNet-152 module.
    pub fn resnet152(num_classes: usize, device: &Device<B>) -> Self {
        ResNetConfig::new(RESNET152_BLOCKS, num_classes, 4).init(device)
    }

    /// ResNet-152 from [`Deep Residual Learning for Image Recognition`](https://arxiv.org/abs/1512.03385)
    /// with pre-trained weights.
    ///
    /// # Arguments
    ///
    /// * `weights`: Pre-trained weights to load.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A ResNet-152 module with pre-trained weights.
    #[cfg(feature = "pretrained")]
    pub fn resnet152_pretrained(
        weights: weights::ResNet152,
        device: &Device<B>,
    ) -> Result<Self, RecorderError> {
        let weights = weights.weights();
        let record = Self::load_weights_record(&weights, device)?;
        let model = ResNet::<B>::resnet152(weights.num_classes, device).load_record(record);

        Ok(model)
    }

    /// Re-initialize the last layer with the specified number of output classes.
    pub fn with_classes(mut self, num_classes: usize) -> Self {
        let [d_input, _d_output] = self.fc.weight.dims();
        self.fc = LinearConfig::new(d_input, num_classes).init(&self.fc.weight.device());
        self
    }
}

#[cfg(feature = "pretrained")]
impl<B: Backend> ResNet<B> {
    /// Load specified pre-trained PyTorch weights as a record.
    fn load_weights_record(
        weights: &weights::Weights,
        device: &Device<B>,
    ) -> Result<ResNetRecord<B>, RecorderError> {
        // Download torch weights
        let torch_weights = weights.download().map_err(|err| {
            RecorderError::Unknown(format!("Could not download weights.\nError: {err}"))
        })?;

        // Load weights from torch state_dict
        let load_args = LoadArgs::new(torch_weights)
            // Map *.downsample.0.* -> *.downsample.conv.*
            .with_key_remap("(.+)\\.downsample\\.0\\.(.+)", "$1.downsample.conv.$2")
            // Map *.downsample.1.* -> *.downsample.bn.*
            .with_key_remap("(.+)\\.downsample\\.1\\.(.+)", "$1.downsample.bn.$2")
            // Map layer[i].[j].* -> layer[i].blocks.[j].*
            .with_key_remap("(layer[1-4])\\.([0-9]+)\\.(.+)", "$1.blocks.$2.$3");
        let record = PyTorchFileRecorder::<FullPrecisionSettings>::new().load(load_args, device)?;

        Ok(record)
    }
}

/// [ResNet](ResNet) configuration.
struct ResNetConfig {
    conv1: Conv2dConfig,
    bn1: BatchNormConfig,
    maxpool: MaxPool2dConfig,
    layer1: LayerBlockConfig,
    layer2: LayerBlockConfig,
    layer3: LayerBlockConfig,
    layer4: LayerBlockConfig,
    avgpool: AdaptiveAvgPool2dConfig,
    fc: LinearConfig,
}

impl ResNetConfig {
    /// Create a new instance of the ResNet [config](ResNetConfig).
    fn new(blocks: [usize; 4], num_classes: usize, expansion: usize) -> Self {
        // `new()` is private but still check just in case...
        assert!(
            expansion == 1 || expansion == 4,
            "ResNet module only supports expansion values [1, 4] for residual blocks"
        );

        // 7x7 conv, 64, /2
        let conv1 = Conv2dConfig::new([3, 64], [7, 7])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(3, 3))
            .with_bias(false);
        let bn1 = BatchNormConfig::new(64);

        // 3x3 maxpool, /2
        let maxpool = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1));

        // Residual blocks
        let bottleneck = expansion > 1;
        let layer1 = LayerBlockConfig::new(blocks[0], 64, 64 * expansion, 1, bottleneck);
        let layer2 =
            LayerBlockConfig::new(blocks[1], 64 * expansion, 128 * expansion, 2, bottleneck);
        let layer3 =
            LayerBlockConfig::new(blocks[2], 128 * expansion, 256 * expansion, 2, bottleneck);
        let layer4 =
            LayerBlockConfig::new(blocks[3], 256 * expansion, 512 * expansion, 2, bottleneck);

        // Average pooling [B, 512 * expansion, H, W] -> [B, 512 * expansion, 1, 1]
        let avgpool = AdaptiveAvgPool2dConfig::new([1, 1]);

        // Output layer
        let fc = LinearConfig::new(512 * expansion, num_classes);

        Self {
            conv1,
            bn1,
            maxpool,
            layer1,
            layer2,
            layer3,
            layer4,
            avgpool,
            fc,
        }
    }
}

impl ResNetConfig {
    /// Initialize a new [ResNet](ResNet) module.
    fn init<B: Backend>(self, device: &Device<B>) -> ResNet<B> {
        // Conv initializer
        let initializer = Initializer::KaimingNormal {
            gain: SQRT_2, // recommended value for ReLU
            fan_out_only: true,
        };

        ResNet {
            conv1: self.conv1.with_initializer(initializer).init(device),
            bn1: self.bn1.init(device),
            relu: Relu::new(),
            maxpool: self.maxpool.init(),
            layer1: self.layer1.init(device),
            layer2: self.layer2.init(device),
            layer3: self.layer3.init(device),
            layer4: self.layer4.init(device),
            avgpool: self.avgpool.init(),
            fc: self.fc.init(device),
        }
    }
}
