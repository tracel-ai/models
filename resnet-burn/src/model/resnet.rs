use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig, Initializer, Linear, LinearConfig, PaddingConfig2d, ReLU,
    },
    tensor::{backend::Backend, Device, Tensor},
};

use super::block::{BasicBlock, Bottleneck, LayerBlock, ResidualBlock};

/// ResNet implementation.
/// Derived from [torchivision.models.resnet.ResNet](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
#[derive(Module, Debug)]
pub struct ResNet<B: Backend, M> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    relu: ReLU,
    maxpool: MaxPool2d,
    layer1: LayerBlock<B, M>,
    layer2: LayerBlock<B, M>,
    layer3: LayerBlock<B, M>,
    layer4: LayerBlock<B, M>,
    avgpool: AdaptiveAvgPool2d,
    fc: Linear<B>,
}

impl<B: Backend, M: ResidualBlock<B>> ResNet<B, M> {
    fn new(blocks: [usize; 4], num_classes: usize, expansion: usize, device: &Device<B>) -> Self {
        // `new()` is private but still check just in case...
        assert!(
            expansion == 1 || expansion == 4,
            "ResNet module only supports expansion values [1, 4] for residual blocks"
        );

        // 7x7 conv, 64, /2
        let conv1 = Conv2dConfig::new([3, 64], [7, 7])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(3, 3))
            .with_bias(false)
            .with_initializer(Initializer::KaimingNormal {
                gain: f64::sqrt(2.0), // recommended value for ReLU
                fan_out_only: true,
            })
            .init(device);
        let bn1 = BatchNormConfig::new(64).init(device);
        let relu = ReLU::new();
        // 3x3 maxpool, /2
        let maxpool = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init();

        // Residual blocks
        let layer1 = LayerBlock::new(blocks[0], 64, 64 * expansion, 1, device);
        let layer2 = LayerBlock::new(blocks[1], 64 * expansion, 128 * expansion, 2, device);
        let layer3 = LayerBlock::new(blocks[2], 128 * expansion, 256 * expansion, 2, device);
        let layer4 = LayerBlock::new(blocks[3], 256 * expansion, 512 * expansion, 2, device);

        // Average pooling [B, 512 * expansion, H, W] -> [B, 512 * expansion, 1, 1]
        let avgpool = AdaptiveAvgPool2dConfig::new([1, 1]).init();

        // Output layer
        let fc = LinearConfig::new(512 * expansion, num_classes).init(device);

        Self {
            conv1,
            bn1,
            relu,
            maxpool,
            layer1,
            layer2,
            layer3,
            layer4,
            avgpool,
            fc,
        }
    }

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
        // let out: Tensor<B, 3> = out.squeeze(3);
        // let out: Tensor<B, 2> = out.squeeze(2);

        self.fc.forward(out)
    }
}

impl<B: Backend> ResNet<B, BasicBlock<B>> {
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
        Self::new([2, 2, 2, 2], num_classes, 1, device)
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
        Self::new([3, 4, 6, 3], num_classes, 1, device)
    }
}

impl<B: Backend> ResNet<B, Bottleneck<B>> {
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
        Self::new([3, 4, 6, 3], num_classes, 4, device)
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
        Self::new([3, 4, 23, 3], num_classes, 4, device)
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
        Self::new([3, 8, 36, 3], num_classes, 4, device)
    }
}
