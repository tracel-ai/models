use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig, Initializer, Linear, LinearConfig, PaddingConfig2d, ReLU,
    },
    tensor::{backend::Backend, Tensor},
};

use super::block::LayerBlock;

/// ResNet implementation.
/// Derived from [torchivision.models.resnet.ResNet](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
#[derive(Module, Debug)]
pub struct ResNet<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    relu: ReLU,
    maxpool: MaxPool2d,
    layer1: LayerBlock<B>,
    layer2: LayerBlock<B>,
    layer3: LayerBlock<B>,
    layer4: LayerBlock<B>,
    avgpool: AdaptiveAvgPool2d,
    fc: Linear<B>,
}

impl<B: Backend> ResNet<B> {
    fn new(blocks: [usize; 4], num_classes: usize) -> Self {
        // 7x7 conv, 64, /2
        let conv1 = Conv2dConfig::new([3, 64], [7, 7])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(3, 3))
            .with_bias(false)
            .with_initializer(Initializer::KaimingNormal {
                gain: f64::sqrt(2.0), // recommended value for ReLU
                fan_out_only: false,  // TODO: switch to true when fixed in burn
            })
            .init();
        let bn1 = BatchNormConfig::new(64).init();
        let relu = ReLU::new();
        // 3x3 maxpool, /2
        let maxpool = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init();

        // Residual blocks
        let layer1 = LayerBlock::new(blocks[0], 64, 64);
        let layer2 = LayerBlock::new(blocks[1], 64, 128);
        let layer3 = LayerBlock::new(blocks[2], 128, 256);
        let layer4 = LayerBlock::new(blocks[2], 256, 512);

        // Average pooling [B, 512, H, W] -> [B, 512, 1, 1]
        let avgpool = AdaptiveAvgPool2dConfig::new([1, 1]).init();

        // Output layer
        let fc = LinearConfig::new(512, num_classes).init();

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

    pub fn resnet18(num_classes: usize) -> Self {
        Self::new([2, 2, 2, 2], num_classes)
    }

    pub fn resnet34(num_classes: usize) -> Self {
        Self::new([3, 4, 6, 3], num_classes)
    }

    // TODO: resnet{50, 101, 152} use different blocks

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
        // println!("Flatten in: {:?}", out.shape());
        // let out = out.flatten(2, 3);
        let out: Tensor<B, 3> = out.squeeze(3);
        let out: Tensor<B, 2> = out.squeeze(2);
        let out = self.fc.forward(out);

        out
    }
}
