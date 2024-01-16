use alloc::vec::Vec;

use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, Initializer, PaddingConfig2d, ReLU,
    },
    tensor::{backend::Backend, Tensor},
};

/// ResNet basic residual block implementation.
/// Derived from [torchivision.models.resnet.BasicBlock](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    relu: ReLU,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    downsample: Option<Downsample<B>>,
}

impl<B: Backend> ResidualBlock<B> {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        // conv3x3
        let conv1 = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false)
            .with_initializer(Initializer::KaimingNormal {
                gain: f64::sqrt(2.0), // recommended value for ReLU
                fan_out_only: false,  // TODO: switch to true when fixed in burn
            })
            .init();
        let bn1 = BatchNormConfig::new(out_channels).init();
        let relu = ReLU::new();
        // conv3x3
        let conv2 = Conv2dConfig::new([out_channels, out_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false)
            .with_initializer(Initializer::KaimingNormal {
                gain: f64::sqrt(2.0), // recommended value for ReLU
                fan_out_only: false,  // TODO: switch to true when fixed in burn
            })
            .init();
        let bn2 = BatchNormConfig::new(out_channels).init();

        let downsample = {
            if in_channels != out_channels {
                Some(Downsample::new(in_channels, out_channels, stride))
            } else {
                None
            }
        };

        Self {
            conv1,
            bn1,
            relu,
            conv2,
            bn2,
            downsample,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let identity = input.clone();

        // Conv block
        let out = self.conv1.forward(input);
        let out = self.bn1.forward(out);
        let out = self.relu.forward(out);
        let out = self.conv2.forward(out);
        let out = self.bn2.forward(out);

        // Skip connection
        let out = {
            match &self.downsample {
                Some(downsample) => out + downsample.forward(identity),
                None => out + identity,
            }
        };

        // Activation
        let out = self.relu.forward(out);

        out
    }
}

/// Downsample layer applies a 1x1 conv to reduce the resolution [H, W] and adjust the number of channels.
#[derive(Module, Debug)]
pub struct Downsample<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B, 2>,
}

impl<B: Backend> Downsample<B> {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        // conv1x1 (default padding = valid)
        let conv = Conv2dConfig::new([in_channels, out_channels], [1, 1])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .with_bias(false)
            .with_initializer(Initializer::KaimingNormal {
                gain: f64::sqrt(2.0), // recommended value for ReLU
                fan_out_only: false,  // TODO: switch to true when fixed in burn
            })
            .init();
        let bn = BatchNormConfig::new(out_channels).init();

        Self { conv, bn }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let out = self.conv.forward(input);
        let out = self.bn.forward(out);

        out
    }
}

/// Collection of sequential residual blocks.
#[derive(Module, Debug)]
pub struct LayerBlock<B: Backend> {
    blocks: Vec<ResidualBlock<B>>,
}

impl<B: Backend> LayerBlock<B> {
    pub fn new(num_blocks: usize, in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let blocks = (0..num_blocks)
            .map(|b| {
                if b == 0 {
                    // First block uses the specified stride
                    ResidualBlock::new(in_channels, out_channels, stride)
                } else {
                    // Other blocks use a stride of 1
                    ResidualBlock::new(out_channels, out_channels, 1)
                }
            })
            .collect();

        Self { blocks }
    }
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut out = input;
        for block in &self.blocks {
            out = block.forward(out);
        }
        out
    }
}
