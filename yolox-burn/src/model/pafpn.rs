use alloc::vec;
use burn::{
    module::Module,
    tensor::{
        backend::Backend,
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
        Device, Tensor,
    },
};

use super::{
    blocks::{expand, BaseConv, BaseConvConfig, Conv, ConvConfig},
    bottleneck::{CspBottleneck, CspBottleneckConfig},
    darknet::{CspDarknet, CspDarknetConfig},
};

pub struct FpnFeatures<B: Backend>(pub Tensor<B, 4>, pub Tensor<B, 4>, pub Tensor<B, 4>);

/// [PAFPN](https://paperswithcode.com/method/pafpn) is the feature pyramid module used in
/// [Path Aggregation Network](https://arxiv.org/abs/1803.01534) that combines FPNs with
/// bottom-up path augmentation.
#[derive(Module, Debug)]
pub struct Pafpn<B: Backend> {
    backbone: CspDarknet<B>,
    lateral_conv0: BaseConv<B>,
    c3_n3: CspBottleneck<B>,
    c3_n4: CspBottleneck<B>,
    c3_p3: CspBottleneck<B>,
    c3_p4: CspBottleneck<B>,
    reduce_conv1: BaseConv<B>,
    bu_conv1: Conv<B>, // bottom-up conv
    bu_conv2: Conv<B>, // bottom-up conv
}

impl<B: Backend> Pafpn<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> FpnFeatures<B> {
        fn upsample<B: Backend>(x_in: Tensor<B, 4>, scale: usize) -> Tensor<B, 4> {
            let [_, _, h, w] = x_in.dims();
            interpolate(
                x_in,
                [h * scale, w * scale],
                InterpolateOptions::new(InterpolateMode::Nearest),
            )
        }

        // Backbone features
        let features = self.backbone.forward(x);

        let fpn_out0 = self.lateral_conv0.forward(features.2);
        let f_out0 = upsample(fpn_out0.clone(), 2);
        let f_out0 = Tensor::cat(vec![f_out0, features.1], 1);
        let f_out0 = self.c3_p4.forward(f_out0);

        let fpn_out1 = self.reduce_conv1.forward(f_out0);
        let f_out1 = upsample(fpn_out1.clone(), 2);
        let f_out1 = Tensor::cat(vec![f_out1, features.0], 1);
        let pan_out2 = self.c3_p3.forward(f_out1);

        let p_out1 = self.bu_conv2.forward(pan_out2.clone());
        let p_out1 = Tensor::cat(vec![p_out1, fpn_out1], 1);
        let pan_out1 = self.c3_n3.forward(p_out1);

        let p_out0 = self.bu_conv1.forward(pan_out1.clone());
        let p_out0 = Tensor::cat(vec![p_out0, fpn_out0], 1);
        let pan_out0 = self.c3_n4.forward(p_out0);

        FpnFeatures(pan_out2, pan_out1, pan_out0)
    }
}

/// [PAFPN block](Pafpn) configuration.
pub struct PafpnConfig {
    backbone: CspDarknetConfig,
    lateral_conv0: BaseConvConfig,
    c3_n3: CspBottleneckConfig,
    c3_n4: CspBottleneckConfig,
    c3_p3: CspBottleneckConfig,
    c3_p4: CspBottleneckConfig,
    reduce_conv1: BaseConvConfig,
    bu_conv1: ConvConfig, // bottom-up conv
    bu_conv2: ConvConfig, // bottom-up conv
}

impl PafpnConfig {
    /// Create a new instance of the PAFPN [config](PafpnConfig).
    pub fn new(depth: f64, width: f64, depthwise: bool) -> Self {
        assert!(
            [0.33, 0.67, 1.0, 1.33].contains(&depth),
            "invalid depth value {depth}"
        );
        assert!(
            [0.25, 0.375, 0.5, 0.75, 1.0, 1.25].contains(&width),
            "invalid width value {width}"
        );

        let in_channels: [usize; 3] = [256, 512, 1024];
        let hidden_channels: [usize; 2] = [
            expand(2 * in_channels[0], width),
            expand(2 * in_channels[1], width),
        ];
        let in_channels: [usize; 3] = [
            expand(in_channels[0], width),
            expand(in_channels[1], width),
            expand(in_channels[2], width),
        ];
        let num_blocks = (3_f64 * depth).round() as usize;

        let backbone = CspDarknetConfig::new(depth, width, depthwise);
        let lateral_conv0 = BaseConvConfig::new(in_channels[2], in_channels[1], 1, 1, 1);
        let c3_p4 = CspBottleneckConfig::new(
            hidden_channels[1],
            in_channels[1],
            num_blocks,
            0.5,
            false,
            depthwise,
        );

        let reduce_conv1 = BaseConvConfig::new(in_channels[1], in_channels[0], 1, 1, 1);
        let c3_p3 = CspBottleneckConfig::new(
            hidden_channels[0],
            in_channels[0],
            num_blocks,
            0.5,
            false,
            depthwise,
        );

        let bu_conv2 = ConvConfig::new(in_channels[0], in_channels[0], 3, 2, depthwise);
        let c3_n3 = CspBottleneckConfig::new(
            hidden_channels[0],
            in_channels[1],
            num_blocks,
            0.5,
            false,
            depthwise,
        );

        let bu_conv1 = ConvConfig::new(in_channels[1], in_channels[1], 3, 2, depthwise);
        let c3_n4 = CspBottleneckConfig::new(
            hidden_channels[1],
            in_channels[2],
            num_blocks,
            0.5,
            false,
            depthwise,
        );

        Self {
            backbone,
            lateral_conv0,
            c3_n3,
            c3_n4,
            c3_p3,
            c3_p4,
            reduce_conv1,
            bu_conv1,
            bu_conv2,
        }
    }

    /// Initialize a new [PAFPN](Pafpn) module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Pafpn<B> {
        Pafpn {
            backbone: self.backbone.init(device),
            lateral_conv0: self.lateral_conv0.init(device),
            c3_n3: self.c3_n3.init(device),
            c3_n4: self.c3_n4.init(device),
            c3_p3: self.c3_p3.init(device),
            c3_p4: self.c3_p4.init(device),
            reduce_conv1: self.reduce_conv1.init(device),
            bu_conv1: self.bu_conv1.init(device),
            bu_conv2: self.bu_conv2.init(device),
        }
    }
}
