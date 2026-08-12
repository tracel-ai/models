use burn::tensor::Device;
use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, PaddingConfig2d,
    },
    tensor::{self, Tensor},
};

/// A rectified linear unit where the activation is limited to a maximum of 6.
#[derive(Module, Debug, Default)]
pub struct ReLU6 {}
impl ReLU6 {
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        tensor::activation::relu(input).clamp_max(6)
    }
}

/// A Conv2d -> BatchNorm -> activation block.
#[derive(Module, Debug)]
pub struct Conv2dNormActivation {
    conv: Conv2d,
    norm: BatchNorm,
    activation: ReLU6,
}

/// [Conv2dNormActivation] configuration.
#[derive(Config, Debug)]
pub struct Conv2dNormActivationConfig {
    pub in_channels: usize,
    pub out_channels: usize,

    #[config(default = "3")]
    pub kernel_size: usize,

    #[config(default = "1")]
    pub stride: usize,

    #[config(default = "None")]
    pub padding: Option<usize>,

    #[config(default = "1")]
    pub groups: usize,

    #[config(default = "1")]
    pub dilation: usize,

    #[config(default = false)]
    pub bias: bool,
}

impl Conv2dNormActivationConfig {
    pub fn init(&self, device: &Device) -> Conv2dNormActivation {
        let padding = if let Some(padding) = self.padding {
            padding
        } else {
            (self.kernel_size - 1) / 2 * self.dilation
        };

        Conv2dNormActivation {
            conv: Conv2dConfig::new(
                [self.in_channels, self.out_channels],
                [self.kernel_size, self.kernel_size],
            )
            .with_padding(PaddingConfig2d::Explicit(
                padding, padding, padding, padding,
            ))
            .with_stride([self.stride, self.stride])
            .with_bias(self.bias)
            .with_dilation([self.dilation, self.dilation])
            .with_groups(self.groups)
            .init(device),
            norm: BatchNormConfig::new(self.out_channels).init(device),
            activation: ReLU6 {},
        }
    }
}
impl Conv2dNormActivation {
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);
        self.activation.forward(x)
    }
}
