use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, PaddingConfig2d,
    },
    tensor::{self, backend::Backend, Tensor},
};
use serde::{Deserialize, Serialize};
#[derive(Module, Debug, Clone, Default)]
pub struct ReLU6 {}
impl ReLU6 {
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        tensor::activation::relu(input).clamp_max(6)
    }
}
#[derive(Module, Debug)]
pub struct Conv2dNormActivation<B: Backend> {
    conv: Conv2d<B>,
    norm_layer: NormalizationLayer<B, 4>,
    activation: ReLU6,
}
#[derive(Module, Debug)]
pub enum NormalizationLayer<B: Backend, const D: usize> {
    BatchNorm(BatchNorm<B, D>),
}
#[derive(Config, Debug)]
pub struct Conv2dNormActivationConfig {
    pub in_channels: usize,
    pub out_channels: usize,

    #[config(default = "3")]
    pub kernel_size: usize,

    #[config(default = "1")]
    pub stride: usize,

    #[config(default = "0")]
    pub padding: usize,

    #[config(default = "1")]
    pub groups: usize,

    #[config(default = "1")]
    pub dilation: usize,

    #[config(default = false)]
    pub bias: bool,

    pub norm_type: NormalizationType,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationType {
    BatchNorm(BatchNormConfig),
}

impl Conv2dNormActivationConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Conv2dNormActivation<B> {
        let norm_layer = match &self.norm_type {
            NormalizationType::BatchNorm(config) => {
                NormalizationLayer::BatchNorm(config.init(device))
            }
        };
        Conv2dNormActivation {
            conv: Conv2dConfig::new(
                [self.in_channels, self.out_channels],
                [self.kernel_size, self.kernel_size],
            )
            .with_padding(PaddingConfig2d::Explicit(self.padding, self.padding))
            .with_stride([self.stride, self.stride])
            .with_bias(self.bias)
            .with_dilation([self.dilation, self.dilation])
            .with_groups(self.groups)
            .init(device),
            norm_layer,
            activation: ReLU6 {},
        }
    }
}
impl<B: Backend> Conv2dNormActivation<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = match &self.norm_layer {
            NormalizationLayer::BatchNorm(norm) => norm.forward(x),
        };
        self.activation.forward(x)
    }
}
