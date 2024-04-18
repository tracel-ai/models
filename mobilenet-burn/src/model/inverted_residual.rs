use super::conv_norm::NormalizationLayer;
use super::conv_norm::NormalizationType;
use super::conv_norm::{Conv2dNormActivation, Conv2dNormActivationConfig};
use alloc::vec;
use alloc::vec::Vec;
use burn::config::Config;
use burn::nn::conv::Conv2dConfig;
use burn::nn::BatchNormConfig;
use burn::tensor::Tensor;
use burn::{module::Module, nn::conv::Conv2d, tensor::backend::Backend};
#[derive(Module, Debug)]
enum InvertedResidualSequentialType<B: Backend> {
    Conv(Conv2d<B>),
    ConvNormActivation(Conv2dNormActivation<B>),
    NormLayer(NormalizationLayer<B, 4>),
}
/// Inverted Residual Block
/// Ref: https://paperswithcode.com/method/inverted-residual-block
#[derive(Module, Debug)]
pub struct InvertedResidual<B: Backend> {
    use_res_connect: bool,
    layers: Vec<InvertedResidualSequentialType<B>>,
}

#[derive(Config, Debug)]
pub struct InvertedResidualConfig {
    pub inp: usize,
    pub oup: usize,
    pub stride: usize,
    pub expand_ratio: usize,
    pub norm_type: NormalizationType,
}

impl InvertedResidualConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> InvertedResidual<B> {
        let mut layers = Vec::new();
        let hidden_dim = self.inp * self.expand_ratio;
        if self.expand_ratio != 1 {
            layers.push(InvertedResidualSequentialType::ConvNormActivation(
                Conv2dNormActivationConfig::new(self.inp, hidden_dim, self.norm_type.clone())
                    .with_kernel_size(1)
                    .init(device),
            ));
        }
        let mut temp_layer: Vec<InvertedResidualSequentialType<B>> = vec![
            InvertedResidualSequentialType::ConvNormActivation(
                Conv2dNormActivationConfig::new(hidden_dim, hidden_dim, self.norm_type.clone())
                    .with_stride(self.stride)
                    .with_groups(hidden_dim)
                    .init(device),
            ),
            InvertedResidualSequentialType::Conv(
                Conv2dConfig::new([hidden_dim, self.oup], [1, 1])
                    .with_stride([1, 1])
                    .with_padding(burn::nn::PaddingConfig2d::Explicit(0, 0))
                    .with_bias(false)
                    .init(device),
            ),
            match self.norm_type {
                NormalizationType::BatchNorm(_) => InvertedResidualSequentialType::NormLayer(
                    NormalizationLayer::BatchNorm(BatchNormConfig::new(self.oup).init(device)),
                ),
            },
        ];
        layers.append(&mut temp_layer);
        InvertedResidual {
            use_res_connect: self.stride == 1 && self.inp == self.oup,
            layers,
        }
    }
}
impl<B: Backend> InvertedResidual<B> {
    pub fn forward(&self, x: &Tensor<B, 4>) -> Tensor<B, 4> {
        let mut out = x.clone();
        for layer in &self.layers {
            match layer {
                InvertedResidualSequentialType::Conv(conv) => out = conv.forward(out),
                InvertedResidualSequentialType::ConvNormActivation(conv_norm) => {
                    out = conv_norm.forward(out)
                }
                InvertedResidualSequentialType::NormLayer(NormalizationLayer::BatchNorm(x)) => {
                    out = x.forward(out)
                }
            }
        }
        if self.use_res_connect {
            out = out + x.clone();
        }
        out
    }
}
