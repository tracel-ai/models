use super::conv_norm::{Conv2dNormActivation, Conv2dNormActivationConfig};
use burn::config::Config;
use burn::nn::conv::Conv2dConfig;
use burn::nn::{BatchNorm, BatchNormConfig};
use burn::tensor::Tensor;
use burn::{module::Module, nn::conv::Conv2d, tensor::backend::Backend};

#[derive(Module, Debug)]
pub struct PointWiseLinear<B: Backend> {
    conv: Conv2d<B>,
    norm: BatchNorm<B, 2>,
}

impl<B: Backend> PointWiseLinear<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.norm.forward(self.conv.forward(x))
    }
}

/// [Inverted Residual Block](https://paperswithcode.com/method/inverted-residual-block).
#[derive(Module, Debug)]
pub struct InvertedResidual<B: Backend> {
    use_res_connect: bool,
    pw: Option<Conv2dNormActivation<B>>, // pointwise, only when expand ratio != 1
    dw: Conv2dNormActivation<B>,
    pw_linear: PointWiseLinear<B>,
}

/// [InvertedResidual](InvertedResidual) configuration.
#[derive(Config, Debug)]
pub struct InvertedResidualConfig {
    pub inp: usize,
    pub oup: usize,
    pub stride: usize,
    pub expand_ratio: usize,
}

impl InvertedResidualConfig {
    /// Initialize a new [InvertedResidual](InvertedResidual) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> InvertedResidual<B> {
        let hidden_dim = self.inp * self.expand_ratio;
        let pw = if self.expand_ratio != 1 {
            Some(
                Conv2dNormActivationConfig::new(self.inp, hidden_dim)
                    .with_kernel_size(1)
                    .init(device),
            )
        } else {
            None
        };
        let dw = Conv2dNormActivationConfig::new(hidden_dim, hidden_dim)
            .with_stride(self.stride)
            .with_groups(hidden_dim)
            .init(device);
        let pw_linear = PointWiseLinear {
            conv: Conv2dConfig::new([hidden_dim, self.oup], [1, 1])
                .with_stride([1, 1])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(0, 0))
                .with_bias(false)
                .init(device),
            norm: BatchNormConfig::new(self.oup).init(device),
        };
        InvertedResidual {
            use_res_connect: self.stride == 1 && self.inp == self.oup,
            pw_linear,
            dw,
            pw,
        }
    }
}

impl<B: Backend> InvertedResidual<B> {
    pub fn forward(&self, x: &Tensor<B, 4>) -> Tensor<B, 4> {
        let mut out = x.clone();
        if let Some(pw) = &self.pw {
            out = pw.forward(out);
        }
        out = self.dw.forward(out);
        out = self.pw_linear.forward(out);

        if self.use_res_connect {
            out = out + x.clone();
        }
        out
    }
}
