use alloc::vec;
use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, PaddingConfig2d,
    },
    tensor::{activation::silu, backend::Backend, Device, Tensor},
};

/// Compute the number of channels based on the provided factor.
pub fn expand(num_channels: usize, factor: f64) -> usize {
    (num_channels as f64 * factor).floor() as usize
}

/// A base convolution block.
/// Allows to switch between regular and depthwise separable convolution blocks based on the
/// architecture.
#[derive(Module, Debug)]
pub enum Conv<B: Backend> {
    /// Basic convolution block used for all variants.
    BaseConv(BaseConv<B>),
    /// Depthwise separable convolution block, used for some blocks by YOLOX-Nano.
    DwsConv(DwsConv<B>),
}

impl<B: Backend> Conv<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Self::BaseConv(conv) => conv.forward(x),
            Self::DwsConv(conv) => conv.forward(x),
        }
    }
}

#[derive(Config)]
pub struct ConvConfig {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    depthwise: bool,
}

impl ConvConfig {
    /// Initialize a new [convolution block](Conv) module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Conv<B> {
        if self.depthwise {
            Conv::DwsConv(
                DwsConvConfig::new(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    self.stride,
                )
                .init(device),
            )
        } else {
            Conv::BaseConv(
                BaseConvConfig::new(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    self.stride,
                    1,
                )
                .init(device),
            )
        }
    }
}

/// A Conv2d -> BatchNorm -> activation block.
#[derive(Module, Debug)]
pub struct BaseConv<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B, 2>,
}

impl<B: Backend> BaseConv<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = self.bn.forward(x);

        silu(x)
    }
}

/// [Base convolution block](BaseConv) configuration.
pub struct BaseConvConfig {
    conv: Conv2dConfig,
    bn: BatchNormConfig,
}

impl BaseConvConfig {
    /// Create a new instance of the base convolution block [config](BaseConvConfig).
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
    ) -> Self {
        // Same padding
        let pad = (kernel_size - 1) / 2;

        let conv = Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(pad, pad))
            .with_groups(groups)
            .with_bias(false);
        let bn = BatchNormConfig::new(out_channels)
            .with_epsilon(1e-3)
            .with_momentum(0.03);

        Self { conv, bn }
    }

    /// Initialize a new [base convolution block](BaseConv) module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> BaseConv<B> {
        BaseConv {
            conv: self.conv.init(device),
            bn: self.bn.init(device),
        }
    }
}

/// A [depthwise separable convolution](https://paperswithcode.com/method/depthwise-separable-convolution)
/// block. Both depthwise and pointwise blocks consist of a Conv2d -> BatchNorm -> activation block.
#[derive(Module, Debug)]
pub struct DwsConv<B: Backend> {
    dconv: BaseConv<B>,
    pconv: BaseConv<B>,
}

impl<B: Backend> DwsConv<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.dconv.forward(x);
        self.pconv.forward(x)
    }
}

/// [Depthwise separable convolution block](DwsConv) configuration.
pub struct DwsConvConfig {
    dconv: BaseConvConfig,
    pconv: BaseConvConfig,
}

impl DwsConvConfig {
    /// Create a new instance of the depthwise separable convolution block [config](DwsConvConfig).
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize) -> Self {
        // Depthwise conv
        let dconv = BaseConvConfig::new(in_channels, in_channels, kernel_size, stride, in_channels);
        // Pointwise conv
        let pconv = BaseConvConfig::new(in_channels, out_channels, 1, 1, 1);

        Self { dconv, pconv }
    }

    /// Initialize a new [depthwise separable convolution block](DwsConv) module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> DwsConv<B> {
        DwsConv {
            dconv: self.dconv.init(device),
            pconv: self.pconv.init(device),
        }
    }
}

/// Focus width and height information into channel space.
#[derive(Module, Debug)]
pub struct Focus<B: Backend> {
    conv: BaseConv<B>,
}

impl<B: Backend> Focus<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let device = x.device();
        let [_, _, h, w] = x.dims();

        // Indexing
        let top_idx = Tensor::arange_step(0..h as i64, 2, &device);
        let bottom_idx = Tensor::arange_step(1..h as i64, 2, &device);
        let left_idx = Tensor::arange_step(0..w as i64, 2, &device);
        let right_idx = Tensor::arange_step(1..w as i64, 2, &device);

        // patch_top_left = x[..., ::2, ::2]
        let patch_top_left = x
            .clone()
            .select(2, top_idx.clone())
            .select(3, left_idx.clone());
        // patch_top_right = x[..., ::2, 1::2]
        let patch_top_right = x.clone().select(2, top_idx).select(3, right_idx.clone());
        // patch_bot_left = x[..., 1::2, ::2]
        let patch_bottom_left = x.clone().select(2, bottom_idx.clone()).select(3, left_idx);
        // patch_bot_right = x[..., 1::2, 1::2]
        let patch_bottom_right = x.select(2, bottom_idx).select(3, right_idx);

        // Shape (b,c,w,h) -> y(b,4c,w/2,h/2)
        let x = Tensor::cat(
            vec![
                patch_top_left,
                patch_bottom_left,
                patch_top_right,
                patch_bottom_right,
            ],
            1,
        );

        self.conv.forward(x)
    }
}

/// [Focus block](Focus) configuration.
pub struct FocusConfig {
    conv: BaseConvConfig,
}

impl FocusConfig {
    /// Create a new instance of the focus block [config](FocusConfig).
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize) -> Self {
        let conv = BaseConvConfig::new(in_channels * 4, out_channels, kernel_size, stride, 1);

        Self { conv }
    }

    /// Initialize a new [focus block](Focus) module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Focus<B> {
        Focus {
            conv: self.conv.init(device),
        }
    }
}

/// Dual convolution block used for feature extraction in the prediction head.
#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv0: Conv<B>,
    conv1: Conv<B>,
}

impl<B: Backend> ConvBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv0.forward(x);
        self.conv1.forward(x)
    }
}

/// [Dual convolution block](ConvBlock) configuration.
pub struct ConvBlockConfig {
    conv0: ConvConfig,
    conv1: ConvConfig,
}

impl ConvBlockConfig {
    /// Create a new instance of the dual convolution block [config](ConvBlockConfig).
    pub fn new(channels: usize, kernel_size: usize, stride: usize, depthwise: bool) -> Self {
        let conv0 = ConvConfig::new(channels, channels, kernel_size, stride, depthwise);
        let conv1 = ConvConfig::new(channels, channels, kernel_size, stride, depthwise);

        Self { conv0, conv1 }
    }

    /// Initialize a new [dual convolution block](ConvBlock) module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> ConvBlock<B> {
        ConvBlock {
            conv0: self.conv0.init(device),
            conv1: self.conv1.init(device),
        }
    }
}
