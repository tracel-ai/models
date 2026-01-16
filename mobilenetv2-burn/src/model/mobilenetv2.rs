use alloc::vec;
use alloc::vec::Vec;
use core::cmp::max;

use burn::{
    config::Config,
    module::Module,
    nn::{
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig,
    },
    tensor::{backend::Backend, Tensor},
};

use super::{
    conv_norm::{Conv2dNormActivation, Conv2dNormActivationConfig},
    inverted_residual::{InvertedResidual, InvertedResidualConfig},
};

#[cfg(feature = "pretrained")]
use {
    super::weights::{self, WeightsMeta},
    burn::tensor::Device,
    burn_store::{ModuleSnapshot, PytorchStore, PytorchStoreError},
};

/// Network blocks structure
const INVERTED_RESIDUAL_SETTINGS: [[usize; 4]; 7] = [
    // (t = expansion factor; c = channels; n = num blocks; s = stride)
    // t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
];
/// Round the number of channels in each layer to be a multiple of this number.
const ROUND_NEAREST: usize = 8;

#[derive(Debug, Module)]
pub struct MobileNetV2<B: Backend> {
    features: Vec<ConvBlock<B>>,
    classifier: Classifier<B>,
    avg_pool: AdaptiveAvgPool2d,
}

impl<B: Backend> MobileNetV2<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut x = input;
        for layer in &self.features {
            match layer {
                ConvBlock::InvertedResidual(block) => {
                    x = block.forward(&x);
                }
                ConvBlock::Conv(conv) => {
                    x = conv.forward(x);
                }
            }
        }
        x = self.avg_pool.forward(x);
        // Reshape [B, C, 1, 1] -> [B, C]
        let x = x.flatten(1, 3);

        self.classifier.forward(x)
    }

    /// Load specified pre-trained PyTorch weights into the model.
    #[cfg(feature = "pretrained")]
    fn load_weights(model: &mut Self, weights: &weights::Weights) -> Result<(), PytorchStoreError> {
        // Download torch weights
        let torch_weights = weights.download().map_err(|err| {
            PytorchStoreError::Other(format!("Could not download weights.\nError: {err}"))
        })?;

        // Load weights from torch state_dict
        let mut store = PytorchStore::from_file(torch_weights)
            // Map features.{0,18}.0.* -> features.{0,18}.conv.*
            .with_key_remapping("features\\.(0|18)\\.0.(.+)", "features.$1.conv.$2")
            // Map features.{0,18}.1.* -> features.{0,18}.norm.*
            .with_key_remapping("features\\.(0|18)\\.1.(.+)", "features.$1.norm.$2")
            // Map features.1.conv.0.0.* -> features.1.dw.conv.*
            .with_key_remapping("features\\.1\\.conv.0.0.(.+)", "features.1.dw.conv.$1")
            // Map features.1.conv.0.1.* -> features.1.dw.conv.*
            .with_key_remapping("features\\.1\\.conv.0.1.(.+)", "features.1.dw.norm.$1")
            // Map features.1.conv.1.* -> features.1.pw_linear.conv.*
            .with_key_remapping("features\\.1\\.conv.1.(.+)", "features.1.pw_linear.conv.$1")
            // Map features.1.conv.2.* -> features.1.pw_linear.norm.*
            .with_key_remapping("features\\.1\\.conv.2.(.+)", "features.1.pw_linear.norm.$1")
            // Map features.[i].conv.0.0.* -> features.[i].pw.conv.*
            .with_key_remapping(
                "features\\.([2-9]|1[0-7])\\.conv.0.0.(.+)", // for i in [2, 17]
                "features.$1.pw.conv.$2",
            )
            // Map features.[i].conv.0.1.* -> features.[i].pw.conv.*
            .with_key_remapping(
                "features\\.([2-9]|1[0-7])\\.conv.0.1.(.+)", // for i in [2, 17]
                "features.$1.pw.norm.$2",
            )
            // Map features.[i].conv.1.0.* -> features.[i].dw.conv.*
            .with_key_remapping(
                "features\\.([2-9]|1[0-7])\\.conv.1.0.(.+)", // for i in [2, 17]
                "features.$1.dw.conv.$2",
            )
            // Map features.[i].conv.1.1.* -> features.[i].dw.norm.*
            .with_key_remapping(
                "features\\.([2-9]|1[0-7])\\.conv.1.1.(.+)", // for i in [2, 17]
                "features.$1.dw.norm.$2",
            )
            // Map features.[i].conv.2.* -> features.[i].pw_linear.conv.*
            .with_key_remapping(
                "features\\.([2-9]|1[0-7])\\.conv.2.(.+)", // for i in [2, 17]
                "features.$1.pw_linear.conv.$2",
            )
            // Map features.[i].conv.3.* -> features.[i].pw_linear.norm.*
            .with_key_remapping(
                "features\\.([2-9]|1[0-7])\\.conv.3.(.+)", // for i in [2, 17]
                "features.$1.pw_linear.norm.$2",
            )
            // Map classifier.1.* -> classifier.linear.*
            .with_key_remapping("classifier.1.(.+)", "classifier.linear.$1");

        model.load_from(&mut store)?;

        Ok(())
    }

    /// MobileNetV2 from [`MobileNetV2: Inverted Residuals and Linear Bottlenecks`](https://arxiv.org/abs/1801.04381)
    /// with pre-trained weights.
    ///
    /// # Arguments
    ///
    /// * `weights`: Pre-trained weights to load.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A MobileNetV2 module with pre-trained weights.
    #[cfg(feature = "pretrained")]
    pub fn pretrained(
        weights: weights::MobileNetV2,
        device: &Device<B>,
    ) -> Result<Self, PytorchStoreError> {
        let weights = weights.weights();
        let mut model = MobileNetV2Config::new()
            .with_num_classes(weights.num_classes)
            .init(device);
        Self::load_weights(&mut model, &weights)?;
        Ok(model)
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
enum ConvBlock<B: Backend> {
    InvertedResidual(InvertedResidual<B>),
    Conv(Conv2dNormActivation<B>),
}

#[derive(Module, Debug)]
struct Classifier<B: Backend> {
    dropout: Dropout,
    linear: Linear<B>,
}
impl<B: Backend> Classifier<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.dropout.forward(input);
        self.linear.forward(x)
    }
}

/// MobileNetV2 from [`MobileNetV2: Inverted Residuals and Linear Bottlenecks`](https://arxiv.org/abs/1801.04381).
#[derive(Debug, Config)]
pub struct MobileNetV2Config {
    #[config(default = "1000")]
    num_classes: usize,

    #[config(default = "1.0")]
    width_mult: f32,

    #[config(default = "0.2")]
    dropout: f64,
}

impl MobileNetV2Config {
    /// Initialize a MobileNetV2 from
    /// [`MobileNetV2: Inverted Residuals and Linear Bottlenecks`](https://arxiv.org/abs/1801.04381).
    ///
    /// # Arguments
    ///
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A MobileNetV2 module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MobileNetV2<B> {
        let input_channel = 32;
        let last_channel = 1280;

        let make_divisible = |v, divisor| {
            let new_v = (v + divisor as f32 / 2.0) as usize / divisor * divisor;
            let mut new_v = max(new_v, divisor);

            // Make sure that round down does not go down by more than 10%
            if (new_v as f32) < 0.9 * v {
                new_v += divisor;
            }

            new_v
        };

        let mut input_channel =
            make_divisible(input_channel as f32 * self.width_mult, ROUND_NEAREST);
        let last_channel = make_divisible(
            last_channel as f32 * f32::max(1.0, self.width_mult),
            ROUND_NEAREST,
        );

        // Feature extraction layers with inverted residual blocks
        let mut features = vec![ConvBlock::Conv(
            Conv2dNormActivationConfig::new(3, input_channel)
                .with_kernel_size(3)
                .with_stride(2)
                .init(device),
        )];
        for [t, c, n, s] in INVERTED_RESIDUAL_SETTINGS.into_iter() {
            let output_channel = make_divisible(c as f32 * self.width_mult, ROUND_NEAREST);
            for i in 0..n {
                let stride = if i == 0 { s } else { 1 };
                features.push(ConvBlock::InvertedResidual(
                    InvertedResidualConfig::new(input_channel, output_channel, stride, t)
                        .init(device),
                ));
                input_channel = output_channel;
            }
        }
        features.push(ConvBlock::Conv(
            Conv2dNormActivationConfig::new(input_channel, last_channel)
                .with_kernel_size(1)
                .init(device),
        ));

        let classifier = Classifier {
            dropout: DropoutConfig::new(self.dropout).init(),
            linear: LinearConfig::new(last_channel, self.num_classes).init(device),
        };

        MobileNetV2 {
            features,
            classifier,
            avg_pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
        }
    }
}
