use alloc::vec;
use alloc::vec::Vec;
use burn::{
    config::Config,
    module::Module,
    nn::{
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig,
    },
    tensor::{backend::Backend, Tensor},
};

use crate::model::{
    conv_norm::Conv2dNormActivationConfig, inverted_residual::InvertedResidualConfig,
    utils::make_divisble,
};
#[cfg(feature = "pretrained")]
use {
    super::weights::{self, WeightsMeta},
    burn::nn::BatchNormConfig,
    burn::record::Recorder,
    burn::record::{FullPrecisionSettings, Recorder, RecorderError},
    burn::tensor::Device,
    burn_import::pytorch::{LoadArgs, PyTorchFileRecorder},
};

use super::{
    conv_norm::{Conv2dNormActivation, NormalizationType},
    inverted_residual::InvertedResidual,
};

#[derive(Debug, Module)]
pub struct MobileNetV2<B: Backend> {
    features: Vec<ConvBlock<B>>,
    classifier: Vec<ClassifierLayersType<B>>,
    avg_pool: AdaptiveAvgPool2d,
}

#[derive(Debug, Config)]
pub struct MobileNetV2Config {
    #[config(default = "1000")]
    num_classes: usize,

    #[config(default = "1.0")]
    width_mult: f32,

    #[config(default = "vec![vec![]]")]
    inverted_residual_setting: Vec<Vec<usize>>,

    #[config(default = "8")]
    round_nearest: usize,

    norm_layer: NormalizationType,

    #[config(default = "0.2")]
    dropout: f64,
}

#[derive(Module, Debug)]
enum ConvBlock<B: Backend> {
    InvertedResidual(InvertedResidual<B>),
    Conv(Conv2dNormActivation<B>),
}

#[derive(Module, Debug)]
enum ClassifierLayersType<B: Backend> {
    Dropout(Dropout),
    Linear(Linear<B>),
}

impl MobileNetV2Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MobileNetV2<B> {
        let input_channel = 32;
        let last_channel = 1280;
        let mut inverted_residual_setting = self.inverted_residual_setting.clone();
        if inverted_residual_setting.is_empty() {
            inverted_residual_setting = vec![
                // t, c, n, s
                vec![1, 16, 1, 1],
                vec![6, 24, 2, 2],
                vec![6, 32, 3, 2],
                vec![6, 64, 4, 2],
                vec![6, 96, 3, 1],
                vec![6, 160, 3, 2],
                vec![6, 320, 1, 1],
            ]
        }
        if inverted_residual_setting[0].len() != 4 {
            panic!(
                "inverted_residual_setting should be non-empty or a 4-element list, got {:#?}",
                self.inverted_residual_setting
            )
        }
        let mut input_channel = make_divisble(
            input_channel as f32 * self.width_mult,
            self.round_nearest as i32,
            None,
        ) as usize;
        let last_channel = make_divisble(
            last_channel as f32 * f32::max(1.0, self.width_mult),
            self.round_nearest as i32,
            None,
        ) as usize;
        let mut features = vec![ConvBlock::Conv(
            Conv2dNormActivationConfig::new(3, input_channel, self.norm_layer.clone())
                .with_kernel_size(3)
                .with_stride(2)
                .init(device),
        )];
        // building inverted residual blocks
        for setting in inverted_residual_setting {
            let t: usize = setting[0];
            let c = setting[1];
            let n = setting[2];
            let s = setting[3];
            let output_channel =
                make_divisble(c as f32 * self.width_mult, self.round_nearest as i32, None) as usize;
            for i in 0..n {
                let stride = if i == 0 { s } else { 1 };
                features.push(ConvBlock::InvertedResidual(
                    InvertedResidualConfig::new(
                        input_channel,
                        output_channel,
                        stride,
                        t,
                        self.norm_layer.clone(),
                    )
                    .init(device),
                ));
                input_channel = output_channel;
            }
        }
        features.push(ConvBlock::Conv(
            Conv2dNormActivationConfig::new(input_channel, last_channel, self.norm_layer.clone())
                .with_kernel_size(1)
                .init(device),
        ));

        let classifier = vec![
            ClassifierLayersType::Dropout(DropoutConfig::new(self.dropout).init()),
            ClassifierLayersType::Linear(
                LinearConfig::new(last_channel, self.num_classes).init(device),
            ),
        ];

        MobileNetV2 {
            features,
            classifier,
            avg_pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
        }
    }
}
impl<B: Backend> MobileNetV2<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
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
        x = x.flatten(1, 1);
        for layer in &self.classifier {
            match layer {
                ClassifierLayersType::Dropout(dropout) => {
                    x = dropout.forward(x);
                }
                ClassifierLayersType::Linear(linear) => {
                    x = linear.forward(x);
                }
            }
        }
        x
    }
}

// #[cfg(feature = "pretrained")]
// impl<B: Backend> MobileNetV2<B> {
//     /// Load specified pre-trained PyTorch weights as a record.
//     fn load_weights_record(
//         weights: &weights::Weights,
//         device: &Device<B>,
//     ) -> Result<MobileNetV2Record<B>, RecorderError> {
//         // Download torch weights
//         let torch_weights = weights.download().map_err(|err| {
//             RecorderError::Unknown(format!("Could not download weights.\nError: {err}"))
//         })?;
//
//         // Load weights from torch state_dict
//         let load_args = LoadArgs::new(torch_weights)
//             // Map *.downsample.0.* -> *.downsample.conv.*
//             .with_key_remap("(.+)\\.downsample\\.0\\.(.+)", "$1.downsample.conv.$2")
//             // Map *.downsample.1.* -> *.downsample.bn.*
//             .with_key_remap("(.+)\\.downsample\\.1\\.(.+)", "$1.downsample.bn.$2")
//             // Map layer[i].[j].* -> layer[i].blocks.[j].*
//             .with_key_remap("(layer[1-4])\\.([0-9]+)\\.(.+)", "$1.blocks.$2.$3");
//         let record = PyTorchFileRecorder::<FullPrecisionSettings>::new().load(load_args, device)?;
//
//         Ok(record)
//     }
//     pub fn mobilenet_v2_pretrained(
//         weights: weights::MobileNetV2,
//         device: &Device<B>,
//     ) -> Result<Self, RecorderError> {
//         let record = Self::load_weights_record(&weights.weights(), device)?;
//         let model = MobileNetV2Config::new(NormalizationType::BatchNorm(BatchNormConfig::new(12)))
//             .init_with(record);
//         Ok(model)
//     }
// }
