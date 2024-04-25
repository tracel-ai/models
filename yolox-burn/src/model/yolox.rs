use burn::{
    module::{ConstantRecord, Module},
    tensor::{backend::Backend, Device, Tensor},
};

use crate::model::bottleneck::SPP_POOLING;

use super::{
    head::{Head, HeadConfig},
    pafpn::{Pafpn, PafpnConfig},
};

#[cfg(feature = "pretrained")]
use {
    super::weights::{self, WeightsMeta},
    burn::record::{FullPrecisionSettings, Recorder, RecorderError},
    burn_import::pytorch::{LoadArgs, PyTorchFileRecorder},
};

/// [YOLOX](https://paperswithcode.com/method/yolox) object detection architecture.
#[derive(Module, Debug)]
pub struct Yolox<B: Backend> {
    backbone: Pafpn<B>,
    head: Head<B>,
}

impl<B: Backend> Yolox<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let features = self.backbone.forward(x);
        self.head.forward(features)
    }

    /// YOLOX-Nano from [`YOLOX: Exceeding YOLO Series in 2021`](https://arxiv.org/abs/2107.08430).
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A YOLOX-Nano module.
    pub fn yolox_nano(num_classes: usize, device: &Device<B>) -> Self {
        YoloxConfig::new(0.33, 0.25, num_classes, true).init(device)
    }

    /// YOLOX-Nano from [`YOLOX: Exceeding YOLO Series in 2021`](https://arxiv.org/abs/2107.08430)
    /// with pre-trained weights.
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A YOLOX-Nano module with pre-trained weights.
    #[cfg(feature = "pretrained")]
    pub fn yolox_nano_pretrained(
        weights: weights::YoloxNano,
        device: &Device<B>,
    ) -> Result<Self, RecorderError> {
        let weights = weights.weights();
        let record = Self::load_weights_record(&weights, device)?;

        let model = Self::yolox_nano(weights.num_classes, device).load_record(record);

        Ok(model)
    }

    /// YOLOX-Tiny from [`YOLOX: Exceeding YOLO Series in 2021`](https://arxiv.org/abs/2107.08430).
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A YOLOX-Tiny module.
    pub fn yolox_tiny(num_classes: usize, device: &Device<B>) -> Self {
        YoloxConfig::new(0.33, 0.375, num_classes, false).init(device)
    }

    /// YOLOX-Tiny from [`YOLOX: Exceeding YOLO Series in 2021`](https://arxiv.org/abs/2107.08430)
    /// with pre-trained weights.
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A YOLOX-Tiny module with pre-trained weights.
    #[cfg(feature = "pretrained")]
    pub fn yolox_tiny_pretrained(
        weights: weights::YoloxTiny,
        device: &Device<B>,
    ) -> Result<Self, RecorderError> {
        let weights = weights.weights();
        let record = Self::load_weights_record(&weights, device)?;

        let model = Self::yolox_tiny(weights.num_classes, device).load_record(record);

        Ok(model)
    }

    /// YOLOX-S from [`YOLOX: Exceeding YOLO Series in 2021`](https://arxiv.org/abs/2107.08430).
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A YOLOX-S module.
    pub fn yolox_s(num_classes: usize, device: &Device<B>) -> Self {
        YoloxConfig::new(0.33, 0.50, num_classes, false).init(device)
    }

    /// YOLOX-S from [`YOLOX: Exceeding YOLO Series in 2021`](https://arxiv.org/abs/2107.08430)
    /// with pre-trained weights.
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A YOLOX-S module with pre-trained weights.
    #[cfg(feature = "pretrained")]
    pub fn yolox_s_pretrained(
        weights: weights::YoloxS,
        device: &Device<B>,
    ) -> Result<Self, RecorderError> {
        let weights = weights.weights();
        let record = Self::load_weights_record(&weights, device)?;

        let model = Self::yolox_s(weights.num_classes, device).load_record(record);

        Ok(model)
    }

    /// YOLOX-M from [`YOLOX: Exceeding YOLO Series in 2021`](https://arxiv.org/abs/2107.08430).
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A YOLOX-M module.
    pub fn yolox_m(num_classes: usize, device: &Device<B>) -> Self {
        YoloxConfig::new(0.67, 0.75, num_classes, false).init(device)
    }

    /// YOLOX-M from [`YOLOX: Exceeding YOLO Series in 2021`](https://arxiv.org/abs/2107.08430)
    /// with pre-trained weights.
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A YOLOX-M module with pre-trained weights.
    #[cfg(feature = "pretrained")]
    pub fn yolox_m_pretrained(
        weights: weights::YoloxM,
        device: &Device<B>,
    ) -> Result<Self, RecorderError> {
        let weights = weights.weights();
        let record = Self::load_weights_record(&weights, device)?;

        let model = Self::yolox_m(weights.num_classes, device).load_record(record);

        Ok(model)
    }

    /// YOLOX-L from [`YOLOX: Exceeding YOLO Series in 2021`](https://arxiv.org/abs/2107.08430).
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A YOLOX-L module.
    pub fn yolox_l(num_classes: usize, device: &Device<B>) -> Self {
        YoloxConfig::new(1., 1., num_classes, false).init(device)
    }

    /// YOLOX-L from [`YOLOX: Exceeding YOLO Series in 2021`](https://arxiv.org/abs/2107.08430)
    /// with pre-trained weights.
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A YOLOX-L module with pre-trained weights.
    #[cfg(feature = "pretrained")]
    pub fn yolox_l_pretrained(
        weights: weights::YoloxL,
        device: &Device<B>,
    ) -> Result<Self, RecorderError> {
        let weights = weights.weights();
        let record = Self::load_weights_record(&weights, device)?;

        let model = Self::yolox_l(weights.num_classes, device).load_record(record);

        Ok(model)
    }

    /// YOLOX-X from [`YOLOX: Exceeding YOLO Series in 2021`](https://arxiv.org/abs/2107.08430).
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A YOLOX-X module.
    pub fn yolox_x(num_classes: usize, device: &Device<B>) -> Self {
        YoloxConfig::new(1.33, 1.25, num_classes, false).init(device)
    }

    /// YOLOX-X from [`YOLOX: Exceeding YOLO Series in 2021`](https://arxiv.org/abs/2107.08430)
    /// with pre-trained weights.
    ///
    /// # Arguments
    ///
    /// * `num_classes`: Number of output classes of the model.
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A YOLOX-X module with pre-trained weights.
    #[cfg(feature = "pretrained")]
    pub fn yolox_x_pretrained(
        weights: weights::YoloxX,
        device: &Device<B>,
    ) -> Result<Self, RecorderError> {
        let weights = weights.weights();
        let record = Self::load_weights_record(&weights, device)?;

        let model = Self::yolox_x(weights.num_classes, device).load_record(record);

        Ok(model)
    }

    /// Load specified pre-trained PyTorch weights as a record.
    fn load_weights_record(
        weights: &weights::Weights,
        device: &Device<B>,
    ) -> Result<YoloxRecord<B>, RecorderError> {
        // Download torch weights
        let torch_weights = weights.download().map_err(|err| {
            RecorderError::Unknown(format!("Could not download weights.\nError: {err}"))
        })?;

        // Load weights from torch state_dict
        let load_args = LoadArgs::new(torch_weights)
            // State dict contains "model", "amp", "optimizer", "start_epoch"
            .with_top_level_key("model")
            // Map backbone.C3_* -> backbone.c3_*
            .with_key_remap("backbone\\.C3_(.+)", "backbone.c3_$1")
            // Map backbone.backbone.dark[i].0.* -> backbone.backbone.dark[i].conv.*
            .with_key_remap("(backbone\\.backbone\\.dark[2-5])\\.0\\.(.+)", "$1.conv.$2")
            // Map backbone.backbone.dark[i].1.* -> backbone.backbone.dark[i].c3.*
            .with_key_remap("(backbone\\.backbone\\.dark[2-4])\\.1\\.(.+)", "$1.c3.$2")
            // Map backbone.backbone.dark5.1.* -> backbone.backbone.dark5.spp.*
            .with_key_remap("(backbone\\.backbone\\.dark5)\\.1\\.(.+)", "$1.spp.$2")
            // Map backbone.backbone.dark5.2.* -> backbone.backbone.dark5.c3.*
            .with_key_remap("(backbone\\.backbone\\.dark5)\\.2\\.(.+)", "$1.c3.$2")
            // Map head.{cls | reg}_convs.x.[i].* -> head.{cls | reg}_convs.x.conv[i].*
            .with_key_remap(
                "(head\\.(cls|reg)_convs\\.[0-9]+)\\.([0-9]+)\\.(.+)",
                "$1.conv$3.$4",
            );

        let mut record: YoloxRecord<B> =
            PyTorchFileRecorder::<FullPrecisionSettings>::new().load(load_args, device)?;

        if let Some(ref mut spp) = record.backbone.backbone.dark5.spp {
            // Handle the initialization for Vec<MaxPool2d>, which has no parameters.
            // Without this, the vector would be initialized as empty and thus no MaxPool2d
            // layers would be applied, which is incorrect.
            if spp.m.is_empty() {
                spp.m = vec![ConstantRecord; SPP_POOLING.len()];
            }
        }

        Ok(record)
    }
}

/// [YOLOX detector](Yolox) configuration.
pub struct YoloxConfig {
    backbone: PafpnConfig,
    head: HeadConfig,
}

impl YoloxConfig {
    /// Create a new instance of the YOLOX detector [config](YoloxConfig).
    pub fn new(depth: f64, width: f64, num_classes: usize, depthwise: bool) -> Self {
        let backbone = PafpnConfig::new(depth, width, depthwise);
        let head = HeadConfig::new(num_classes, width, depthwise);

        Self { backbone, head }
    }

    /// Initialize a new [YOLOX detector](Yolox) module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolox<B> {
        Yolox {
            backbone: self.backbone.init(device),
            head: self.head.init(device),
        }
    }
}
