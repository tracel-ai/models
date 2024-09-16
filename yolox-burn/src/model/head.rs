use alloc::{vec, vec::Vec};
use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Initializer, PaddingConfig2d,
    },
    tensor::{activation::sigmoid, backend::Backend, Device, Int, Shape, Tensor},
};
use itertools::{izip, multiunzip};

use super::{
    blocks::{expand, BaseConv, BaseConvConfig, ConvBlock, ConvBlockConfig},
    pafpn::FpnFeatures,
};

const STRIDES: [usize; 3] = [8, 16, 32];
const IN_CHANNELS: [usize; 3] = [256, 512, 1024];
const PRIOR_PROB: f64 = 1e-2;

/// Create a 2D coordinate grid for the specified dimensions.
/// Similar to [`numpy.indices`](https://numpy.org/doc/stable/reference/generated/numpy.indices.html)
/// but specific to two dimensions.
fn create_2d_grid<B: Backend>(x: usize, y: usize, device: &Device<B>) -> Tensor<B, 3, Int> {
    let y_idx = Tensor::arange(0..y as i64, device)
        .reshape(Shape::new([y, 1]))
        .repeat_dim(1, x)
        .reshape(Shape::new([y, x]));
    let x_idx = Tensor::arange(0..x as i64, device)
        .reshape(Shape::new([1, x])) // can only repeat with dim=1
        .repeat_dim(0, y)
        .reshape(Shape::new([y, x]));

    Tensor::stack(vec![x_idx, y_idx], 2)
}

/// YOLOX head.
#[derive(Module, Debug)]
pub struct Head<B: Backend> {
    stems: Vec<BaseConv<B>>,
    cls_convs: Vec<ConvBlock<B>>,
    reg_convs: Vec<ConvBlock<B>>,
    cls_preds: Vec<Conv2d<B>>,
    reg_preds: Vec<Conv2d<B>>,
    obj_preds: Vec<Conv2d<B>>,
}

impl<B: Backend> Head<B> {
    pub fn forward(&self, x: FpnFeatures<B>) -> Tensor<B, 3> {
        let features: [Tensor<B, 4>; 3] = [x.0, x.1, x.2];

        // Outputs for each feature map
        let (outputs, shapes): (Vec<Tensor<B, 3>>, Vec<(usize, usize)>) = izip!(
            features,
            &self.stems,
            &self.cls_convs,
            &self.cls_preds,
            &self.reg_convs,
            &self.reg_preds,
            &self.obj_preds,
            &STRIDES
        )
        .map(
            |(feat, stem, cls_conv, cls_pred, reg_conv, reg_pred, obj_pred, _stride)| {
                let feat = stem.forward(feat);

                let cls_feat = cls_conv.forward(feat.clone());
                let cls_out = cls_pred.forward(cls_feat);

                let reg_feat = reg_conv.forward(feat);
                let reg_out = reg_pred.forward(reg_feat.clone());

                let obj_out = obj_pred.forward(reg_feat);

                // Output [B, 5 + num_classes, num_anchors]
                let out = Tensor::cat(vec![reg_out, sigmoid(obj_out), sigmoid(cls_out)], 1);
                let [_, _, h, w] = out.dims();
                (out.flatten(2, 3), (h, w))
            },
        )
        .unzip();

        // 1. Concat all regression outputs
        // 2. Permute shape to [B, num_anchors_total, 5 + num_classes]
        // 3. Decode absolute bounding box values
        self.decode(Tensor::cat(outputs, 2).swap_dims(2, 1), shapes.as_ref())
    }

    /// Decode bounding box absolute values from regression output offsets.
    fn decode(&self, outputs: Tensor<B, 3>, shapes: &[(usize, usize)]) -> Tensor<B, 3> {
        let device = outputs.device();
        let [b, num_anchors, num_outputs] = outputs.dims();

        let (grids, strides) = shapes
            .iter()
            .zip(STRIDES)
            .map(|((h, w), stride)| {
                // Grid (x, y) coordinates
                let num_anchors = w * h;
                let grid =
                    create_2d_grid::<B>(*w, *h, &device).reshape(Shape::new([1, num_anchors, 2]));
                let strides: Tensor<B, 3, Int> =
                    Tensor::full(Shape::new([1, num_anchors, 1]), stride as i64, &device);

                (grid, strides)
            })
            .unzip();

        let grids = Tensor::cat(grids, 1).float();
        let strides = Tensor::cat(strides, 1).float();

        Tensor::cat(
            vec![
                // Add grid offset to center coordinates and scale to image dimensions
                (outputs.clone().slice([0..b, 0..num_anchors, 0..2]) + grids) * strides.clone(),
                // Decode `log` encoded boxes with `exp`and scale to image dimensions
                outputs.clone().slice([0..b, 0..num_anchors, 2..4]).exp() * strides,
                // Classification outputs
                outputs.slice([0..b, 0..num_anchors, 4..num_outputs]),
            ],
            2,
        )
    }
}

/// [YOLOX head](Head) configuration.
pub struct HeadConfig {
    stems: Vec<BaseConvConfig>,
    cls_convs: Vec<ConvBlockConfig>,
    reg_convs: Vec<ConvBlockConfig>,
    cls_preds: Vec<Conv2dConfig>,
    reg_preds: Vec<Conv2dConfig>,
    obj_preds: Vec<Conv2dConfig>,
}

impl HeadConfig {
    /// Create a new instance of the YOLOX head [config](HeadConfig).
    pub fn new(num_classes: usize, width: f64, depthwise: bool) -> Self {
        let hidden_channels: usize = 256;
        // Initialize conv2d biases for classification and objectness heads
        let bias = -f64::ln((1.0 - PRIOR_PROB) / PRIOR_PROB);

        let (stems, cls_convs, reg_convs, cls_preds, reg_preds, obj_preds) =
            multiunzip(IN_CHANNELS.into_iter().map(|in_channels| {
                let stem = BaseConvConfig::new(
                    expand(in_channels, width),
                    expand(hidden_channels, width),
                    1,
                    1,
                    1,
                );

                let cls_conv =
                    ConvBlockConfig::new(expand(hidden_channels, width), 3, 1, depthwise);
                let reg_conv =
                    ConvBlockConfig::new(expand(hidden_channels, width), 3, 1, depthwise);

                let cls_pred =
                    Conv2dConfig::new([expand(hidden_channels, width), num_classes], [1, 1])
                        .with_padding(PaddingConfig2d::Explicit(0, 0))
                        .with_initializer(Initializer::Constant { value: bias });
                let reg_pred = Conv2dConfig::new([expand(hidden_channels, width), 4], [1, 1])
                    .with_padding(PaddingConfig2d::Explicit(0, 0));
                let obj_pred = Conv2dConfig::new([expand(hidden_channels, width), 1], [1, 1])
                    .with_padding(PaddingConfig2d::Explicit(0, 0))
                    .with_initializer(Initializer::Constant { value: bias });

                (stem, cls_conv, reg_conv, cls_pred, reg_pred, obj_pred)
            }));

        Self {
            stems,
            cls_convs,
            reg_convs,
            cls_preds,
            reg_preds,
            obj_preds,
        }
    }

    /// Initialize a new [YOLOX head](Head) module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Head<B> {
        Head {
            stems: self.stems.iter().map(|m| m.init(device)).collect(),
            cls_convs: self.cls_convs.iter().map(|m| m.init(device)).collect(),
            reg_convs: self.reg_convs.iter().map(|m| m.init(device)).collect(),
            cls_preds: self.cls_preds.iter().map(|m| m.init(device)).collect(),
            reg_preds: self.reg_preds.iter().map(|m| m.init(device)).collect(),
            obj_preds: self.obj_preds.iter().map(|m| m.init(device)).collect(),
        }
    }
}
