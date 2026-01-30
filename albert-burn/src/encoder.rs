use burn::module::Module;
use burn::nn::transformer::{TransformerEncoderConfig, TransformerEncoderLayer};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor};

/// ALBERT encoder with cross-layer parameter sharing.
///
/// Contains a single `TransformerEncoderLayer` that is applied
/// `num_hidden_layers` times, sharing all parameters across layers.
#[derive(Module, Debug)]
pub struct AlbertEncoder<B: Backend> {
    /// The single shared transformer layer.
    pub layer: TransformerEncoderLayer<B>,
    /// Number of times to apply the shared layer.
    pub num_hidden_layers: usize,
}

impl<B: Backend> AlbertEncoder<B> {
    /// Create a new ALBERT encoder from a transformer encoder config.
    ///
    /// The config's `n_layers` field is ignored; `num_hidden_layers` controls repetition.
    pub fn new(
        config: &TransformerEncoderConfig,
        num_hidden_layers: usize,
        device: &B::Device,
    ) -> Self {
        let layer = TransformerEncoderLayer::new(config, device);
        Self {
            layer,
            num_hidden_layers,
        }
    }

    /// Forward pass: apply the shared layer `num_hidden_layers` times.
    pub fn forward(
        &self,
        mut x: Tensor<B, 3>,
        mask_pad: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        for _ in 0..self.num_hidden_layers {
            x = self.layer.forward(x, mask_pad.clone(), None);
        }
        x
    }
}
