use burn::module::Module;
use burn::nn::activation::ActivationConfig;
use burn::nn::transformer::{TransformerEncoderConfig, TransformerEncoderLayer};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor};

/// ALBERT encoder with cross-layer parameter sharing.
///
/// Contains a projection from `embedding_size` to `hidden_size` followed by
/// a single `TransformerEncoderLayer` applied `num_hidden_layers` times.
#[derive(Module, Debug)]
pub struct AlbertEncoder<B: Backend> {
    /// Projects from embedding_size to hidden_size.
    pub projection: Linear<B>,
    /// The single shared transformer layer.
    pub layer: TransformerEncoderLayer<B>,
    /// Number of times to apply the shared layer.
    pub num_hidden_layers: usize,
}

impl<B: Backend> AlbertEncoder<B> {
    /// Create a new ALBERT encoder.
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_attention_heads: usize,
        embedding_size: usize,
        num_hidden_layers: usize,
        dropout: f64,
        layer_norm_eps: f64,
        device: &B::Device,
    ) -> Self {
        let projection = LinearConfig::new(embedding_size, hidden_size).init(device);

        let encoder_config = TransformerEncoderConfig::new(
            hidden_size,
            intermediate_size,
            num_attention_heads,
            1, // single layer — we loop manually for weight sharing
        )
        .with_dropout(dropout)
        .with_norm_first(false)
        .with_activation(ActivationConfig::GeluApproximate)
        .with_layer_norm_eps(layer_norm_eps);

        let layer = TransformerEncoderLayer::new(&encoder_config, device);

        Self {
            projection,
            layer,
            num_hidden_layers,
        }
    }

    /// Forward pass: project embeddings, then apply the shared layer `num_hidden_layers` times.
    pub fn forward(&self, x: Tensor<B, 3>, mask_pad: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 3> {
        let mut x = self.projection.forward(x);
        for _ in 0..self.num_hidden_layers {
            x = self.layer.forward(x, mask_pad.clone(), None);
        }
        x
    }
}
