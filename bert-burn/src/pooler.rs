use burn::tensor::Device;
use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::Tensor,
};
use derive_new::new;

/// Pooler
#[derive(Module, Debug, new)]
pub struct Pooler {
    /// Linear output
    output: Linear,
}

impl Pooler {
    /// Forward pass
    pub fn forward(&self, encoder_output: Tensor<3>) -> Tensor<3> {
        let [batch_size, _, _] = encoder_output.dims();

        self.output
            .forward(encoder_output.slice([0..batch_size, 0..1]))
            .tanh()
    }
}

/// Pooler Configuration
#[derive(Config, Debug)]
pub struct PoolerConfig {
    /// Hidden size
    pub hidden_size: usize,
}

impl PoolerConfig {
    /// Initialize a new Pooler module.
    pub fn init(&self, device: &Device) -> Pooler {
        let output = LinearConfig::new(self.hidden_size, self.hidden_size).init(device);

        Pooler::new(output)
    }
}
