use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};
use derive_new::new;

/// Pooler
#[derive(Module, Debug, new)]
pub struct Pooler<B: Backend> {
    /// Linear output
    output: Linear<B>,
}

impl<B: Backend> Pooler<B> {
    /// Forward pass
    pub fn forward(&self, encoder_output: Tensor<B, 3>) -> Tensor<B, 3> {
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
    pub fn init<B: Backend>(&self, device: &B::Device) -> Pooler<B> {
        let output = LinearConfig::new(self.hidden_size, self.hidden_size).init(device);

        Pooler::new(output)
    }
}
