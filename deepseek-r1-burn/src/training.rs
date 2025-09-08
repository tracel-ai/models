use burn::{
    data::dataset::Dataset,
    module::Module,
    nn::loss::CrossEntropyLoss,
    optim::AdamConfig,
    tensor::{backend::Backend, Tensor},
    train::{metric::LossMetric, LearnerBuilder},
};

use crate::{DeepSeekR1, DeepSeekR1Config};

pub struct TrainingConfig {
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub epochs: usize,
    pub batch_size: usize,
    pub max_seq_len: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            weight_decay: 0.01,
            epochs: 10,
            batch_size: 32,
            max_seq_len: 2048,
        }
    }
}

pub fn train<B: Backend, D: Dataset<Tensor<B, 2>>>(
    model: DeepSeekR1<B>,
    dataset: D,
    config: TrainingConfig,
) -> DeepSeekR1<B> {
    let device = B::Device::default();
    let optim = AdamConfig::new()
        .with_learning_rate(config.learning_rate)
        .with_weight_decay(config.weight_decay)
        .init();

    let learner = LearnerBuilder::new(&device)
        .num_epochs(config.epochs)
        .batch_size(config.batch_size)
        .grads_accumulation(1)
        .metric_train_plot(LossMetric::new())
        .metric_valid_plot(LossMetric::new())
        .build(model, optim);

    let model_trained = learner.fit(dataset);

    model_trained
}

pub fn compute_loss<B: Backend>(
    model: &DeepSeekR1<B>,
    input: Tensor<B, 2>,
    target: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let output = model.forward(input);
    let loss = CrossEntropyLoss::new(None);
    loss.forward(output, target)
} 