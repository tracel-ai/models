use std::time::Instant;

use crate::{
    data::{ClassificationBatch, ClassificationBatcher},
    dataset::{PlanetLoader, CLASSES},
};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::ImageFolderDataset},
    nn::loss::BinaryCrossEntropyLossConfig,
    optim::{decay::WeightDecayConfig, AdamConfig},
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{HammingScore, LossMetric},
        LearnerBuilder, MultiLabelClassificationOutput, TrainOutput, TrainStep, ValidStep,
    },
};
use resnet_burn::{weights, ResNet};

const NUM_CLASSES: usize = CLASSES.len();

pub trait MultiLabelClassification<B: Backend> {
    fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 2, Int>,
    ) -> MultiLabelClassificationOutput<B>;
}

impl<B: Backend> MultiLabelClassification<B> for ResNet<B> {
    fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 2, Int>,
    ) -> MultiLabelClassificationOutput<B> {
        let output = self.forward(images);
        let loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        MultiLabelClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<ClassificationBatch<B>, MultiLabelClassificationOutput<B>>
    for ResNet<B>
{
    fn step(
        &self,
        batch: ClassificationBatch<B>,
    ) -> TrainOutput<MultiLabelClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<ClassificationBatch<B>, MultiLabelClassificationOutput<B>>
    for ResNet<B>
{
    fn step(&self, batch: ClassificationBatch<B>) -> MultiLabelClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    #[config(default = 5)]
    pub num_epochs: usize,

    #[config(default = 128)]
    pub batch_size: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    #[config(default = 1e-3)]
    pub learning_rate: f64,

    #[config(default = 5e-5)]
    pub weight_decay: f32,

    #[config(default = 70)]
    pub train_percentage: u8,

    pub num_classes: usize,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, device: B::Device) {
    create_artifact_dir(artifact_dir);

    // Config
    let config = TrainingConfig::new(NUM_CLASSES);
    let optimizer = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(config.weight_decay)))
        .init();

    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    // Dataloaders
    let batcher_train = ClassificationBatcher::<B>::new(device.clone());
    let batcher_valid = ClassificationBatcher::<B::InnerBackend>::new(device.clone());

    let (train, valid) =
        ImageFolderDataset::planet_train_val_split(config.train_percentage, config.seed).unwrap();

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(valid);

    // Pre-trained ResNet-18 adapted for num_classes in this task
    let model = ResNet::resnet18_pretrained(weights::ResNet18::ImageNet1kV1, &device)
        .unwrap()
        .with_classes(NUM_CLASSES);

    // Learner config
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(HammingScore::new())
        .metric_valid_numeric(HammingScore::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, optimizer, config.learning_rate);

    // Training
    let now = Instant::now();
    let model_trained = learner.fit(dataloader_train, dataloader_test);
    let elapsed = now.elapsed().as_secs();
    println!("Training completed in {}m{}s", (elapsed / 60), elapsed % 60);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
