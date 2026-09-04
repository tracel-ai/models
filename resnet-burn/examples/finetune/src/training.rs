use std::time::Instant;

use crate::{
    data::ClassificationBatcher,
    dataset::{PlanetLoader, CLASSES},
};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::ImageFolderDataset},
    optim::{decay::WeightDecayConfig, AdamConfig},
    prelude::*,
    train::{
        metric::{HammingScore, LossMetric},
        Learner, SupervisedTraining,
    },
};
use resnet_burn::{weights, ResNet};

const NUM_CLASSES: usize = CLASSES.len();

#[derive(Config, Debug)]
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

pub fn train(artifact_dir: &str, device: Device) {
    create_artifact_dir(artifact_dir);

    // Config
    let config = TrainingConfig::new(NUM_CLASSES);
    let optimizer = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(config.weight_decay)))
        .init();

    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    device.seed(config.seed);
    let autodiff_device = device.clone().autodiff();

    // Dataloaders
    let batcher_train = ClassificationBatcher::new(autodiff_device.clone());
    let batcher_valid = ClassificationBatcher::new(device.clone());

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
    let model = ResNet::resnet18_pretrained(weights::ResNet18::ImageNet1kV1, &autodiff_device)
        .unwrap()
        .with_classes(NUM_CLASSES);

    // Training config
    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metrics((HammingScore::new(), LossMetric::new()))
        .with_default_checkpointers()
        .num_epochs(config.num_epochs)
        .summary();

    // Training
    let now = Instant::now();
    let training_result = training.launch(Learner::new(model, optimizer, config.learning_rate));
    let elapsed = now.elapsed().as_secs();
    println!("Training completed in {}m{}s", (elapsed / 60), elapsed % 60);

    training_result
        .model
        .into_record()
        .save(format!("{artifact_dir}/model"))
        .expect("Trained model should be saved successfully");
}
