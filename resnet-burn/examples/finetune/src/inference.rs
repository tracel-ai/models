use crate::{
    data::ClassificationBatcher,
    dataset::{PlanetLoader, CLASSES},
    training::TrainingConfig,
};
use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{
            vision::{Annotation, ImageFolderDataset},
            Dataset,
        },
    },
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::activation::sigmoid,
};
use resnet_burn::ResNet;

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, threshold: f32) {
    // Load trained ResNet-18
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model: ResNet<B> = ResNet::resnet18(config.num_classes, &device).load_record(record);

    // Get an item from validation split with multiple labels
    let (_train, valid) =
        ImageFolderDataset::planet_train_val_split(config.train_percentage, config.seed).unwrap();
    let item = valid.get(20).unwrap();

    let label = if let Annotation::MultiLabel(ref categories) = item.annotation {
        categories.iter().map(|&i| CLASSES[i]).collect::<Vec<_>>()
    } else {
        panic!("Annotation should be multilabel")
    };

    // Forward pass with sigmoid activation function
    let batcher = ClassificationBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = sigmoid(model.forward(batch.images));

    // Get predicted class names over the specified threshold
    let predicted = output.greater_equal_elem(threshold).nonzero()[1]
        .to_data()
        .iter::<B::IntElem>()
        .map(|i| CLASSES[i.elem::<i64>() as usize])
        .collect::<Vec<_>>();

    println!("Predicted: {:?}\nExpected: {:?}", predicted, label);
}
