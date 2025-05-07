use std::path::Path;

use burn::{
    backend::{Autodiff, Wgpu},
    data::dataloader::DataLoader,
    module::Module,
    nn::loss::Loss,
    optim::AdamConfig,
    record::{FileRecorder, FullPrecisionSettings, Recorder},
    tensor::{backend::Backend, Device, Element, Tensor, TensorData},
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
};

use yolox_burn::{
    dataset::coco::CocoDatasetLoader,
    model::{
        boxes::nms,
        weights::{self, WeightsMeta},
        yolox::Yolox,
    },
};

const HEIGHT: usize = 640;
const WIDTH: usize = 640;
const BATCH_SIZE: usize = 4;
const NUM_EPOCHS: usize = 50;
const LEARNING_RATE: f64 = 0.001;

fn to_tensor<B: Backend, T: Element>(
    data: Vec<T>,
    shape: [usize; 3],
    device: &Device<B>,
) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(
        TensorData::new(data, shape).convert::<B::FloatElem>(),
        device,
    )
    // [H, W, C] -> [C, H, W]
    .permute([2, 0, 1])
}

struct YoloxLoss<B: Backend> {
    device: Device<B>,
}

impl<B: Backend> YoloxLoss<B> {
    pub fn new(device: Device<B>) -> Self {
        Self { device }
    }
}

impl<B: Backend> Loss<B> for YoloxLoss<B> {
    type Input = (Tensor<B, 3>, Tensor<B, 3>);
    type Output = Tensor<B, 1>;

    fn forward(&self, (pred, target): Self::Input) -> Self::Output {
        // Implement YOLOX loss function
        // This is a simplified version - you'll need to implement the full loss function
        // based on the YOLOX paper
        let reg_loss = (pred.slice([0..1, 0..pred.dims()[1], 0..4]) - target.slice([0..1, 0..target.dims()[1], 0..4]))
            .powf(2.0)
            .mean();
        
        let obj_loss = (pred.slice([0..1, 0..pred.dims()[1], 4..5]) - target.slice([0..1, 0..target.dims()[1], 4..5]))
            .powf(2.0)
            .mean();
        
        let cls_loss = (pred.slice([0..1, 0..pred.dims()[1], 5..pred.dims()[2]]) - target.slice([0..1, 0..target.dims()[1], 5..target.dims()[2]]))
            .powf(2.0)
            .mean();

        reg_loss + obj_loss + cls_loss
    }
}

pub fn main() {
    // Parse arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        println!("Usage: {} <input_coco_dir> <output_model_path>", args[0]);
        return;
    }
    let input_dir = &args[1];
    let output_path = &args[2];

    // Create device with wgpu backend
    let device = Device::<Wgpu>::default();

    // Create YOLOX model with wgpu backend
    let model: Yolox<Autodiff<Wgpu>> = Yolox::yolox_tiny_pretrained(weights::YoloxTiny::Coco, &device)
        .map_err(|err| format!("Failed to load pre-trained weights.\nError: {err}"))
        .unwrap();

    // Create dataset loader
    let dataset = CocoDatasetLoader::new(
        &format!("{}/annotations/instances_train2017.json", input_dir),
        &format!("{}/train2017", input_dir),
        device.clone(),
    );

    // Create data loader with reduced number of workers
    let dataloader = DataLoader::builder(dataset)
        .batch_size(BATCH_SIZE)
        .shuffle(true)
        .num_workers(2)
        .build();

    // Create optimizer with adjusted parameters for wgpu
    let optim = AdamConfig::new()
        .with_lr(LEARNING_RATE)
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_eps(1e-8)
        .with_weight_decay(0.0005)
        .init();

    // Create loss function
    let loss_fn = YoloxLoss::new(device.clone());

    // Create metrics
    let metrics = [AccuracyMetric::new(), LossMetric::new()];

    // Create learner with adjusted parameters
    let learner = LearnerBuilder::new(&model)
        .with_optimizer(optim)
        .with_num_epochs(NUM_EPOCHS)
        .with_batch_size(BATCH_SIZE)
        .with_loss_fn(loss_fn)
        .with_metrics(metrics)
        .with_grad_accumulation(2)
        .build();

    // Train the model
    let model_trained = learner.fit(dataloader);

    // Save the trained model
    let recorder = FileRecorder::<FullPrecisionSettings>::new();
    recorder
        .record(model_trained.into_record(), Path::new(output_path))
        .unwrap();
} 