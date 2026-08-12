use burn::tensor::Device;
use finetune::{inference::infer, training::train};

const ARTIFACT_DIR: &str = "/tmp/resnet-finetune";

fn main() {
    let device = Device::flex();
    train(ARTIFACT_DIR, device.clone());
    infer(ARTIFACT_DIR, device, 0.5);
}
