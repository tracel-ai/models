use burn::backend::Autodiff;
use burn_flex::{Flex, FlexDevice};
use finetune::{inference::infer, training::train};

const ARTIFACT_DIR: &str = "/tmp/resnet-finetune";

fn main() {
    let device = FlexDevice;
    train::<Autodiff<Flex>>(ARTIFACT_DIR, device);
    infer::<Flex>(ARTIFACT_DIR, device, 0.5);
}
