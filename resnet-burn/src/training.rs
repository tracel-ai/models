use burn::{
    nn::loss::BinaryCrossEntropyLossConfig,
    prelude::*,
    train::{InferenceStep, MultiLabelClassificationOutput, TrainOutput, TrainStep},
};

use crate::ResNet;

impl ResNet {
    fn forward_classification(
        &self,
        images: Tensor<4>,
        targets: Tensor<2, Int>,
    ) -> MultiLabelClassificationOutput {
        let output = self.forward(images);
        let loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        MultiLabelClassificationOutput::new(loss, output, targets)
    }
}

#[derive(Clone, Debug)]
pub struct ClassificationBatch {
    pub images: Tensor<4>,
    pub targets: Tensor<2, Int>,
}

impl TrainStep for ResNet {
    type Input = ClassificationBatch;
    type Output = MultiLabelClassificationOutput;

    fn step(&self, batch: ClassificationBatch) -> TrainOutput<MultiLabelClassificationOutput> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl InferenceStep for ResNet {
    type Input = ClassificationBatch;
    type Output = MultiLabelClassificationOutput;

    fn step(&self, batch: ClassificationBatch) -> MultiLabelClassificationOutput {
        self.forward_classification(batch.images, batch.targets)
    }
}
