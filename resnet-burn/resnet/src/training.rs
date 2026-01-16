use burn::{
    nn::loss::BinaryCrossEntropyLossConfig,
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{InferenceStep, MultiLabelClassificationOutput, TrainOutput, TrainStep},
};

use crate::ResNet;

impl<B: Backend> ResNet<B> {
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

#[derive(Clone, Debug)]
pub struct ClassificationBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: AutodiffBackend> TrainStep for ResNet<B> {
    type Input = ClassificationBatch<B>;
    type Output = MultiLabelClassificationOutput<B>;

    fn step(
        &self,
        batch: ClassificationBatch<B>,
    ) -> TrainOutput<MultiLabelClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for ResNet<B> {
    type Input = ClassificationBatch<B>;
    type Output = MultiLabelClassificationOutput<B>;

    fn step(&self, batch: ClassificationBatch<B>) -> MultiLabelClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}
