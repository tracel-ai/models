#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub(crate) mod cache;
pub mod llama;
pub mod pretrained;
pub mod sampling;
pub mod tokenizer;
mod transformer;

#[cfg(test)]
mod tests {
    pub type TestBackend = burn_flex::Flex;
    pub type TestTensor<const D: usize> = burn::tensor::Tensor<TestBackend, D>;
}
