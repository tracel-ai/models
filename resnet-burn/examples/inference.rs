use resnet_burn::model::resnet::ResNet;

use burn::tensor::Tensor;
use burn_ndarray::NdArray;

type Backend = NdArray<f32>;

pub fn main() {
    let x = Tensor::<Backend, 4>::ones([1, 3, 224, 224]);

    let model = ResNet::<Backend>::resnet18(10);

    let out = model.forward(x);

    println!("Output scores: {}", out);
}
