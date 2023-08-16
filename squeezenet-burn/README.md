# SqueezeNet Burn - from ONNX to Rust

SqueezeNet is a small CNN that can be used for image classification. It was trained on the ImageNet
dataset and can classify images into 1000 different classes. The included ONNX model is copied from
the [ONNX model zoo](https://github.com/onnx/models/tree/main/vision/classification/squeezenet), and
the details of the model can be found in the [paper](https://arxiv.org/abs/1602.07360).

The ONNX model is converted into a [Burn](https://github.com/burn-rs/burn/tree/main) model in Rust
using the [burn-import](https://github.com/burn-rs/burn/tree/main/burn-import) crate during build
time. The weights are saved in a binary file during build time in Burn compatible format, and the
model is loaded at runtime.

The labels for the classes are included in the crate and generated from the
[`labels.txt`](src/model/label.txt) during build time.

The data normalizer for the model is included in the crate. See
[Normalizer](src/model/normalizer.rs).

See the [classify example](examples/classify.rs) for how to use the model.

## Usage

### To include the model in your project

Add this to your `Cargo.toml`:

```toml
[dependencies]
squeezenet-burn = { git = "https://github.com/burn-rs/models", package = "squeezenet-burn" }

```

### To run the example

```shell
cargo r --release --example classify samples/flamingo.jpg
```
