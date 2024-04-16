# SqueezeNet Burn - from ONNX to Rust

SqueezeNet is a small CNN that can be used for image classification. It was trained on the ImageNet
dataset and can classify images into 1000 different classes. The included ONNX model is copied from
the [ONNX model zoo](https://github.com/onnx/models/tree/main/vision/classification/squeezenet), and
the details of the model can be found in the [paper](https://arxiv.org/abs/1602.07360).

The ONNX model is converted into a [Burn](https://github.com/burn-rs/burn/tree/main) model in Rust
using the [burn-import](https://github.com/burn-rs/burn/tree/main/burn-import) crate during build
time. The weights are saved in a binary file during build time in Burn compatible format, and the
model is loaded at runtime.

It is worth noting that the model can be fine-tuned to improve the accuracy, since the ONNX model is
fully converted to a Burn model. The model is trained with the ImageNet dataset, which contains 1.2
million images. The model can be fine-tuned with a smaller dataset to improve the accuracy for a
specific use case.

The labels for the classes are included in the crate and generated from the
[`labels.txt`](src/model/label.txt) during build time.

The data normalizer for the model is included in the crate. See
[Normalizer](src/model/normalizer.rs).

The model is [no_std compatible](https://docs.rust-embedded.org/book/intro/no-std.html).

See the [classify example](examples/classify.rs) for how to use the model.

## Usage

### To include the model in your project

Add this to your `Cargo.toml`:

```toml
[dependencies]
squeezenet-burn = { git = "https://github.com/tracel-ai/models", package = "squeezenet-burn", features = ["weights_embedded"], default-features = false }
```

### To run the example

1. Use the `weights_embedded` feature to embed the weights in the binary.

```shell
cargo r --release --features weights_embedded --no-default-features --example classify samples/flamingo.jpg
```

2. Use the `weights_file` feature to load the weights from a file.

```shell
cargo r --release --features weights_file  --example classify samples/flamingo.jpg
```

3. Use the `weights_f16` feature to use 16-bit floating point numbers for the weights.

```shell
cargo r --release --features "weights_embedded, weights_f16" --no-default-features --example classify samples/flamingo.jpg
```

Or

```shell
cargo r --release --features "weights_file, weights_f16"  --example classify samples/flamingo.jpg
```

## Feature Flags

- `weights_file`: Load the weights from a file (enabled by default).
- `weights_embedded`: Embed the weights in the binary.
- `weights_f16`: Use 16-bit floating point numbers for the weights. (by default 32-bit is used)
