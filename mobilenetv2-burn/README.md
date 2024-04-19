# MobileNetV2 Burn

[MobileNetV2](https://arxiv.org/abs/1801.04381) is a convolutional neural network architecture for
classification tasks which seeks to perform well on mobile devices. You can find the
[Burn](https://github.com/tracel-ai/burn) implementation for the MobileNetV2 in
[src/model/mobilenetv2.rs](src/model/mobilenetv2.rs).

The model is [no_std compatible](https://docs.rust-embedded.org/book/intro/no-std.html).

## Usage

### `Cargo.toml`

Add this to your `Cargo.toml`:

```toml
[dependencies]
resnet-burn = { git = "https://github.com/tracel-ai/models", package = "mobilenetv2-burn", default-features = false }
```

If you want to get the pre-trained ImageNet weights, enable the `pretrained` feature flag.

```toml
[dependencies]
resnet-burn = { git = "https://github.com/tracel-ai/models", package = "mobilenetv2-burn", features = ["pretrained"] }
```

**Important:** this feature requires `std`.

### Example Usage

The [inference example](examples/inference.rs) initializes a MobileNetV2 from the ImageNet
[pre-trained weights](https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html#torchvision.models.MobileNet_V2_Weights)
with the `NdArray` backend and performs inference on the provided input image.

You can run the example with the following command:

```sh
cargo run --release --features pretrained --example inference samples/dog.jpg
```
