# ResNet Burn

To this day, [ResNet](https://arxiv.org/abs/1512.03385)s are still a strong baseline for your image
classification tasks. You can find the [Burn](https://github.com/tracel-ai/burn) implementation for
the ResNet variants in [src/model/resnet.rs](src/model/resnet.rs).

The model is [no_std compatible](https://docs.rust-embedded.org/book/intro/no-std.html).

## Usage

### `Cargo.toml`

Add this to your `Cargo.toml`:

```toml
[dependencies]
resnet-burn = { git = "https://github.com/burn-rs/models", package = "resnet-burn", default-features = false }
```

### Example Usage

The [inference example](examples/inference.rs) initializes a ResNet-18 with the `NdArray` backend,
imports the ImageNet pre-trained weights from
[`torchvision`](https://download.pytorch.org/models/resnet18-f37072fd.pth) and performs inference on
the provided input image.

After downloading the
[pre-trained weights](https://download.pytorch.org/models/resnet18-f37072fd.pth) to the current
directory, you can run the example with the following command:

```sh
cargo run --release --example inference samples/dog.jpg
```
