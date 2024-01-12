# ResNet Burn

To this day, [ResNet](https://arxiv.org/abs/1512.03385)s are still a strong baseline for your image classification tasks. You can find the [Burn](https://github.com/tracel-ai/burn) implementation for the ResNet variants in [src/model/resnet.rs](src/model/resnet.rs).

The model is [no_std compatible](https://docs.rust-embedded.org/book/intro/no-std.html).

## Usage

### `Cargo.toml`

Add this to your `Cargo.toml`:

<!-- ```toml
[dependencies]
resnet-burn = { git = "https://github.com/burn-rs/models", package = "resnet-burn", features = ["weights_embedded"], default-features = false }
``` -->

### Example Usage

The [inference example](examples/inference.rs) initializes a ResNet-18 with the `NdArray` backend and performs inference on a single input image. You can also run it yourself with the following command:

```shell
cargo run --release --example inference
```


TODO:
- [ ] Load pre-trained weights
- [ ] Replace dummy input with actual image
- [ ] Training example