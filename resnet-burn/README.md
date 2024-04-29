# ResNet Burn

To this day, [ResNet](https://arxiv.org/abs/1512.03385)s are still a strong baseline for your image
classification tasks. You can find the [Burn](https://github.com/tracel-ai/burn) implementation for
the ResNet variants in [resnet.rs](resnet/src/resnet.rs).

The model is [no_std compatible](https://docs.rust-embedded.org/book/intro/no-std.html).

## Usage

### `Cargo.toml`

Add this to your `Cargo.toml`:

```toml
[dependencies]
resnet-burn = { git = "https://github.com/tracel-ai/models", package = "resnet-burn", default-features = false }
```

If you want to get the pre-trained ImageNet weights, enable the `pretrained` feature flag.

```toml
[dependencies]
resnet-burn = { git = "https://github.com/tracel-ai/models", package = "resnet-burn", features = ["pretrained"] }
```

**Important:** this feature requires `std`.

### Example Usage

#### Inference

The [inference example](examples/inference/examples/inference.rs) initializes a ResNet-18 from the
ImageNet
[pre-trained weights](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html#torchvision.models.ResNet18_Weights)
with the `NdArray` backend and performs inference on the provided input image.

You can run the example with the following command:

```sh
cargo run --release --example inference samples/dog.jpg --release
```

#### Fine-tuning

For this [multi-label image classification fine-tuning example](examples/finetune), a sample of the
planets dataset from the Kaggle competition
[Planet: Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)
is downloaded from a
[fastai mirror](https://github.com/fastai/fastai/blob/master/fastai/data/external.py#L55). The
sample dataset is a collection of satellite images with multiple labels describing the scene, as
illustrated below.

<img src="./samples/dataset.jpg" alt="Planet dataset sample" width="1000"/>

To achieve this task, a ResNet-18 pre-trained on the ImageNet dataset is fine-tuned on the target
planets dataset. The training recipe used is fairly simple. The main objective is to demonstrate how to re-use a
pre-trained model for a different downstream task.

Without any bells and whistle, our model achieves over 90% multi-label accuracy (i.e., hamming
score) on the validation set after just 5 epochs.

Run the example with the Torch GPU backend:

```sh
export TORCH_CUDA_VERSION=cu121
cargo run --release --example finetune --features tch-gpu
```

Run it with our WGPU backend:

```sh
cargo run --release --example finetune --features wgpu
```
