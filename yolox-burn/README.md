# YOLOX Burn

There have been many different object detection models with the YOLO prefix released in the recent
years, though most of them carry a GPL or AGPL license which restricts their usage. For this reason,
we selected [YOLOX](https://arxiv.org/abs/2107.08430) as the first object detection architecture
since both the original code and pre-trained weights are released under the
[Apache 2.0](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE) open source license.

You can find the [Burn](https://github.com/tracel-ai/burn) implementation for the YOLOX variants in
[src/model/yolox.rs](src/model/yolox.rs).

The model is [no_std compatible](https://docs.rust-embedded.org/book/intro/no-std.html).

## Usage

### `Cargo.toml`

Add this to your `Cargo.toml`:

```toml
[dependencies]
yolox-burn = { git = "https://github.com/tracel-ai/models", package = "yolox-burn", default-features = false }
```

If you want to get the COCO pre-trained weights, enable the `pretrained` feature flag.

```toml
[dependencies]
yolox-burn = { git = "https://github.com/tracel-ai/models", package = "yolox-burn", features = ["pretrained"] }
```

**Important:** this feature requires `std`.

### Example Usage

The [inference example](examples/inference.rs) initializes a YOLOX-Tiny from the COCO
[pre-trained weights](https://github.com/Megvii-BaseDetection/YOLOX?tab=readme-ov-file#standard-models)
with the `NdArray` backend and performs inference on the provided input image.

You can run the example with the following command:

```sh
cargo run --release --features pretrained --example inference samples/dog_bike_man.jpg
```
