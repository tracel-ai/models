# Qwen3.5 Burn

The [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-0.8B) text model with Burn. Three of every
four layers are [Gated DeltaNet](https://arxiv.org/abs/2412.06464), the fourth is gated softmax
attention. Everything is plain Burn tensor ops: it runs on every backend and trains with
autodiff. The vision tower and the multi-token-prediction head are not included.

Burn has no Gated DeltaNet layer yet, so it lives in `src/gdn/` for now. It moves to `burn-nn`
once that lands.

Weights are loaded from HF safetensors with [`burn-store`](https://crates.io/crates/burn-store).
The tokenizer is Qwen's byte-level BPE in pure Rust, read from `tokenizer.json`.

## Usage

```toml
[dependencies]
qwen35-burn = { git = "https://github.com/tracel-ai/models", package = "qwen35-burn" }
```

```rust
use burn::tensor::Device;
use qwen35_burn::{Tokenizer, loader};

let device = Device::default();
let model = loader::from_dir("path/to/Qwen3.5-0.8B", &device)?;
let tokenizer = Tokenizer::from_file("path/to/Qwen3.5-0.8B/tokenizer.json")?;
```

Greedy next tokens for a prompt:

```bash
cargo run -p qwen35-burn --release --example next-token -- path/to/model "The capital of Sweden is"
```

## Testing

```bash
cargo test -p qwen35-burn
```

compares every layer and the logits of a tiny random model against
[Crane](https://github.com/lucasjinreal/Crane) (`tests/fixtures/tiny`), and checks that
gradients flow.

With a checkpoint and Crane's logits for it:

```bash
QWEN35_DIR=path/to/model cargo test -p qwen35-burn --release -- --ignored
```

checks the tokenizer against HF `tokenizers` and the full model's logits against Crane. On a
Qwen3.5-0.8B fine-tune ([openjev](https://huggingface.co/AlexWortega/openjev)) the logits match
with cosine 1.0000000 at 7, 64 and 300 tokens, same top token.

## Credits

Ported from [Crane](https://github.com/lucasjinreal/Crane) (MIT), `crane-core/src/ops/gdn/` and
`crane-core/src/models/qwen3_5/`, whose GDN code comes from
[mistral.rs](https://github.com/EricLBuehler/mistral.rs) (MIT). The chunked delta rule follows
HF Transformers' `torch_chunk_gated_delta_rule`.

## License

MIT OR Apache-2.0
