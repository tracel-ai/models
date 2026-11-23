//! Qwen3.5's building blocks besides Gated DeltaNet: RMSNorm, rotary position, gated attention,
//! SwiGLU MLP and the decoder layer. Ported from Crane (`models/qwen3_5/modeling.rs`, MIT); see
//! the README.

use crate::config::{LayerKind, Qwen35Config};
use crate::gdn::{GatedDeltaNet, GdnState};
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{sigmoid, silu, softmax};
use burn::tensor::{Bool, Device, Tensor, TensorData};

/// RMSNorm as Qwen3.5 stores it: the weight is an offset, the scale is `1 + weight`. Not named
/// `RmsNorm` on purpose: `burn-store`'s PyTorch adapter renames `weight` to `gamma` inside a
/// module of that name.
#[derive(Module, Debug)]
pub struct OffsetRmsNorm {
    pub weight: Param<Tensor<1>>,
    pub eps: f64,
}

impl OffsetRmsNorm {
    pub fn new(size: usize, eps: f64, device: &Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::zeros([size], device)),
            eps,
        }
    }

    /// Normalises the last dim.
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let r = (x.clone().powi_scalar(2).mean_dim(D - 1) + self.eps).sqrt();
        (x / r) * (self.weight.val() + 1.0).unsqueeze::<D>()
    }
}

/// Rotary position for positions `start..start + len`: `cos`, `sin` of shape `[len, rot / 2]`.
/// Computed in f32 like HF (`inv_freq = 1 / theta^(2i / rot)`). For text, Qwen3.5's interleaved
/// MRoPE has equal T/H/W positions and reduces to exactly this.
pub fn rope_tables(
    theta: f64,
    rot: usize,
    start: usize,
    len: usize,
    device: &Device,
) -> (Tensor<2>, Tensor<2>) {
    let half = rot / 2;
    let base = theta as f32;
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / base.powf(i as f32 * 2.0 / rot as f32))
        .collect();
    let mut cos = Vec::with_capacity(len * half);
    let mut sin = Vec::with_capacity(len * half);
    for p in start..start + len {
        for f in &inv {
            let a = p as f32 * f;
            cos.push(a.cos());
            sin.push(a.sin());
        }
    }
    (
        Tensor::from_data(TensorData::new(cos, [len, half]), device),
        Tensor::from_data(TensorData::new(sin, [len, half]), device),
    )
}

/// Rotate-half RoPE on the first `rot` dims of `x: [B, H, S, D]`; the rest pass through.
pub fn apply_rope(x: Tensor<4>, cos: Tensor<2>, sin: Tensor<2>) -> Tensor<4> {
    let [b, h, s, d] = x.dims();
    let half = cos.dims()[1];
    let rot = 2 * half;
    let cos = cos.reshape([1, 1, s, half]);
    let sin = sin.reshape([1, 1, s, half]);
    let x1 = x.clone().narrow(3, 0, half);
    let x2 = x.clone().narrow(3, half, half);
    let r1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
    let r2 = x2 * cos + x1 * sin;
    if rot == d {
        Tensor::cat(vec![r1, r2], 3)
    } else {
        Tensor::cat(vec![r1, r2, x.narrow(3, rot, d - rot)], 3).reshape([b, h, s, d])
    }
}

/// Qwen3.5's full attention: grouped KV heads, per-head RMSNorm on Q and K, partial RoPE, and an
/// output gate (`q_proj` emits `[query | gate]` per head; the result is scaled by
/// `sigmoid(gate)` before `o_proj`).
#[derive(Module, Debug)]
pub struct Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub q_norm: OffsetRmsNorm,
    pub k_norm: OffsetRmsNorm,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub output_gate: bool,
}

impl Attention {
    pub fn new(c: &Qwen35Config, device: &Device) -> Self {
        let lin = |i, o| LinearConfig::new(i, o).with_bias(false).init(device);
        let (nh, nkv, hd) = (c.num_attention_heads, c.num_key_value_heads, c.head_dim);
        let q_out = if c.attn_output_gate {
            2 * nh * hd
        } else {
            nh * hd
        };
        Self {
            q_proj: lin(c.hidden_size, q_out),
            k_proj: lin(c.hidden_size, nkv * hd),
            v_proj: lin(c.hidden_size, nkv * hd),
            o_proj: lin(nh * hd, c.hidden_size),
            q_norm: OffsetRmsNorm::new(hd, c.rms_norm_eps, device),
            k_norm: OffsetRmsNorm::new(hd, c.rms_norm_eps, device),
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            output_gate: c.attn_output_gate,
        }
    }

    /// Causal self-attention over `x: [B, S, hidden]`.
    pub fn forward(&self, x: Tensor<3>, cos: Tensor<2>, sin: Tensor<2>) -> Tensor<3> {
        let [b, s, _] = x.dims();
        let (nh, nkv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);
        let q_out = self.q_proj.forward(x.clone());
        let (q, gate) = if self.output_gate {
            let qg = q_out.reshape([b, s, nh, 2 * hd]);
            (
                qg.clone().narrow(3, 0, hd),
                Some(qg.narrow(3, hd, hd).reshape([b, s, nh * hd])),
            )
        } else {
            (q_out.reshape([b, s, nh, hd]), None)
        };
        let q = self.q_norm.forward(q.swap_dims(1, 2)); // [B, H, S, D]
        let k = self.k_norm.forward(
            self.k_proj
                .forward(x.clone())
                .reshape([b, s, nkv, hd])
                .swap_dims(1, 2),
        );
        let v = self
            .v_proj
            .forward(x)
            .reshape([b, s, nkv, hd])
            .swap_dims(1, 2);
        let q = apply_rope(q, cos.clone(), sin.clone());
        let k = apply_rope(k, cos, sin);

        // Query head h reads KV head h / (H / H_kv).
        let rep = nh / nkv;
        let share = |t: Tensor<4>| {
            if rep == 1 {
                t
            } else {
                t.unsqueeze_dim::<5>(2)
                    .expand([b, nkv, rep, s, hd])
                    .reshape([b, nh, s, hd])
            }
        };
        let (k, v) = (share(k), share(v));
        let scores = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar(1.0 / (hd as f64).sqrt()); // [B, H, S, S]
        let future = Tensor::<2, Bool>::tril_mask([s, s], 0, &scores.device())
            .unsqueeze::<4>()
            .expand([b, nh, s, s]);
        let weights = softmax(scores.mask_fill(future, f32::NEG_INFINITY), 3);
        let y = weights.matmul(v).swap_dims(1, 2).reshape([b, s, nh * hd]);
        let y = match gate {
            Some(g) => y * sigmoid(g),
            None => y,
        };
        self.o_proj.forward(y)
    }
}

/// SwiGLU: `down(silu(gate(x)) * up(x))`.
#[derive(Module, Debug)]
pub struct Mlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl Mlp {
    pub fn new(c: &Qwen35Config, device: &Device) -> Self {
        let lin = |i, o| LinearConfig::new(i, o).with_bias(false).init(device);
        Self {
            gate_proj: lin(c.hidden_size, c.intermediate_size),
            up_proj: lin(c.hidden_size, c.intermediate_size),
            down_proj: lin(c.intermediate_size, c.hidden_size),
        }
    }
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.down_proj
            .forward(silu(self.gate_proj.forward(x.clone())) * self.up_proj.forward(x))
    }
}

/// One decoder layer: pre-norm mixer (Gated DeltaNet or attention) + residual, pre-norm MLP +
/// residual. Exactly one of `linear_attn` / `self_attn` is set.
#[derive(Module, Debug)]
pub struct DecoderLayer {
    pub input_layernorm: OffsetRmsNorm,
    pub linear_attn: Option<GatedDeltaNet>,
    pub self_attn: Option<Attention>,
    pub post_attention_layernorm: OffsetRmsNorm,
    pub mlp: Mlp,
}

impl DecoderLayer {
    pub fn new(c: &Qwen35Config, kind: LayerKind, device: &Device) -> Self {
        Self {
            input_layernorm: OffsetRmsNorm::new(c.hidden_size, c.rms_norm_eps, device),
            linear_attn: (kind == LayerKind::Linear).then(|| c.gdn().init(device)),
            self_attn: (kind == LayerKind::Full).then(|| Attention::new(c, device)),
            post_attention_layernorm: OffsetRmsNorm::new(c.hidden_size, c.rms_norm_eps, device),
            mlp: Mlp::new(c, device),
        }
    }

    pub fn forward(&self, x: Tensor<3>, cos: Tensor<2>, sin: Tensor<2>) -> Tensor<3> {
        let h = self.input_layernorm.forward(x.clone());
        let mixed = match (&self.linear_attn, &self.self_attn) {
            (Some(gdn), _) => gdn.forward(h, None::<GdnState>).0,
            (None, Some(attn)) => attn.forward(h, cos, sin),
            (None, None) => unreachable!("a decoder layer has a mixer"),
        };
        let x = x + mixed;
        let h = self.post_attention_layernorm.forward(x.clone());
        x + self.mlp.forward(h)
    }
}
