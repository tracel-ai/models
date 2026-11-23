//! Ported from Crane (https://github.com/lucasjinreal/Crane, MIT), `crane-core/src/ops/gdn/{layer,conv,norm,projection}.rs`,
//! itself from mistral.rs (MIT). See the README.
//!
//! The Gated DeltaNet layer as Qwen3.5 uses it (`linear_attn` in the HF checkpoint).
//!
//! ```text
//! x ─► in_proj_qkv ─► causal conv (kernel 4) ─► SiLU ─► split Q K V into heads
//!   │                                                   Q, K: L2-normalise, repeat to V heads
//!   ├► in_proj_b ─► beta = sigmoid(b)                    │
//!   ├► in_proj_a ─► g = -exp(A_log) * softplus(a + dt_bias)
//!   │                                                   ▼
//!   │                                         gated delta rule (chunked)
//!   └► in_proj_z ─────────────► gated RMSNorm: rms(y) * norm * silu(z) ─► out_proj ─► out
//! ```
//!
//! Field names follow the checkpoint (`linear_attn.in_proj_qkv.weight`, `conv1d.weight`,
//! `dt_bias`, `A_log`, `norm.weight`, `out_proj.weight`); [`crate::loader`] maps them.

use super::chunked::gated_delta_rule;
use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{sigmoid, silu, softplus};
use burn::tensor::{Device, Distribution, Tensor};

/// Sizes. The defaults are Qwen3.5-0.8B's.
#[derive(Config, Debug)]
pub struct GatedDeltaNetConfig {
    #[config(default = 1024)]
    pub hidden_size: usize,
    #[config(default = 16)]
    pub num_k_heads: usize,
    #[config(default = 16)]
    pub num_v_heads: usize,
    #[config(default = 128)]
    pub head_k_dim: usize,
    #[config(default = 128)]
    pub head_v_dim: usize,
    #[config(default = 4)]
    pub conv_kernel: usize,
    /// Epsilon of the gated RMSNorm (`rms_norm_eps`).
    #[config(default = 1e-6)]
    pub eps: f64,
    /// Tokens per chunk in the rule.
    #[config(default = 64)]
    pub chunk: usize,
}

impl GatedDeltaNetConfig {
    pub fn key_dim(&self) -> usize {
        self.num_k_heads * self.head_k_dim
    }
    pub fn value_dim(&self) -> usize {
        self.num_v_heads * self.head_v_dim
    }
    /// Width of the conv input: Q, K and V side by side.
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    /// A layer with fresh weights (A_log ~ log U(1, 16) as Qwen3.5 initialises it).
    pub fn init(&self, device: &Device) -> GatedDeltaNet {
        assert!(
            self.num_v_heads.is_multiple_of(self.num_k_heads),
            "num_v_heads must be a multiple of num_k_heads"
        );
        let lin = |i, o| LinearConfig::new(i, o).with_bias(false).init(device);
        let nv = self.num_v_heads;
        GatedDeltaNet {
            in_proj_qkv: lin(self.hidden_size, self.conv_dim()),
            in_proj_z: lin(self.hidden_size, self.value_dim()),
            in_proj_b: lin(self.hidden_size, nv),
            in_proj_a: lin(self.hidden_size, nv),
            conv1d: Param::from_tensor(Tensor::random(
                [self.conv_dim(), 1, self.conv_kernel],
                Distribution::Uniform(-0.5, 0.5),
                device,
            )),
            dt_bias: Param::from_tensor(Tensor::ones([nv], device)),
            a_log: Param::from_tensor(
                Tensor::<1>::random([nv], Distribution::Uniform(1.0, 16.0), device).log(),
            ),
            norm: Param::from_tensor(Tensor::ones([self.head_v_dim], device)),
            out_proj: lin(self.value_dim(), self.hidden_size),
            num_k_heads: self.num_k_heads,
            num_v_heads: nv,
            head_k_dim: self.head_k_dim,
            head_v_dim: self.head_v_dim,
            eps: self.eps,
            chunk: self.chunk,
        }
    }
}

/// One Gated DeltaNet layer.
#[derive(Module, Debug)]
pub struct GatedDeltaNet {
    pub in_proj_qkv: Linear,
    pub in_proj_z: Linear,
    pub in_proj_b: Linear,
    pub in_proj_a: Linear,
    /// Depthwise causal conv over Q|K|V, `[conv_dim, 1, kernel]` (PyTorch's `Conv1d` layout), no bias.
    pub conv1d: Param<Tensor<3>>,
    /// `[num_v_heads]`
    pub dt_bias: Param<Tensor<1>>,
    /// `[num_v_heads]`; the decay rate is `exp(a_log)`.
    pub a_log: Param<Tensor<1>>,
    /// Gated RMSNorm weight, `[head_v_dim]` (used as is, no `1 +`).
    pub norm: Param<Tensor<1>>,
    pub out_proj: Linear,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub eps: f64,
    pub chunk: usize,
}

/// What a layer remembers between calls, so a long text can be fed in pieces (or a shared
/// prefix computed once).
#[derive(Debug, Clone)]
pub struct GdnState {
    /// The last `kernel - 1` conv inputs, `[B, conv_dim, kernel - 1]`.
    pub conv: Tensor<3>,
    /// The delta-rule memory, `[B, num_v_heads, head_k_dim, head_v_dim]`, f32.
    pub recurrent: Tensor<4>,
}

impl GatedDeltaNet {
    /// `x: [B, S, hidden]` → `[B, S, hidden]` and the state after the last token.
    /// `state` is what an earlier call returned, or `None` to start fresh.
    pub fn forward(&self, x: Tensor<3>, state: Option<GdnState>) -> (Tensor<3>, GdnState) {
        let [b, s, _] = x.dims();
        let (nk, nv, dk, dv) = (
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
        );
        let (key_dim, value_dim) = (nk * dk, nv * dv);
        let (conv_state, rec_state) = match state {
            Some(st) => (Some(st.conv), Some(st.recurrent)),
            None => (None, None),
        };

        let (qkv, conv) = self.causal_conv(self.in_proj_qkv.forward(x.clone()), conv_state);
        let q = qkv.clone().narrow(2, 0, key_dim).reshape([b, s, nk, dk]);
        let k = qkv
            .clone()
            .narrow(2, key_dim, key_dim)
            .reshape([b, s, nk, dk]);
        let v = qkv
            .narrow(2, 2 * key_dim, value_dim)
            .reshape([b, s, nv, dv]);
        let (q, k) = (
            self.to_v_heads(l2_normalise(q)),
            self.to_v_heads(l2_normalise(k)),
        );

        let beta = sigmoid(self.in_proj_b.forward(x.clone())); // [B, S, Hv]
        let decay_rate = self.a_log.val().exp().reshape([1, 1, nv]);
        let a = self.in_proj_a.forward(x.clone()) + self.dt_bias.val().reshape([1, 1, nv]);
        let g = (softplus(a, 1.0) * decay_rate).neg(); // [B, S, Hv]

        let heads_first = |t: Tensor<4>| t.swap_dims(1, 2); // [B, S, H, D] → [B, H, S, D]
        let (y, recurrent) = gated_delta_rule(
            heads_first(q),
            heads_first(k),
            heads_first(v),
            g.swap_dims(1, 2),
            beta.swap_dims(1, 2),
            rec_state,
            self.chunk,
        );
        let y = y.swap_dims(1, 2); // [B, S, Hv, Dv]

        let z = self.in_proj_z.forward(x).reshape([b, s, nv, dv]);
        let y = rms_normalise(y, self.eps) * self.norm.val().reshape([1, 1, 1, dv]) * silu(z);
        (
            self.out_proj.forward(y.reshape([b, s, value_dim])),
            GdnState { conv, recurrent },
        )
    }

    /// Depthwise causal conv + SiLU over `[B, S, C]`. Output `t` sees inputs `t-3..=t`
    /// (kernel 4); before the first token it sees the carried state (zeros when fresh).
    fn causal_conv(&self, x: Tensor<3>, state: Option<Tensor<3>>) -> (Tensor<3>, Tensor<3>) {
        let [b, s, c] = x.dims();
        let w = self.conv1d.val(); // [C, 1, K]
        let kernel = w.dims()[2];
        let w = w.reshape([c, kernel]);
        let x = x.swap_dims(1, 2); // [B, C, S]
        let past = state.unwrap_or_else(|| Tensor::zeros([b, c, kernel - 1], &x.device()));
        let full = Tensor::cat(vec![past, x], 2); // [B, C, K - 1 + S]
        let mut out = Tensor::<3>::zeros([b, c, s], &full.device());
        for j in 0..kernel {
            out = out + full.clone().narrow(2, j, s) * w.clone().narrow(1, j, 1).reshape([1, c, 1]);
        }
        let carry = full.narrow(2, s, kernel - 1); // the last K - 1 inputs
        (silu(out).swap_dims(1, 2), carry)
    }

    /// `[B, S, Hk, D]` → `[B, S, Hv, D]`: each key head is shared by `Hv / Hk` consecutive value
    /// heads (HF's interleaved order).
    fn to_v_heads(&self, t: Tensor<4>) -> Tensor<4> {
        let r = self.num_v_heads / self.num_k_heads;
        if r == 1 {
            return t;
        }
        let [b, s, hk, d] = t.dims();
        t.unsqueeze_dim::<5>(3)
            .expand([b, s, hk, r, d])
            .reshape([b, s, hk * r, d])
    }
}

/// `x / sqrt(sum(x^2) + 1e-6)` over the last dim (Qwen3.5's QK-norm).
fn l2_normalise(x: Tensor<4>) -> Tensor<4> {
    let n = (x.clone().powi_scalar(2).sum_dim(3) + 1e-6).sqrt();
    x / n
}

/// `x / sqrt(mean(x^2) + eps)` over the last dim.
fn rms_normalise(x: Tensor<4>, eps: f64) -> Tensor<4> {
    let r = (x.clone().powi_scalar(2).mean_dim(3) + eps).sqrt();
    x / r
}

impl Default for GatedDeltaNetConfig {
    fn default() -> Self {
        Self::new()
    }
}
