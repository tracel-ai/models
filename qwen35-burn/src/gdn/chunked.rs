//! Chunked formulation after Crane (`ops/gdn/chunked.rs`, MIT) and HF Transformers; see the README.
//!
//! The gated delta rule, chunked.
//!
//! Per token the rule is (state `S` is `[K, V]` per head):
//!
//! ```text
//! S = S * exp(g)                  decay
//! delta = beta * (v - S^T k)      what the memory gets wrong about v
//! S = S + k delta^T               write the correction
//! y = S^T q                       read
//! ```
//!
//! Done token by token that is a long chain of tiny steps, which a GPU runs badly. Here the
//! sequence is cut into chunks of `C` tokens. Inside a chunk the dependencies between tokens are
//! one triangular linear system, solved with batched matrix products; only the state carry from
//! chunk to chunk is sequential (`S / C` steps instead of `S`). Same maths as HF's
//! `torch_chunk_gated_delta_rule`, llama.cpp's `build_delta_net_chunking` and Crane's
//! `ops/gdn/chunked.rs`.
//!
//! The one step that usually needs a hand-written kernel is the inverse of the per-chunk matrix
//! `I + A` (`A` strictly lower-triangular). Here it is a blocked forward substitution in plain
//! tensor ops, see [`unit_lower_inverse`], so the whole function runs on every backend and
//! autodiff differentiates through it.

use burn::tensor::Tensor;
use burn::tensor::ops::PadMode;

/// Default chunk size (llama.cpp, HF and Crane use 64).
pub const DEFAULT_CHUNK: usize = 64;

/// The gated delta rule over a whole sequence.
///
/// Shapes (`B` batch, `H` heads, `S` tokens):
/// * `q`, `k`: `[B, H, S, K]`, already L2-normalised per head (Qwen3.5 does that before the
///   rule). `q` is scaled by `1 / sqrt(K)` **here**, not by the caller.
/// * `v`: `[B, H, S, V]`
/// * `g`: `[B, H, S]`, log of the per-token decay (`<= 0`)
/// * `beta`: `[B, H, S]`, write strength in `(0, 1)`
/// * `state`: `[B, H, K, V]` carried in from earlier tokens, or `None` for zeros
///
/// Returns `y: [B, H, S, V]` and the state after the last token, `[B, H, K, V]`. Keep `q`, `k`,
/// `v`, `g`, `beta` and the state in f32 (Qwen3.5's `mamba_ssm_dtype` is float32).
///
/// # Panics
/// If `S == 0`, `chunk == 0`, or the shapes disagree.
pub fn gated_delta_rule(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<3>,
    beta: Tensor<3>,
    state: Option<Tensor<4>>,
    chunk: usize,
) -> (Tensor<4>, Tensor<4>) {
    let [b, h, s, kd] = q.dims();
    let vd = v.dims()[3];
    assert!(
        s > 0 && chunk > 0,
        "need at least one token and a chunk size"
    );
    assert_eq!(k.dims(), [b, h, s, kd], "k shape");
    assert_eq!(v.dims(), [b, h, s, vd], "v shape");
    assert_eq!(g.dims(), [b, h, s], "g shape");
    assert_eq!(beta.dims(), [b, h, s], "beta shape");
    let device = q.device();
    let bh = b * h;

    // A sequence shorter than one chunk is one (smaller) chunk.
    let c = chunk.min(s);
    let n = s.div_ceil(c);
    let pad = n * c - s;
    // Padded tokens have beta = 0 and g = 0: they write nothing and add no decay, so they
    // change neither the real tokens' outputs nor the final state. Their outputs are dropped.
    let pad3 = |x: Tensor<4>| {
        if pad > 0 {
            x.pad([(0, 0), (0, 0), (0, pad), (0, 0)], PadMode::Constant(0.0))
        } else {
            x
        }
    };
    let pad2 = |x: Tensor<3>| {
        if pad > 0 {
            x.pad([(0, 0), (0, 0), (0, pad)], PadMode::Constant(0.0))
        } else {
            x
        }
    };

    let scale = 1.0 / (kd as f64).sqrt();
    let rows = bh * n;
    let q = pad3(q.mul_scalar(scale)).reshape([rows, c, kd]);
    let k = pad3(k).reshape([rows, c, kd]);
    let v = pad3(v).reshape([rows, c, vd]);
    let g = pad2(g).reshape([rows, c]);
    let beta = pad2(beta).reshape([rows, c, 1]);

    let g_cum = g.cumsum(1); // [rows, C]
    let k_beta = k.clone() * beta.clone();
    let v_beta = v * beta;

    // decay[i, j] = exp(g_cum[i] - g_cum[j]) for i >= j. Zeroing the exponent above the
    // diagonal before exp keeps those entries at 1 instead of overflowing; they are masked next.
    let diff = (g_cum.clone().unsqueeze_dim::<3>(2) - g_cum.clone().unsqueeze_dim::<3>(1)).tril(0);
    let decay = diff.exp();
    let k_t = k.clone().swap_dims(1, 2); // [rows, K, C]
    let a = (k_beta.clone().matmul(k_t.clone()) * decay.clone()).tril(-1);
    let qk = (q.clone().matmul(k_t) * decay).tril(0);

    let t_inv = unit_lower_inverse(a);
    let g_exp = g_cum.clone().exp().unsqueeze_dim::<3>(2); // [rows, C, 1]
    let v_fixed = t_inv.clone().matmul(v_beta); // [rows, C, V]
    let k_decayed = t_inv.matmul(k_beta * g_exp.clone()); // [rows, C, K]
    let q_decayed = q * g_exp; // [rows, C, K]
    let g_last = g_cum.clone().narrow(1, c - 1, 1); // [rows, 1]
    let k_to_end = k * (g_last.clone() - g_cum).exp().unsqueeze_dim::<3>(2); // [rows, C, K]
    let g_last_exp = g_last.exp().reshape([bh, n, 1, 1]);

    let v_fixed = v_fixed.reshape([bh, n, c, vd]);
    let k_decayed = k_decayed.reshape([bh, n, c, kd]);
    let qk = qk.reshape([bh, n, c, c]);
    let q_decayed = q_decayed.reshape([bh, n, c, kd]);
    let k_to_end_t = k_to_end.reshape([bh, n, c, kd]).swap_dims(2, 3); // [BH, N, K, C]

    let mut st = match state {
        Some(st) => {
            assert_eq!(st.dims(), [b, h, kd, vd], "state shape");
            st.reshape([bh, kd, vd])
        }
        None => Tensor::zeros([bh, kd, vd], &device),
    };
    let take = |x: &Tensor<4>, i: usize| -> Tensor<3> {
        let [a, _, r, cc] = x.dims();
        x.clone().narrow(1, i, 1).reshape([a, r, cc])
    };
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let v_new = take(&v_fixed, i) - take(&k_decayed, i).matmul(st.clone()); // [BH, C, V]
        ys.push(take(&q_decayed, i).matmul(st.clone()) + take(&qk, i).matmul(v_new.clone()));
        let decay = g_last_exp.clone().narrow(1, i, 1).reshape([bh, 1, 1]);
        st = st * decay + take(&k_to_end_t, i).matmul(v_new);
    }
    let y = Tensor::cat(ys, 1).narrow(1, 0, s).reshape([b, h, s, vd]);
    (y, st.reshape([b, h, kd, vd]))
}

/// Diagonal block size of [`unit_lower_inverse`] (FLA uses 16 as well).
pub const INVERSE_BLOCK: usize = 16;

/// `(I + A)^-1` for a batch of strictly lower-triangular `A`: `[rows, C, C]`.
///
/// Blocked forward substitution, the way FLA's `solve_tril` does it:
/// 1. the `16 x 16` diagonal blocks are inverted by exact row-by-row substitution, all blocks of
///    all rows in one batch (16 small steps);
/// 2. block row `i` of the inverse is then `-X_ii (A_i,<i X_<i,<i)`: two matrix products per
///    block row (3 rows for `C = 64`).
///
/// Substitution is backward stable. A Neumann/power series is not: on real text the keys in a
/// chunk are similar, the entries of `A` approach 1, and its powers grow to ~1e18 before they
/// cancel (see the `similar_keys_stay_accurate` test).
pub fn unit_lower_inverse(a: Tensor<3>) -> Tensor<3> {
    let [rows, c, c2] = a.dims();
    assert_eq!(c, c2, "square matrices");
    let b = INVERSE_BLOCK.min(c);
    let nb = c.div_ceil(b);
    let cp = nb * b;
    // Zero padding keeps A strictly lower; the top-left C x C of the padded inverse is the answer.
    let a = if cp > c {
        a.pad([(0, 0), (0, cp - c), (0, cp - c)], PadMode::Constant(0.0))
    } else {
        a
    };

    // 1. All diagonal blocks at once: [nb * rows, b, b].
    let diag = Tensor::cat(
        (0..nb)
            .map(|j| a.clone().narrow(1, j * b, b).narrow(2, j * b, b))
            .collect(),
        0,
    );
    let diag_inv = substitute(diag);
    let block = |j: usize| diag_inv.clone().narrow(0, j * rows, rows); // [rows, b, b]

    // 2. Grow the inverse one block row at a time.
    let mut x = block(0);
    for i in 1..nb {
        let a_row = a.clone().narrow(1, i * b, b).narrow(2, 0, i * b); // [rows, b, i*b]
        let x_ii = block(i);
        let left = x_ii.clone().matmul(a_row.matmul(x.clone())).neg(); // [rows, b, i*b]
        let top = Tensor::cat(vec![x, Tensor::zeros([rows, i * b, b], &left.device())], 2);
        let bottom = Tensor::cat(vec![left, x_ii], 2);
        x = Tensor::cat(vec![top, bottom], 1);
    }
    if cp > c {
        x.narrow(1, 0, c).narrow(2, 0, c)
    } else {
        x
    }
}

/// Exact `(I + A)^-1` by forward substitution, one row per step: row `i` is
/// `e_i - A[i, :i] X[:i, :]`.
fn substitute(a: Tensor<3>) -> Tensor<3> {
    let [n, b, _] = a.dims();
    let eye = Tensor::<2>::eye(b, &a.device())
        .unsqueeze_dim::<3>(0)
        .expand([n, b, b]);
    let mut x = eye.clone().narrow(1, 0, 1); // row 0: e_0
    for i in 1..b {
        let row = eye.clone().narrow(1, i, 1)
            - a.clone().narrow(1, i, 1).narrow(2, 0, i).matmul(x.clone());
        x = Tensor::cat(vec![x, row], 1);
    }
    x
}
