use burn::backend::libtorch::{LibTorch, TchTensor};
use burn::tensor::ops::FloatTensorOps;
use burn::{
    backend::libtorch::TchElement,
    tensor::{
        activation::softmax,
        f16,
        ops::{BoolTensor, FloatTensor},
        Bool, Tensor,
    },
};

pub trait Backend: burn::tensor::backend::Backend {
    fn scaled_dot_product_attention(
        query: FloatTensor<Self, 4>,
        key: FloatTensor<Self, 4>,
        value: FloatTensor<Self, 4>,
        mask: Option<BoolTensor<Self, 4>>,
        scale: f64,
    ) -> FloatTensor<Self, 4> {
        default::<Self>(query, key, value, mask, scale)
    }
}

impl<F: TchElement> Backend for LibTorch<F> {
    fn scaled_dot_product_attention(
        query: FloatTensor<Self, 4>,
        key: FloatTensor<Self, 4>,
        value: FloatTensor<Self, 4>,
        mask: Option<BoolTensor<Self, 4>>,
        scale: f64,
    ) -> FloatTensor<Self, 4> {
        if mask.is_none() || true {
            return default::<Self>(query, key, value, mask, scale);
        }

        let mask = mask.map(|m| m.tensor.logical_not());
        let tensor = tch::Tensor::f_scaled_dot_product_attention::<tch::Tensor>(
            &query.tensor,
            &key.tensor,
            &value.tensor,
            mask,
            0.0,
            false,
            scale,
        )
        .unwrap();

        TchTensor::new(tensor)
    }
}

fn default<B: Backend>(
    query: FloatTensor<B, 4>,
    key: FloatTensor<B, 4>,
    value: FloatTensor<B, 4>,
    mask: Option<BoolTensor<B, 4>>,
    scale: f64,
) -> FloatTensor<B, 4> {
    let query: Tensor<B, 4> = Tensor::from_primitive(query);
    let key = Tensor::from_primitive(key);
    let value = Tensor::from_primitive(value);
    let mask = mask.map(Tensor::from_primitive);

    let mut scores = query.matmul(key.swap_dims(2, 3)).div_scalar(scale);

    if let Some(mask) = mask {
        scores = scores.mask_fill(mask, f32::NEG_INFINITY);
    }

    let scores = softmax(scores, 3);

    // Output [batch_size, num_heads, seq_len, head_dim]
    scores.matmul(value).into_primitive()
}
