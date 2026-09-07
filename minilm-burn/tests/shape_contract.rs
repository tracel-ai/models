//! The shape assertions on the public entry points should reject mismatched masks.

use burn::tensor::Tensor;
use minilm_burn::mean_pooling;

#[test]
fn mean_pooling_accepts_matching_mask() {
    let device = Default::default();
    let hidden = Tensor::<3>::zeros([2, 3, 4], &device);
    let mask = Tensor::<2>::zeros([2, 3], &device);

    let pooled = mean_pooling(hidden, mask);

    assert_eq!(pooled.dims(), [2, 4]);
}

#[test]
#[should_panic(expected = "axis 1 expected 3, got 2")]
fn mean_pooling_rejects_mask_with_wrong_seq_len() {
    let device = Default::default();
    let hidden = Tensor::<3>::zeros([2, 3, 4], &device);
    let mask = Tensor::<2>::zeros([2, 2], &device);

    let _ = mean_pooling(hidden, mask);
}
