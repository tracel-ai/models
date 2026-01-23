use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

// TODO: Consider proposing MeanPooler for Burn's nn module

/// Mean pooling over sequence dimension with attention mask.
///
/// This implements the standard mean pooling used in sentence transformers:
/// - Masks out padding tokens using the attention mask
/// - Computes mean over the sequence dimension
///
/// # Arguments
/// - `hidden_states`: Hidden states from the encoder [batch_size, seq_len, hidden_size]
/// - `attention_mask`: Attention mask where 1 = real token, 0 = padding [batch_size, seq_len]
///
/// # Returns
/// Pooled sentence embeddings [batch_size, hidden_size]
pub fn mean_pooling<B: Backend>(
    hidden_states: Tensor<B, 3>,
    attention_mask: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let [batch_size, seq_len, hidden_size] = hidden_states.dims();

    // Expand mask to match hidden states: [batch, seq_len] -> [batch, seq_len, hidden]
    let mask_expanded = attention_mask
        .clone()
        .reshape([batch_size, seq_len, 1])
        .expand([batch_size, seq_len, hidden_size]);

    // Apply mask and sum
    let masked_hidden = hidden_states * mask_expanded;
    let sum_hidden: Tensor<B, 2> = masked_hidden.sum_dim(1).reshape([batch_size, hidden_size]);

    // Count non-padding tokens per batch
    let token_counts: Tensor<B, 2> = attention_mask
        .sum_dim(1)
        .reshape([batch_size, 1])
        .expand([batch_size, hidden_size])
        .clamp_min(1e-9); // Avoid division by zero

    // Mean = sum / count
    sum_hidden / token_counts
}

/// L2 normalize embeddings (each row to unit length).
///
/// This matches the default behavior of sentence-transformers.
pub fn normalize_l2<B: Backend>(embeddings: Tensor<B, 2>) -> Tensor<B, 2> {
    use burn::tensor::linalg::{Norm, vector_normalize};
    vector_normalize(embeddings, Norm::L2, 1, 1e-12)
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;
    use burn::tensor::Tolerance;

    type B = NdArray<f32>;

    #[test]
    fn test_mean_pooling() {
        let device = Default::default();

        // Hidden states: [2, 3, 4] (batch=2, seq=3, hidden=4)
        let hidden = Tensor::<B, 3>::from_floats(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [0.0, 0.0, 0.0, 0.0],
                ], // seq 3 is padding
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0, 3.0],
                ], // all real tokens
            ],
            &device,
        );

        // Attention mask: [2, 3]
        let mask = Tensor::<B, 2>::from_floats([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], &device);

        let pooled = mean_pooling(hidden, mask);

        // Expected:
        // Batch 0: mean of [[1,2,3,4], [5,6,7,8]] = [3, 4, 5, 6]
        // Batch 1: mean of [[1,1,1,1], [2,2,2,2], [3,3,3,3]] = [2, 2, 2, 2]
        let expected = TensorData::from([[3.0f32, 4.0, 5.0, 6.0], [2.0, 2.0, 2.0, 2.0]]);
        pooled
            .into_data()
            .assert_approx_eq(&expected, Tolerance::<f32>::rel_abs(1e-5, 1e-5));
    }
}
