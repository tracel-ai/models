use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

/// Tokenize sentences and return padded input tensors.
///
/// # Returns
/// `(input_ids, attention_mask)` — both shaped `[batch_size, max_seq_len]`.
pub fn tokenize_batch<B: Backend>(
    tokenizer: &tokenizers::Tokenizer,
    sentences: &[&str],
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 2>) {
    let encodings = tokenizer
        .encode_batch(sentences.to_vec(), true)
        .expect("Failed to tokenize");

    let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap();
    let batch_size = sentences.len();

    let mut input_ids_data = vec![0i64; batch_size * max_len];
    let mut attention_mask_data = vec![0.0f32; batch_size * max_len];

    for (i, encoding) in encodings.iter().enumerate() {
        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        for (j, &id) in ids.iter().enumerate() {
            input_ids_data[i * max_len + j] = id as i64;
            attention_mask_data[i * max_len + j] = mask[j] as f32;
        }
    }

    let input_ids = Tensor::<B, 1, Int>::from_data(input_ids_data.as_slice(), device)
        .reshape([batch_size, max_len]);
    let attention_mask = Tensor::<B, 1>::from_data(attention_mask_data.as_slice(), device)
        .reshape([batch_size, max_len]);

    (input_ids, attention_mask)
}
