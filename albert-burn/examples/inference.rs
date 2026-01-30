use albert_burn::{AlbertMaskedLM, tokenize_batch};
use burn::backend::ndarray::NdArray;
use burn::tensor::Tensor;

type B = NdArray<f32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();

    // Load pretrained ALBERT model and tokenizer
    println!("Loading ALBERT model...");
    let (model, tokenizer) = AlbertMaskedLM::<B>::pretrained(&device, None)?;

    // Fill-mask example
    let sentence = "The capital of France is [MASK].";
    println!("\nInput: \"{}\"", sentence);

    let (input_ids, attention_mask) = tokenize_batch::<B>(&tokenizer, &[sentence], &device);

    // Find the [MASK] token position
    let input_ids_data = input_ids.to_data();
    let ids: &[i64] = input_ids_data.as_slice().unwrap();
    let mask_token_id = tokenizer
        .token_to_id("[MASK]")
        .expect("[MASK] token not found");
    let mask_pos = ids
        .iter()
        .position(|&id| id == mask_token_id as i64)
        .expect("[MASK] position not found");

    // Forward pass
    let logits = model.forward(input_ids, attention_mask, None);

    // Get logits for the masked position: [1, seq_len, vocab_size] → [vocab_size]
    let [_, _, vocab_size] = logits.dims();
    let mask_logits: Tensor<B, 1> = logits
        .slice([0..1, mask_pos..mask_pos + 1, 0..vocab_size])
        .reshape([vocab_size]);

    // Top-5 predictions
    let top_k = mask_logits.sort_descending_with_indices(0);
    let top_indices = top_k.1;
    let top_values = top_k.0;

    let top_values_data = top_values.to_data();
    let top_indices_data = top_indices.to_data();
    let scores: &[f32] = top_values_data.as_slice().unwrap();
    let indices: &[i64] = top_indices_data.as_slice().unwrap();

    println!("\nTop 5 predictions for [MASK]:");
    for i in 0..5 {
        let token_id = indices[i] as u32;
        let token = tokenizer.id_to_token(token_id).unwrap_or("?".to_string());
        println!("  {}: \"{}\" (logit: {:.2})", i + 1, token, scores[i]);
    }

    Ok(())
}
