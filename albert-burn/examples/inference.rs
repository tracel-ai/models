use albert_burn::{AlbertMaskedLM, AlbertVariant, tokenize_batch};
use burn::backend::ndarray::NdArray;
use burn::tensor::Tensor;
use clap::Parser;

type B = NdArray<f32>;

#[derive(Parser)]
#[command(about = "ALBERT fill-mask inference")]
struct Args {
    /// Model variant: base, large, xlarge, xxlarge
    #[arg(default_value = "base")]
    variant: String,
}

fn parse_variant(s: &str) -> AlbertVariant {
    match s {
        "base" => AlbertVariant::BaseV2,
        "large" => AlbertVariant::LargeV2,
        "xlarge" => AlbertVariant::XLargeV2,
        "xxlarge" => AlbertVariant::XXLargeV2,
        other => {
            eprintln!("Unknown variant '{}', using base", other);
            AlbertVariant::BaseV2
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let variant = parse_variant(&args.variant);
    let device = Default::default();

    println!("Loading ALBERT {:?}...", variant);
    let (model, tokenizer) = AlbertMaskedLM::<B>::pretrained(&device, variant, None)?;

    let sentence = "The capital of France is [MASK].";
    println!("\nInput: \"{}\"", sentence);

    let (input_ids, attention_mask) = tokenize_batch::<B>(&tokenizer, &[sentence], &device);

    let input_ids_data = input_ids.to_data();
    let ids: &[i64] = input_ids_data.as_slice().unwrap();
    let mask_token_id = tokenizer
        .token_to_id("[MASK]")
        .expect("[MASK] token not found");
    let mask_pos = ids
        .iter()
        .position(|&id| id == mask_token_id as i64)
        .expect("[MASK] position not found");

    let logits = model.forward(input_ids, attention_mask, None);

    let [_, _, vocab_size] = logits.dims();
    let mask_logits: Tensor<B, 1> = logits
        .slice([0..1, mask_pos..mask_pos + 1, 0..vocab_size])
        .reshape([vocab_size]);

    let top_k = mask_logits.sort_descending_with_indices(0);
    let top_values_data = top_k.0.to_data();
    let top_indices_data = top_k.1.to_data();
    let scores: &[f32] = top_values_data.as_slice().unwrap();
    let indices: &[i64] = top_indices_data.as_slice().unwrap();

    println!("\nTop 5 predictions for [MASK]:");
    for i in 0..5 {
        let token_id = indices[i] as u32;
        let token = tokenizer
            .id_to_token(token_id)
            .unwrap_or("?".to_string())
            .trim_start_matches('▁')
            .to_string();
        println!("  {}: \"{}\" (logit: {:.4})", i + 1, token, scores[i]);
    }

    Ok(())
}
