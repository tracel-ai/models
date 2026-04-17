use bert_burn::data::{BertInputBatcher, BertTokenizer};
use bert_burn::fill_mask::fill_mask;
use bert_burn::loader::{download_hf_model, load_model_config, load_pretrained_masked_lm};
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::backend::Backend;
use burn_flex::{Flex, FlexDevice};
use std::env;
use std::sync::Arc;

pub fn launch<B: Backend>(device: B::Device) {
    let args: Vec<String> = env::args().collect();
    let default_model = "roberta-base".to_string();
    let model_variant = if args.len() > 1 {
        &args[1]
    } else {
        &default_model
    };

    println!("Model variant: {}", model_variant);

    let text_samples = vec![
        "Paris is the <mask> of France.".to_string(),
        "The goal of life is <mask>.".to_string(),
    ];

    let (config_file, model_file) =
        download_hf_model(model_variant).expect("Failed to download BERT model from HF Hub");
    let model_config = load_model_config(config_file).expect("Failed to load BERT config");

    let mut model = model_config.init_with_lm_head::<B>(&device);
    load_pretrained_masked_lm(&mut model, &model_file)
        .expect("Failed to load pretrained BERT masked LM weights");

    let tokenizer = Arc::new(BertTokenizer::new(
        model_variant.to_string(),
        model_config.pad_token_id,
    ));

    let batcher = Arc::new(BertInputBatcher::new(
        tokenizer.clone(),
        model_config.max_seq_len.unwrap(),
    ));

    let input = batcher.batch(text_samples.clone(), &device);
    let [batch_size, _seq_len] = input.tokens.dims();
    println!("Input: {:?} // (Batch Size, Seq_len)", input.tokens.shape());

    let output = fill_mask(&model, &model_config, tokenizer.as_ref(), input);

    for i in 0..batch_size {
        let input = &text_samples[i];
        let result = &output[i];
        println!("Input: {}", input);
        for fill_mask_result in result.iter() {
            let mask_idx = fill_mask_result.mask_idx;
            let top_k = &fill_mask_result.top_k;
            for (k, (score, token)) in top_k.iter().enumerate() {
                println!(
                    "Top {} Prediction for {}: {} (Score: {:.4})",
                    k + 1,
                    mask_idx,
                    token,
                    score
                );
            }
        }
    }
}

fn main() {
    launch::<Flex>(FlexDevice);
}
