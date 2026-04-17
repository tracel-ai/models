use bert_burn::data::{BertInputBatcher, BertTokenizer};
use bert_burn::loader::{download_hf_model, load_model_config, load_pretrained};
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
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
        "Jays power up to take finale Contrary to popular belief, the power never really \
                 snapped back at SkyDome on Sunday. The lights came on after an hour delay, but it \
                 took some extra time for the batting orders to provide some extra wattage."
            .to_string(),
        "Yemen Sentences 15 Militants on Terror Charges A court in Yemen has sentenced one \
                 man to death and 14 others to prison terms for a series of attacks and terrorist \
                 plots in 2002, including the bombing of a French oil tanker."
            .to_string(),
        "IBM puts grids to work at U.S. Open IBM will put a collection of its On \
                 Demand-related products and technologies to this test next week at the U.S. Open \
                 tennis championships, implementing a grid-based infrastructure capable of running \
                 multiple workloads including two not associated with the tournament."
            .to_string(),
    ];

    let (config_file, model_file) = download_hf_model(model_variant);
    let model_config = load_model_config(config_file);

    let mut model = model_config.init::<B>(&device);
    load_pretrained(&mut model, &model_file).expect("Failed to load pretrained BERT weights");

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
    println!("Input: {}", input.tokens);

    let output = model.forward(input);

    let cls_token_idx = 0;
    let d_model = model_config.hidden_size;
    let sentence_embedding = output.hidden_states.clone().slice([
        0..batch_size,
        cls_token_idx..cls_token_idx + 1,
        0..d_model,
    ]);

    let sentence_embedding: Tensor<B, 2> = sentence_embedding.squeeze_dim(1);
    println!("Roberta Sentence embedding: {}", sentence_embedding);
}

fn main() {
    launch::<Flex>(FlexDevice);
}
