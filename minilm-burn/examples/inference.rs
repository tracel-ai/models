use burn::backend::ndarray::NdArray;
use burn::tensor::{Int, Tensor};
use minilm_burn::{mean_pooling, MiniLmConfig, MiniLmModel};
use tokenizers::Tokenizer;

type B = NdArray<f32>;

const MODEL_NAME: &str = "sentence-transformers/all-MiniLM-L12-v2";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();

    println!("Downloading model from {}...", MODEL_NAME);

    // Download model files from HuggingFace
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(MODEL_NAME.to_string());

    let config_path = repo.get("config.json")?;
    let weights_path = repo.get("model.safetensors")?;
    let tokenizer_path = repo.get("tokenizer.json")?;

    println!("Loading model...");
    let config = MiniLmConfig::load_from_hf(&config_path)?;
    let mut model: MiniLmModel<B> = config.init(&device);
    minilm_burn::load_pretrained(&mut model, &weights_path)?;

    println!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

    // Example sentences
    let sentences = vec![
        "This is an example sentence.",
        "Each sentence is converted to a vector.",
        "Rust is a systems programming language.",
    ];

    println!("\nEncoding {} sentences...", sentences.len());

    // Tokenize all sentences
    let encodings = tokenizer
        .encode_batch(sentences.clone(), true)
        .map_err(|e| format!("Failed to encode: {}", e))?;

    // Find max length for padding
    let max_len = encodings
        .iter()
        .map(|e: &tokenizers::Encoding| e.get_ids().len())
        .max()
        .unwrap();

    // Prepare input tensors
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

    let input_ids: Tensor<B, 2, Int> =
        Tensor::<B, 1, Int>::from_data(input_ids_data.as_slice(), &device)
            .reshape([batch_size, max_len]);
    let attention_mask: Tensor<B, 2> =
        Tensor::<B, 1>::from_data(attention_mask_data.as_slice(), &device)
            .reshape([batch_size, max_len]);

    println!("Input shape: {:?}", input_ids.dims());

    // Forward pass
    println!("Running inference...");
    let output = model.forward(input_ids, attention_mask.clone(), None);

    // Mean pooling
    let embeddings = mean_pooling(output.hidden_states, attention_mask);

    println!("\nSentence embeddings shape: {:?}", embeddings.dims());

    // Print embeddings (first 5 dimensions of each)
    let embeddings_data = embeddings.to_data();
    for (i, sentence) in sentences.iter().enumerate() {
        let start = i * 384;
        let values: Vec<f32> = (0..5)
            .map(|j| embeddings_data.as_slice::<f32>().unwrap()[start + j])
            .collect();
        println!("\n\"{}\"", sentence);
        println!("  First 5 dims: {:?}...", values);
    }

    // Compute cosine similarity between sentences
    let emb0: Tensor<B, 1> = embeddings.clone().slice([0..1, 0..384]).squeeze();
    let emb1: Tensor<B, 1> = embeddings.clone().slice([1..2, 0..384]).squeeze();
    let emb2: Tensor<B, 1> = embeddings.clone().slice([2..3, 0..384]).squeeze();

    let sim_01 = cosine_similarity(&emb0, &emb1);
    let sim_02 = cosine_similarity(&emb0, &emb2);
    let sim_12 = cosine_similarity(&emb1, &emb2);

    println!("\nCosine similarities:");
    println!("  Sentence 0 vs 1: {:.4}", sim_01);
    println!("  Sentence 0 vs 2: {:.4}", sim_02);
    println!("  Sentence 1 vs 2: {:.4}", sim_12);

    Ok(())
}

fn cosine_similarity(a: &Tensor<B, 1>, b: &Tensor<B, 1>) -> f32 {
    let dot = a.clone().mul(b.clone()).sum().into_scalar();
    let norm_a = a.clone().powf_scalar(2.0).sum().sqrt().into_scalar();
    let norm_b = b.clone().powf_scalar(2.0).sum().sqrt().into_scalar();
    dot / (norm_a * norm_b)
}
