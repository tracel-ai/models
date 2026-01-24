use burn::backend::ndarray::NdArray;
use burn::tensor::linalg::cosine_similarity;
use burn::tensor::{Int, Tensor};
use minilm_burn::{MiniLmModel, mean_pooling, normalize_l2};

type B = NdArray<f32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();

    // Load pretrained model and tokenizer (downloads from HuggingFace)
    println!("Loading model...");
    let (model, tokenizer) = MiniLmModel::<B>::pretrained(&device, Default::default(), None)?;

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

    // Mean pooling + L2 normalize (sentence-transformers default)
    let embeddings = mean_pooling(output.hidden_states, attention_mask);
    let embeddings = normalize_l2(embeddings);

    let [_, hidden_size] = embeddings.dims();
    println!("\nSentence embeddings shape: {:?}", embeddings.dims());

    // Print embeddings (first 5 dimensions of each)
    let embeddings_data = embeddings.to_data();
    for (i, sentence) in sentences.iter().enumerate() {
        let start = i * hidden_size;
        let values: Vec<f32> = (0..5)
            .map(|j| embeddings_data.as_slice::<f32>().unwrap()[start + j])
            .collect();
        println!("\n\"{}\"", sentence);
        println!("  First 5 dims: {:?}...", values);
    }

    // Compute cosine similarity between sentences
    let emb0: Tensor<B, 1> = embeddings.clone().slice([0..1, 0..hidden_size]).squeeze();
    let emb1: Tensor<B, 1> = embeddings.clone().slice([1..2, 0..hidden_size]).squeeze();
    let emb2: Tensor<B, 1> = embeddings.clone().slice([2..3, 0..hidden_size]).squeeze();

    let sim_01: f32 = cosine_similarity(emb0.clone(), emb1.clone(), 0, None).into_scalar();
    let sim_02: f32 = cosine_similarity(emb0, emb2.clone(), 0, None).into_scalar();
    let sim_12: f32 = cosine_similarity(emb1, emb2, 0, None).into_scalar();

    println!("\nCosine similarities:");
    println!("  Sentence 0 vs 1: {:.4}", sim_01);
    println!("  Sentence 0 vs 2: {:.4}", sim_02);
    println!("  Sentence 1 vs 2: {:.4}", sim_12);

    Ok(())
}
