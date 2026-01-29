use burn::backend::ndarray::NdArray;
use burn::tensor::Tensor;
use burn::tensor::linalg::cosine_similarity;
use minilm_burn::{MiniLmModel, mean_pooling, normalize_l2, tokenize_batch};

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

    // Tokenize and prepare input tensors
    let (input_ids, attention_mask) = tokenize_batch::<B>(&tokenizer, &sentences, &device);

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
