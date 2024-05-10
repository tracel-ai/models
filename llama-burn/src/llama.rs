use std::time::Instant;

use burn::{
    config::Config,
    module::Module,
    nn::{RotaryEncoding, RotaryEncodingConfig},
    record::{FileRecorder, HalfPrecisionSettings, Recorder, RecorderError},
    tensor::{
        activation::softmax, backend::Backend, Data, Device, ElementConversion, Int, Shape, Tensor,
    },
};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

use crate::{
    sampling::{Sampler, TopP},
    tokenizer::Tiktoken,
    transformer::{KeyValueCache, Transformer, TransformerConfig},
};

#[derive(Config, Debug)]
pub struct LlamaConfig {
    /// The size of the model.
    #[config(default = "4096")]
    pub d_model: usize,
    /// The size of the feed-forward hidden inner features.
    pub hidden_size: usize,
    /// The number of transformer blocks.
    #[config(default = "32")]
    pub num_hidden_layers: usize,
    /// The number of attention heads.
    #[config(default = "32")]
    pub num_attention_heads: usize,
    /// The number of key-value heads.
    pub num_key_value_heads: Option<usize>,
    /// The vocabulary size.
    pub vocab_size: usize,
    /// RMSNorm epsilon
    #[config(default = "1e-5")]
    pub norm_eps: f64,
    /// Rotary positional encoding (RoPE) theta.
    #[config(default = "10000.0")]
    pub rope_theta: f32,
    /// Maximum sequence length for input text.
    #[config(default = "512")]
    pub max_seq_len: usize,
    /// The tokenizer path.
    pub tokenizer: String,
}

impl LlamaConfig {
    /// Llama-3-8B configuration.
    pub fn llama3_8b(tokenizer_path: &str) -> Self {
        // hidden_size = 14336; vocab_size = 128256
        Self::new(14336, 128256, tokenizer_path.to_string())
            .with_num_key_value_heads(Some(8))
            .with_rope_theta(500000.0)
    }

    /// Initialize a new [Llama](Llama) module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Result<Llama<B>, String> {
        let tokenizer = Tiktoken::new(&self.tokenizer)?;
        let num_key_value_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);
        let model = TransformerConfig::new(
            self.vocab_size,
            self.num_hidden_layers,
            self.d_model,
            self.hidden_size,
            self.num_attention_heads,
            num_key_value_heads,
        )
        .with_max_seq_len(self.max_seq_len)
        .with_norm_eps(self.norm_eps)
        .init(device);

        let cache = (0..self.num_hidden_layers)
            .map(|_| KeyValueCache::new(self.max_seq_len))
            .collect::<Vec<_>>();

        let rope = RotaryEncodingConfig::new(
            self.max_seq_len * 2,
            self.d_model / self.num_attention_heads,
        )
        .with_theta(self.rope_theta)
        .init(device);

        Ok(Llama {
            tokenizer,
            model,
            cache,
            rope,
            device: device.clone(),
        })
    }

    /// Load pre-trained Llama checkpoint.
    pub fn load_pretrained<B: Backend>(
        &self,
        checkpoint: &str,
        device: &Device<B>,
    ) -> Result<Llama<B>, String> {
        let mut llama = self.init(device)?;

        // Load weights from torch state_dict
        let load_args = LoadArgs::new(checkpoint.into())
            // Map layers.[i].feed_forward.w1.* -> layers.[i].feed_forward.swiglu.linear_inner.*
            .with_key_remap(
                "(layers\\.[0-9]+\\.feed_forward)\\.w1\\.(.+)",
                "$1.swiglu.linear_inner.$2",
            )
            // Map layers.[i].feed_forward.w3.* -> layers.[i].feed_forward.swiglu.linear_outer.*
            .with_key_remap(
                "(layers\\.[0-9]+\\.feed_forward)\\.w3\\.(.+)",
                "$1.swiglu.linear_outer.$2",
            )
            // Map norm.weight -> norm.gamma for all layers
            .with_key_remap("(.*)norm\\.weight", "${1}norm.gamma");
        println!("Loading record...");
        let now = Instant::now();
        let record = PyTorchFileRecorder::<HalfPrecisionSettings>::new()
            .load(load_args, device)
            .map_err(|e| e.to_string())?;
        let elapsed = now.elapsed().as_secs();
        println!("Loaded in {}s", elapsed);

        llama.model = llama.model.load_record(record);
        println!("Llama record loaded");

        Ok(llama)
    }
}

/// Meta Llama large language model and tokenizer.
pub struct Llama<B: Backend> {
    /// The tokenizer.
    tokenizer: Tiktoken,
    /// Llama decoder-only transformer.
    model: Transformer<B>,
    /// Key-value cache for each transformer block.
    cache: Vec<KeyValueCache<B>>,
    /// Rotary positional encoding (RoPE).
    rope: RotaryEncoding<B>,
    device: Device<B>,
}

impl<B: Backend> Llama<B> {
    /// Generate text sample based on the provided prompt.
    ///
    /// # Arguments
    /// - `prompt`: The prompt string to use for generating the samples.
    /// - `sample_len`: The number of new tokens to generate (i.e., the number of generation steps to take).
    /// - `temperature`: Temperature value for controlling randomness in sampling. High values result in more random sampling.
    /// - `top_p`: Top-p probability threshold for nucleus sampling.
    /// - `seed`: The seed to use when generating random samples.
    ///
    /// # Returns
    /// The generated text.
    pub fn generate(
        &mut self,
        prompt: &str,
        sample_len: usize,
        temperature: f64,
        top_p: f64,
        seed: u64,
    ) -> String {
        let mut tokens = self.tokenize(prompt).unsqueeze::<2>();
        let eos_token = self.tokenizer.eos_id() as i64;

        let mut sampler = if temperature > 0.0 {
            Sampler::TopP(TopP::new(top_p, seed))
        } else {
            Sampler::Argmax
        };

        for _ in 0..sample_len {
            let logits = self
                .model
                .forward(tokens.clone(), &mut self.cache, &self.rope);
            let [batch_size, seq_len, _vocab_size] = logits.dims();
            let mut next_token_logits = logits
                .slice([0..batch_size, seq_len - 1..seq_len])
                .squeeze(1); // [batch_size=1, vocab_size]

            // TODO: naive sampling w/o cumsum tensor op to first test llama implementation correctness
            if temperature > 0.0 {
                next_token_logits = softmax(next_token_logits / temperature, 1);
            };

            let next_token = sampler.sample(next_token_logits);

            // Concatenate the new generated token
            tokens = Tensor::cat(vec![tokens, next_token.clone()], 1);

            if next_token.equal_elem(eos_token).all().into_scalar() {
                break;
            }
        }

        let tokens = tokens
            .into_data()
            .value
            .iter()
            .map(|t| t.elem::<i64>() as usize)
            .collect::<Vec<_>>();

        self.tokenizer.decode(tokens).unwrap()
    }

    /// Encode a string into a tensor of tokens.
    fn tokenize(&self, text: &str) -> Tensor<B, 1, Int> {
        let tokens = self
            .tokenizer
            .encode(text, true, false)
            .into_iter()
            .map(|t| t as i64)
            .collect::<Vec<_>>();

        let shape = Shape::new([tokens.len()]);
        Tensor::<B, 1, Int>::from_data(Data::new(tokens, shape).convert(), &self.device)
    }

    /// Save Llama model to file using the specified recorder.
    pub fn save<R: FileRecorder<B>>(
        self,
        file_path: &str,
        recorder: &R,
    ) -> Result<(), RecorderError> {
        println!("Saving record...");
        let now = Instant::now();
        self.model.save_file(file_path, recorder)?;
        let elapsed = now.elapsed().as_secs();
        println!("Saved in {}s", elapsed);

        Ok(())
    }

    /// Load Llama model from file using the specified recorder.
    pub fn load<R: FileRecorder<B>>(
        mut self,
        file_path: &str,
        recorder: &R,
    ) -> Result<Self, RecorderError> {
        println!("Loading record...");
        let now = Instant::now();
        self.model = self.model.load_file(file_path, recorder, &self.device)?;
        let elapsed = now.elapsed().as_secs();
        println!("Loaded in {}s", elapsed);

        Ok(self)
    }
}
