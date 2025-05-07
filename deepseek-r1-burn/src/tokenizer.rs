use std::path::Path;
use thiserror::Error;
use tokenizers::Tokenizer;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Failed to load tokenizer: {0}")]
    LoadError(String),
    #[error("Failed to encode text: {0}")]
    EncodeError(String),
    #[error("Failed to decode tokens: {0}")]
    DecodeError(String),
}

pub struct DeepSeekTokenizer {
    tokenizer: Tokenizer,
}

impl DeepSeekTokenizer {
    pub fn new(tokenizer_path: &Path) -> Result<Self, TokenizerError> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| TokenizerError::LoadError(e.to_string()))?;
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.tokenizer
            .encode(text, true)
            .map_err(|e| TokenizerError::EncodeError(e.to_string()))
            .map(|encoding| encoding.get_tokens().to_vec())
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| TokenizerError::DecodeError(e.to_string()))
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
} 