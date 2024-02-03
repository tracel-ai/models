pub trait Tokenizer: Send + Sync {
    /// Converts a text string into a sequence of tokens.
    fn encode(&self, value: &str) -> Vec<usize>;

    /// Converts a sequence of tokens back into a text string.
    fn decode(&self, tokens: &[usize]) -> String;

    /// Gets the size of the tokenizer's vocabulary.
    fn vocab_size(&self) -> usize;

    /// Gets the token used for padding sequences to a consistent length.
    fn pad_token(&self) -> usize;

    /// Gets the string representation of the padding token.
    /// The default implementation uses `decode` on the padding token.
    fn pad_token_value(&self) -> String {
        self.decode(&[self.pad_token()])
    }
}

/// Struct represents a specific tokenizer using the Roberta BPE tokenization strategy.
pub struct BertTokenizer {
    // The underlying tokenizer from the `tokenizers` library.
    tokenizer: tokenizers::Tokenizer,
    pad_token: usize,
}

// Default implementation for creating a new BertTokenizer.
// Downloads tokenizer from given model_name (eg: "roberta-base").
// Pad_token_id is the id of the padding token used to convert sequences to a consistent length.
// specified in the model's config.json.
impl BertTokenizer {
    pub fn new(model_name: String, pad_token_id: usize) -> Self {
        Self {
            tokenizer: tokenizers::Tokenizer::from_pretrained(model_name, None).unwrap(),
            pad_token: pad_token_id,
        }
    }
}

// Implementation of the Tokenizer trait for BertTokenizer.
impl Tokenizer for BertTokenizer {
    /// Convert a text string into a sequence of tokens using the BERT model's tokenization strategy.
    fn encode(&self, value: &str) -> Vec<usize> {
        let tokens = self.tokenizer.encode(value, true).unwrap();
        tokens.get_ids().iter().map(|t| *t as usize).collect()
    }

    /// Converts a sequence of tokens back into a text string.
    fn decode(&self, tokens: &[usize]) -> String {
        let tokens = tokens.iter().map(|t| *t as u32).collect::<Vec<u32>>();
        self.tokenizer.decode(&tokens, false).unwrap()
    }

    /// Gets the size of the BERT tokenizer's vocabulary.
    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Gets the token used for padding sequences to a consistent length.
    fn pad_token(&self) -> usize {
        self.pad_token
    }
}
