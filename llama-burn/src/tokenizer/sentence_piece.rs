use rust_tokenizers::tokenizer::{
    SentencePieceBpeTokenizer, Tokenizer as BaseTokenizer, TruncationStrategy,
};

use super::Tokenizer;

const BOS_TOKEN_ID: i64 = 1;
const EOS_TOKEN_ID: i64 = 2;

pub struct SentiencePieceTokenizer {
    bpe: SentencePieceBpeTokenizer,
    bos_token_id: i64,
    eos_token_id: i64,
}

impl Tokenizer for SentiencePieceTokenizer {
    /// Load the [SentenciePiece](https://github.com/google/sentencepiece) tokenizer.
    fn new(tokenizer_path: &str) -> Result<Self, String> {
        let bpe = SentencePieceBpeTokenizer::from_file(tokenizer_path, false)
            .map_err(|e| e.to_string())?;

        Ok(Self {
            bpe,
            bos_token_id: BOS_TOKEN_ID,
            eos_token_id: EOS_TOKEN_ID,
        })
    }

    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<i64> {
        let bos_token = if bos { vec![self.bos_token_id] } else { vec![] };
        let eos_token = if eos { vec![self.eos_token_id] } else { vec![] };

        // No text combination
        let tokens = self
            .bpe
            .encode(text, None, usize::MAX, &TruncationStrategy::LongestFirst, 0)
            .token_ids;

        [bos_token, tokens, eos_token]
            .into_iter()
            .flat_map(|t| t.into_iter())
            .map(|t| t as i64)
            .collect()
    }

    fn decode(&self, tokens: Vec<i64>) -> String {
        self.bpe.decode(&tokens, true, false)
    }

    fn bos_id(&self) -> i64 {
        self.bos_token_id as i64
    }

    fn eos_id(&self) -> i64 {
        self.eos_token_id as i64
    }
}
