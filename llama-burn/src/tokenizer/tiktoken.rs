use std::{
    collections::HashSet,
    fs::File,
    io::{BufRead, BufReader},
};

use base64::{engine::general_purpose::STANDARD, Engine};
use rustc_hash::FxHashMap as HashMap;
use tiktoken_rs::CoreBPE;

use super::Tokenizer;

const BOS_TOKEN: &str = "<|begin_of_text|>";
const EOS_TOKEN: &str = "<|end_of_text|>";

const NUM_RESERVED_SPECIAL_TOKENS: usize = 256;
const SPECIAL_TOKENS: [&str; 10] = [
    BOS_TOKEN,
    EOS_TOKEN,
    "<|reserved_special_token_0|>",
    "<|reserved_special_token_1|>",
    "<|reserved_special_token_2|>",
    "<|reserved_special_token_3|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|reserved_special_token_4|>",
    "<|eot_id|>", // end of turn
];
const PATTERN: &str = r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#;

#[derive(Debug, Clone)]
pub struct Tiktoken {
    bpe: CoreBPE,
    bos_token_id: usize,
    eos_token_id: usize,
}

impl Tokenizer for Tiktoken {
    /// Load the [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    fn new(tiktoken_bpe_file: &str) -> Result<Self, String> {
        let file = File::open(tiktoken_bpe_file).map_err(|e| e.to_string())?;
        let mut mergeable_ranks: HashMap<Vec<u8>, usize> = HashMap::default();

        for line in BufReader::new(file).lines().flatten() {
            let mut parts = line.split(' ');
            let token = STANDARD
                .decode(parts.next().ok_or("Missing token")?)
                .map_err(|e| e.to_string())?;
            let rank = parts
                .next()
                .ok_or("Missing rank")?
                .parse::<usize>()
                .map_err(|e| e.to_string())?;

            mergeable_ranks.insert(token, rank);
        }

        let special_tokens = [
            SPECIAL_TOKENS
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>(),
            (5..NUM_RESERVED_SPECIAL_TOKENS - 5)
                .into_iter()
                .map(|i| format!("<|reserved_special_token_{i}|>"))
                .collect::<Vec<_>>(),
        ]
        .concat();
        let special_tokens = special_tokens
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s, i))
            .collect::<HashMap<String, usize>>();

        let num_base_tokens = mergeable_ranks.len();
        let bos_token_id = special_tokens[BOS_TOKEN] + num_base_tokens;
        let eos_token_id = special_tokens[EOS_TOKEN] + num_base_tokens;

        let bpe =
            CoreBPE::new(mergeable_ranks, special_tokens, PATTERN).map_err(|e| e.to_string())?;
        Ok(Self {
            bpe,
            bos_token_id,
            eos_token_id,
        })
    }

    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<i64> {
        let bos_token = if bos { vec![self.bos_token_id] } else { vec![] };
        let eos_token = if eos { vec![self.eos_token_id] } else { vec![] };

        // `allowed_special` is an empty set
        let tokens = self.bpe.encode(text, HashSet::new());

        [bos_token, tokens, eos_token]
            .into_iter()
            .flat_map(|t| t.into_iter())
            .map(|t| t as i64)
            .collect()
    }

    fn decode(&self, tokens: Vec<i64>) -> String {
        self.bpe
            .decode(tokens.into_iter().map(|t| t as usize).collect())
            .expect("Should decode tokens")
    }

    fn bos_id(&self) -> i64 {
        self.bos_token_id as i64
    }

    fn eos_id(&self) -> i64 {
        self.eos_token_id as i64
    }
}
