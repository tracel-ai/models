pub trait Tokenizer {
    /// Load the tokenizer from the provided path.
    fn new(tokenizer_path: &str) -> Result<Self, String>
    where
        Self: Sized;

    /// Encode a string into a list of token identifiers.
    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<i64>;

    /// Decode a list of token identifiers into a string.
    fn decode(&self, tokens: Vec<i64>) -> String;

    /// Beginning of sentence token identifier.
    fn bos_id(&self) -> i64;

    /// End of sentence token identifier.
    fn eos_id(&self) -> i64;
}
