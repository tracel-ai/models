//! Qwen's byte-level BPE tokenizer, read from the checkpoint's `tokenizer.json`. Pure Rust:
//! `regex` for the pre-split, `unicode-normalization` for NFC, the BPE itself here.
//!
//! The pipeline is the one `tokenizer.json` describes (and HF `tokenizers` runs):
//! 1. added tokens (`<|endoftext|>`, ...) are cut out of the raw text first;
//! 2. the rest is NFC-normalised;
//! 3. split by Qwen's pattern (contractions, letter runs, single digits, punctuation runs,
//!    newlines, whitespace);
//! 4. each piece's UTF-8 bytes are mapped to GPT-2's printable byte alphabet and merged by BPE
//!    rank.
//!
//! No BOS/EOS is added (Qwen adds none).

use regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use unicode_normalization::UnicodeNormalization;

/// Qwen's pre-tokenizer pattern with its last two alternatives `\s+(?!\S)|\s+` folded into
/// `\s+`; the lookahead is applied by hand in [`Tokenizer::split`].
const SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+";

pub struct Tokenizer {
    /// `(left id, right id)` → `(rank, merged id)`.
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Id of each byte's one-character token.
    byte_ids: [u32; 256],
    /// Added (special) tokens, longest first so a longer one wins at the same position.
    added: Vec<(String, u32)>,
    split: Regex,
    pad_id: u32,
    /// The bytes of every token, by id (added tokens as their text).
    bytes: Vec<Vec<u8>>,
}

#[derive(Deserialize)]
struct File {
    model: Model,
    added_tokens: Vec<Added>,
}
#[derive(Deserialize)]
struct Model {
    vocab: HashMap<String, u32>,
    merges: Vec<(String, String)>,
}
#[derive(Deserialize)]
struct Added {
    id: u32,
    content: String,
}

impl Tokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_json(&text)
    }

    pub fn from_json(json: &str) -> Result<Self, String> {
        let f: File = serde_json::from_str(json).map_err(|e| format!("tokenizer.json: {e}"))?;
        let alphabet = byte_alphabet();
        let mut byte_ids = [0u32; 256];
        for (b, ch) in alphabet.iter().enumerate() {
            byte_ids[b] = *f
                .model
                .vocab
                .get(&ch.to_string())
                .ok_or(format!("byte {b} has no token"))?;
        }
        let mut merges = HashMap::with_capacity(f.model.merges.len());
        for (rank, (l, r)) in f.model.merges.iter().enumerate() {
            let id = |s: &str| {
                f.model
                    .vocab
                    .get(s)
                    .copied()
                    .ok_or(format!("merge part `{s}` not in vocab"))
            };
            let joined = format!("{l}{r}");
            merges.insert((id(l)?, id(r)?), (rank as u32, id(&joined)?));
        }
        let mut added: Vec<(String, u32)> = f
            .added_tokens
            .into_iter()
            .map(|a| (a.content, a.id))
            .collect();
        added.sort_by_key(|a| std::cmp::Reverse(a.0.len()));
        let mut bytes = vec![
            Vec::new();
            f.model
                .vocab
                .values()
                .chain(added.iter().map(|a| &a.1))
                .max()
                .map_or(0, |m| *m as usize + 1)
        ];
        let unbyte: HashMap<char, u8> = alphabet
            .iter()
            .enumerate()
            .map(|(b, c)| (*c, b as u8))
            .collect();
        for (s, &id) in &f.model.vocab {
            bytes[id as usize] = s
                .chars()
                .map(|c| {
                    unbyte
                        .get(&c)
                        .copied()
                        .ok_or(format!("token `{s}` has a char outside the byte alphabet"))
                })
                .collect::<Result<_, _>>()?;
        }
        for (s, id) in &added {
            bytes[*id as usize] = s.as_bytes().to_vec();
        }
        let pad_id = added
            .iter()
            .find(|(s, _)| s == "<|endoftext|>")
            .map(|a| a.1)
            .ok_or("no <|endoftext|>")?;
        Ok(Self {
            merges,
            byte_ids,
            added,
            split: Regex::new(SPLIT).map_err(|e| e.to_string())?,
            pad_id,
            bytes,
        })
    }

    /// `<|endoftext|>`, the usual padding token.
    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }

    /// Token ids of `text`, appended to `out` (no special tokens added).
    pub fn encode_into(&self, text: &str, out: &mut Vec<u32>) {
        let mut rest = text;
        while !rest.is_empty() {
            // The earliest added token, if any; plain text before it goes through the pipeline.
            let next = self
                .added
                .iter()
                .filter_map(|(s, id)| rest.find(s.as_str()).map(|at| (at, s.len(), *id)))
                .min_by_key(|&(at, len, _)| (at, std::cmp::Reverse(len)));
            match next {
                Some((at, len, id)) => {
                    self.encode_plain(&rest[..at], out);
                    out.push(id);
                    rest = &rest[at + len..];
                }
                None => {
                    self.encode_plain(rest, out);
                    rest = "";
                }
            }
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 1);
        self.encode_into(text, &mut out);
        out
    }

    /// The text of `ids` (invalid UTF-8, from a token cut mid-character, becomes U+FFFD).
    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&id| self.bytes.get(id as usize).into_iter().flatten().copied())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let nfc: String = text.nfc().collect();
        let mut word = Vec::new();
        for piece in self.split(&nfc) {
            word.clear();
            word.extend(piece.bytes().map(|b| self.byte_ids[b as usize]));
            self.merge(&mut word);
            out.extend_from_slice(&word);
        }
    }

    /// Pieces of `text` by Qwen's pattern, leftmost-first, with `\s+(?!\S)` applied: a run of
    /// whitespace followed by a non-space gives up its last character to the next piece
    /// (so `"  x"` → `" "`, `" x"`), unless the run is a single character.
    fn split<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut pieces = Vec::new();
        let mut pos = 0;
        while pos < text.len() {
            let m = self
                .split
                .find_at(text, pos)
                .expect("the pattern covers every character");
            debug_assert_eq!(m.start(), pos, "pieces are contiguous");
            let mut end = m.end();
            let piece = m.as_str();
            let all_space = piece.chars().all(char::is_whitespace);
            let ends_line = piece.ends_with(['\r', '\n']);
            if all_space && !ends_line && end < text.len() && piece.chars().count() > 1 {
                // followed by a non-space (otherwise the run would have continued)
                end -= piece.chars().next_back().unwrap().len_utf8();
            }
            pieces.push(&text[pos..end]);
            pos = end;
        }
        pieces
    }

    /// Byte-pair merges by rank, lowest first, until no adjacent pair has a merge.
    fn merge(&self, word: &mut Vec<u32>) {
        while word.len() > 1 {
            let best = word
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| {
                    self.merges
                        .get(&(w[0], w[1]))
                        .map(|&(rank, id)| (rank, i, id))
                })
                .min();
            let Some((_, i, id)) = best else { break };
            // Merge every occurrence of this pair, left to right (HF does the same).
            let (a, b) = (word[i], word[i + 1]);
            let mut w = 0;
            let mut r = 0;
            while r < word.len() {
                if r + 1 < word.len() && word[r] == a && word[r + 1] == b {
                    word[w] = id;
                    r += 2;
                } else {
                    word[w] = word[r];
                    r += 1;
                }
                w += 1;
            }
            word.truncate(w);
        }
    }
}

/// GPT-2's map from bytes to printable characters (`bytes_to_unicode`).
fn byte_alphabet() -> [char; 256] {
    let mut out = ['\0'; 256];
    let printable =
        |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
    let mut extra = 0;
    for b in 0..256u32 {
        out[b as usize] = if printable(b) {
            char::from_u32(b).unwrap()
        } else {
            extra += 1;
            char::from_u32(255 + extra).unwrap()
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_alphabet_matches_gpt2() {
        let a = byte_alphabet();
        assert_eq!(a[b' ' as usize], 'Ġ');
        assert_eq!(a[b'\n' as usize], 'Ċ');
        assert_eq!(a[b'A' as usize], 'A');
        assert_eq!(a[0], 'Ā');
    }
}
