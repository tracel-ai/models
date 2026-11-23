//! The tokenizer against HF `tokenizers` 0.23 on hard texts: whitespace runs, newlines,
//! contractions, digits, Swedish, decomposed Unicode (NFC), emoji, CJK, other scripts, special
//! tokens inside text, code, XML. Reference ids are in `fixtures/tokenizer/cases.json`. Needs the
//! checkpoint's `tokenizer.json` (20 MB), so it is ignored by default:
//!
//! ```sh
//! QWEN35_DIR=path/to/model cargo test -p qwen35-burn --release --test tokenizer -- --ignored
//! ```

use qwen35_burn::Tokenizer;
use std::path::PathBuf;
use unicode_normalization::UnicodeNormalization;

#[test]
#[ignore = "needs the checkpoint's tokenizer.json"]
fn matches_hf_tokenizers() {
    let dir = PathBuf::from(std::env::var("QWEN35_DIR").expect("set QWEN35_DIR"));
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let cases: Vec<serde_json::Value> =
        serde_json::from_str(include_str!("fixtures/tokenizer/cases.json")).unwrap();
    let mut failed = 0;
    for c in &cases {
        let text = c["text"].as_str().unwrap();
        let want: Vec<u32> = c["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let got = tok.encode(text);
        if got != want {
            failed += 1;
            println!("MISMATCH {text:?}\n  ours {got:?}\n  hf   {want:?}");
        }
        let nfc: String = text.nfc().collect();
        assert_eq!(tok.decode(&got), nfc, "decode {text:?}");
    }
    assert_eq!(failed, 0, "{failed} of {} cases differ", cases.len());
}
