use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use burn_import::onnx::ModelGen;

const LABEL_SOURCE_FILE: &str = "src/model/label.txt";
const LABEL_DEST_FILE: &str = "model/label.rs";

fn main() {
    // Re-run the build script if model files change.
    println!("cargo:rerun-if-changed=src/model");

    // Generate the model code from the ONNX file.
    ModelGen::new()
        .input("src/model/squeezenet1.onnx")
        .out_dir("model/")
        .run_from_script();

    // Generate the labels from the synset.txt file.
    generate_labels_from_txt_file().unwrap();
}

/// Read labels from synset.txt and store them in a vector of strings in a Rust file.
fn generate_labels_from_txt_file() -> std::io::Result<()> {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join(LABEL_DEST_FILE);
    let mut f = File::create(&dest_path)?;

    let file = File::open(LABEL_SOURCE_FILE)?;
    let reader = BufReader::new(file);

    writeln!(f, "pub static LABELS: &[&str] = &[")?;
    for line in reader.lines() {
        writeln!(f, "    \"{}\",", line.unwrap())?;
    }
    writeln!(f, "];")?;

    Ok(())
}
