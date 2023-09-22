use std::env;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use burn_import::burn::graph::RecordType;
use burn_import::onnx::ModelGen;

const LABEL_SOURCE_FILE: &str = "src/model/label.txt";
const LABEL_DEST_FILE: &str = "model/label.rs";
const GENERATED_MODEL_WEIGHTS_FILE: &str = "squeezenet1.mpk.gz";
const INPUT_ONNX_FILE: &str = "src/model/squeezenet1.onnx";
const OUT_DIR: &str = "model/";

fn main() {
    // Re-run the build script if model files change.
    println!("cargo:rerun-if-changed=src/model");

    // Make sure either weights_file or weights_embedded is enabled.
    if cfg!(feature = "weights_file") && cfg!(feature = "weights_embedded") {
        panic!("Only one of the features weights_file and weights_embedded can be enabled");
    }

    // Make sure at least one of weights_file or weights_embedded is enabled.
    if !cfg!(feature = "weights_file") && !cfg!(feature = "weights_embedded") {
        panic!("One of the features weights_file and weights_embedded must be enabled");
    }

    // Check if the weights are embedded.
    let (record_type, embed_states) = if cfg!(feature = "weights_embedded") {
        (RecordType::Bincode, true)
    } else {
        (RecordType::NamedMpkGz, false)
    };

    // Check if half precision is enabled.
    let half_precision = cfg!(feature = "half_precision");

    // Generate the model code from the ONNX file.
    ModelGen::new()
        .input(INPUT_ONNX_FILE)
        .out_dir(OUT_DIR)
        .record_type(record_type)
        .embed_states(embed_states)
        .half_precision(half_precision)
        .run_from_script();

    // Copy the weights next to the executable.
    if cfg!(feature = "weights_file") {
        copy_weights_next_to_executable();
    }

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

/// Copy the weights file next to the executable.
fn copy_weights_next_to_executable() {
    // Obtain the OUT_DIR path from the environment variable.
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not defined");

    // Weights file in OUT_DIR that you want to copy.
    let source_path = Path::new(&out_dir)
        .join("model")
        .join(GENERATED_MODEL_WEIGHTS_FILE);

    // Determine the profile (debug or release) to set the appropriate destination directory.
    let profile = env::var("PROFILE").expect("PROFILE not defined");
    let target_dir = format!("target/{}", profile);

    // Specify the destination path.
    let destination_path = Path::new(&target_dir)
        .join("examples")
        .join(GENERATED_MODEL_WEIGHTS_FILE);

    // Copy the file.
    fs::copy(source_path, destination_path).expect("Failed to copy generated file");
}
