use std::env;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use burn_onnx::{LoadStrategy, ModelGen};

const LABEL_SOURCE_FILE: &str = "src/model/label.txt";
const LABEL_DEST_FILE: &str = "model/label.rs";
const GENERATED_MODEL_WEIGHTS_FILE: &str = "squeezenet1.bpk";
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

    // Select weight-loading strategy based on feature flags.
    let load_strategy = if cfg!(feature = "weights_embedded") {
        LoadStrategy::Embedded
    } else {
        LoadStrategy::File
    };

    // Generate the model code from the ONNX file.
    ModelGen::new()
        .input(INPUT_ONNX_FILE)
        .out_dir(OUT_DIR)
        .load_strategy(load_strategy)
        .run_from_script();

    // Copy the weights next to the executable.
    if cfg!(feature = "weights_file") && cfg!(feature = "weights_file_dump") {
        copy_weights_next_to_executable();
    }

    // Generate the labels from the synset.txt file.
    generate_labels_from_txt_file().unwrap();
}

/// Read labels from synset.txt and store them in a vector of strings in a Rust file.
fn generate_labels_from_txt_file() -> std::io::Result<()> {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join(LABEL_DEST_FILE);
    let mut f = File::create(dest_path)?;

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
    let target_dir = format!("target/{profile}");

    let destination_dir = Path::new(&target_dir).join("examples");
    // The examples directory is created by cargo the first time an example is
    // built. On a fresh clone, `cargo check` / `cargo clippy` may run this
    // build script before any example has ever been built, so we create it
    // ourselves.
    fs::create_dir_all(&destination_dir).expect("Failed to create examples directory");
    let destination_path = destination_dir.join(GENERATED_MODEL_WEIGHTS_FILE);

    fs::copy(source_path, destination_path).expect("Failed to copy generated file");
}
