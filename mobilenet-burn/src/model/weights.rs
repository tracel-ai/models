/// Pre-trained weights metadata.
pub struct Weights {
    pub(super) url: &'static str,
    pub(super) num_classes: usize,
}

#[cfg(feature = "pretrained")]
mod downloader {
    use super::*;
    use burn::data::network::downloader;
    use std::fs::{create_dir_all, File};
    use std::io::Write;
    use std::path::PathBuf;

    impl Weights {
        /// Download the pre-trained weights to the local cache directory.
        pub fn download(&self) -> Result<PathBuf, std::io::Error> {
            // Model cache directory
            let model_dir = dirs::home_dir()
                .expect("Should be able to get home directory")
                .join(".cache")
                .join("mobilenetv2-burn");

            if !model_dir.exists() {
                create_dir_all(&model_dir)?;
            }

            let file_base_name = self.url.rsplit_once('/').unwrap().1;
            let file_name = model_dir.join(file_base_name);
            if !file_name.exists() {
                // Download file content
                let bytes = downloader::download_file_as_bytes(self.url, file_base_name);

                // Write content to file
                let mut output_file = File::create(&file_name)?;
                let bytes_written = output_file.write(&bytes)?;

                if bytes_written != bytes.len() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Failed to write the whole model weights file.",
                    ));
                }
            }

            Ok(file_name)
        }
    }
}

pub trait WeightsMeta {
    fn weights(&self) -> Weights;
}

/// MobileNetV2 pre-trained weights.
pub enum MobileNetV2 {
    // /// These weights reproduce closely the results of the original paper.
    // /// Top-1 accuracy: 71.878%.
    // /// Top-5 accuracy: 90.286%.
    // ImageNet1kV1,
    /// These weights improve upon the results of the original paper with a new training
    /// [recipe](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives).
    /// Top-1 accuracy: 72.154%.
    /// Top-5 accuracy: 90.822%.
    ImageNet1kV2,
}
impl WeightsMeta for MobileNetV2 {
    fn weights(&self) -> Weights {
        let url = match *self {
            // https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
            // MobileNetV2::ImageNet1kV1 => {
            //     // NOTE: The zip file at this URL cannot be properly parsed with zip-rs
            //     // Invalid Zip archive: Could not find central directory end
            //     "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"
            // }
            MobileNetV2::ImageNet1kV2 => {
                "https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth"
            }
        };
        Weights {
            url,
            num_classes: 1000,
        }
    }
}
