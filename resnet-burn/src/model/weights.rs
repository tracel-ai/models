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
                .join("resnet-burn");

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

/// ResNet-18 pre-trained weights.
pub enum ResNet18 {
    /// These weights reproduce closely the results of the original paper.
    /// Top-1 accuracy: 69.758%.
    /// Top-5 accuracy: 89.078%.
    ImageNet1kV1,
}
impl WeightsMeta for ResNet18 {
    fn weights(&self) -> Weights {
        Weights {
            url: "https://download.pytorch.org/models/resnet18-f37072fd.pth",
            num_classes: 1000,
        }
    }
}

/// ResNet-34 pre-trained weights.
pub enum ResNet34 {
    /// These weights reproduce closely the results of the original paper.
    /// Top-1 accuracy: 73.314%.
    /// Top-5 accuracy: 91.420%.
    ImageNet1kV1,
}
impl WeightsMeta for ResNet34 {
    fn weights(&self) -> Weights {
        Weights {
            url: "https://download.pytorch.org/models/resnet34-b627a593.pth",
            num_classes: 1000,
        }
    }
}

/// ResNet-50 pre-trained weights.
pub enum ResNet50 {
    /// These weights reproduce closely the results of the original paper.
    /// Top-1 accuracy: 76.130%.
    /// Top-5 accuracy: 92.862%.
    ImageNet1kV1,
    /// These weights improve upon the results of the original paper with a new training
    /// [recipe](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives).
    /// Top-1 accuracy: 80.858%.
    /// Top-5 accuracy: 95.434%.
    ImageNet1kV2,
}
impl WeightsMeta for ResNet50 {
    fn weights(&self) -> Weights {
        let url = match *self {
            ResNet50::ImageNet1kV1 => "https://download.pytorch.org/models/resnet50-0676ba61.pth",
            ResNet50::ImageNet1kV2 => "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        };
        Weights {
            url,
            num_classes: 1000,
        }
    }
}

/// ResNet-101 pre-trained weights.
pub enum ResNet101 {
    /// These weights reproduce closely the results of the original paper.
    /// Top-1 accuracy: 77.374%.
    /// Top-5 accuracy: 93.546%.
    ImageNet1kV1,
    /// These weights improve upon the results of the original paper with a new training
    /// [recipe](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives).
    /// Top-1 accuracy: 81.886%.
    /// Top-5 accuracy: 95.780%.
    ImageNet1kV2,
}
impl WeightsMeta for ResNet101 {
    fn weights(&self) -> Weights {
        let url = match *self {
            ResNet101::ImageNet1kV1 => "https://download.pytorch.org/models/resnet101-63fe2227.pth",
            ResNet101::ImageNet1kV2 => "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
        };
        Weights {
            url,
            num_classes: 1000,
        }
    }
}

/// ResNet-152 pre-trained weights.
pub enum ResNet152 {
    /// These weights reproduce closely the results of the original paper.
    /// Top-1 accuracy: 78.312%.
    /// Top-5 accuracy: 94.046%.
    ImageNet1kV1,
    /// These weights improve upon the results of the original paper with a new training
    /// [recipe](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives).
    /// Top-1 accuracy: 82.284%.
    /// Top-5 accuracy: 96.002%.
    ImageNet1kV2,
}
impl WeightsMeta for ResNet152 {
    fn weights(&self) -> Weights {
        let url = match *self {
            ResNet152::ImageNet1kV1 => "https://download.pytorch.org/models/resnet152-394f9c45.pth",
            ResNet152::ImageNet1kV2 => "https://download.pytorch.org/models/resnet152-f82ba261.pth",
        };
        Weights {
            url,
            num_classes: 1000,
        }
    }
}
