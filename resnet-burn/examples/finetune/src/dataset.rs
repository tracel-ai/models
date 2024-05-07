use flate2::read::GzDecoder;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};
use tar::Archive;

use burn::data::{
    dataset::vision::{ImageFolderDataset, ImageLoaderError},
    network::downloader,
};

/// Planets dataset sample mirror from [fastai](https://github.com/fastai/fastai/blob/master/fastai/data/external.py#L55).
/// Licensed under the [Appache License](https://github.com/fastai/fastai/blob/master/LICENSE).
const URL: &str = "https://s3.amazonaws.com/fast-ai-sample/planet_sample.tgz";
const LABELS: &str = "labels.csv";
pub const CLASSES: [&str; 17] = [
    "agriculture",
    "artisinal_mine",
    "bare_ground",
    "blooming",
    "blow_down",
    "clear",
    "cloudy",
    "conventional_mine",
    "cultivation",
    "habitation",
    "haze",
    "partly_cloudy",
    "primary",
    "road",
    "selective_logging",
    "slash_burn",
    "water",
];

/// A sample of the planets dataset from the Kaggle competition
/// [Planet: Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space).
///
/// This version of the multi-label classification dataset contains 1,000 256x256 image patches
/// with possibly multiple labels per patch. The labels can broadly be broken into three groups:
/// atmospheric conditions, common land cover/land use phenomena, and rare land cover/land use
/// phenomena. Each patch will have one and potentially more than one atmospheric label and zero
/// or more common and rare labels.
///
/// The data is downloaded from the web from the [fastai mirror](https://github.com/fastai/fastai/blob/master/fastai/data/external.py#L55).
pub trait PlanetLoader: Sized {
    fn planet_train_val_split(
        train_percentage: u8,
        seed: u64,
    ) -> Result<(Self, Self), ImageLoaderError>;
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct PlanetSample {
    image_name: String,
    tags: String,
}

impl PlanetLoader for ImageFolderDataset {
    /// Creates new Planet dataset for train and validation splits.
    ///
    /// # Arguments
    ///
    /// * `train_percentage` - Percentage of the training split. The remainder will be used for the validation split.
    /// * `seed` - Controls the shuffling applied to the data before applying the split.
    ///
    fn planet_train_val_split(
        train_percentage: u8,
        seed: u64,
    ) -> Result<(Self, Self), ImageLoaderError> {
        assert!(
            train_percentage > 0 && train_percentage < 100,
            "Training split percentage must be between (0, 100)"
        );
        let root = download();

        // Load items from csv
        let mut rdr = csv::ReaderBuilder::new()
            .from_path(root.join(LABELS))
            .map_err(|err| ImageLoaderError::Unknown(err.to_string()))?;

        // Collect items (image path, labels)
        let mut classes = HashSet::new();
        let mut items = rdr
            .deserialize()
            .map(|result| {
                let item: PlanetSample =
                    result.map_err(|err| ImageLoaderError::Unknown(err.to_string()))?;
                let tags = item
                    .tags
                    .split(' ')
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>();

                for tag in tags.iter() {
                    classes.insert(tag.clone());
                }

                Ok((
                    // Full path to image
                    root.join("train")
                        .join(item.image_name)
                        .with_extension("jpg"),
                    // Multiple labels per image (e.g., ["haze", "primary", "water"])
                    tags,
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Sort class names
        let mut classes = classes.iter().collect::<Vec<_>>();
        classes.sort();
        assert_eq!(classes, CLASSES, "Invalid categories"); // just in case the labels unexpectedly change

        // Shuffle items
        items.shuffle(&mut StdRng::seed_from_u64(seed));

        // Split train and validation
        let size = items.len();
        let train_slice = (size as f32 * (train_percentage as f32 / 100.0)) as usize;

        let train = Self::new_multilabel_classification_with_items(
            items[..train_slice].to_vec(),
            &classes,
        )?;
        let valid = Self::new_multilabel_classification_with_items(
            items[train_slice..].to_vec(),
            &classes,
        )?;

        Ok((train, valid))
    }
}

/// Download the Planet dataset from the web to the current example directory.
fn download() -> PathBuf {
    // Point to current example directory
    let example_dir = Path::new(file!()).parent().unwrap().parent().unwrap();
    let planet_dir = example_dir.join("planet_sample");

    // Check for already downloaded content
    let labels_file = planet_dir.join(LABELS);
    if !labels_file.exists() {
        // Download gzip file
        let bytes = downloader::download_file_as_bytes(URL, "planet_sample.tgz");

        // Decode gzip file content and unpack archive
        let gz_buffer = GzDecoder::new(&bytes[..]);
        let mut archive = Archive::new(gz_buffer);
        archive.unpack(example_dir).unwrap();
    }

    planet_dir
}
