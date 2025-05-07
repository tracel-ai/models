use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::string::String;
use std::vec::Vec;

use burn::{
    data::dataset::Dataset,
    tensor::{backend::Backend, TensorData, Int, Tensor},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CocoAnnotation {
    pub id: u32,
    pub image_id: u32,
    pub category_id: u32,
    pub bbox: [f32; 4], // [x, y, width, height]
    pub area: f32,
    pub iscrowd: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CocoImage {
    pub id: u32,
    pub file_name: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CocoCategory {
    pub id: u32,
    pub name: String,
    pub supercategory: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CocoDataset {
    pub images: Vec<CocoImage>,
    pub annotations: Vec<CocoAnnotation>,
    pub categories: Vec<CocoCategory>,
}

#[derive(Debug, Clone)]
pub struct CocoDatasetItem<B: Backend> {
    pub image: Tensor<B, 3>,
    pub boxes: Tensor<B, 2>,
    pub labels: Tensor<B, 1, Int>,
}

pub struct CocoDatasetLoader<B: Backend> {
    dataset: CocoDataset,
    image_dir: String,
    device: B::Device,
}

impl<B: Backend> CocoDatasetLoader<B> {
    pub fn new(annotation_path: &str, image_dir: &str, device: B::Device) -> Self {
        let mut file = File::open(annotation_path).expect("Failed to open annotation file");
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("Failed to read annotation file");
        let dataset: CocoDataset = serde_json::from_str(&contents).expect("Failed to parse JSON");

        Self {
            dataset,
            image_dir: image_dir.to_string(),
            device,
        }
    }
}

impl<B: Backend> Dataset<CocoDatasetItem<B>> for CocoDatasetLoader<B> {
    fn get(&self, index: usize) -> Option<CocoDatasetItem<B>> {
        let coco_image = &self.dataset.images[index];
        let image_path = Path::new(&self.image_dir).join(&coco_image.file_name);
        let img = image::open(image_path).expect("Failed to open image");
        let img = img.to_rgb8();
        let (height, width) = (img.height() as usize, img.width() as usize);
        let img_data: Vec<f32> = img
            .pixels()
            .flat_map(|p| {
                vec![
                    p[0] as f32 / 255.0,
                    p[1] as f32 / 255.0,
                    p[2] as f32 / 255.0,
                ]
            })
            .collect();

        let image_tensor = Tensor::<B, 3>::from_data(
            TensorData::new(img_data, [height, width, 3]),
            &self.device,
        )
        .permute([2, 0, 1]); // Change from HWC to CHW format

        // Get annotations for this image
        let annotations: Vec<&CocoAnnotation> = self
            .dataset
            .annotations
            .iter()
            .filter(|ann| ann.image_id == coco_image.id)
            .collect();

        let boxes: Vec<f32> = annotations
            .iter()
            .flat_map(|ann| {
                let [x, y, w, h] = ann.bbox;
                vec![x, y, x + w, y + h] // Convert to [x1, y1, x2, y2] format
            })
            .collect();

        let labels: Vec<i64> = annotations
            .iter()
            .map(|ann| ann.category_id as i64)
            .collect();

        let boxes = Tensor::<B, 2>::from_data(
            TensorData::new(boxes, [annotations.len(), 4]),
            &self.device,
        );
        let labels = Tensor::<B, 1, Int>::from_data(
            TensorData::new(labels, [annotations.len()]),
            &self.device,
        );

        Some(CocoDatasetItem {
            image: image_tensor,
            boxes,
            labels,
        })
    }

    fn len(&self) -> usize {
        self.dataset.images.len()
    }
} 