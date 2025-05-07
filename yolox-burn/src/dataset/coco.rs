use std::path::Path;

use burn::{
    data::dataset::{Dataset, DatasetItem},
    tensor::{backend::Backend, Device, Element, Tensor, TensorData},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CocoAnnotation {
    pub id: u64,
    pub image_id: u64,
    pub category_id: u64,
    pub bbox: [f32; 4], // [x, y, width, height]
    pub area: f32,
    pub iscrowd: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CocoImage {
    pub id: u64,
    pub width: u32,
    pub height: u32,
    pub file_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CocoCategory {
    pub id: u64,
    pub name: String,
    pub supercategory: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CocoDataset {
    pub images: Vec<CocoImage>,
    pub annotations: Vec<CocoAnnotation>,
    pub categories: Vec<CocoCategory>,
}

pub struct CocoDatasetItem<B: Backend> {
    pub image: Tensor<B, 3>,
    pub target: Tensor<B, 3>,
}

impl<B: Backend> DatasetItem for CocoDatasetItem<B> {
    type Input = Tensor<B, 3>;
    type Output = Tensor<B, 3>;

    fn input(&self) -> Self::Input {
        self.image.clone()
    }

    fn output(&self) -> Self::Output {
        self.target.clone()
    }
}

pub struct CocoDatasetLoader<B: Backend> {
    dataset: CocoDataset,
    device: Device<B>,
    image_dir: String,
}

impl<B: Backend> CocoDatasetLoader<B> {
    pub fn new(dataset_path: &str, image_dir: &str, device: Device<B>) -> Self {
        let dataset = serde_json::from_str::<CocoDataset>(
            &std::fs::read_to_string(dataset_path).unwrap(),
        )
        .unwrap();

        Self {
            dataset,
            device,
            image_dir: image_dir.to_string(),
        }
    }

    fn load_image(&self, image: &CocoImage) -> Tensor<B, 3> {
        let image_path = Path::new(&self.image_dir).join(&image.file_name);
        let img = image::open(image_path).unwrap();
        
        // Resize to model input size
        let img = img.resize_exact(
            640,
            640,
            image::imageops::FilterType::Triangle,
        );

        // Convert to tensor
        let img_data = img.into_rgb8().into_raw();
        let shape = [640, 640, 3];
        
        Tensor::<B, 3>::from_data(
            TensorData::new(img_data, shape).convert::<B::FloatElem>(),
            &self.device,
        )
        .permute([2, 0, 1]) // [H, W, C] -> [C, H, W]
    }

    fn create_target(&self, image_id: u64) -> Tensor<B, 3> {
        // Get annotations for this image
        let annotations: Vec<&CocoAnnotation> = self
            .dataset
            .annotations
            .iter()
            .filter(|ann| ann.image_id == image_id)
            .collect();

        // Create target tensor
        let mut target = Tensor::zeros([1, 100, 85], &self.device); // [batch, max_boxes, 4+1+80]

        for (i, ann) in annotations.iter().enumerate() {
            if i >= 100 {
                break; // Limit to 100 boxes per image
            }

            // Convert bbox to [x1, y1, x2, y2] format
            let [x, y, w, h] = ann.bbox;
            let x1 = x;
            let y1 = y;
            let x2 = x + w;
            let y2 = y + h;

            // Set bbox coordinates
            target.slice_mut([0..1, i..i+1, 0..4]).copy_from(&Tensor::from_data(
                TensorData::new(vec![x1, y1, x2, y2], [1, 1, 4]).convert::<B::FloatElem>(),
                &self.device,
            ));

            // Set objectness score
            target.slice_mut([0..1, i..i+1, 4..5]).copy_from(&Tensor::ones([1, 1, 1], &self.device));

            // Set class score
            let class_idx = ann.category_id as usize - 1; // COCO categories start at 1
            target.slice_mut([0..1, i..i+1, 5+class_idx..6+class_idx])
                .copy_from(&Tensor::ones([1, 1, 1], &self.device));
        }

        target
    }
}

impl<B: Backend> Dataset<CocoDatasetItem<B>> for CocoDatasetLoader<B> {
    fn get(&self, index: usize) -> CocoDatasetItem<B> {
        let image = &self.dataset.images[index];
        let image_tensor = self.load_image(image);
        let target_tensor = self.create_target(image.id);

        CocoDatasetItem {
            image: image_tensor,
            target: target_tensor,
        }
    }

    fn len(&self) -> usize {
        self.dataset.images.len()
    }
} 