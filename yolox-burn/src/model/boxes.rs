use burn::tensor::{backend::Backend, ElementConversion, Tensor};
use itertools::Itertools;

pub struct BoundingBox {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
    pub confidence: f32,
}

/// Non-maximum suppression (NMS) filters overlapping bounding boxes that have an intersection-over-
/// union (IoU) greater or equal than the specified `iou_threshold` with previously selected boxes.
///
/// Boxes are filtered based on `score_threshold` and ranked based on their score. As such, lower
/// scoring boxes are removed when overlapping with another (higher scoring) box.
///
/// # Arguments
///
/// * `boxes`: Bounding box coordinates. Shape: `[batch_size, num_boxes, 4]`.
/// * `scores` - Classification scores for each box. Shape: `[batch_size, num_boxes, num_classes]`.
/// * `iou_threshold` - Scalar threshold for IoU.
/// * `score_threshold` - Scalar threshold for scores.
///
/// # Returns
///
/// Vector of bounding boxes grouped by class for each batch. The boxes are sorted in decreasing
/// order of scores for each class.
pub fn nms<B: Backend>(
    boxes: Tensor<B, 3>,
    scores: Tensor<B, 3>,
    iou_threshold: f32,
    score_threshold: f32,
) -> Vec<Vec<Vec<BoundingBox>>> {
    let [batch_size, num_boxes, num_classes] = scores.dims();

    // Bounding boxes grouped by batch and by (maximum) class index
    let mut bboxes = boxes
        .iter_dim(0)
        .zip(scores.iter_dim(0))
        .enumerate()
        // Per-batch
        .map(|(_, (candidate_boxes, candidate_scores))| {
            // Keep max scoring boxes only ([num_boxes, 1], [num_boxes, 1])
            let (cls_score, cls_idx) = candidate_scores.squeeze_dim::<2>(0).max_dim_with_indices(1);
            let cls_score: Vec<_> = cls_score
                .into_data()
                .iter::<B::FloatElem>()
                .map(|v| v.elem::<f32>())
                .collect();
            let cls_idx: Vec<_> = cls_idx
                .into_data()
                .iter::<B::IntElem>()
                .map(|v| v.elem::<i64>() as usize)
                .collect();

            // [num_boxes, 4]
            let candidate_boxes: Vec<_> = candidate_boxes
                .into_data()
                .iter::<B::FloatElem>()
                .map(|v| v.elem::<f32>())
                .collect();

            // Per-class filtering based on score
            (0..num_classes)
                .map(|cls_id| {
                    // [num_boxes, 1]
                    (0..num_boxes)
                        .filter_map(|box_idx| {
                            let box_cls_idx = cls_idx[box_idx];
                            if box_cls_idx != cls_id {
                                return None;
                            }
                            let box_cls_score = cls_score[box_idx];
                            if box_cls_score >= score_threshold {
                                let bbox = &candidate_boxes[box_idx * 4..box_idx * 4 + 4];
                                Some(BoundingBox {
                                    xmin: bbox[0] - bbox[2] / 2.,
                                    ymin: bbox[1] - bbox[3] / 2.,
                                    xmax: bbox[0] + bbox[2] / 2.,
                                    ymax: bbox[1] + bbox[3] / 2.,
                                    confidence: box_cls_score,
                                })
                            } else {
                                None
                            }
                        })
                        .sorted_unstable_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    for batch_bboxes in bboxes.iter_mut().take(batch_size) {
        non_maximum_suppression(batch_bboxes, iou_threshold);
    }

    bboxes
}

/// Intersection over union of two bounding boxes.
pub fn iou(b1: &BoundingBox, b2: &BoundingBox) -> f32 {
    let b1_area = (b1.xmax - b1.xmin + 1.) * (b1.ymax - b1.ymin + 1.);
    let b2_area = (b2.xmax - b2.xmin + 1.) * (b2.ymax - b2.ymin + 1.);
    let i_xmin = b1.xmin.max(b2.xmin);
    let i_xmax = b1.xmax.min(b2.xmax);
    let i_ymin = b1.ymin.max(b2.ymin);
    let i_ymax = b1.ymax.min(b2.ymax);
    let i_area = (i_xmax - i_xmin + 1.).max(0.) * (i_ymax - i_ymin + 1.).max(0.);
    i_area / (b1_area + b2_area - i_area)
}

/// Perform non-maximum suppression over boxes of the same class.
pub fn non_maximum_suppression(bboxes: &mut [Vec<BoundingBox>], threshold: f32) {
    for bboxes_for_class in bboxes.iter_mut() {
        bboxes_for_class.sort_by(|b1, b2| b2.confidence.partial_cmp(&b1.confidence).unwrap());
        let mut current_index = 0;
        for index in 0..bboxes_for_class.len() {
            let mut drop = false;
            for prev_index in 0..current_index {
                let iou = iou(&bboxes_for_class[prev_index], &bboxes_for_class[index]);
                if iou > threshold {
                    drop = true;
                    break;
                }
            }
            if !drop {
                bboxes_for_class.swap(current_index, index);
                current_index += 1;
            }
        }
        bboxes_for_class.truncate(current_index);
    }
}
