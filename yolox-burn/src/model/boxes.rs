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
///
/// Selected indices of the boxes kept by NMS, sorted in decreasing order of scores.
/// The selected index format is `[batch_index, class_index, box_index]`.
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
            let (cls_score, cls_idx) = candidate_scores.squeeze::<2>(0).max_dim_with_indices(1);
            let cls_score: Vec<_> = cls_score
                .into_data()
                .value
                .iter()
                .map(|v| v.elem::<f32>())
                .collect();
            let cls_idx: Vec<_> = cls_idx
                .into_data()
                .value
                .iter()
                .map(|v| v.elem::<i64>() as usize)
                .collect();

            // [num_boxes, 4]
            let candidate_boxes: Vec<_> = candidate_boxes
                .into_data()
                .value
                .iter()
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

// pub fn nms<B: Backend>(
//     boxes: Tensor<B, 3>,
//     scores: Tensor<B, 3>,
//     iou_threshold: f64,
//     score_threshold: f64,
// ) -> () {
//     // Tensor<B, 2>
//     let [_, num_boxes, num_classes] = scores.shape().dims;
//     let device = boxes.device();
//     let is_valid = Tensor::<B, 1, Bool>::from_data([true; 1], &device);
//     // let values: Tensor<B, 1, _> = Tensor::from_data([1, 0, 3, 4, -1], &device);

//     let x = Tensor::arange(0..6 as i64, &device).reshape(Shape::new([2, 3]));
//     let mask = x.greater_elem(1);

//     let selected_indices = boxes
//         .iter_dim(0)
//         .enumerate()
//         // Per-batch NMS
//         .map(|(batch_idx, candidate_boxes)| {
//             // Per-class filtering
//             (0..num_classes).map(|cls_idx| {
//                 scores.select(
//                     0,
//                     Tensor::<B, 1, Int>::full(Shape::new([1]), batch_idx as i64, &device),
//                 );
//                 // [batch_size, num_boxes, 1]
//                 let scores_mask = scores.greater_equal_elem(score_threshold).any_dim(2);
//                 // 1. `non_zero` to get the mask's valid boxes indices
//                 // 2. `select` to return only valid boxes (by score threshold)
//                 // 3. `argsort` to get the indices ranked by classification score
//                 // NOTE: we could do it all on CPU (preds to_vec) to iterate and compare as done in
//                 // [candle](https://github.com/huggingface/candle/blob/main/candle-examples/examples/yolo-v3/main.rs#L38)
//                 // ([nms](https://github.com/huggingface/candle/blob/main/candle-transformers/src/object_detection.rs#L31))
//                 // but this is not so nice
//                 let box_idx = scores_mask
//                     .iter_dim(1)
//                     .filter_map(|box_idx| {
//                         // NOTE: bool tensors do not support `select` and `into_scalar()`
//                         if scores_mask
//                             .slice([batch_idx..batch_idx + 1])
//                             .any()
//                             .int()
//                             .into_scalar() as bool
//                         {
//                             Some(box_idx)
//                         } else {
//                             None
//                         }
//                     })
//                     .collect();
//                 // scores[batch_idx, ]
//                 // [batch_size, num_boxes, 1]
//                 let scores_max = scores.gather(dim, indices);
//             })
//         });
// }

// std::vector<SelectedIndex> selected_indices;
//   std::vector<BoxInfoPtr> selected_boxes_inside_class;
//   selected_boxes_inside_class.reserve(std::min<size_t>(static_cast<size_t>(max_output_boxes_per_class), pc.num_boxes_));

//   for (int64_t batch_index = 0; batch_index < pc.num_batches_; ++batch_index) {
//     for (int64_t class_index = 0; class_index < pc.num_classes_; ++class_index) {
//       int64_t box_score_offset = (batch_index * pc.num_classes_ + class_index) * pc.num_boxes_;
//       const float* batch_boxes = boxes_data + (batch_index * pc.num_boxes_ * 4);
//       std::vector<BoxInfoPtr> candidate_boxes;
//       candidate_boxes.reserve(pc.num_boxes_);

//       // Filter by score_threshold_
//       const auto* class_scores = scores_data + box_score_offset;
//       if (pc.score_threshold_ != nullptr) {
//         for (int64_t box_index = 0; box_index < pc.num_boxes_; ++box_index, ++class_scores) {
//           if (*class_scores > score_threshold) {
//             candidate_boxes.emplace_back(*class_scores, box_index);
//           }
//         }
//       } else {
//         for (int64_t box_index = 0; box_index < pc.num_boxes_; ++box_index, ++class_scores) {
//           candidate_boxes.emplace_back(*class_scores, box_index);
//         }
//       }
//       std::priority_queue<BoxInfoPtr, std::vector<BoxInfoPtr>> sorted_boxes(std::less<BoxInfoPtr>(), std::move(candidate_boxes));

//       selected_boxes_inside_class.clear();
//       // Get the next box with top score, filter by iou_threshold
//       while (!sorted_boxes.empty() && static_cast<int64_t>(selected_boxes_inside_class.size()) < max_output_boxes_per_class) {
//         const BoxInfoPtr& next_top_score = sorted_boxes.top();

//         bool selected = true;
//         // Check with existing selected boxes for this class, suppress if exceed the IOU (Intersection Over Union) threshold
//         for (const auto& selected_index : selected_boxes_inside_class) {
//           if (SuppressByIOU(batch_boxes, next_top_score.index_, selected_index.index_, center_point_box, iou_threshold)) {
//             selected = false;
//             break;
//           }
//         }

//         if (selected) {
//           selected_boxes_inside_class.push_back(next_top_score);
//           selected_indices.emplace_back(batch_index, class_index, next_top_score.index_);
//         }
//         sorted_boxes.pop();
//       }  // while
//     }    // for class_index
//   }      // for batch_index
