from ensemble_boxes import weighted_boxes_fusion
import numpy as np
from PIL import Image
import os

def box_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    area_a = box_area(boxes_a.T)
    area_b = box_area(boxes_b.T)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])
    area_inter = np.prod(
        np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
        
    return area_inter / (area_a[:, None] + area_b - area_inter)

def get_img(tile_img_path):
    tile_img = np.array(Image.open(tile_img_path))  # HWC, RGB
    return tile_img

def process_yolo_preds(preds, conf_threshold=0.25):
    # Convert boxes
    xy = preds[:, 0:2]
    wh = preds[:, 2:4]
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)

    # Objectness
    obj_conf = preds[:, 4]

    # Class probabilities
    class_probs = preds[:, 5:]

    # Class ID
    class_ids = np.argmax(class_probs, axis=1)

    # Class score of selected class
    class_scores = class_probs[np.arange(len(class_probs)), class_ids]

    # Final confidence
    final_conf = obj_conf * class_scores

    # Filter
    mask = final_conf >= conf_threshold

    return boxes[mask], final_conf[mask], class_ids[mask]

def draw_cluster(draw, cluster_info, img_size, label="Tile-level-Cluster", color="blue"):
    W, H = img_size

    x1, y1, x2, y2 = cluster_info["bounding_box"]

    # Clip to image bounds
    x1 = int(max(0, min(W - 1, x1)))
    y1 = int(max(0, min(H - 1, y1)))
    x2 = int(max(0, min(W - 1, x2)))
    y2 = int(max(0, min(H - 1, y2)))

    # Draw thick rectangle for cluster
    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
    
    # Label cluster
    label = label

    draw.text((x1 + 3, y1 + 3), label, fill=color)
    
def helper_os_walk(file_path):
    formats = ['.jpeg', '.jpg', '.png']
    for root, _, files in os.walk(file_path):
        for file in files:
            format = os.path.splitext(file)[1]
            file_name = os.path.splitext(file)[0]
            if format in formats:
                img_path = os.path.join(root, file)
                yield img_path, file, file_name, format
                
def normalize_box(box, img_w, img_h):
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    return [
            (x1) / img_w,
            (y1) / img_h,
            (x2) / img_w,
            (y2) / img_h
        ]

def denormalize_box(box, img_w, img_h):
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    return [
            x1 * img_w, # x_min
            y1 * img_h, # y_min
            x2 * img_w, # x_max
            y2 * img_h  # y_max
        ]

def weighted_boxes_fusion_helper(cluster_boxes, cluster_confs, img_size, iou_thr=0.45):
    W, H = img_size
    cluster_box_normalized = [
        normalize_box(box, W, H)
        for box in cluster_boxes
    ]
    boxes, scores, _ = weighted_boxes_fusion([cluster_box_normalized], [cluster_confs], [[1] * len(cluster_boxes)], conf_type='max', iou_thr=iou_thr)
    denormalized_boxes = [
        denormalize_box(box, W, H)
        for box in boxes
    ]
    return np.array(denormalized_boxes), scores

def iqr_filter_aspect_ratio_preds(preds, factor=1.5, min_width=10, min_height=10):
    """
    Filter bounding boxes that have:
    1. Outlier aspect ratios using IQR (width / height)
    2. Very small width or height
    
    Returns filtered boxes and confidences.
    """
    boxes = preds[:, :4]

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    aspect_ratios = widths / (heights + 1e-6)
    
    Q1 = np.percentile(aspect_ratios, 25)
    Q3 = np.percentile(aspect_ratios, 75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    
    mask = (aspect_ratios >= lower) & (aspect_ratios <= upper)
    mask &= (widths >= min_width) & (heights >= min_height)


    return preds[mask]

def remove_box_inside_box_preds(preds):
    """
    Remove bounding boxes that are fully inside another box.
    """
    boxes = preds[:, :4]
    keep = np.ones(len(boxes), dtype=bool)

    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i == j:
                continue

            xi1, yi1, xi2, yi2 = boxes[i]
            xj1, yj1, xj2, yj2 = boxes[j]

            if (xi1 >= xj1 and yi1 >= yj1 and
                xi2 <= xj2 and yi2 <= yj2):

                area_i = (xi2 - xi1) * (yi2 - yi1)
                area_j = (xj2 - xj1) * (yj2 - yj1)

                if area_i < area_j:
                    keep[i] = False
                    
    return preds[keep]

def filter_pipeline_preds(preds):
    """
    preds: (N, 6) → [x1, y1, x2, y2, conf, class]
    """
    if len(preds) == 0:
        return preds

    preds = iqr_filter_aspect_ratio_preds(preds)
    preds = remove_box_inside_box_preds(preds)

    return preds

def get_thyrocytes_inside_cluster(cluster_info, final_preds):
    boxes = np.array(cluster_info["boxes"])
    thyrocytes_inside_the_cluster = []
    for preds in final_preds:
        pred_box = preds[:4]
        for boxe in boxes:
            iou = box_iou_batch(
                pred_box[np.newaxis, :],
                boxe[np.newaxis, :]
            )[0, 0]
            if iou  != 0:
                # print("IoU: ", iou)
                # print("Prediction: ", preds)
                thyrocytes_inside_the_cluster.append(preds)
                break  # No need to check other cluster boxes for this prediction
    return thyrocytes_inside_the_cluster

def draw_thyrocytes_inside_cluster(draw, thyrocytes_inside_the_cluster, color):
    for preds in thyrocytes_inside_the_cluster:
        x1, y1, x2, y2, conf, class_id = preds
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y2), f"{conf:.2f}", fill=color)