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

def xywh_to_xyxy(preds):
    """
    preds: np.ndarray of shape (N, 6)
           [x, y, w, h, conf, class_id]
    returns:
        boxes: (N, 4) → [x1, y1, x2, y2]
        confs: (N,)
        classes: (N,)
    """
    xy = preds[:, 0:2]
    wh = preds[:, 2:4]

    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)
    confs = preds[:, 4]
    classes = preds[:, 5].astype(np.int32)

    return boxes, confs, classes

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
                image_path = os.path.join(root, file)
                yield image_path, file, file_name, format
                
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

def get_thyrocytes_inside_cluster(cluster_info, filtered_final_preds):
    cluster_box = np.array(cluster_info["bounding_box"])
    thyrocytes_inside_the_cluster = []
    for preds in filtered_final_preds:
        pred_box = preds[:4]
        iou = box_iou_batch(
            pred_box[np.newaxis, :],
            cluster_box[np.newaxis, :]
        )[0, 0]
        if iou  != 0:
            print("IoU: ", iou)
            print("Prediction: ", preds)
            thyrocytes_inside_the_cluster.append(preds)
    return thyrocytes_inside_the_cluster