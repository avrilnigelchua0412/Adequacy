import numpy as np
from PIL import Image

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

def draw_cluster(draw, cluster_info, img_size, color="blue"):
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
    label = (
        f"Cluster | "
        f"N={cluster_info['num_boxes']} | "
        f"μconf={cluster_info['mean_confidence']:.2f}"
    )

    draw.text((x1 + 3, y1 + 3), label, fill=color)