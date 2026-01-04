from tile_inferencer import TileInferencer
import onnxruntime as ort
from nms_processor import NMSProcessor
import numpy as np
import json, yaml
from PIL import Image, ImageDraw

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

if __name__ == "__main__":
    # model_path = "YOLOv5 Nano Level 3 Model/best.onnx"
    model_path = "YOLOv7 Tiny Level 3 Model/best.onnx"
    input_spec_path = "input_spec.json"
    preprocessing_path = "preprocessing.yaml"
    tile_img_path = "input_tile_image.jpg"

    inferencer = TileInferencer(model_path, input_spec_path, preprocessing_path)
    tile_img = inferencer.get_img(tile_img_path)
    preprocessed_img = inferencer.preprocess(tile_img)
    print("Preprocessed image shape:", preprocessed_img.shape)
    
    outputs = inferencer.infer(tile_img)
    
    postprocessing_path = "postprocessing.yaml"
    with open(postprocessing_path) as f:
        post_cfg = yaml.safe_load(f)
    mask = outputs[0][0][:, 4] >= post_cfg['conf_threshold']
    preds = outputs[0][0][mask] 

    print(f"Number of predictions after filtering: {len(preds)}")

    boxes, confs, classes = xywh_to_xyxy(preds)
    
    predictions = np.concatenate([boxes, confs[:, None], classes[:, None]], axis=1)
    nms_processor = NMSProcessor(post_cfg)
    keep = nms_processor.non_max_suppression(predictions)
    final_preds = predictions[keep]

    print(f"Final Number of predictions after filtering: {len(final_preds)}")
    
    pil_img = Image.fromarray(tile_img)
    draw = ImageDraw.Draw(pil_img)
    
    for box in final_preds:
        x1, y1, x2, y2, conf, class_id = box
        # print( f'{int(x1)}, {int(y1)},{int(x2)}, {int(y2)}, { conf:.2f}')
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"{int(class_id)}:{conf:.2f}", fill="red")
        
    
    from IoU_adjacency_matrix import iou_based_clustering
    clustered_info = iou_based_clustering(final_preds[:, :4], final_preds[:, 4])
    
    for cluster in clustered_info:
        draw_cluster(draw, cluster, pil_img.size, color="blue")
    
    pil_img.save("output_tile_image.jpg")