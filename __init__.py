from utils import xywh_to_xyxy, draw_cluster, get_img, helper_os_walk, weighted_boxes_fusion_helper, get_thyrocytes_inside_cluster
from IoU_adjacency_matrix import iou_based_clustering
from tile_inferencer import TileInferencer
import onnxruntime as ort
from nms_processor import NMSProcessor
import numpy as np
import json, yaml
from PIL import Image, ImageDraw
from image_tiler import ImageTiler
"""
Tile-Based Object Detection and Clustering Pipeline
==================================================

This module implements an end-to-end inference pipeline for detecting,
filtering, and spatially clustering object predictions from large images
using a tiled inference strategy.

The pipeline is designed for:
- Large images processed via overlapping tiles
- ONNX-based YOLO inference
- Multi-stage suppression and clustering
- Biologically interpretable spatial aggregation

Intended use:
- Research experiments
- FastAPI backend inference
- Figure generation for publications
"""

if __name__ == "__main__":
    data_path = "Data"
    
    # model_path = "YOLOv5 Nano Level 3 Model/best.onnx"
    model_path = "YOLOv7 Tiny Level 3 Model/best.onnx"
    input_spec_path = "input_spec.json"
    preprocessing_path = "preprocessing.yaml"
    postprocessing_path = "postprocessing.yaml"
    
    """
    Initialization Phase
    --------------------
    This section initializes all components required for inference:

    - Loads the ONNX detection model
    - Loads preprocessing and postprocessing configurations
    - Instantiates the image tiler for overlapping tile generation

    No inference or computation happens at this stage.
    """
    inferencer = TileInferencer(model_path, input_spec_path, preprocessing_path)
    with open(postprocessing_path) as f:
        post_cfg = yaml.safe_load(f)
        
    nms_processor = NMSProcessor(post_cfg)
    img_tiler = ImageTiler()
    
    for image_path, file, file_name, format in helper_os_walk(data_path):
        pass
    
    """
    Input Image Loading
    -------------------
    The full-resolution image is loaded into memory as a NumPy array.
    This image serves as the global spatial reference frame for all
    subsequent detections and clusters.
    """
    img_path = "input_tile_image.jpg"
    img = get_img(img_path)
    all_preds = []
    all_clustered_info = []
    
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    """
    Tile-Level Inference Loop
    ------------------------
    The input image is decomposed into overlapping tiles.

    For each tile:
    1. The tile is preprocessed and passed through the ONNX model
    2. Low-confidence predictions are filtered
    3. Tile-local Non-Maximum Suppression (NMS) is applied
    4. Bounding boxes are reprojected into global image coordinates
    5. IoU-based clustering is performed within the tile

    Tile-level results are accumulated for global aggregation.
    """
    for tile, x0, y0, tile_id in img_tiler.tile(img):
        preprocessed_img = inferencer.preprocess(tile)
    
        outputs = inferencer.infer(tile)
        
        mask = outputs[0][0][:, 4] >= post_cfg['conf_threshold']
        preds = outputs[0][0][mask] 

        boxes, confs, classes = xywh_to_xyxy(preds)
        
        predictions = np.concatenate([boxes, confs[:, None], classes[:, None]], axis=1)
        
        """
        Tile-Level Filtering and Suppression
        -----------------------------------
        Raw model predictions are filtered using a confidence threshold,
        followed by Non-Maximum Suppression (NMS) / Weighted Boxes Fusion to remove duplicate detections
        within the same tile.

        Remaining bounding boxes are reprojected into global image coordinates
        using the tile's top-left offset (x0, y0).
        """
        
        """Non-Maximum Suppression (NMS)"""
        keep = nms_processor.non_max_suppression(predictions)
        final_preds = predictions[keep]
        final_preds[:, [0, 2]] += x0
        final_preds[:, [1, 3]] += y0
        try:
            """Weighted Boxes Fusion"""
            # boxes, confidences = predictions[:, :4], predictions[:, 4]
            # new_boxes, new_scores = weighted_boxes_fusion_helper(boxes, confidences, pil_img.size, iou_thr=0.25)
            # new_label = np.ones((len(new_boxes), 1), dtype=np.int64)
            # final_preds = np.concatenate([new_boxes, new_scores[:, None], new_label] , axis=1)
            # final_preds[:, [0, 2]] += x0
            # final_preds[:, [1, 3]] += y0
            """
            Tile-Level IoU-Based Clustering
            -------------------------------
            Within each tile, detections are grouped using an IoU-based graph:

            - Each detection is treated as a node
            - Edges connect detections with non-zero IoU
            - Connected components represent spatially linked detections

            Cluster-level features (bounding box, mean confidence, size)
            are extracted and stored for later global aggregation.
            """
            all_preds.append(final_preds)
            
            clustered_info = iou_based_clustering(final_preds[:, :4], final_preds[:, 4], pil_img.size)
            
            # for info in clustered_info:
            #     if info['num_boxes'] > 1:
            #         draw_cluster(draw, info, pil_img.size)
            
            all_clustered_info.extend(clustered_info)
        except:
            pass
        
    """
    Global Detection Aggregation
    ----------------------------
    All tile-level detections are concatenated into a single
    image-level array. At this stage, all bounding boxes share
    a common global coordinate system.
    """
    all_preds = np.vstack(all_preds)
    
    boxes, confidences = all_preds[:, :4], all_preds[:, 4]
    new_boxes, new_scores = weighted_boxes_fusion_helper(boxes, confidences, pil_img.size, iou_thr=0.25)
    new_label = np.ones((len(new_boxes), 1), dtype=np.int64)
    filtered_final_preds = np.concatenate([new_boxes, new_scores[:, None], new_label] , axis=1)
    
    
    for preds in filtered_final_preds:
        x1, y1, x2, y2, conf, class_id = preds
        print(preds)
        draw.rectangle([x1, y1, x2, y2], outline="black", width=1)
        draw.text((x1, y2), f"{int(class_id)}:{conf:.2f}", fill="black")
    
    """
    Second-Stage (Global) Clustering
    --------------------------------
    Cluster bounding boxes obtained from individual tiles are
    clustered again using the same IoU-based graph strategy.

    This step merges spatially related clusters that span multiple
    tiles and resolves duplication caused by overlapping tiles.
    """
    boxes = []
    confidences = []
    for cluster in all_clustered_info:
        if cluster['num_boxes'] > 1:
            boxes.append(list(cluster['bounding_box']))
            confidences.append(cluster['mean_confidence'])

    boxes = np.array(boxes)
    confidences = np.array(confidences)
    
    infos = iou_based_clustering(boxes, confidences, pil_img.size)
    
    
    """
    Visualization (Qualitative Analysis)
    ------------------------------------
    For qualitative inspection and debugging:

    - Individual detections are drawn in red
    - Final cluster bounding boxes are drawn in blue

    Visualization is strictly separated from inference logic and
    does not influence computational results.
    """
    
    for info in infos:
        thyrocytes_inside_the_cluster = []
        thyrocytes_inside_the_cluster.extend(get_thyrocytes_inside_cluster(info, filtered_final_preds))
        for preds in thyrocytes_inside_the_cluster:
            x1, y1, x2, y2, conf, class_id = preds
            draw.rectangle([x1, y1, x2, y2], outline="red", width=1)
            draw.text((x1, y2), f"{conf:.2f}", fill="red")
        number_of_thyrocytes = len(thyrocytes_inside_the_cluster)
        if number_of_thyrocytes >= 10:
            draw_cluster(draw, info, pil_img.size, label=f"Adequate with {len(thyrocytes_inside_the_cluster)} Thyrocytes", color='black')
        else:
            draw_cluster(draw, info, pil_img.size, label=f"Inadequate with {len(thyrocytes_inside_the_cluster)} Thyrocytes", color='blue')
        
    """
    Output Generation
    -----------------
    The final visualization image, containing both detections
    and clustered regions, is saved to disk.
    """
    pil_img.save("output_tile_image.jpg")