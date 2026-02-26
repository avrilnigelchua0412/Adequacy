from utils import xywh_to_xyxy, draw_cluster, get_img, helper_os_walk, weighted_boxes_fusion_helper, get_thyrocytes_inside_cluster, filter_pipeline, draw_thyrocytes_inside_cluster
from IoU_adjacency_matrix import iou_based_clustering
from tile_inferencer import TileInferencer
import onnxruntime as ort
from nms_processor import NMSProcessor
import numpy as np
import json, yaml
from PIL import Image, ImageDraw
from image_tiler import ImageTiler
import os
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

def tile_inference_pipeline(img, pil_img, draw, inferencer, img_tiler, nms_processor, conf_threshold):
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
    all_preds = []
    all_clustered_info = []
    for tile, x0, y0, tile_id in img_tiler.tile(img):
        preprocessed_img = inferencer.preprocess(tile)

        outputs = inferencer.infer(tile)

        mask = outputs[0][0][:, 4] >= conf_threshold
        preds = outputs[0][0][mask]     
        boxes, confs, classes = xywh_to_xyxy(preds)

        predictions = np.concatenate([boxes, confs[:, None], classes[:, None]], axis=1)

        """
        Tile-Level Filtering and Suppression
        -----------------------------------
        Raw model predictions are filtered using a confidence threshold,
        followed by Non-Maximum Suppression (NMS) to remove duplicate detections
        within the same tile.
        """

        """Non-Maximum Suppression (NMS)"""
        keep = nms_processor.non_max_suppression(predictions)
        final_preds = predictions[keep]

        if len(final_preds) == 0:
            continue
        # Separate boxes and confidences
        boxes = final_preds[:, :4]
        confidences = final_preds[:, 4]
        """1st Filter out boxes with extreme aspect ratios and remove boxes that are fully inside another box"""
        boxes, confidences = filter_pipeline(boxes, confidences)
        final_preds = np.c_[boxes, confidences, np.ones((len(boxes), 1))]

        """
        Remaining bounding boxes are reprojected into global image coordinates
        using the tile's top-left offset (x0, y0).
        """

        final_preds[:, [0, 2]] += x0
        final_preds[:, [1, 3]] += y0

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

        # """Visualization of tile-level clusters (for debugging)"""
        # for info in clustered_info:
        #     if info['num_boxes'] > 1:
        #         draw_cluster(draw, info, pil_img.size)

        all_clustered_info.extend(clustered_info)
        
    return np.vstack(all_preds), all_clustered_info

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
    for conf_threshold in post_cfg["conf_threshold"]:
        for img_path, file, file_name, format in helper_os_walk(data_path):
        
            """
            Input Image Loading
            -------------------
            The full-resolution image is loaded into memory as a NumPy array.
            This image serves as the global spatial reference frame for all
            subsequent detections and clusters.
            """
            img = get_img(img_path)
            
            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)
            
            """Tile-Level Inference"""
            all_preds, all_clustered_info = tile_inference_pipeline(img, pil_img, draw, inferencer, img_tiler, nms_processor, conf_threshold)
            
            """2nd Filter out boxes with extreme aspect ratios and remove boxes that are fully inside another box"""
            boxes, confidences = filter_pipeline(all_preds[:, :4], all_preds[:, 4])
            
            #################################################################################################################################################
            
            """
            Global Detection Aggregation
            ----------------------------
            All tile-level detections are concatenated into a single
            image-level array. At this stage, all bounding boxes share
            a common global coordinate system.
            """
            new_boxes, new_scores = weighted_boxes_fusion_helper(boxes, confidences, pil_img.size, iou_thr=0.25)
            new_label = np.ones((len(new_boxes), 1), dtype=np.int64)
            global_final_preds = np.concatenate([new_boxes, new_scores[:, None], new_label] , axis=1)
            
            
            """
            Second-Stage (Global) Clustering
            --------------------------------
            Method 1: 3-Step Clustering
            
            Cluster bounding boxes obtained from individual tiles are
            clustered again using the same IoU-based graph strategy.
            
            Method 2: 2-Step Clustering
            Directly clustering the filtered bounding boxes from the global aggregation step without the intermediate step of clustering tile-level clusters.
            This approach may yield more biologically relevant clusters by considering the spatial relationships of all detections at once,
            rather than relying on the potentially noisy intermediate clusters from individual tiles.
            """
            
            #######################################################################################################################################################################
            #######################################################################################################################################################################
            #######################################################################################################################################################################
            # Method 1: 3-Step Clustering
            # Cluster bounding boxes obtained from individual tiles are clustered again using the same IoU-based graph strategy.
            #######################################################################################################################################################################
            
            # cluster_boxes = []
            # cluster_confs = []
            # for cluster in all_clustered_info:
            #     if cluster["num_boxes"] > 1:
            #         cluster_boxes.append(cluster["bounding_box"])
            #         cluster_confs.append(cluster["mean_confidence"])

            # cluster_boxes = np.array(cluster_boxes)
            # cluster_confs = np.array(cluster_confs)
            
            # if len(cluster_boxes) > 0:
            #     infos = iou_based_clustering(cluster_boxes, cluster_confs, pil_img.size)
            # else:
            #     infos = []
            
            #######################################################################################################################################################################
            # Method 2: 2-Step Clustering directly on the filtered bounding boxes from the global aggregation step without the intermediate step of clustering tile-level clusters.
            #######################################################################################################################################################################
            if len(new_boxes) > 0:
                infos = iou_based_clustering(new_boxes, new_scores, pil_img.size)
            else:
                infos = []
            #######################################################################################################################################################################
            #######################################################################################################################################################################
            #######################################################################################################################################################################
            """
            Visualization (Qualitative Analysis)
            ------------------------------------
            For qualitative inspection and debugging:

            - Individual detections are drawn in red
            - Final cluster bounding boxes are drawn in blue

            Visualization is strictly separated from inference logic and
            does not influence computational results.
            """
            
            thyrocytes = [
                {"bbox": preds[:4].tolist(), "confidence": float(preds[4])}
                for preds in global_final_preds
            ]
            
            for info in infos:
                thyrocytes_inside_the_cluster = []
                thyrocytes_inside_the_cluster.extend(get_thyrocytes_inside_cluster(info, global_final_preds))
                number_of_thyrocytes = len(thyrocytes_inside_the_cluster)
                # draw_thyrocytes_inside_cluster(draw, thyrocytes_inside_the_cluster, color="red")
                if number_of_thyrocytes >= 10:
                    draw_cluster(draw, info, pil_img.size, label=f"Adequate with {len(thyrocytes_inside_the_cluster)} Thyrocytes", color='red')
                    draw_thyrocytes_inside_cluster(draw, thyrocytes_inside_the_cluster, color="red")
                elif number_of_thyrocytes < 10 and number_of_thyrocytes > 1:
                    draw_cluster(draw, info, pil_img.size, label=f"Inadequate with {len(thyrocytes_inside_the_cluster)} Thyrocytes", color='green')
                    draw_thyrocytes_inside_cluster(draw, thyrocytes_inside_the_cluster, color="green")
                else:
                    draw_thyrocytes_inside_cluster(draw, thyrocytes_inside_the_cluster, color="black")
                
            """
            Output Generation
            -----------------
            The final visualization image, containing both detections
            and clustered regions, is saved to disk.
            """
            dir_path = os.path.join("2_step_test_data_for_validation (level 1 - 3)", f'confidence_value: {str(conf_threshold)}', file_name)
            os.makedirs(dir_path, exist_ok=True)
            save_path = os.path.join(dir_path, file)
            pil_img.save(save_path)