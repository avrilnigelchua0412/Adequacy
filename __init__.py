from utils import process_yolo_preds, draw_cluster, get_img, helper_os_walk, weighted_boxes_fusion_helper, get_thyrocytes_inside_cluster, filter_pipeline_preds, draw_thyrocytes_inside_cluster
from IoU_adjacency_matrix import iou_based_clustering
from tile_inferencer import TileInferencer
import onnxruntime as ort
from nms_processor import NMSProcessor
import numpy as np
import json, yaml
from PIL import Image, ImageDraw
from image_tiler import ImageTiler
import os
import pandas as pd
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
        # preprocessed_img = inferencer.preprocess(tile)

        outputs = inferencer.infer(tile)
        
        preds = outputs[0][0]
        
        boxes, confs, classes = process_yolo_preds(preds, conf_threshold)
        
        # print(classes)
        
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

        """1st Filter out boxes with extreme aspect ratios and remove boxes that are fully inside another box"""
        
        thyrocyte_mask = final_preds[:, 5] == 0
        thyro_preds = final_preds[thyrocyte_mask]

        # Apply filtering directly on full prediction rows
        thyro_preds = filter_pipeline_preds(thyro_preds)

        # Combine back with confusants
        confusant_preds = final_preds[final_preds[:, 5] == 1]

        if len(confusant_preds) > 0:
            final_preds = np.vstack([thyro_preds, confusant_preds])
        else:
            final_preds = thyro_preds

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

        thyro_preds_global = final_preds[final_preds[:, 5] == 0]
        
        clustered_info = iou_based_clustering(thyro_preds_global[:, :4], thyro_preds_global[:, 4], pil_img.size)

        """Visualization of tile-level clusters (for debugging)"""
        for info in clustered_info:
            if info['num_boxes'] > 1:
                draw_cluster(draw, info, pil_img.size)

        all_clustered_info.extend(clustered_info)
        
    return np.vstack(all_preds), all_clustered_info

if __name__ == "__main__":
    
    test_files = pd.read_csv("test_df_summary.csv")['File'].to_list()
    
    data_path = "fnab"
    
    # model_path = "fnab_models/YoloV5/modelv2-confusant-level3/weights/best.onnx"
    model_path = "fnab_models/YoloV7/modelv2-confusant-level3/weights/best.onnx"
    
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
    
    conf_threshold = post_cfg["conf_threshold"]
    for img_path, file, file_name, format in helper_os_walk(data_path):
        if file in test_files: 
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
            
            thyrocyte_mask = all_preds[:, 5] == 0
            thyro_preds = all_preds[thyrocyte_mask]
            """2nd Filter out boxes with extreme aspect ratios and remove boxes that are fully inside another box"""
            
            thyro_preds = filter_pipeline_preds(thyro_preds)
            
            thyro_boxes = thyro_preds[:, :4]
            thyro_confs = thyro_preds[:, 4]
            thyro_classes = thyro_preds[:, 5]
            
            
            
            #################################################################################################################################################
            
            """
            Global Detection Aggregation
            ----------------------------
            All tile-level detections are concatenated into a single
            image-level array. At this stage, all bounding boxes share
            a common global coordinate system.
            """
            new_boxes, new_scores = weighted_boxes_fusion_helper(thyro_boxes, thyro_confs, pil_img.size, iou_thr=0.25)
            thyro_classes = np.zeros(len(new_boxes))
            thyro_final_preds = np.c_[new_boxes, new_scores, thyro_classes]
            
            # Combine back with confusants
            confusant_preds = all_preds[all_preds[:, 5] == 1]
            if len(confusant_preds) > 0:
                final_preds = np.vstack([thyro_final_preds, confusant_preds])
            else:
                final_preds = thyro_final_preds
            
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
            
            cluster_boxes = []
            cluster_confs = []
            for cluster in all_clustered_info:
                if cluster["num_boxes"] > 1:
                    cluster_boxes.append(cluster["bounding_box"])
                    cluster_confs.append(cluster["mean_confidence"])

            cluster_boxes = np.array(cluster_boxes)
            cluster_confs = np.array(cluster_confs)
            
            if len(cluster_boxes) > 0:
                infos = iou_based_clustering(cluster_boxes, cluster_confs, pil_img.size)
            else:
                infos = []
            
            #######################################################################################################################################################################
            # Method 2: 2-Step Clustering directly on the filtered bounding boxes from the global aggregation step without the intermediate step of clustering tile-level clusters.
            #######################################################################################################################################################################
            # if len(new_boxes) > 0:
            #     infos = iou_based_clustering(new_boxes, new_scores, pil_img.size)
            # else:
            #     infos = []
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
                for preds in thyro_final_preds
            ]
            
            for t in thyrocytes:
                draw.rectangle(t["bbox"], outline="black", width=3)
                draw.text((t["bbox"][0], t["bbox"][3]), f"{t['confidence']:.2f}", fill="black")
                
            for info in infos:
                thyrocytes_inside_the_cluster = []
                thyrocytes_inside_the_cluster.extend(get_thyrocytes_inside_cluster(info, thyro_final_preds))
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
            dir_path = os.path.join("3_step_test_data_for_validation (level 1 - 3)")
            os.makedirs(dir_path, exist_ok=True)
            save_path = os.path.join(dir_path, file)
            pil_img.save(save_path)