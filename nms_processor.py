"""
Source: https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/
"""
import numpy as np
from utils import box_iou_batch
class NMSProcessor:
    def __init__(self, postprocessing_cfg):
        # self.conf_thres = postprocessing_cfg["conf_threshold"]
        self.iou_thres = postprocessing_cfg["iou_threshold"]


    def non_max_suppression(self, predictions: np.ndarray) -> np.ndarray:
        rows, columns = predictions.shape

        sort_index = np.flip(predictions[:, 4].argsort())
        predictions = predictions[sort_index]

        boxes = predictions[:, :4]
        categories = predictions[:, 5]
        ious = box_iou_batch(boxes, boxes)
        ious = ious - np.eye(rows)

        keep = np.ones(rows, dtype=bool)

        for index, (iou, category) in enumerate(zip(ious, categories)):
            if not keep[index]:
                continue

            condition = (iou > self.iou_thres) & (categories == category)
            keep = keep & ~condition

        return keep[sort_index.argsort()]