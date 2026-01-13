"""
# boxes = [b₁, b₂, …, bₙ]
# where bᵢ = [x1, y1, x2, y2, conf, class_id]
# Connectectivity matrix
# A[i, j] = 1  if IoU(bᵢ, bⱼ) ≥ τ_cluster
# A[i, j] = 0  otherwise

Graph interpretation (important)
Each node = one detection box
An edge exists if the two boxes overlap “meaningfully”
The graph is undirected
This graph answers the question:
“Which detections are spatially linked?”

Biological intuition
A single thyrocyte → many overlapping boxes → dense clique
Two touching thyrocytes → two cliques connected by a bridge
Separate cells → disconnected subgraphs
"""


import numpy as np
from nms_processor import NMSProcessor

def build_iou_graph_batch(boxes):
    """
    boxes: (N, 4)
    returns adjacency list graph
    """
    N = len(boxes)
    iou_matrix = NMSProcessor.box_iou_batch(boxes, boxes)

    graph = {i: set() for i in range(N)}

    for i in range(N):
        neighbors = np.where(iou_matrix[i] != 0.0)[0]
        for j in neighbors:
            if i != j:
                graph[i].add(j)
                graph[j].add(i)

    return graph

def connected_components(graph):
    visited = set()
    components = []

    for node in graph:
        if node not in visited:
            stack = [node]
            component = []

            while stack:
                n = stack.pop()
                if n not in visited:
                    visited.add(n)
                    component.append(n)
                    stack.extend(graph[n] - visited)

            components.append(component)

    return components
def cluster_features(boxes, confidences, clusters):
    cluster_info = []
    
    for cluster in clusters:
        cluster_boxes = boxes[cluster]
        cluster_confs = confidences[cluster]

        x1 = np.min(cluster_boxes[:, 0])
        y1 = np.min(cluster_boxes[:, 1])
        x2 = np.max(cluster_boxes[:, 2])
        y2 = np.max(cluster_boxes[:, 3])
        cluster_info.append({
            "num_boxes": len(cluster),
            "mean_confidence": float(np.mean(cluster_confs)),
            "bounding_box": (x1, y1, x2, y2),
            "area": (x2 - x1) * (y2 - y1)
        })

    return cluster_info

@staticmethod
def iou_based_clustering(boxes, confidences):
    graph = build_iou_graph_batch(boxes)
    clusters = connected_components(graph)
    return cluster_features(boxes, confidences, clusters)