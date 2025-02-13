import json, os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_gemini(results_dir):
    pass

def evaluate_gpt4(results_dir):
    pass

def extract_result_files(result_directory):
    pass


import math
from collections import defaultdict

def compute_iou(bbox1, bbox2):
    """
    Computes the Intersection-over-Union (IoU) of two bounding boxes.
    Each bbox is a list in the form [x1, y1, x2, y2].
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0
    return inter_area / union_area

def semantic_similarity(entity1: str, entity2: str, embedding_model: SentenceTransformerEmbeddings) -> float:

    em_1 = embedding_model.embed_query(entity1)
    em_2 = embedding_model.embed_query(entity2)
    similarity = cosine_similarity([em_1], [em_2])[0][0]
    
    return similarity

def compute_statistics(ground_truths, predictions, iou_threshold=0.5):
    """
    For each detection in the ground_truths list, finds matching predictions on the same frame,
    computes IoU between bounding boxes, and if IoU >= threshold, computes semantic similarity.
    
    Both ground_truths and predictions are lists of dictionaries with keys:
      - role (e.g., "what")
      - entity (a string)
      - frame (an integer)
      - bbox (list of four numbers [x1, y1, x2, y2])
      - certainty (a numeric value)
      - ocr_flag (boolean)
      
    Returns:
      A dictionary mapping (role, certainty) to a list of tuples (iou, similarity).
    """
    # Sort the lists by frame for clarity (if not already sorted)
    ground_truths = sorted(ground_truths, key=lambda d: d["frame"])
    predictions = sorted(predictions, key=lambda d: d["frame"])

    # Dictionary to store results keyed by (role, certainty)
    results = defaultdict(list)
    
    # For each ground truth detection, find predictions on the same frame
    for gt in ground_truths:
        gt_frame = gt["frame"]
        # Filter predictions that occur on the same frame.
        frame_preds = [pred for pred in predictions if pred["frame"] == gt_frame]
        for pred in frame_preds:
            iou = compute_iou(gt["bbox"], pred["bbox"])
            if iou >= iou_threshold:
                sim = semantic_similarity(gt["entity"], pred["entity"])
                key = (gt["role"], gt["certainty"])
                results[key].append((iou, sim))
    
    return results

def aggregate_stats(results):
    """
    Given a dictionary where keys are (role, certainty) and values are lists of (iou, similarity)
    tuples, compute summary statistics for each key.
    
    Returns a dictionary mapping (role, certainty) to aggregated stats.
    """
    aggregated = {}
    for key, values in results.items():
        if values:
            iou_vals, sim_vals = zip(*values)
            aggregated[key] = {
                "count": len(values),
                "avg_iou": sum(iou_vals) / len(iou_vals),
                "avg_similarity": sum(sim_vals) / len(sim_vals)
            }
        else:
            aggregated[key] = {
                "count": 0,
                "avg_iou": None,
                "avg_similarity": None
            }
    return aggregated


def evaluate_multivent_g(result_dir, multivent_g_json_file, threshold, model_name = "sentence-transformers/all-MiniLM-L12-v1"):
    
    embedding_model = SentenceTransformerEmbeddings(model_name=model_name, model_kwargs={"trust_remote_code" : True})

    
    with open(multivent_g_json_file, "r") as f:
        multivent_g_ground_truth = json.load(f)
        
    videos = [name for name in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, name))]
    
    for video in videos:
        video_ground_truth = multivent_g_ground_truth[video]
        with open(os.path.join(result_dir, video, "results.json")) as f:
            results = json.load(f)
        
        
    
        # Hallucinations
        
        # Retrieval of objects
        
        # Labeling of objects
        