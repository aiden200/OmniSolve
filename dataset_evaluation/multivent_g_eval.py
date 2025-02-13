import json, os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from depth_extraction.extract_depth import DepthCalculator
from dataset_evaluation.visualize_charts import *
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import cv2

import math
from collections import defaultdict

def compute_iou(bbox1, bbox2, evaluation):
    """
    Computes the Intersection-over-Union (IoU) of two bounding boxes.
    Each bbox is a list in the form [x1, y1, x2, y2].
    """

    if evaluation == "gemini":
        # Some reason Gemini normalizes by 10 times more
        bbox2[0] = bbox2[0] / 10 
        bbox2[1] = bbox2[1] / 10 
        bbox2[2] = bbox2[2] / 10 
        bbox2[3] = bbox2[3] / 10 

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


def load_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    # Set the video position to the desired frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {frame_number} from {video_path}")
        cap.release()
        return None
    
    cap.release()
    return frame


def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0  # or you can return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames



def semantic_similarity(entity1: str, entity2: str, embedding_model: SentenceTransformerEmbeddings) -> float:

    em_1 = embedding_model.embed_query(entity1)
    em_2 = embedding_model.embed_query(entity2)
    similarity = cosine_similarity([em_1], [em_2])[0][0]
    
    return similarity


def frame_depth_level(frame, depth_extractor: DepthCalculator):
    depth = depth_extractor.extract_image_depth(frame)
    return depth




def aggregate_per_category(overall_stats):
    """
    For each category (e.g., "confidence", "depth", etc.), this function
    aggregates performance metrics by grouping all items with the same
    category value and computing the average precision, recall, and semantic similarity.
    
    overall_stats: dict
      Keys are category names and values are lists of tuples:
         (category_value, (precision, recall, avg_sem_sim))
    
    Returns:
      aggregated: dict
         Structure:
         {
           "confidence": {
               value1: {"avg_precision": ..., "avg_recall": ..., "avg_sem_sim": ..., "count": n},
               value2: { ... },
               ...
           },
           "depth": { ... },
           ...
         }
    """
    from collections import defaultdict
    aggregated = {}
    for category, entries in overall_stats.items():
        groups = defaultdict(list)
        # Group all entries by the category value.
        for cat_value, performance in entries:
            cat_value = round(float(cat_value), 3) if isinstance(cat_value, np.floating) else cat_value
            groups[cat_value].append(performance)
        aggregated[category] = {}
        for cat_value, performances in groups.items():
            cat_value = round(float(cat_value), 3) if isinstance(cat_value, np.floating) else cat_value
            # Separate out each performance component. We ignore None values.
            precisions = [p for (p, r, s) in performances if p is not None]
            recalls    = [r for (p, r, s) in performances if r is not None]
            sem_sims   = [s for (p, r, s) in performances if s is not None]
            
            avg_precision = round(float(np.mean(precisions)), 3) if precisions else 0
            avg_recall    = round(float(np.mean(recalls)), 3) if recalls else 0
            avg_sem_sim   = round(float(np.mean(sem_sims)), 3) if sem_sims else 0

            
            aggregated[category][cat_value] = {
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
                "avg_sem_sim": avg_sem_sim,
                "count": len(performances)
            }
    
    # print(aggregated)

    return aggregated




def evaluate_multivent_g(result_dir, 
                         multivent_g_json_file, 
                         multivent_yt_path,
                         threshold,
                         evaluation="gemini",
                         evaluation_file="gemini_results.json",
                         model_name="sentence-transformers/all-MiniLM-L12-v1",
                         visualize=False):
    # Initialize models.
    embedding_model = SentenceTransformerEmbeddings(model_name=model_name, model_kwargs={"trust_remote_code": True})
    depth_extractor = DepthCalculator()
    
    # Load ground truth.
    with open(multivent_g_json_file, "r") as f:
        multivent_g_ground_truth = json.load(f)
    
    # Each video is a sub-directory in result_dir.
    videos = [name for name in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, name))]
    
    overall_stats = defaultdict(list)
    
    # Global counters for overall metrics.
    total_gt_objects_global = 0
    total_pred_objects_global = 0
    detected_gt_global = 0   # Number of GT objects matched.
    detected_pred_global = 0 # Number of predictions matched.
    all_sem_sims = []        # All semantic similarity scores across matches.
    
    # Store per-video metrics.
    video_metrics = {}
    
    #TODO: Get rid of this
    videos = ["EDLy6c3jH8U"]

    # Load evaluation results 
    if evaluation == "gemini":
        with open(evaluation_file, "r") as f:
            predictions = json.load(f)
        
    
    for video in videos:
        video_load_path = os.path.join(multivent_yt_path, f"{video}.mp4")
        video_path = os.path.join(result_dir, video)
        video_ground_truth = multivent_g_ground_truth.get(video, {})
        gt_objects = video_ground_truth.get("spatial", [])
        
            
        # Organize GT by frame.
        gt_frames = defaultdict(list)
        for obj in gt_objects:
            frame_number = obj["frame"]
            gt_frames[frame_number].append(obj)
        
        # Video missing due to corruption
        if video not in predictions:
            continue

        # Organize predictions by frame.
        video_predictions = predictions[video]
        pred_frames = defaultdict(list)
        for pred in video_predictions:
            pred_frames[pred["frame"]].append(pred)
        
        
        # All frames present in GT or predictions.
        all_frames = set(gt_frames.keys()).union(pred_frames.keys())
        
        # Per-video counters.
        video_total_gt = 0
        video_total_pred = 0
        video_detected_gt = 0
        video_detected_pred = 0
        video_sem_sims = []  # Semantic similarities for this video.

        video_total_frames = get_total_frames(video_load_path)
        
        for frame_number in sorted(all_frames):
            frame = load_frame(video_load_path, frame_number)
            if frame is None:
                print(f"Error: Missing frame {frame_number} in video {video}")
                continue
            
            avg_depth = frame_depth_level(frame, depth_extractor)
            
            frame_gt_objects = gt_frames.get(frame_number, [])
            frame_pred_objects = pred_frames.get(frame_number, [])
            
            video_total_gt += len(frame_gt_objects)
            video_total_pred += len(frame_pred_objects)
            
            # For matching in the current frame.
            gt_matches = [False] * len(frame_gt_objects)
            pred_matches = [False] * len(frame_pred_objects)
            semantic_similarities = []
            
            # Loop over all GT and predictions.
            for i, gt_obj in enumerate(frame_gt_objects):
                for j, pred_obj in enumerate(frame_pred_objects):
                    iou = compute_iou(gt_obj["bbox"], pred_obj["bbox"], evaluation)
                    if iou >= threshold:
                        gt_matches[i] = True
                        pred_matches[j] = True
                        sim = semantic_similarity(gt_obj["entity"], pred_obj["entity"], embedding_model)
                        semantic_similarities.append(sim)
                        all_sem_sims.append(sim)
                        video_sem_sims.append(sim)
            
            frame_detected_gt = sum(gt_matches)
            frame_detected_pred = sum(pred_matches)
            
            # Frame-level recall and precision.
            frame_recall = frame_detected_gt / len(frame_gt_objects) if frame_gt_objects else 0
            frame_precision = frame_detected_pred / len(frame_pred_objects) if frame_pred_objects else 0
            avg_sem_sim = np.mean(semantic_similarities) if semantic_similarities else 0
            
            # Performance tuple for this frame.
            performance_tuple = (frame_precision, frame_recall, avg_sem_sim)
            
            # Record per-object (ground truth) stats with performance tuple.
            object_count = len(frame_gt_objects)
            for obj in frame_gt_objects:
                certainty = obj["certainty"]
                role = obj["role"]
                entity = obj["entity"]
                bbox = obj["bbox"]
                x1, y1, x2, y2 = bbox
                object_size = abs(x2 - x1) * abs(y2 - y1)

                current_frame_percentage = round(frame_number/video_total_frames, 3)
                
                overall_stats["confidence"].append((certainty, performance_tuple))
                overall_stats["depth"].append((avg_depth, performance_tuple))
                overall_stats["role"].append((role, performance_tuple))
                overall_stats["object_size"].append((object_size, performance_tuple))
                overall_stats["object_cluster"].append((object_count, performance_tuple))
                overall_stats["frame_number"].append((current_frame_percentage, performance_tuple))
            
            video_detected_gt += frame_detected_gt
            video_detected_pred += frame_detected_pred
        
        # Video-level recall and precision.
        video_recall = video_detected_gt / video_total_gt if video_total_gt > 0 else 0
        video_precision = video_detected_pred / video_total_pred if video_total_pred > 0 else 0
        video_avg_sem_sim = np.mean(video_sem_sims) if video_sem_sims else 0
        
        print(f"Video: {video}")
        print(f"  Retrieval Recall: {video_recall:.2f} (matched {video_detected_gt}/{video_total_gt})")
        print(f"  Hallucination Precision: {video_precision:.2f} (matched {video_detected_pred}/{video_total_pred})")
        
        video_metrics[video] = {
            "recall": video_recall,
            "precision": video_precision,
            "avg_sem_sim": video_avg_sem_sim,
            "total_gt": video_total_gt,
            "total_pred": video_total_pred,
            "detected_gt": video_detected_gt,
            "detected_pred": video_detected_pred
        }
        
        total_gt_objects_global += video_total_gt
        total_pred_objects_global += video_total_pred
        detected_gt_global += video_detected_gt
        detected_pred_global += video_detected_pred
    
    overall_recall = detected_gt_global / total_gt_objects_global if total_gt_objects_global > 0 else 0
    overall_precision = detected_pred_global / total_pred_objects_global if total_pred_objects_global > 0 else 0
    overall_avg_sem_sim = np.mean(all_sem_sims) if all_sem_sims else 0
    
    print("Overall Metrics:")
    print(f"  Overall Retrieval Recall: {overall_recall:.2f} ({detected_gt_global}/{total_gt_objects_global})")
    print(f"  Overall Hallucination Precision: {overall_precision:.2f} ({detected_pred_global}/{total_pred_objects_global})")
    
    # Aggregate the per-object stats.
    aggregated_results = {}
    for category, data in overall_stats.items():
        data_sorted = sorted(data, key=lambda x: x[0])
        aggregated_results[category] = data_sorted

    # Prepare an evaluation metrics dictionary.
    evaluation_metrics = {
        "overall_metrics": {
            "overall_recall": overall_recall,
            "overall_precision": overall_precision,
            "overall_avg_sem_sim": overall_avg_sem_sim,
            "total_gt_objects": total_gt_objects_global,
            "total_pred_objects": total_pred_objects_global,
            "detected_gt": detected_gt_global,
            "detected_pred": detected_pred_global
        },
        "video_metrics": video_metrics,
        "aggregated_stats": aggregated_results
    }
    
    # Save the evaluation metrics to a JSON file.
    with open("benchmark_results/evaluation_metrics.json", "w") as f:
        json.dump(evaluation_metrics, f, indent=2, default=lambda o: round(float(o), 3) if isinstance(o, np.floating) else o)
    

    aggregated_by_category = aggregate_per_category(evaluation_metrics["aggregated_stats"])
    with open("benchmark_results/category_evaluation.json", "w") as f:
        json.dump(aggregated_by_category, f, indent=2, default=lambda o: round(float(o), 3) if isinstance(o, np.floating) else o)


    # Optionally visualize the metrics.
    if visualize:
        visualize_metrics(evaluation, evaluation_metrics)
        visualize_aggregated_categories(evaluation, aggregated_by_category)

    return evaluation_metrics




result_dir = "/data/multivent_processed/"
multivent_g_json_file = "/home/aiden/Documents/cs/multiVENT/data/multivent_g.json"
multivent_yt_path = "/data/multivent_yt_videos/"
evaluation = "gemini"
evaluation_file = "benchmark_results/gemini_results.json"
threshold = .1


evaluation_metrics = evaluate_multivent_g(result_dir, multivent_g_json_file, multivent_yt_path, threshold, evaluation=evaluation, evaluation_file=evaluation_file, visualize=True)
# aggregated_by_category = aggregate_per_category(evaluation_metrics["aggregated_stats"])
# print(json.dumps(aggregated_by_category, indent=2, default=lambda o: round(float(o), 3) if isinstance(o, np.floating) else o))
