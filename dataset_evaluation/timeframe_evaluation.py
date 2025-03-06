import os, json
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

def get_subclip_segments(subclip_folder):
    """
    Given a folder containing subclip files (named 0.mp4, 1.mp4, etc.),
    compute the predicted temporal segments as (start, end) tuples.
    """
    # Sort files based on their numeric name
    i = 0
    
    segments = []
    current_time = 0.0

    while os.path.exists(os.path.join(subclip_folder, f"{i}.mp4")):
        file_path = os.path.join(subclip_folder, f"{i}.mp4")
        # Use MoviePy to load the clip and get its duration
        clip = VideoFileClip(file_path)
        duration = clip.duration
        segments.append((current_time, current_time + duration))
        current_time += duration
        clip.reader.close()  # Close the clip to free resources
        if clip.audio:
            clip.audio.reader.close_proc()
        i += 1
    
    return segments

def compute_iou(seg1, seg2):
    """
    Compute the Intersection over Union (IoU) for two segments.
    Each segment is a tuple: (start, end).
    """
    start1, end1 = seg1
    start2, end2 = seg2
    
    # Compute the intersection duration
    intersection = max(0, min(end1, end2) - max(start1, start2))
    # Compute the union duration
    union = max(end1, end2) - min(start1, start2)
    return intersection / union if union > 0 else 0

def evaluate_segments(ground_truth, predictions, iou_threshold=0.5):
    """
    Given lists of ground truth segments and predicted segments (each a tuple of (start, end)),
    evaluate segmentation performance using precision, recall, F1 score, and average IoU.
    
    Matching is done by considering a predicted segment as a true positive if it has an IoU 
    above the threshold with any unmatched ground truth segment.
    """
    true_positives = 0
    matched_gt = set()  # indices of matched ground truth segments
    matched_pred = set()  # indices of matched predicted segments
    total_iou = 0
    count_iou = 0
    
    # Iterate over each predicted segment and try to find a matching ground truth segment.
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truth):
            if j in matched_gt:
                continue  # already matched this ground truth segment
            iou = compute_iou(pred, gt)
            if iou >= iou_threshold:
                true_positives += 1
                matched_gt.add(j)
                matched_pred.add(i)
                total_iou += iou
                count_iou += 1
                break  # move on to next predicted segment after a match
    
    precision = true_positives / len(predictions) if predictions else 0
    recall = true_positives / len(ground_truth) if ground_truth else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    average_iou = total_iou / count_iou if count_iou > 0 else 0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'average_iou': average_iou,
        'num_predictions': len(predictions),
        'num_ground_truth': len(ground_truth)
    }
    return metrics

def compute_cumulative_metrics_over_time(ground_truth, predictions, iou_threshold=0.5, time_step=1.0):
    """
    Compute cumulative evaluation metrics over time.
    For each time t, consider only segments that have ended by time t.
    Returns a tuple of lists: (times, precision_series, recall_series, f1_series, iou_series)
    """
    # Determine maximum time using both ground truth and predictions
    max_gt = max([seg[1] for seg in ground_truth]) if ground_truth else 0
    max_pred = max([seg[1] for seg in predictions]) if predictions else 0
    max_time = max(max_gt, max_pred)
    
    times = []
    precision_series = []
    recall_series = []
    f1_series = []
    iou_series = []
    
    t = time_step
    while t <= max_time:
        gt_t = [seg for seg in ground_truth if seg[1] <= t]
        pred_t = [seg for seg in predictions if seg[1] <= t]
        m = evaluate_segments(gt_t, pred_t, iou_threshold)
        times.append(t)
        precision_series.append(m['precision'])
        recall_series.append(m['recall'])
        f1_series.append(m['f1_score'])
        iou_series.append(m['average_iou'])
        t += time_step
        
    return times, precision_series, recall_series, f1_series, iou_series


def eval_multivent_g(result_dir, multivent_g_json_file, output_dir, time_step=1.0):
    with open(multivent_g_json_file, "r") as f:
        multivent_g_ground_truth = json.load(f)
    
    video_metrics = []
    # Each video is a sub-directory in result_dir.
    videos = [name for name in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, name))]
    
    # Create output directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for video in tqdm(videos, desc="Evaluating videos"):
        ground_truth_segments = []
        subclip_folder = os.path.join(result_dir, video)
        predicted_segments = get_subclip_segments(subclip_folder)
        
        # Get the ground truth temporal spans from JSON
        temporal_spans = multivent_g_ground_truth[video]["temporal"]
        for t in temporal_spans:
            ground_truth_segments.append(t["time"])
        
        # Overall metrics for the full video
        metrics = evaluate_segments(ground_truth_segments, predicted_segments, iou_threshold=0.5)
        metrics["video"] = video
        video_metrics.append(metrics)
        
        # print(f"Video: {video}")
        # for key, value in metrics.items():
        #     if key != "video":
        #         print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
        # print("-------------------------------------------------")
        
        # # Compute cumulative metrics over time for this video.
        # times, precision_series, recall_series, f1_series, iou_series = \
        #     compute_cumulative_metrics_over_time(ground_truth_segments, predicted_segments, iou_threshold=0.5, time_step=time_step)
        
        # # Plot cumulative metrics over time.
        # plt.figure(figsize=(10, 6))
        # plt.plot(times, precision_series, label="Precision")
        # plt.plot(times, recall_series, label="Recall")
        # plt.plot(times, f1_series, label="F1 Score")
        # plt.plot(times, iou_series, label="Average IoU")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Score")
        # plt.title(f"Cumulative Metrics Over Time for {video}")
        # plt.legend()
        # plt.tight_layout()
        # cumulative_plot_path = os.path.join(output_dir, f"{video}_cumulative_metrics.png")
        # plt.savefig(cumulative_plot_path)
        # plt.close()
        # print(f"Saved cumulative metrics plot for {video} to {cumulative_plot_path}")
    
    # Create a DataFrame for overall metrics across videos.
    df = pd.DataFrame(video_metrics)
    print("\nAggregated Metrics DataFrame:")
    print(df)
    
    # Save aggregated metrics to CSV and JSON files.
    metrics_csv = os.path.join(output_dir, "evaluation_metrics.csv")
    df.to_csv(metrics_csv, index=False)
    metrics_json = os.path.join(output_dir, "evaluation_metrics.json")
    df.to_json(metrics_json, orient="records", indent=4)
    print(f"Saved evaluation metrics to {metrics_csv} and {metrics_json}")
    
    # Plot bar charts for key metrics per video.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    metrics_to_plot = ['precision', 'recall', 'f1_score', 'average_iou']
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        df.plot.bar(x='video', y=metric, ax=ax, legend=False)
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Video")
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    bar_chart_path = os.path.join(output_dir, "bar_charts.png")
    fig.savefig(bar_chart_path)
    print(f"Saved bar charts plot to {bar_chart_path}")
    plt.close(fig)
    
    # Plot comparison of the number of ground truth segments vs. predicted segments for each video.
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    width = 0.35
    ind = range(len(df))
    ax2.bar(ind, df['num_ground_truth'], width, label='Ground Truth')
    ax2.bar([i + width for i in ind], df['num_predictions'], width, label='Predictions')
    ax2.set_xlabel("Video")
    ax2.set_ylabel("Count")
    ax2.set_title("Number of Ground Truth Segments vs Predicted Segments")
    ax2.set_xticks([i + width/2 for i in ind])
    ax2.set_xticklabels(df['video'], rotation=45, ha="right")
    ax2.legend()
    plt.tight_layout()
    comparison_chart_path = os.path.join(output_dir, "segment_comparison.png")
    fig2.savefig(comparison_chart_path)
    print(f"Saved segment comparison plot to {comparison_chart_path}")
    plt.close(fig2)



    overall_avg = {
        'precision': df['precision'].mean(),
        'recall': df['recall'].mean(),
        'f1_score': df['f1_score'].mean(),
        'average_iou': df['average_iou'].mean()
    }
    
    plt.figure(figsize=(8, 6))
    metrics_names = list(overall_avg.keys())
    avg_values = list(overall_avg.values())
    bars = plt.bar(metrics_names, avg_values)
    plt.xlabel("Metric")
    plt.ylabel("Average Score")
    plt.title("Overall Average Metrics Across All Videos")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    overall_avg_path = os.path.join(output_dir, "overall_average_metrics.png")
    plt.savefig(overall_avg_path)
    print(f"Saved overall average metrics plot to {overall_avg_path}")
    plt.close()
        


# Example usage:
if __name__ == "__main__":
    result_dir = "/data/multivent_processed/"
    multivent_g_json_file = "/home/aiden/Documents/cs/multiVENT/data/multivent_g.json"
    output_dir = "benchmark_results/timespan_results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    eval_multivent_g(result_dir, multivent_g_json_file, output_dir, time_step=1.0)
    



    # # Assume the ground truth segments for video 'x' are defined as a list of tuples.
    # # For example: video x is segmented as 0-8, 8-16, etc.
    # ground_truth_segments = [(0, 8), (8, 16), (16, 24)]
    
    # # Folder where the model's subclips for video x are stored.
    # subclip_folder = "path/to/video_x_subclips"  # update with your actual path
    
    # # Get predicted segments from subclips
    # predicted_segments = get_subclip_segments(subclip_folder)
    
    # # Evaluate the segments using the defined metrics
    # metrics = evaluate_segments(ground_truth_segments, predicted_segments, iou_threshold=0.5)
    
    # print("Evaluation Metrics:")
    # for key, value in metrics.items():
    #     print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
