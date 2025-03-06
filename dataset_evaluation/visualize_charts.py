import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_continuous_category(evaluation, category_name, data):
    """
    Visualize a continuous category (regression charts) for performance metrics.
    
    Parameters:
      - category_name: str, name of the category (e.g. "confidence", "depth", etc.)
      - data: dict, mapping category value to a dict with keys "avg_precision", "avg_recall", "avg_sem_sim", and "count".
    """
    # Sort data by the category value.
    sorted_items = sorted(data.items(), key=lambda x: x[0])
    # if category_name == "confidence":
    #     print(sorted_items)
    #     exit(0)
    xs = [float(x[0]) for x in sorted_items]  # Convert keys to float
    avg_precision = [x[1]["avg_precision"] for x in sorted_items]
    avg_recall = [x[1]["avg_recall"] for x in sorted_items]
    avg_sem_sim = [x[1]["avg_sem_sim"] for x in sorted_items]

    # Create a figure with 3 subplots: one for each metric.
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [("Precision", avg_precision), ("Recall", avg_recall), ("Semantic Similarity", avg_sem_sim)]
    
    for ax, (metric_name, y_values) in zip(axs, metrics):
        ax.scatter(xs, y_values, label="Data Points")
        # Compute regression line if we have at least 2 points.
        if len(xs) >= 2:
            coeffs = np.polyfit(xs, y_values, 1)
            poly_eqn = np.poly1d(coeffs)
            x_fit = np.linspace(min(xs), max(xs), 100)
            y_fit = poly_eqn(x_fit)
            ax.plot(x_fit, y_fit, color="red", label="Regression Line")
        ax.set_xlabel(category_name)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} vs {category_name}")
        ax.legend()

            
    fig.tight_layout()
    plt.savefig(f'benchmark_results/{evaluation}_{category_name}.png')
    plt.show()


def visualize_categorical_category(evaluation, category_name, data):
    """
    Visualize a categorical category (bar charts) for performance metrics.
    
    Parameters:
      - category_name: str, name of the category (should be "role" in your case)
      - data: dict, mapping each category value to a dict with performance metrics.
    """
    sorted_items = sorted(data.items(), key=lambda x: x[0])
    xs = [str(x[0]) for x in sorted_items]  # convert keys to strings for x-axis labels
    avg_precision = [x[1]["avg_precision"] for x in sorted_items]
    avg_recall = [x[1]["avg_recall"] for x in sorted_items]
    avg_sem_sim = [x[1]["avg_sem_sim"] for x in sorted_items]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [("Precision", avg_precision), ("Recall", avg_recall), ("Semantic Similarity", avg_sem_sim)]
    
    for ax, (metric_name, y_values) in zip(axs, metrics):
        ax.bar(xs, y_values, color="blue")
        ax.set_xlabel(category_name)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} by {category_name}")
    fig.tight_layout()
    plt.savefig(f'benchmark_results/{evaluation}_{category_name}.png')
    plt.show()


def visualize_aggregated_categories(evaluation, aggregated):
    """
    For each category in the aggregated dictionary, visualize the performance.
    For continuous categories, we show regression charts.
    For categorical ones (e.g., "role"), we show bar charts.
    """
    for category, data in aggregated.items():
        if category == "role":
            visualize_categorical_category(evaluation, category, data)
        else:
            visualize_continuous_category(evaluation, category, data)


import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_category_metrics_comparison(evaluations, category_metrics):
    roles = ["emergency-response", "outcome-occurred", "what", "when", "where", "who-affected"]
    metric_keys = ["avg_precision", "avg_recall", "avg_sem_sim"]
    metric_names = {"avg_precision": "Precision", "avg_recall": "Recall", "avg_sem_sim": "Semantic Similarity"}
    
    n_roles = len(roles)
    n_evals = len(evaluations)
    
    # Use matplotlib's default color cycle for consistency across plots
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Loop through each metric to create a separate plot.
    for metric in metric_keys:
        plt.figure(figsize=(8, 6))
        # x positions: one per role
        x = np.arange(n_roles)
        # Determine bar width based on number of evaluations.
        bar_width = 0.8 / n_evals
        # Calculate horizontal offsets for each evaluation's bars.
        offsets = [((i - (n_evals - 1) / 2) * bar_width) for i in range(n_evals)]
        
        # Plot bars for each evaluation.
        for i in range(n_evals):
            # Extract the metric values for each role in the given evaluation.
            values = [category_metrics[i]["role"][role][metric] for role in roles]
            plt.bar(x + offsets[i], values, bar_width, label=evaluations[i], color=colors[i % len(colors)])
        
        plt.xticks(x, roles, rotation=45)
        plt.ylabel("Metric Value")
        plt.ylim(0, 1)
        plt.title(f"Overall Category {metric_names[metric]} Comparison")
        plt.legend()
        
        # Save the figure
        filename = f'benchmark_results/category_{"_vs_".join(evaluations)}_overall_{metric}.png'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.show()




def visualize_metrics_comparison(evaluations, evaluation_metrics):
    # ---------------------------
    # Top: Overall Metrics Extraction
    # ---------------------------
    # Define metric labels and corresponding values for each evaluation
    labels = ["Recall", "Precision", "Avg Semantic Similarity"]
    overall_metrics = []
    values = []
    video_metrics = []
    for i in range(len(evaluations)):
        overall_metrics.append(evaluation_metrics[i]["overall_metrics"])
        values.append([
            evaluation_metrics[i]["overall_metrics"]["overall_recall"],
            evaluation_metrics[i]["overall_metrics"]["overall_precision"],
            evaluation_metrics[i]["overall_metrics"]["overall_avg_sem_sim"]
        ])
        video_metrics.append(evaluation_metrics[i]["video_metrics"])
        
    # ---------------------------
    # Overall Metrics Comparison Plot
    # ---------------------------
    x = np.arange(len(labels))  # Positions for labels on x-axis
    n_evals = len(evaluations)
    bar_width = 0.8 / n_evals  # Allocate 80% of the group width
    offsets = [((i - (n_evals - 1) / 2) * bar_width) for i in range(n_evals)]
    
    # Use matplotlib's default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.figure(figsize=(8, 6))
    for i in range(n_evals):
        plt.bar(x + offsets[i], values[i], bar_width, label=evaluations[i], color=colors[i % len(colors)])
    
    plt.ylim(0, 1)
    plt.xticks(x, labels)
    plt.ylabel("Metric Value")
    plt.title("Overall Metrics Comparison")
    plt.legend()
    overall_filename = f'benchmark_results/{"_vs_".join(evaluations)}_overall_metrics.png'
    plt.savefig(overall_filename)
    plt.show()
    
    # ---------------------------
    # Per-Video Metrics Comparison Plot
    # ---------------------------
    # Assume each evaluation has the same set of video keys
    videos = list(video_metrics[0].keys())
    x_videos = np.arange(len(videos))  # positions for videos on x-axis
    bar_width = 0.8 / n_evals  # same dynamic bar width
    offsets = [((i - (n_evals - 1) / 2) * bar_width) for i in range(n_evals)]
    
    plt.figure(figsize=(12, 8))
    
    # Recall subplot
    plt.subplot(3, 1, 1)
    for i in range(n_evals):
        recalls = [video_metrics[i][v]["recall"] for v in videos]
        plt.bar(x_videos + offsets[i], recalls, bar_width, label=evaluations[i], color=colors[i % len(colors)])
    plt.ylabel("Recall")
    plt.title("Per-Video Metrics Comparison")
    plt.xticks(ticks=x_videos, labels=videos, rotation=45)
    plt.legend()
    
    # Precision subplot
    plt.subplot(3, 1, 2)
    for i in range(n_evals):
        precisions = [video_metrics[i][v]["precision"] for v in videos]
        plt.bar(x_videos + offsets[i], precisions, bar_width, label=evaluations[i], color=colors[i % len(colors)])
    plt.ylabel("Precision")
    plt.xticks(ticks=x_videos, labels=videos, rotation=45)
    plt.legend()
    
    # Avg Semantic Similarity subplot
    plt.subplot(3, 1, 3)
    for i in range(n_evals):
        avg_sem_sims = [video_metrics[i][v]["avg_sem_sim"] for v in videos]
        plt.bar(x_videos + offsets[i], avg_sem_sims, bar_width, label=evaluations[i], color=colors[i % len(colors)])
    plt.ylabel("Avg Semantic Similarity")
    plt.xticks(ticks=x_videos, labels=videos, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    per_video_filename = f'benchmark_results/{"_vs_".join(evaluations)}_per_video_metrics.png'
    plt.savefig(per_video_filename)
    plt.show()



def visualize_metrics(evaluation, evaluation_metrics):
    overall_metrics = evaluation_metrics["overall_metrics"]
    video_metrics = evaluation_metrics["video_metrics"]
    
    # Plot overall metrics.
    labels = ["Recall", "Precision", "Avg Semantic Similarity"]
    values = [
        overall_metrics["overall_recall"],
        overall_metrics["overall_precision"],
        overall_metrics["overall_avg_sem_sim"]
    ]
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=["blue", "green", "orange"])
    plt.ylim(0, 1)
    plt.title("Overall Metrics")
    plt.savefig(f'benchmark_results/{evaluation}_overall_metrics.png')
    plt.show()
    
    # Plot per-video metrics.
    videos = list(video_metrics.keys())
    recalls = [video_metrics[v]["recall"] for v in videos]
    precisions = [video_metrics[v]["precision"] for v in videos]
    avg_sem_sims = [video_metrics[v]["avg_sem_sim"] for v in videos]
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(3, 1, 1)
    plt.bar(videos, recalls, color="blue")
    plt.ylabel("Recall")
    plt.title("Per-Video Metrics")
    
    plt.subplot(3, 1, 2)
    plt.bar(videos, precisions, color="green")
    plt.ylabel("Precision")
    
    plt.subplot(3, 1, 3)
    plt.bar(videos, avg_sem_sims, color="orange")
    plt.ylabel("Avg Semantic Similarity")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'benchmark_results/{evaluation}_per_video_metrics.png')
    plt.show()




def compare_continuous_category(evaluation1, evaluation2, category_name, data1, data2):
    """
    Compare two continuous categories (regression charts) for performance metrics.
    
    Parameters:
      - evaluation1, evaluation2: str, names/labels for the two evaluation results.
      - category_name: str, name of the category (e.g. "confidence", "depth", etc.)
      - data1, data2: dicts, mapping category value to a dict with keys "avg_precision", 
        "avg_recall", "avg_sem_sim", and "count".
    """
    # Sort data by the category value.
    sorted_data1 = sorted(data1.items(), key=lambda x: float(x[0]))
    sorted_data2 = sorted(data2.items(), key=lambda x: float(x[0]))
    
    xs1 = [float(item[0]) for item in sorted_data1]
    xs2 = [float(item[0]) for item in sorted_data2]
    
    precision1 = [item[1]["avg_precision"] for item in sorted_data1]
    recall1    = [item[1]["avg_recall"] for item in sorted_data1]
    semsim1    = [item[1]["avg_sem_sim"] for item in sorted_data1]
    
    precision2 = [item[1]["avg_precision"] for item in sorted_data2]
    recall2    = [item[1]["avg_recall"] for item in sorted_data2]
    semsim2    = [item[1]["avg_sem_sim"] for item in sorted_data2]
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("Precision", precision1, precision2),
        ("Recall",    recall1,    recall2),
        ("Semantic Similarity", semsim1, semsim2)
    ]
    
    colors = {evaluation1: "blue", evaluation2: "green"}
    
    for ax, (metric_name, y1, y2) in zip(axs, metrics):
        # Plot evaluation1
        ax.scatter(xs1, y1, color=colors[evaluation1], label=f"{evaluation1} Data")
        if len(xs1) >= 2:
            coeffs1 = np.polyfit(xs1, y1, 1)
            poly_eqn1 = np.poly1d(coeffs1)
            x_fit1 = np.linspace(min(xs1), max(xs1), 100)
            y_fit1 = poly_eqn1(x_fit1)
            ax.plot(x_fit1, y_fit1, color=colors[evaluation1], linestyle="--", label=f"{evaluation1} Regression")
        
        # Plot evaluation2
        ax.scatter(xs2, y2, color=colors[evaluation2], label=f"{evaluation2} Data")
        if len(xs2) >= 2:
            coeffs2 = np.polyfit(xs2, y2, 1)
            poly_eqn2 = np.poly1d(coeffs2)
            x_fit2 = np.linspace(min(xs2), max(xs2), 100)
            y_fit2 = poly_eqn2(x_fit2)
            ax.plot(x_fit2, y_fit2, color=colors[evaluation2], linestyle="--", label=f"{evaluation2} Regression")
        
        ax.set_xlabel(category_name)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} vs {category_name}")
        ax.legend()
        
    fig.tight_layout()
    plt.savefig(f'benchmark_results/compare_{evaluation1}_{evaluation2}_{category_name}.png')
    plt.show()



def compare_categorical_category(evaluation1, evaluation2, category_name, data1, data2):
    """
    Compare two categorical categories (bar charts) for performance metrics.
    
    Parameters:
      - evaluation1, evaluation2: str, names/labels for the two evaluation results.
      - category_name: str, name of the category (e.g., "role").
      - data1, data2: dicts, mapping each category value to a dict with performance metrics.
    """
    # Sort data; we assume both results have the same categorical keys.
    sorted_items = sorted(data1.items(), key=lambda x: x[0])
    xs = [str(item[0]) for item in sorted_items]
    
    precision1 = [item[1]["avg_precision"] for item in sorted_items]
    recall1    = [item[1]["avg_recall"] for item in sorted_items]
    semsim1    = [item[1]["avg_sem_sim"] for item in sorted_items]
    
    # Assume keys in data2 match data1; otherwise, adjust accordingly.
    precision2 = [data2[k]["avg_precision"] for k in xs]
    recall2    = [data2[k]["avg_recall"] for k in xs]
    semsim2    = [data2[k]["avg_sem_sim"] for k in xs]
    
    x_pos = np.arange(len(xs))
    width = 0.35  # width of each bar
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("Precision", precision1, precision2),
        ("Recall",    recall1,    recall2),
        ("Semantic Similarity", semsim1, semsim2)
    ]
    
    for ax, (metric_name, m1, m2) in zip(axs, metrics):
        ax.bar(x_pos - width/2, m1, width=width, color="blue", label=evaluation1)
        ax.bar(x_pos + width/2, m2, width=width, color="green", label=evaluation2)
        ax.set_xlabel(category_name)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} by {category_name}")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(xs)
        ax.legend()
    
    fig.tight_layout()
    plt.savefig(f'benchmark_results/compare_{evaluation1}_{evaluation2}_{category_name}.png')
    plt.show()



# def compare_aggregated_categories(evaluation1, evaluation2, aggregated1, aggregated2):
#     """
#     For each category in the aggregated dictionary, visualize the performance.
#     For continuous categories, we show regression charts.
#     For categorical ones (e.g., "role"), we show bar charts.
#     """
#     for category, data in aggregated1.items():
#         if category == "role":
#             compare_aggregated_categories(evaluation1,evaluation2, category, data)
#         else:
#             visualize_continuous_category(evaluation, category, data)