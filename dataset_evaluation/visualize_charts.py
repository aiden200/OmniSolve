import matplotlib.pyplot as plt
import numpy as np

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