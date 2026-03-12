"""
Visualization utilities for Bayesian Neural Networks and uncertainty quantification.

This module provides comprehensive visualization tools for uncertainty analysis,
calibration plots, and model interpretability.
"""

from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class UncertaintyVisualizer:
    """Comprehensive uncertainty visualization toolkit."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_uncertainty_distribution(
        self,
        uncertainties: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        method_name: str = "Model",
        save_path: Optional[str] = None
    ) -> None:
        """Plot uncertainty distribution analysis."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Convert to numpy
        unc_np = uncertainties.detach().cpu().numpy()
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Get predicted classes
        predicted_classes = np.argmax(pred_np, axis=1)
        correct_predictions = (predicted_classes == target_np)
        max_uncertainties = np.max(unc_np, axis=1)
        max_confidences = np.max(pred_np, axis=1)
        
        # 1. Uncertainty distribution
        axes[0, 0].hist(unc_np.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title(f'Uncertainty Distribution - {method_name}')
        axes[0, 0].set_xlabel('Uncertainty (Std Dev)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Uncertainty vs Correctness
        correct_unc = max_uncertainties[correct_predictions]
        incorrect_unc = max_uncertainties[~correct_predictions]
        
        axes[0, 1].hist(correct_unc, bins=30, alpha=0.7, label='Correct', color='green')
        axes[0, 1].hist(incorrect_unc, bins=30, alpha=0.7, label='Incorrect', color='red')
        axes[0, 1].set_title(f'Uncertainty by Correctness - {method_name}')
        axes[0, 1].set_xlabel('Max Uncertainty')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Confidence vs Uncertainty scatter
        axes[0, 2].scatter(max_confidences, max_uncertainties, alpha=0.6, s=20)
        axes[0, 2].set_title(f'Confidence vs Uncertainty - {method_name}')
        axes[0, 2].set_xlabel('Max Confidence')
        axes[0, 2].set_ylabel('Max Uncertainty')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Uncertainty heatmap for first 50 samples
        if len(unc_np) > 50:
            sample_unc = unc_np[:50]
        else:
            sample_unc = unc_np
        
        im = axes[1, 0].imshow(sample_unc.T, cmap='viridis', aspect='auto')
        axes[1, 0].set_title(f'Uncertainty Heatmap (First {len(sample_unc)} Samples) - {method_name}')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Class')
        plt.colorbar(im, ax=axes[1, 0])
        
        # 5. Uncertainty ranking
        sorted_indices = np.argsort(max_uncertainties)[::-1]
        sorted_correct = correct_predictions[sorted_indices]
        
        # Plot cumulative accuracy as we go from high to low uncertainty
        cumulative_accuracy = np.cumsum(sorted_correct) / np.arange(1, len(sorted_correct) + 1)
        
        axes[1, 1].plot(cumulative_accuracy, alpha=0.7, linewidth=2)
        axes[1, 1].set_title(f'Cumulative Accuracy by Uncertainty Ranking - {method_name}')
        axes[1, 1].set_xlabel('Sample Rank (High to Low Uncertainty)')
        axes[1, 1].set_ylabel('Cumulative Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Class-wise uncertainty
        unique_classes = np.unique(target_np)
        class_uncertainties = []
        class_labels = []
        
        for cls in unique_classes:
            class_mask = (target_np == cls)
            class_unc = max_uncertainties[class_mask]
            class_uncertainties.append(class_unc)
            class_labels.append(f'Class {cls}')
        
        axes[1, 2].boxplot(class_uncertainties, labels=class_labels)
        axes[1, 2].set_title(f'Uncertainty by True Class - {method_name}')
        axes[1, 2].set_xlabel('True Class')
        axes[1, 2].set_ylabel('Max Uncertainty')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_calibration_analysis(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        method_name: str = "Model",
        num_bins: int = 10,
        save_path: Optional[str] = None
    ) -> None:
        """Plot comprehensive calibration analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Convert to probabilities
        if predictions.dim() == 2 and predictions.size(1) > 1:
            probs = F.softmax(predictions, dim=1)
        else:
            probs = predictions
        
        # Get max confidence and predicted class
        max_confidences, predicted_classes = torch.max(probs, dim=1)
        
        # Convert to numpy
        confidences = max_confidences.cpu().numpy()
        predicted = predicted_classes.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Compute accuracy for each prediction
        correct = (predicted == targets_np).astype(float)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                bin_accuracies.append(correct[in_bin].mean())
                bin_confidences.append(confidences[in_bin].mean())
                bin_counts.append(in_bin.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)
        
        bin_accuracies = np.array(bin_accuracies)
        bin_confidences = np.array(bin_confidences)
        bin_counts = np.array(bin_counts)
        
        # 1. Reliability diagram
        axes[0, 0].bar(bin_confidences, bin_accuracies, width=0.1, alpha=0.7, 
                      color='skyblue', edgecolor='black')
        axes[0, 0].plot([0, 1], [0, 1], 'r--', label='Perfect Calibration', linewidth=2)
        axes[0, 0].set_title(f'Reliability Diagram - {method_name}')
        axes[0, 0].set_xlabel('Confidence')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Confidence distribution
        axes[0, 1].hist(confidences, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title(f'Confidence Distribution - {method_name}')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Accuracy vs Confidence scatter
        axes[1, 0].scatter(confidences, correct, alpha=0.5, s=10)
        axes[1, 0].plot([0, 1], [0, 1], 'r--', label='Perfect Calibration', linewidth=2)
        axes[1, 0].set_title(f'Accuracy vs Confidence - {method_name}')
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Sample count per bin
        axes[1, 1].bar(bin_confidences, bin_counts, width=0.1, alpha=0.7, 
                      color='orange', edgecolor='black')
        axes[1, 1].set_title(f'Sample Count per Bin - {method_name}')
        axes[1, 1].set_xlabel('Confidence')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(
        self,
        model_results: Dict[str, Dict[str, torch.Tensor]],
        metrics_to_plot: List[str] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Plot comparison across multiple models."""
        if metrics_to_plot is None:
            metrics_to_plot = ['accuracy', 'calibration_error', 'mean_uncertainty']
        
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        model_names = list(model_results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        for i, metric in enumerate(metrics_to_plot):
            values = []
            for model_name in model_names:
                if metric in model_results[model_name]:
                    values.append(model_results[model_name][metric])
                else:
                    values.append(0)
            
            bars = axes[i].bar(model_names, values, color=colors, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_uncertainty_vs_performance(
        self,
        uncertainties: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        method_name: str = "Model",
        save_path: Optional[str] = None
    ) -> None:
        """Plot uncertainty vs performance analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Convert to numpy
        unc_np = uncertainties.detach().cpu().numpy()
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        predicted_classes = np.argmax(pred_np, axis=1)
        correct_predictions = (predicted_classes == target_np)
        max_uncertainties = np.max(unc_np, axis=1)
        max_confidences = np.max(pred_np, axis=1)
        
        # 1. Uncertainty vs Accuracy scatter
        axes[0, 0].scatter(max_uncertainties, correct_predictions.astype(float), 
                          alpha=0.6, s=20)
        axes[0, 0].set_title(f'Uncertainty vs Accuracy - {method_name}')
        axes[0, 0].set_xlabel('Max Uncertainty')
        axes[0, 0].set_ylabel('Correct (1) / Incorrect (0)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Confidence vs Accuracy scatter
        axes[0, 1].scatter(max_confidences, correct_predictions.astype(float), 
                          alpha=0.6, s=20)
        axes[0, 1].set_title(f'Confidence vs Accuracy - {method_name}')
        axes[0, 1].set_xlabel('Max Confidence')
        axes[0, 1].set_ylabel('Correct (1) / Incorrect (0)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Uncertainty bins vs accuracy
        unc_bins = np.linspace(0, max_uncertainties.max(), 10)
        bin_accuracies = []
        bin_centers = []
        
        for i in range(len(unc_bins) - 1):
            bin_mask = (max_uncertainties >= unc_bins[i]) & (max_uncertainties < unc_bins[i + 1])
            if bin_mask.sum() > 0:
                bin_accuracies.append(correct_predictions[bin_mask].mean())
                bin_centers.append((unc_bins[i] + unc_bins[i + 1]) / 2)
        
        axes[1, 0].plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=8)
        axes[1, 0].set_title(f'Uncertainty Bins vs Accuracy - {method_name}')
        axes[1, 0].set_xlabel('Uncertainty Bin Center')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Confidence bins vs accuracy
        conf_bins = np.linspace(0, 1, 10)
        bin_accuracies = []
        bin_centers = []
        
        for i in range(len(conf_bins) - 1):
            bin_mask = (max_confidences >= conf_bins[i]) & (max_confidences < conf_bins[i + 1])
            if bin_mask.sum() > 0:
                bin_accuracies.append(correct_predictions[bin_mask].mean())
                bin_centers.append((conf_bins[i] + conf_bins[i + 1]) / 2)
        
        axes[1, 1].plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=8)
        axes[1, 1].plot([0, 1], [0, 1], 'r--', label='Perfect Calibration', linewidth=2)
        axes[1, 1].set_title(f'Confidence Bins vs Accuracy - {method_name}')
        axes[1, 1].set_xlabel('Confidence Bin Center')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()


class InteractiveVisualizer:
    """Interactive visualization using Plotly."""
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set3
    
    def create_uncertainty_dashboard(
        self,
        uncertainties: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        method_name: str = "Model"
    ) -> go.Figure:
        """Create interactive uncertainty dashboard."""
        # Convert to numpy
        unc_np = uncertainties.detach().cpu().numpy()
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        predicted_classes = np.argmax(pred_np, axis=1)
        correct_predictions = (predicted_classes == target_np)
        max_uncertainties = np.max(unc_np, axis=1)
        max_confidences = np.max(pred_np, axis=1)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Uncertainty Distribution',
                'Uncertainty vs Correctness',
                'Confidence vs Uncertainty',
                'Uncertainty Heatmap'
            ),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # 1. Uncertainty distribution
        fig.add_trace(
            go.Histogram(x=unc_np.flatten(), name='Uncertainty', nbinsx=50),
            row=1, col=1
        )
        
        # 2. Uncertainty vs Correctness
        fig.add_trace(
            go.Histogram(x=max_uncertainties[correct_predictions], 
                        name='Correct', nbinsx=30, opacity=0.7),
            row=1, col=2
        )
        fig.add_trace(
            go.Histogram(x=max_uncertainties[~correct_predictions], 
                        name='Incorrect', nbinsx=30, opacity=0.7),
            row=1, col=2
        )
        
        # 3. Confidence vs Uncertainty scatter
        fig.add_trace(
            go.Scatter(x=max_confidences, y=max_uncertainties,
                      mode='markers', name='Samples',
                      marker=dict(size=5, opacity=0.6)),
            row=2, col=1
        )
        
        # 4. Uncertainty heatmap
        if len(unc_np) > 50:
            sample_unc = unc_np[:50]
        else:
            sample_unc = unc_np
        
        fig.add_trace(
            go.Heatmap(z=sample_unc.T, colorscale='Viridis'),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'Uncertainty Analysis Dashboard - {method_name}',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_calibration_dashboard(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        method_name: str = "Model",
        num_bins: int = 10
    ) -> go.Figure:
        """Create interactive calibration dashboard."""
        # Convert to probabilities
        if predictions.dim() == 2 and predictions.size(1) > 1:
            probs = F.softmax(predictions, dim=1)
        else:
            probs = predictions
        
        # Get max confidence and predicted class
        max_confidences, predicted_classes = torch.max(probs, dim=1)
        
        # Convert to numpy
        confidences = max_confidences.cpu().numpy()
        predicted = predicted_classes.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Compute accuracy for each prediction
        correct = (predicted == targets_np).astype(float)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                bin_accuracies.append(correct[in_bin].mean())
                bin_confidences.append(confidences[in_bin].mean())
                bin_counts.append(in_bin.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Reliability Diagram',
                'Confidence Distribution',
                'Accuracy vs Confidence',
                'Sample Count per Bin'
            )
        )
        
        # 1. Reliability diagram
        fig.add_trace(
            go.Bar(x=bin_confidences, y=bin_accuracies, name='Accuracy',
                  marker_color='skyblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect Calibration',
                      line=dict(dash='dash', color='red')),
            row=1, col=1
        )
        
        # 2. Confidence distribution
        fig.add_trace(
            go.Histogram(x=confidences, name='Confidence', nbinsx=20,
                        marker_color='lightgreen'),
            row=1, col=2
        )
        
        # 3. Accuracy vs Confidence scatter
        fig.add_trace(
            go.Scatter(x=confidences, y=correct, mode='markers', name='Samples',
                      marker=dict(size=5, opacity=0.5)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect Calibration',
                      line=dict(dash='dash', color='red')),
            row=2, col=1
        )
        
        # 4. Sample count per bin
        fig.add_trace(
            go.Bar(x=bin_confidences, y=bin_counts, name='Count',
                  marker_color='orange'),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'Calibration Analysis Dashboard - {method_name}',
            height=800,
            showlegend=True
        )
        
        return fig


def create_summary_report(
    model_results: Dict[str, Dict[str, torch.Tensor]],
    save_path: Optional[str] = None
) -> None:
    """Create a comprehensive summary report."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    model_names = list(model_results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    
    # Extract metrics
    accuracies = []
    calibration_errors = []
    mean_uncertainties = []
    
    for model_name in model_names:
        results = model_results[model_name]
        predictions = results['predictions']
        targets = results['targets']
        
        # Calculate accuracy
        if predictions.dim() == 2 and predictions.size(1) > 1:
            probs = F.softmax(predictions, dim=1)
        else:
            probs = predictions
        
        predicted_classes = torch.argmax(probs, dim=1)
        accuracy = (predicted_classes == targets).float().mean().item()
        accuracies.append(accuracy)
        
        # Calculate calibration error (simplified)
        max_confidences, _ = torch.max(probs, dim=1)
        confidences = max_confidences.cpu().numpy()
        predicted = predicted_classes.cpu().numpy()
        targets_np = targets.cpu().numpy()
        correct = (predicted == targets_np).astype(float)
        
        # Simple calibration error
        bin_boundaries = np.linspace(0, 1, 11)
        ece = 0
        for i in range(len(bin_boundaries) - 1):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * in_bin.mean()
        
        calibration_errors.append(ece)
        
        # Mean uncertainty
        if 'uncertainties' in results:
            mean_unc = results['uncertainties'].mean().item()
        else:
            mean_unc = 0
        mean_uncertainties.append(mean_unc)
    
    # Plot metrics
    metrics = [accuracies, calibration_errors, mean_uncertainties]
    metric_names = ['Accuracy', 'Calibration Error', 'Mean Uncertainty']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        bars = axes[0, i].bar(model_names, metric, color=colors, alpha=0.7, edgecolor='black')
        axes[0, i].set_title(name)
        axes[0, i].tick_params(axis='x', rotation=45)
        axes[0, i].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, metric):
            height = bar.get_height()
            axes[0, i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
    
    # Additional analysis plots
    # Uncertainty vs accuracy scatter
    axes[1, 0].scatter(mean_uncertainties, accuracies, c=colors, s=100, alpha=0.7)
    for i, name in enumerate(model_names):
        axes[1, 0].annotate(name, (mean_uncertainties[i], accuracies[i]), 
                           xytext=(5, 5), textcoords='offset points')
    axes[1, 0].set_xlabel('Mean Uncertainty')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Uncertainty vs Accuracy')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Calibration vs accuracy scatter
    axes[1, 1].scatter(calibration_errors, accuracies, c=colors, s=100, alpha=0.7)
    for i, name in enumerate(model_names):
        axes[1, 1].annotate(name, (calibration_errors[i], accuracies[i]), 
                           xytext=(5, 5), textcoords='offset points')
    axes[1, 1].set_xlabel('Calibration Error')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Calibration vs Accuracy')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Model ranking
    # Combine metrics for ranking (higher accuracy, lower calibration error, lower uncertainty is better)
    normalized_acc = np.array(accuracies)
    normalized_cal = 1 - np.array(calibration_errors)  # Lower is better
    normalized_unc = 1 - np.array(mean_uncertainties) / max(mean_uncertainties)  # Lower is better
    
    combined_scores = normalized_acc + normalized_cal + normalized_unc
    ranking_indices = np.argsort(combined_scores)[::-1]
    
    axes[1, 2].bar(range(len(model_names)), [combined_scores[i] for i in ranking_indices], 
                  color=[colors[i] for i in ranking_indices], alpha=0.7, edgecolor='black')
    axes[1, 2].set_xticks(range(len(model_names)))
    axes[1, 2].set_xticklabels([model_names[i] for i in ranking_indices], rotation=45)
    axes[1, 2].set_title('Model Ranking (Combined Score)')
    axes[1, 2].set_ylabel('Combined Score')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Visualization module loaded successfully!")
    print("This module provides comprehensive visualization tools for Bayesian Neural Networks.")
