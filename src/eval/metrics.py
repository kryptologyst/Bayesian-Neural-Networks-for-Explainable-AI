"""
Evaluation metrics for Bayesian Neural Networks and uncertainty quantification.

This module provides comprehensive evaluation metrics for uncertainty estimation,
calibration, and model reliability.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import (
    Accuracy, CalibrationError, 
    Precision, Recall, F1Score, AUROC
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, brier_score_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


class UncertaintyMetrics:
    """Comprehensive uncertainty evaluation metrics."""
    
    def __init__(self, num_classes: int, device: torch.device = None):
        self.num_classes = num_classes
        self.device = device or torch.device("cpu")
        
        # Initialize torchmetrics
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        self.calibration_error = CalibrationError(task="multiclass", num_classes=num_classes).to(self.device)
        self.precision = Precision(task="multiclass", num_classes=num_classes, average="macro").to(self.device)
        self.recall = Recall(task="multiclass", num_classes=num_classes, average="macro").to(self.device)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(self.device)
        self.auroc = AUROC(task="multiclass", num_classes=num_classes, average="macro").to(self.device)
    
    def compute_all_metrics(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        confidences: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute all uncertainty and calibration metrics."""
        metrics = {}
        
        # Convert to probabilities if needed
        if predictions.dim() == 2 and predictions.size(1) > 1:
            probs = F.softmax(predictions, dim=1)
        else:
            probs = predictions
        
        # Basic classification metrics
        metrics["accuracy"] = self.accuracy(probs, targets).item()
        metrics["precision"] = self.precision(probs, targets).item()
        metrics["recall"] = self.recall(probs, targets).item()
        metrics["f1_score"] = self.f1(probs, targets).item()
        metrics["auroc"] = self.auroc(probs, targets).item()
        
        # Calibration metrics
        metrics["calibration_error"] = self.calibration_error(probs, targets).item()
        # Use sklearn's brier_score_loss for binary, log_loss for multiclass
        if num_classes == 2:
            metrics["brier_score"] = brier_score_loss(targets.cpu().numpy(), probs[:, 1].cpu().numpy())
        else:
            metrics["brier_score"] = log_loss(targets.cpu().numpy(), probs.cpu().numpy())
        
        # Uncertainty metrics
        metrics["mean_uncertainty"] = uncertainties.mean().item()
        metrics["max_uncertainty"] = uncertainties.max().item()
        metrics["uncertainty_std"] = uncertainties.std().item()
        
        # Confidence metrics
        if confidences is not None:
            metrics["mean_confidence"] = confidences.mean().item()
            metrics["confidence_std"] = confidences.std().item()
        else:
            max_confidences = torch.max(probs, dim=1)[0]
            metrics["mean_confidence"] = max_confidences.mean().item()
            metrics["confidence_std"] = max_confidences.std().item()
        
        # Additional uncertainty analysis
        predicted_classes = torch.argmax(probs, dim=1)
        correct_predictions = (predicted_classes == targets)
        
        # Uncertainty for correct vs incorrect predictions
        correct_uncertainty = uncertainties[correct_predictions]
        incorrect_uncertainty = uncertainties[~correct_predictions]
        
        if len(correct_uncertainty) > 0:
            metrics["uncertainty_correct_mean"] = correct_uncertainty.mean().item()
        else:
            metrics["uncertainty_correct_mean"] = 0.0
            
        if len(incorrect_uncertainty) > 0:
            metrics["uncertainty_incorrect_mean"] = incorrect_uncertainty.mean().item()
        else:
            metrics["uncertainty_incorrect_mean"] = 0.0
        
        # Uncertainty calibration
        metrics["uncertainty_calibration"] = self._compute_uncertainty_calibration(
            uncertainties, correct_predictions
        )
        
        return metrics
    
    def _compute_uncertainty_calibration(
        self, 
        uncertainties: torch.Tensor, 
        correct_predictions: torch.Tensor
    ) -> float:
        """Compute uncertainty calibration (how well uncertainty predicts correctness)."""
        # Use max uncertainty as the uncertainty measure
        max_uncertainties = torch.max(uncertainties, dim=1)[0]
        
        # Compute AUROC for uncertainty predicting correctness
        try:
            auroc = roc_auc_score(
                correct_predictions.cpu().numpy().astype(int),
                max_uncertainties.cpu().numpy()
            )
            return auroc
        except ValueError:
            return 0.5  # Random performance if all predictions are same class
    
    def compute_reliability_diagram(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_bins: int = 10,
        save_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute reliability diagram for calibration assessment."""
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
        
        # Compute reliability
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
        
        # Plot reliability diagram
        plt.figure(figsize=(8, 6))
        plt.bar(bin_confidences, bin_accuracies, width=0.1, alpha=0.7, 
                color='skyblue', edgecolor='black')
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Reliability Diagram')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return bin_confidences, bin_accuracies, bin_counts
    
    def compute_uncertainty_ranking(
        self,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        predictions: torch.Tensor
    ) -> Dict[str, float]:
        """Compute uncertainty ranking metrics."""
        # Convert to probabilities
        if predictions.dim() == 2 and predictions.size(1) > 1:
            probs = F.softmax(predictions, dim=1)
        else:
            probs = predictions
        
        predicted_classes = torch.argmax(probs, dim=1)
        correct_predictions = (predicted_classes == targets)
        
        # Use max uncertainty as ranking criterion
        max_uncertainties = torch.max(uncertainties, dim=1)[0]
        
        # Sort by uncertainty (descending)
        sorted_indices = torch.argsort(max_uncertainties, descending=True)
        sorted_correct = correct_predictions[sorted_indices]
        
        # Compute metrics
        total_samples = len(correct_predictions)
        num_incorrect = (~correct_predictions).sum().item()
        
        # Precision@K for uncertainty ranking
        precision_at_k = []
        for k in [10, 20, 50, 100]:
            if k <= total_samples:
                top_k_correct = sorted_correct[:k]
                precision_k = (~top_k_correct).sum().item() / k
                precision_at_k.append(precision_k)
            else:
                precision_at_k.append(num_incorrect / total_samples)
        
        # Area Under the Precision-Recall curve for uncertainty ranking
        try:
            from sklearn.metrics import precision_recall_curve, auc
            precision, recall, _ = precision_recall_curve(
                (~correct_predictions).cpu().numpy().astype(int),
                max_uncertainties.cpu().numpy()
            )
            auprc = auc(recall, precision)
        except:
            auprc = 0.0
        
        return {
            "precision_at_10": precision_at_k[0],
            "precision_at_20": precision_at_k[1],
            "precision_at_50": precision_at_k[2],
            "precision_at_100": precision_at_k[3],
            "auprc_uncertainty": auprc
        }


class CalibrationAnalyzer:
    """Advanced calibration analysis for Bayesian models."""
    
    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins
    
    def expected_calibration_error(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """Compute Expected Calibration Error (ECE)."""
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
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def maximum_calibration_error(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """Compute Maximum Calibration Error (MCE)."""
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
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def plot_calibration_curves(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        method_name: str = "Model",
        save_path: Optional[str] = None
    ) -> None:
        """Plot calibration curves."""
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
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
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
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Reliability diagram
        plt.subplot(2, 2, 1)
        plt.bar(bin_confidences, bin_accuracies, width=0.1, alpha=0.7, 
                color='skyblue', edgecolor='black')
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'Reliability Diagram - {method_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Confidence distribution
        plt.subplot(2, 2, 2)
        plt.hist(confidences, bins=20, alpha=0.7, color='lightgreen')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title(f'Confidence Distribution - {method_name}')
        plt.grid(True, alpha=0.3)
        
        # Accuracy vs Confidence scatter
        plt.subplot(2, 2, 3)
        plt.scatter(confidences, correct, alpha=0.5, s=1)
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy vs Confidence - {method_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Bin counts
        plt.subplot(2, 2, 4)
        plt.bar(bin_confidences, bin_counts, width=0.1, alpha=0.7, 
                color='orange', edgecolor='black')
        plt.xlabel('Confidence')
        plt.ylabel('Number of Samples')
        plt.title(f'Sample Count per Bin - {method_name}')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def compare_models_calibration(
    model_results: Dict[str, Dict[str, torch.Tensor]],
    save_path: Optional[str] = None
) -> None:
    """Compare calibration across multiple models."""
    plt.figure(figsize=(15, 10))
    
    analyzer = CalibrationAnalyzer()
    
    for i, (model_name, results) in enumerate(model_results.items()):
        predictions = results['predictions']
        targets = results['targets']
        
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
        bin_boundaries = np.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                bin_accuracies.append(correct[in_bin].mean())
                bin_confidences.append(confidences[in_bin].mean())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
        
        bin_accuracies = np.array(bin_accuracies)
        bin_confidences = np.array(bin_confidences)
        
        # Plot reliability diagram
        plt.subplot(2, 3, i + 1)
        plt.bar(bin_confidences, bin_accuracies, width=0.1, alpha=0.7, 
                edgecolor='black')
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'{model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Overall comparison
    plt.subplot(2, 3, 6)
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for i, (model_name, results) in enumerate(model_results.items()):
        predictions = results['predictions']
        targets = results['targets']
        
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
        bin_boundaries = np.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                bin_accuracies.append(correct[in_bin].mean())
                bin_confidences.append(confidences[in_bin].mean())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
        
        bin_accuracies = np.array(bin_accuracies)
        bin_confidences = np.array(bin_confidences)
        
        plt.plot(bin_confidences, bin_accuracies, 'o-', 
                color=colors[i % len(colors)], label=model_name, alpha=0.7)
    
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Calibration Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Uncertainty evaluation metrics module loaded successfully!")
    print("This module provides comprehensive evaluation tools for Bayesian Neural Networks.")
