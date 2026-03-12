"""
Quick start script for Bayesian Neural Networks demo.

This script demonstrates the basic usage of the Bayesian Neural Networks toolkit.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import numpy as np
from src.models.bayesian_nn import (
    BayesianNeuralNetwork, DeepEnsemble, 
    train_bayesian_model, evaluate_uncertainty,
    set_seed, get_device
)
from src.data.pipeline import DataPipeline
from src.eval.metrics import UncertaintyMetrics
from src.viz.visualizer import UncertaintyVisualizer

def quick_demo():
    """Run a quick demonstration of the Bayesian Neural Networks."""
    print("=" * 60)
    print("Bayesian Neural Networks - Quick Demo")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading Iris dataset...")
    pipeline = DataPipeline()
    train_dataset, val_dataset, test_dataset, metadata = pipeline.load_iris_dataset()
    
    print(f"Dataset: {metadata.name}")
    print(f"Features: {len(metadata.features)}")
    print(f"Classes: {metadata.target_info['num_classes']}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Test different methods
    methods = ["mc_dropout", "variational"]
    results = {}
    
    for method in methods:
        print(f"\n{'='*20} {method.upper()} {'='*20}")
        
        # Create model
        model = BayesianNeuralNetwork(
            input_dim=4,
            hidden_dims=[64, 32],
            output_dim=3,
            method=method,
            dropout_rate=0.5,
            num_samples=50
        )
        
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Train model
        print("Training model...")
        history = train_bayesian_model(
            model, train_loader, val_loader,
            num_epochs=20, learning_rate=0.001, device=device
        )
        
        # Evaluate uncertainty
        print("Evaluating uncertainty...")
        eval_results = evaluate_uncertainty(model, test_loader, device, num_samples=50)
        
        # Store results
        results[method] = eval_results
        
        # Display results
        print(f"\n{method} Results:")
        print(f"  Accuracy: {eval_results['accuracy']:.4f}")
        print(f"  Mean Uncertainty: {eval_results['mean_uncertainty']:.4f}")
        print(f"  Max Uncertainty: {eval_results['max_uncertainty']:.4f}")
        
        # Compute detailed metrics
        metrics = UncertaintyMetrics(num_classes=3)
        all_metrics = metrics.compute_all_metrics(
            eval_results['predictions'],
            eval_results['uncertainties'],
            eval_results['targets']
        )
        
        print(f"  Calibration Error: {all_metrics['calibration_error']:.4f}")
        print(f"  Brier Score: {all_metrics['brier_score']:.4f}")
        print(f"  Uncertainty Calibration: {all_metrics['uncertainty_calibration']:.4f}")
    
    # Compare methods
    print(f"\n{'='*20} COMPARISON {'='*20}")
    print("Method Comparison:")
    print(f"{'Method':<15} {'Accuracy':<10} {'Calibration':<12} {'Mean Unc':<10}")
    print("-" * 50)
    
    for method, eval_results in results.items():
        metrics = UncertaintyMetrics(num_classes=3)
        all_metrics = metrics.compute_all_metrics(
            eval_results['predictions'],
            eval_results['uncertainties'],
            eval_results['targets']
        )
        
        print(f"{method:<15} {eval_results['accuracy']:<10.4f} {all_metrics['calibration_error']:<12.4f} {eval_results['mean_uncertainty']:<10.4f}")
    
    print(f"\n{'='*60}")
    print("Demo completed successfully!")
    print("For interactive exploration, run: streamlit run demo/app.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        quick_demo()
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
