"""
Test script for Bayesian Neural Networks implementation.

This script tests the core functionality and ensures everything works correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.models.bayesian_nn import (
    BayesianNeuralNetwork, DeepEnsemble, 
    train_bayesian_model, evaluate_uncertainty,
    load_iris_dataset, set_seed, get_device
)
from src.data.pipeline import DataPipeline
from src.eval.metrics import UncertaintyMetrics, CalibrationAnalyzer
from src.viz.visualizer import UncertaintyVisualizer

def test_basic_functionality():
    """Test basic functionality of the Bayesian Neural Network."""
    print("Testing basic functionality...")
    
    # Set random seed
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading Iris dataset...")
    X_train, X_test, y_train, y_test, scaler = load_iris_dataset()
    
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {len(torch.unique(y_train))}")
    
    # Create data loaders
    from torch.utils.data import DataLoader, TensorDataset
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Test MC-Dropout model
    print("\nTesting MC-Dropout model...")
    model_mc = BayesianNeuralNetwork(
        input_dim=4,
        hidden_dims=[32, 16],
        output_dim=3,
        method="mc_dropout",
        dropout_rate=0.5
    )
    
    # Quick training
    print("Training MC-Dropout model...")
    history_mc = train_bayesian_model(
        model_mc, train_loader, test_loader,
        num_epochs=10, learning_rate=0.001, device=device
    )
    
    # Evaluate uncertainty
    print("Evaluating uncertainty...")
    eval_results_mc = evaluate_uncertainty(model_mc, test_loader, device, num_samples=50)
    
    print(f"MC-Dropout Results:")
    print(f"  Accuracy: {eval_results_mc['accuracy']:.4f}")
    print(f"  Mean Uncertainty: {eval_results_mc['mean_uncertainty']:.4f}")
    print(f"  Max Uncertainty: {eval_results_mc['max_uncertainty']:.4f}")
    
    # Test Variational model
    print("\nTesting Variational model...")
    model_var = BayesianNeuralNetwork(
        input_dim=4,
        hidden_dims=[32, 16],
        output_dim=3,
        method="variational",
        prior_std=1.0
    )
    
    # Quick training
    print("Training Variational model...")
    history_var = train_bayesian_model(
        model_var, train_loader, test_loader,
        num_epochs=10, learning_rate=0.001, device=device
    )
    
    # Evaluate uncertainty
    print("Evaluating uncertainty...")
    eval_results_var = evaluate_uncertainty(model_var, test_loader, device, num_samples=50)
    
    print(f"Variational Results:")
    print(f"  Accuracy: {eval_results_var['accuracy']:.4f}")
    print(f"  Mean Uncertainty: {eval_results_var['mean_uncertainty']:.4f}")
    print(f"  Max Uncertainty: {eval_results_var['max_uncertainty']:.4f}")
    
    # Test Deep Ensemble
    print("\nTesting Deep Ensemble...")
    model_ensemble = DeepEnsemble(
        input_dim=4,
        hidden_dims=[32, 16],
        output_dim=3,
        num_models=3,
        dropout_rate=0.5
    )
    
    # Quick training
    print("Training Deep Ensemble...")
    history_ensemble = train_bayesian_model(
        model_ensemble, train_loader, test_loader,
        num_epochs=10, learning_rate=0.001, device=device
    )
    
    # Evaluate uncertainty
    print("Evaluating uncertainty...")
    eval_results_ensemble = evaluate_uncertainty(model_ensemble, test_loader, device, num_samples=50)
    
    print(f"Deep Ensemble Results:")
    print(f"  Accuracy: {eval_results_ensemble['accuracy']:.4f}")
    print(f"  Mean Uncertainty: {eval_results_ensemble['mean_uncertainty']:.4f}")
    print(f"  Max Uncertainty: {eval_results_ensemble['max_uncertainty']:.4f}")
    
    return {
        'mc_dropout': eval_results_mc,
        'variational': eval_results_var,
        'deep_ensemble': eval_results_ensemble
    }

def test_evaluation_metrics():
    """Test evaluation metrics."""
    print("\nTesting evaluation metrics...")
    
    # Create dummy data
    predictions = torch.randn(100, 3)
    uncertainties = torch.rand(100, 3)
    targets = torch.randint(0, 3, (100,))
    
    # Test UncertaintyMetrics
    metrics = UncertaintyMetrics(num_classes=3)
    all_metrics = metrics.compute_all_metrics(predictions, uncertainties, targets)
    
    print("Uncertainty Metrics:")
    for key, value in all_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test CalibrationAnalyzer
    analyzer = CalibrationAnalyzer()
    ece = analyzer.expected_calibration_error(predictions, targets)
    mce = analyzer.maximum_calibration_error(predictions, targets)
    
    print(f"Calibration Metrics:")
    print(f"  ECE: {ece:.4f}")
    print(f"  MCE: {mce:.4f}")

def test_data_pipeline():
    """Test data pipeline functionality."""
    print("\nTesting data pipeline...")
    
    pipeline = DataPipeline()
    
    # Test Iris dataset
    train_dataset, val_dataset, test_dataset, metadata = pipeline.load_iris_dataset()
    
    print(f"Iris Dataset:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Features: {len(metadata.features)}")
    print(f"  Classes: {metadata.target_info['num_classes']}")
    
    # Test synthetic dataset
    train_syn, val_syn, test_syn, metadata_syn = pipeline.create_synthetic_dataset(
        n_samples=500, n_features=8, n_classes=3
    )
    
    print(f"Synthetic Dataset:")
    print(f"  Train: {len(train_syn)} samples")
    print(f"  Val: {len(val_syn)} samples")
    print(f"  Test: {len(test_syn)} samples")
    print(f"  Features: {len(metadata_syn.features)}")
    print(f"  Classes: {metadata_syn.target_info['num_classes']}")

def test_visualization():
    """Test visualization functionality."""
    print("\nTesting visualization...")
    
    # Create dummy data
    predictions = torch.randn(50, 3)
    uncertainties = torch.rand(50, 3)
    targets = torch.randint(0, 3, (50,))
    
    # Test visualizer
    visualizer = UncertaintyVisualizer()
    
    print("Creating uncertainty distribution plot...")
    visualizer.plot_uncertainty_distribution(
        uncertainties, predictions, targets, method_name="Test Model"
    )
    
    print("Creating calibration analysis plot...")
    visualizer.plot_calibration_analysis(
        predictions, targets, method_name="Test Model"
    )

def main():
    """Run all tests."""
    print("=" * 60)
    print("Bayesian Neural Networks - Test Suite")
    print("=" * 60)
    
    try:
        # Test basic functionality
        results = test_basic_functionality()
        
        # Test evaluation metrics
        test_evaluation_metrics()
        
        # Test data pipeline
        test_data_pipeline()
        
        # Test visualization (commented out to avoid display issues in CI)
        # test_visualization()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
        # Summary
        print("\nSummary of Results:")
        for method, results in results.items():
            print(f"{method}: Accuracy = {results['accuracy']:.4f}, "
                  f"Mean Uncertainty = {results['mean_uncertainty']:.4f}")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
