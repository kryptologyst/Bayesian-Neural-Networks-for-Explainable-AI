"""
Simple test script to verify basic functionality without external dependencies.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test basic imports
        import torch
        import numpy as np
        print("✓ PyTorch and NumPy imported successfully")
        
        # Test our modules
        from src.models.bayesian_nn import BayesianNeuralNetwork
        print("✓ BayesianNeuralNetwork imported successfully")
        
        from src.data.pipeline import DataPipeline
        print("✓ DataPipeline imported successfully")
        
        from src.eval.metrics import UncertaintyMetrics
        print("✓ UncertaintyMetrics imported successfully")
        
        from src.viz.visualizer import UncertaintyVisualizer
        print("✓ UncertaintyVisualizer imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without training."""
    print("\nTesting basic functionality...")
    
    try:
        import torch
        import numpy as np
        from src.models.bayesian_nn import BayesianNeuralNetwork, set_seed, get_device
        
        # Set seed
        set_seed(42)
        print("✓ Random seed set")
        
        # Get device
        device = get_device()
        print(f"✓ Device detected: {device}")
        
        # Create model
        model = BayesianNeuralNetwork(
            input_dim=4,
            hidden_dims=[32, 16],
            output_dim=3,
            method="mc_dropout"
        )
        print("✓ MC-Dropout model created")
        
        # Test forward pass
        x = torch.randn(10, 4)
        output, uncertainty = model(x)
        print(f"✓ Forward pass successful: output shape {output.shape}, uncertainty shape {uncertainty.shape}")
        
        # Create variational model
        model_var = BayesianNeuralNetwork(
            input_dim=4,
            hidden_dims=[32, 16],
            output_dim=3,
            method="variational"
        )
        print("✓ Variational model created")
        
        # Test forward pass
        output_var, uncertainty_var = model_var(x)
        print(f"✓ Variational forward pass successful: output shape {output_var.shape}, uncertainty shape {uncertainty_var.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in basic functionality test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_pipeline():
    """Test data pipeline functionality."""
    print("\nTesting data pipeline...")
    
    try:
        from src.data.pipeline import DataPipeline
        
        # Create pipeline
        pipeline = DataPipeline()
        print("✓ DataPipeline created")
        
        # Test Iris dataset loading
        train_dataset, val_dataset, test_dataset, metadata = pipeline.load_iris_dataset()
        print(f"✓ Iris dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples")
        print(f"✓ Metadata: {metadata.name} with {len(metadata.features)} features and {metadata.target_info['num_classes']} classes")
        
        # Test synthetic dataset creation
        train_syn, val_syn, test_syn, metadata_syn = pipeline.create_synthetic_dataset(
            n_samples=100, n_features=5, n_classes=2
        )
        print(f"✓ Synthetic dataset created: {len(train_syn)} train, {len(val_syn)} val, {len(test_syn)} test samples")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in data pipeline test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Bayesian Neural Networks - Basic Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Data Pipeline", test_data_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The implementation is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
