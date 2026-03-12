"""
Utility functions for Bayesian Neural Networks.

This module provides common utility functions used across the project.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
import numpy as np
import random


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results(results: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """Save results dictionary to JSON file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert tensors to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            serializable_results[key] = value.detach().cpu().tolist()
        elif isinstance(value, dict):
            serializable_results[key] = save_results(value, None)  # Recursive
        else:
            serializable_results[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def load_results(load_path: Union[str, Path]) -> Dict[str, Any]:
    """Load results dictionary from JSON file."""
    load_path = Path(load_path)
    
    if not load_path.exists():
        raise FileNotFoundError(f"Results file not found: {load_path}")
    
    with open(load_path, 'r') as f:
        results = json.load(f)
    
    return results


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_number(number: float, precision: int = 4) -> str:
    """Format number with appropriate precision."""
    if abs(number) < 1e-6:
        return f"{number:.2e}"
    elif abs(number) < 1e-3:
        return f"{number:.6f}"
    elif abs(number) < 1:
        return f"{number:.4f}"
    elif abs(number) < 100:
        return f"{number:.3f}"
    else:
        return f"{number:.2f}"


def get_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """Get model size information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of parameters in model."""
    return sum(p.numel() for p in model.parameters())


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage."""
    import psutil
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent()
    }


def print_system_info():
    """Print system information."""
    print("=" * 50)
    print("System Information")
    print("=" * 50)
    
    # Python version
    import sys
    print(f"Python Version: {sys.version}")
    
    # PyTorch version
    print(f"PyTorch Version: {torch.__version__}")
    
    # CUDA availability
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # MPS availability (Apple Silicon)
    if hasattr(torch.backends, 'mps'):
        print(f"MPS Available: {torch.backends.mps.is_available()}")
    
    # Memory usage
    memory_info = get_memory_usage()
    print(f"Memory Usage: {memory_info['rss_mb']:.2f} MB ({memory_info['percent']:.1f}%)")
    
    print("=" * 50)


def create_experiment_dir(base_dir: Union[str, Path], experiment_name: str) -> Path:
    """Create experiment directory with timestamp."""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    return exp_dir


def log_experiment_info(exp_dir: Path, config: Dict[str, Any], results: Dict[str, Any]):
    """Log experiment information to directory."""
    # Save config
    save_config(config, exp_dir / "config.yaml")
    
    # Save results
    save_results(results, exp_dir / "results.json")
    
    # Save system info
    with open(exp_dir / "system_info.txt", 'w') as f:
        import sys
        f.write(f"Python Version: {sys.version}\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA Version: {torch.version.cuda}\n")
            f.write(f"GPU Count: {torch.cuda.device_count()}\n")


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary."""
    required_keys = ['model', 'training', 'data']
    
    for key in required_keys:
        if key not in config:
            print(f"Missing required config key: {key}")
            return False
    
    # Validate model config
    model_config = config['model']
    if 'method' not in model_config:
        print("Missing model.method in config")
        return False
    
    valid_methods = ['mc_dropout', 'variational', 'deep_ensemble']
    if model_config['method'] not in valid_methods:
        print(f"Invalid model.method: {model_config['method']}. Must be one of {valid_methods}")
        return False
    
    # Validate training config
    training_config = config['training']
    required_training_keys = ['num_epochs', 'learning_rate', 'batch_size']
    for key in required_training_keys:
        if key not in training_config:
            print(f"Missing training.{key} in config")
            return False
    
    # Validate data config
    data_config = config['data']
    if 'dataset' not in data_config:
        print("Missing data.dataset in config")
        return False
    
    return True


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    import logging
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test seed setting
    set_seed(42)
    print("Random seed set to 42")
    
    # Test device detection
    device = get_device()
    print(f"Detected device: {device}")
    
    # Test system info
    print_system_info()
    
    # Test config validation
    test_config = {
        'model': {'method': 'mc_dropout'},
        'training': {'num_epochs': 100, 'learning_rate': 0.001, 'batch_size': 32},
        'data': {'dataset': 'iris'}
    }
    
    is_valid = validate_config(test_config)
    print(f"Config validation: {'PASSED' if is_valid else 'FAILED'}")
    
    print("Utility functions test completed!")
