"""
Data pipeline and preprocessing utilities for Bayesian Neural Networks.

This module provides data loading, preprocessing, and augmentation utilities
for various datasets used in uncertainty quantification experiments.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, 
    make_classification, make_regression
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


class DatasetMetadata:
    """Metadata container for dataset information."""
    
    def __init__(
        self,
        name: str,
        description: str,
        features: List[Dict[str, Any]],
        target_info: Dict[str, Any],
        sensitive_attributes: Optional[List[str]] = None,
        monotonic_features: Optional[List[str]] = None
    ):
        self.name = name
        self.description = description
        self.features = features
        self.target_info = target_info
        self.sensitive_attributes = sensitive_attributes or []
        self.monotonic_features = monotonic_features or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "features": self.features,
            "target_info": self.target_info,
            "sensitive_attributes": self.sensitive_attributes,
            "monotonic_features": self.monotonic_features
        }
    
    def save(self, filepath: str) -> None:
        """Save metadata to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'DatasetMetadata':
        """Load metadata from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class BayesianDataset(Dataset):
    """Custom dataset class for Bayesian Neural Networks."""
    
    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None,
        metadata: Optional[DatasetMetadata] = None
    ):
        self.features = features
        self.targets = targets
        self.uncertainties = uncertainties
        self.metadata = metadata
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.uncertainties is not None:
            return self.features[idx], self.targets[idx], self.uncertainties[idx]
        return self.features[idx], self.targets[idx], None
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced datasets."""
        unique_classes, counts = torch.unique(self.targets, return_counts=True)
        total_samples = len(self.targets)
        weights = total_samples / (len(unique_classes) * counts.float())
        return weights


class DataPipeline:
    """Comprehensive data pipeline for Bayesian Neural Networks."""
    
    def __init__(
        self,
        scaler_type: str = "standard",
        test_size: float = 0.3,
        val_size: float = 0.2,
        random_state: int = 42
    ):
        self.scaler_type = scaler_type
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')
    
    def load_iris_dataset(self) -> Tuple[BayesianDataset, BayesianDataset, BayesianDataset, DatasetMetadata]:
        """Load and preprocess the Iris dataset."""
        # Load data
        data = load_iris()
        X, y = data.data, data.target
        
        # Create metadata
        features = [
            {"name": "sepal_length", "type": "continuous", "range": [4.3, 7.9], "description": "Sepal length in cm"},
            {"name": "sepal_width", "type": "continuous", "range": [2.0, 4.4], "description": "Sepal width in cm"},
            {"name": "petal_length", "type": "continuous", "range": [1.0, 6.9], "description": "Petal length in cm"},
            {"name": "petal_width", "type": "continuous", "range": [0.1, 2.5], "description": "Petal width in cm"}
        ]
        
        target_info = {
            "name": "species",
            "type": "categorical",
            "classes": ["setosa", "versicolor", "virginica"],
            "num_classes": 3
        }
        
        metadata = DatasetMetadata(
            name="Iris",
            description="Classic iris flower classification dataset",
            features=features,
            target_info=target_info
        )
        
        # Split data
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=self.val_size, 
            random_state=self.random_state, stratify=y_train_val
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        
        y_train_tensor = torch.LongTensor(y_train)
        y_val_tensor = torch.LongTensor(y_val)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create datasets
        train_dataset = BayesianDataset(X_train_tensor, y_train_tensor, metadata=metadata)
        val_dataset = BayesianDataset(X_val_tensor, y_val_tensor, metadata=metadata)
        test_dataset = BayesianDataset(X_test_tensor, y_test_tensor, metadata=metadata)
        
        return train_dataset, val_dataset, test_dataset, metadata
    
    def load_wine_dataset(self) -> Tuple[BayesianDataset, BayesianDataset, BayesianDataset, DatasetMetadata]:
        """Load and preprocess the Wine dataset."""
        # Load data
        data = load_wine()
        X, y = data.data, data.target
        
        # Create metadata
        features = [
            {"name": f"feature_{i}", "type": "continuous", "description": f"Wine feature {i}"}
            for i in range(X.shape[1])
        ]
        
        target_info = {
            "name": "wine_class",
            "type": "categorical",
            "classes": ["class_0", "class_1", "class_2"],
            "num_classes": 3
        }
        
        metadata = DatasetMetadata(
            name="Wine",
            description="Wine classification dataset",
            features=features,
            target_info=target_info
        )
        
        # Split data
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=self.val_size, 
            random_state=self.random_state, stratify=y_train_val
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        
        y_train_tensor = torch.LongTensor(y_train)
        y_val_tensor = torch.LongTensor(y_val)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create datasets
        train_dataset = BayesianDataset(X_train_tensor, y_train_tensor, metadata=metadata)
        val_dataset = BayesianDataset(X_val_tensor, y_val_tensor, metadata=metadata)
        test_dataset = BayesianDataset(X_test_tensor, y_test_tensor, metadata=metadata)
        
        return train_dataset, val_dataset, test_dataset, metadata
    
    def load_breast_cancer_dataset(self) -> Tuple[BayesianDataset, BayesianDataset, BayesianDataset, DatasetMetadata]:
        """Load and preprocess the Breast Cancer dataset."""
        # Load data
        data = load_breast_cancer()
        X, y = data.data, data.target
        
        # Create metadata
        features = [
            {"name": f"feature_{i}", "type": "continuous", "description": f"Cancer feature {i}"}
            for i in range(X.shape[1])
        ]
        
        target_info = {
            "name": "diagnosis",
            "type": "binary",
            "classes": ["malignant", "benign"],
            "num_classes": 2
        }
        
        metadata = DatasetMetadata(
            name="Breast Cancer",
            description="Breast cancer classification dataset",
            features=features,
            target_info=target_info
        )
        
        # Split data
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=self.val_size, 
            random_state=self.random_state, stratify=y_train_val
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        
        y_train_tensor = torch.LongTensor(y_train)
        y_val_tensor = torch.LongTensor(y_val)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create datasets
        train_dataset = BayesianDataset(X_train_tensor, y_train_tensor, metadata=metadata)
        val_dataset = BayesianDataset(X_val_tensor, y_val_tensor, metadata=metadata)
        test_dataset = BayesianDataset(X_test_tensor, y_test_tensor, metadata=metadata)
        
        return train_dataset, val_dataset, test_dataset, metadata
    
    def create_synthetic_dataset(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        n_classes: int = 3,
        n_informative: int = 5,
        n_redundant: int = 0,
        noise: float = 0.1,
        class_sep: float = 1.0
    ) -> Tuple[BayesianDataset, BayesianDataset, BayesianDataset, DatasetMetadata]:
        """Create a synthetic classification dataset."""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_clusters_per_class=1,
            random_state=self.random_state
        )
        
        # Create metadata
        features = [
            {"name": f"feature_{i}", "type": "continuous", "description": f"Synthetic feature {i}"}
            for i in range(n_features)
        ]
        
        target_info = {
            "name": "class",
            "type": "categorical",
            "classes": [f"class_{i}" for i in range(n_classes)],
            "num_classes": n_classes
        }
        
        metadata = DatasetMetadata(
            name="Synthetic",
            description=f"Synthetic classification dataset with {n_samples} samples and {n_features} features",
            features=features,
            target_info=target_info
        )
        
        # Split data
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=self.val_size, 
            random_state=self.random_state, stratify=y_train_val
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        
        y_train_tensor = torch.LongTensor(y_train)
        y_val_tensor = torch.LongTensor(y_val)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create datasets
        train_dataset = BayesianDataset(X_train_tensor, y_train_tensor, metadata=metadata)
        val_dataset = BayesianDataset(X_val_tensor, y_val_tensor, metadata=metadata)
        test_dataset = BayesianDataset(X_test_tensor, y_test_tensor, metadata=metadata)
        
        return train_dataset, val_dataset, test_dataset, metadata
    
    def create_data_loaders(
        self,
        train_dataset: BayesianDataset,
        val_dataset: BayesianDataset,
        test_dataset: BayesianDataset,
        batch_size: int = 32,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for training."""
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        return train_loader, val_loader, test_loader
    
    def analyze_dataset(self, dataset: BayesianDataset, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Analyze dataset characteristics."""
        features = dataset.features.numpy()
        targets = dataset.targets.numpy()
        
        analysis = {
            "num_samples": len(dataset),
            "num_features": features.shape[1],
            "num_classes": len(np.unique(targets)),
            "class_distribution": np.bincount(targets).tolist(),
            "feature_stats": {
                "mean": np.mean(features, axis=0).tolist(),
                "std": np.std(features, axis=0).tolist(),
                "min": np.min(features, axis=0).tolist(),
                "max": np.max(features, axis=0).tolist()
            }
        }
        
        # Plot analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Class distribution
        unique_classes, counts = np.unique(targets, return_counts=True)
        axes[0, 0].bar(unique_classes, counts, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Class Distribution')
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Count')
        
        # Feature correlation heatmap
        if features.shape[1] <= 20:  # Only plot if not too many features
            corr_matrix = np.corrcoef(features.T)
            im = axes[0, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[0, 1].set_title('Feature Correlation Matrix')
            plt.colorbar(im, ax=axes[0, 1])
        
        # Feature distributions
        axes[1, 0].hist(features.flatten(), bins=50, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Feature Value Distribution')
        axes[1, 0].set_xlabel('Feature Value')
        axes[1, 0].set_ylabel('Frequency')
        
        # Feature box plots (first 10 features)
        n_features_to_plot = min(10, features.shape[1])
        axes[1, 1].boxplot([features[:, i] for i in range(n_features_to_plot)])
        axes[1, 1].set_title(f'Feature Distributions (First {n_features_to_plot})')
        axes[1, 1].set_xlabel('Feature Index')
        axes[1, 1].set_ylabel('Feature Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return analysis


def save_dataset_metadata(
    train_dataset: BayesianDataset,
    val_dataset: BayesianDataset,
    test_dataset: BayesianDataset,
    metadata: DatasetMetadata,
    save_dir: str
) -> None:
    """Save dataset and metadata to disk."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata.save(save_dir / "metadata.json")
    
    # Save datasets
    torch.save(train_dataset, save_dir / "train_dataset.pt")
    torch.save(val_dataset, save_dir / "val_dataset.pt")
    torch.save(test_dataset, save_dir / "test_dataset.pt")
    
    print(f"Dataset saved to {save_dir}")


def load_dataset_metadata(save_dir: str) -> Tuple[BayesianDataset, BayesianDataset, BayesianDataset, DatasetMetadata]:
    """Load dataset and metadata from disk."""
    save_dir = Path(save_dir)
    
    # Load metadata
    metadata = DatasetMetadata.load(save_dir / "metadata.json")
    
    # Load datasets
    train_dataset = torch.load(save_dir / "train_dataset.pt")
    val_dataset = torch.load(save_dir / "val_dataset.pt")
    test_dataset = torch.load(save_dir / "test_dataset.pt")
    
    return train_dataset, val_dataset, test_dataset, metadata


if __name__ == "__main__":
    # Example usage
    pipeline = DataPipeline()
    
    # Load Iris dataset
    print("Loading Iris dataset...")
    train_dataset, val_dataset, test_dataset, metadata = pipeline.load_iris_dataset()
    
    print(f"Dataset: {metadata.name}")
    print(f"Description: {metadata.description}")
    print(f"Features: {len(metadata.features)}")
    print(f"Classes: {metadata.target_info['num_classes']}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = pipeline.create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=32
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Analyze dataset
    analysis = pipeline.analyze_dataset(train_dataset)
    print(f"Analysis: {analysis}")
    
    print("Data pipeline example completed successfully!")
