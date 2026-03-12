# Bayesian Neural Networks for Explainable AI

A comprehensive research and educational toolkit for Bayesian Neural Networks with uncertainty quantification, calibration analysis, and interpretability methods.

## ⚠️ Important Disclaimer

**This is a research and educational tool only.** The uncertainty estimates and explanations provided by this toolkit may be unstable, misleading, or incorrect. They should not be used for making regulated decisions without human review and validation.

Always consult domain experts and validate results before applying to real-world problems.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Bayesian-Neural-Networks-for-Explainable-AI.git
cd Bayesian-Neural-Networks-for-Explainable-AI

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Basic Usage

```python
from src.models.bayesian_nn import BayesianNeuralNetwork, train_bayesian_model
from src.data.pipeline import DataPipeline
from src.eval.metrics import UncertaintyMetrics

# Load dataset
pipeline = DataPipeline()
train_dataset, val_dataset, test_dataset, metadata = pipeline.load_iris_dataset()

# Create model
model = BayesianNeuralNetwork(
    input_dim=4,
    hidden_dims=[64, 32],
    output_dim=3,
    method="mc_dropout"
)

# Train model
history = train_bayesian_model(model, train_loader, val_loader)

# Evaluate uncertainty
eval_results = evaluate_uncertainty(model, test_loader)
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py
```

## 📁 Project Structure

```
bayesian-neural-networks-xai/
├── src/                          # Source code
│   ├── models/                   # Bayesian neural network models
│   │   └── bayesian_nn.py        # Core BNN implementations
│   ├── data/                     # Data pipeline and preprocessing
│   │   └── pipeline.py           # Dataset loading and preprocessing
│   ├── eval/                     # Evaluation metrics and analysis
│   │   └── metrics.py            # Uncertainty and calibration metrics
│   ├── viz/                      # Visualization utilities
│   │   └── visualizer.py         # Plotting and interactive visualizations
│   ├── methods/                  # Additional Bayesian methods
│   ├── explainers/               # Model interpretability methods
│   └── utils/                    # Utility functions
├── data/                         # Dataset storage
│   ├── raw/                      # Raw datasets
│   └── processed/                # Preprocessed datasets
├── configs/                      # Configuration files
├── scripts/                      # Training and evaluation scripts
├── notebooks/                    # Jupyter notebooks for exploration
├── tests/                        # Unit tests
├── assets/                       # Generated plots and results
├── demo/                         # Interactive demo application
│   └── app.py                    # Streamlit demo
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
└── README.md                     # This file
```

## Bayesian Methods

### Monte Carlo Dropout (MC-Dropout)
- **Description**: Uses dropout during inference to estimate uncertainty
- **Advantages**: Simple, computationally efficient
- **Use Case**: Quick uncertainty estimation for existing models

### Variational Inference
- **Description**: Learns distribution over model parameters
- **Advantages**: Theoretically principled, learnable uncertainty
- **Use Case**: When you need to learn uncertainty from data

### Deep Ensembles
- **Description**: Trains multiple models with different initializations
- **Advantages**: Robust uncertainty estimates, good calibration
- **Use Case**: When computational resources allow multiple training runs

## Datasets

### Built-in Datasets

1. **Iris Dataset**
   - 4 features, 3 classes
   - Classic classification benchmark
   - Well-balanced classes

2. **Wine Dataset**
   - 13 features, 3 classes
   - Wine classification task
   - Moderate complexity

3. **Breast Cancer Dataset**
   - 30 features, 2 classes
   - Binary classification
   - Medical domain

### Synthetic Datasets
- Configurable number of samples, features, and classes
- Control over noise levels and class separation
- Useful for controlled experiments

## 🔧 Configuration

### Model Configuration

```yaml
model:
  method: "mc_dropout"  # or "variational", "deep_ensemble"
  hidden_dims: [64, 32]
  dropout_rate: 0.5
  prior_std: 1.0
  num_samples: 100

training:
  num_epochs: 100
  learning_rate: 0.001
  batch_size: 32
  kl_weight: 0.01

evaluation:
  num_samples: 100
  num_bins: 10
```

### Data Configuration

```yaml
data:
  scaler_type: "standard"  # or "minmax"
  test_size: 0.3
  val_size: 0.2
  random_state: 42
```

## Evaluation Metrics

### Uncertainty Metrics
- **Mean Uncertainty**: Average uncertainty across predictions
- **Max Uncertainty**: Maximum uncertainty observed
- **Uncertainty Calibration**: How well uncertainty predicts correctness

### Calibration Metrics
- **Expected Calibration Error (ECE)**: Average calibration error
- **Maximum Calibration Error (MCE)**: Maximum calibration error
- **Brier Score**: Probability calibration measure

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class performance
- **AUROC**: Area under ROC curve

## Visualization

### Uncertainty Analysis
- Uncertainty distribution histograms
- Uncertainty vs correctness analysis
- Confidence vs uncertainty scatter plots
- Uncertainty heatmaps

### Calibration Analysis
- Reliability diagrams
- Confidence distributions
- Accuracy vs confidence plots
- Sample count per confidence bin

### Model Comparison
- Side-by-side metric comparisons
- Uncertainty vs accuracy scatter plots
- Model ranking visualizations

## Experiments

### Training Scripts

```bash
# Train MC-Dropout model
python scripts/train.py --method mc_dropout --dataset iris

# Train Variational model
python scripts/train.py --method variational --dataset wine

# Train Deep Ensemble
python scripts/train.py --method deep_ensemble --dataset synthetic
```

### Evaluation Scripts

```bash
# Evaluate uncertainty
python scripts/evaluate.py --model_path models/mc_dropout_model.pt

# Compare methods
python scripts/compare.py --methods mc_dropout variational deep_ensemble

# Generate visualizations
python scripts/visualize.py --results_path results/experiment_1/
```

## Examples

### Basic Training Example

```python
import torch
from src.models.bayesian_nn import BayesianNeuralNetwork, train_bayesian_model
from src.data.pipeline import DataPipeline
from src.eval.metrics import UncertaintyMetrics

# Set random seed
torch.manual_seed(42)

# Load data
pipeline = DataPipeline()
train_dataset, val_dataset, test_dataset, metadata = pipeline.load_iris_dataset()

# Create data loaders
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create model
model = BayesianNeuralNetwork(
    input_dim=4,
    hidden_dims=[64, 32],
    output_dim=3,
    method="mc_dropout",
    dropout_rate=0.5
)

# Train model
history = train_bayesian_model(
    model, train_loader, val_loader,
    num_epochs=50,
    learning_rate=0.001
)

# Evaluate uncertainty
eval_results = evaluate_uncertainty(model, test_loader, num_samples=100)

# Compute metrics
metrics = UncertaintyMetrics(num_classes=3)
all_metrics = metrics.compute_all_metrics(
    eval_results['predictions'],
    eval_results['uncertainties'],
    eval_results['targets']
)

print(f"Accuracy: {all_metrics['accuracy']:.4f}")
print(f"Calibration Error: {all_metrics['calibration_error']:.4f}")
print(f"Mean Uncertainty: {all_metrics['mean_uncertainty']:.4f}")
```

### Visualization Example

```python
from src.viz.visualizer import UncertaintyVisualizer

# Create visualizer
visualizer = UncertaintyVisualizer()

# Plot uncertainty analysis
visualizer.plot_uncertainty_distribution(
    eval_results['uncertainties'],
    eval_results['predictions'],
    eval_results['targets'],
    method_name="MC-Dropout",
    save_path="assets/uncertainty_analysis.png"
)

# Plot calibration analysis
visualizer.plot_calibration_analysis(
    eval_results['predictions'],
    eval_results['targets'],
    method_name="MC-Dropout",
    save_path="assets/calibration_analysis.png"
)
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_bayesian_nn.py

# Run with verbose output
pytest -v
```

## Results and Benchmarks

### Expected Performance (Iris Dataset)

| Method | Accuracy | Calibration Error | Mean Uncertainty |
|--------|----------|-------------------|------------------|
| MC-Dropout | 0.95+ | 0.05-0.10 | 0.10-0.20 |
| Variational | 0.94+ | 0.03-0.08 | 0.08-0.15 |
| Deep Ensemble | 0.96+ | 0.02-0.06 | 0.12-0.25 |

*Note: Results may vary based on hyperparameters and random seeds*

## Research Applications

### Uncertainty Quantification
- Medical diagnosis with confidence intervals
- Autonomous systems with safety guarantees
- Financial risk assessment

### Model Calibration
- Improving prediction reliability
- Reducing overconfident predictions
- Enhancing model trustworthiness

### Interpretability
- Understanding model decision boundaries
- Identifying uncertain regions
- Explaining prediction confidence

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run linting
black src/
ruff check src/

# Run tests
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

1. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. ICML.
2. Blundell, C., et al. (2015). Weight uncertainty in neural networks. ICML.
3. Lakshminarayanan, B., et al. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS.
4. Guo, C., et al. (2017). On calibration of modern neural networks. ICML.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Streamlit team for the interactive web framework
- The Bayesian deep learning research community
- Contributors and users of this project

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation
- Review the examples and notebooks

---

**Remember**: This toolkit is for research and educational purposes only. Always validate results and consult domain experts before applying to real-world problems.
# Bayesian-Neural-Networks-for-Explainable-AI
