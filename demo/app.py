"""
Streamlit demo application for Bayesian Neural Networks and uncertainty quantification.

This interactive demo allows users to explore uncertainty estimation methods,
visualize calibration, and compare different Bayesian approaches.
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.bayesian_nn import (
    BayesianNeuralNetwork, DeepEnsemble, 
    train_bayesian_model, evaluate_uncertainty,
    load_iris_dataset, create_synthetic_dataset, set_seed, get_device
)
from eval.metrics import UncertaintyMetrics, CalibrationAnalyzer
from viz.visualizer import UncertaintyVisualizer, InteractiveVisualizer
from data.pipeline import DataPipeline, DatasetMetadata

# Page configuration
st.set_page_config(
    page_title="Bayesian Neural Networks Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🧠 Bayesian Neural Networks Demo</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Important Disclaimer</h4>
    <p><strong>This is a research and educational tool only.</strong> The uncertainty estimates and explanations provided by this demo may be unstable, misleading, or incorrect. They should not be used for making regulated decisions without human review and validation.</p>
    <p>Always consult domain experts and validate results before applying to real-world problems.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("Configuration")

# Dataset selection
st.sidebar.markdown("### Dataset")
dataset_choice = st.sidebar.selectbox(
    "Choose dataset:",
    ["Iris", "Synthetic Classification", "Custom Upload"]
)

# Model configuration
st.sidebar.markdown("### Model Configuration")
method = st.sidebar.selectbox(
    "Bayesian Method:",
    ["mc_dropout", "variational", "deep_ensemble"]
)

hidden_dims = st.sidebar.multiselect(
    "Hidden Layer Dimensions:",
    [32, 64, 128, 256],
    default=[64, 32]
)

dropout_rate = st.sidebar.slider(
    "Dropout Rate:",
    min_value=0.1,
    max_value=0.8,
    value=0.5,
    step=0.1
)

num_epochs = st.sidebar.slider(
    "Training Epochs:",
    min_value=10,
    max_value=200,
    value=50,
    step=10
)

learning_rate = st.sidebar.slider(
    "Learning Rate:",
    min_value=0.0001,
    max_value=0.01,
    value=0.001,
    step=0.0001,
    format="%.4f"
)

# Evaluation configuration
st.sidebar.markdown("### Evaluation")
num_samples = st.sidebar.slider(
    "MC Samples for Uncertainty:",
    min_value=10,
    max_value=200,
    value=100,
    step=10
)

batch_size = st.sidebar.selectbox(
    "Batch Size:",
    [16, 32, 64, 128],
    index=1
)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = {}

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📊 Dataset", "🤖 Model Training", "📈 Analysis"])

with tab1:
    st.markdown('<h2 class="section-header">Welcome to Bayesian Neural Networks Demo</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    This interactive demo showcases Bayesian Neural Networks for uncertainty quantification in machine learning. 
    Explore different uncertainty estimation methods and their applications in trustworthy AI.
    
    ### What You Can Do:
    
    1. **📊 Dataset Exploration**: Load and analyze different datasets
    2. **🤖 Model Training**: Train Bayesian neural networks with various methods
    3. **📈 Uncertainty Analysis**: Visualize and evaluate uncertainty estimates
    4. **🔍 Model Comparison**: Compare different Bayesian approaches
    
    ### Bayesian Methods Available:
    
    - **MC-Dropout**: Monte Carlo Dropout for uncertainty estimation
    - **Variational Inference**: Learnable uncertainty parameters
    - **Deep Ensembles**: Multiple model ensemble for uncertainty
    
    ### Key Features:
    
    - Interactive uncertainty visualization
    - Calibration analysis and reliability diagrams
    - Model comparison and ranking
    - Real-time training progress
    - Export capabilities for results
    """)
    
    # Quick start
    st.markdown('<h3 class="section-header">Quick Start</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1. Choose Dataset**
        - Select from Iris, Synthetic, or upload your own
        - Configure dataset parameters
        """)
    
    with col2:
        st.markdown("""
        **2. Configure Model**
        - Select Bayesian method
        - Set architecture and hyperparameters
        - Choose evaluation settings
        """)
    
    with col3:
        st.markdown("""
        **3. Train & Analyze**
        - Train the model
        - Explore uncertainty estimates
        - Compare with other methods
        """)

with tab2:
    st.markdown('<h2 class="section-header">Dataset Information</h2>', unsafe_allow_html=True)
    
    # Load dataset based on selection
    if dataset_choice == "Iris":
        st.info("Loading Iris dataset...")
        
        # Set random seed for reproducibility
        set_seed(42)
        
        # Load dataset
        pipeline = DataPipeline()
        train_dataset, val_dataset, test_dataset, metadata = pipeline.load_iris_dataset()
        
        # Store in session state
        st.session_state.dataset_info = {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'metadata': metadata,
            'dataset_name': 'Iris'
        }
        
        # Display dataset information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Dataset Overview")
            st.write(f"**Name**: {metadata.name}")
            st.write(f"**Description**: {metadata.description}")
            st.write(f"**Features**: {len(metadata.features)}")
            st.write(f"**Classes**: {metadata.target_info['num_classes']}")
            st.write(f"**Train Samples**: {len(train_dataset)}")
            st.write(f"**Val Samples**: {len(val_dataset)}")
            st.write(f"**Test Samples**: {len(test_dataset)}")
        
        with col2:
            st.markdown("### Feature Information")
            feature_df = pd.DataFrame(metadata.features)
            st.dataframe(feature_df, use_container_width=True)
        
        # Dataset analysis
        st.markdown("### Dataset Analysis")
        
        # Class distribution
        train_targets = train_dataset.targets.numpy()
        unique_classes, counts = np.unique(train_targets, return_counts=True)
        
        fig = px.bar(
            x=unique_classes, 
            y=counts,
            title="Class Distribution",
            labels={'x': 'Class', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature distributions
        train_features = train_dataset.features.numpy()
        feature_names = [f['name'] for f in metadata.features]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=feature_names
        )
        
        for i, name in enumerate(feature_names):
            row = i // 2 + 1
            col = i % 2 + 1
            fig.add_trace(
                go.Histogram(x=train_features[:, i], name=name, nbinsx=20),
                row=row, col=col
            )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    elif dataset_choice == "Synthetic Classification":
        st.info("Generating synthetic classification dataset...")
        
        # Synthetic dataset parameters
        col1, col2 = st.columns(2)
        
        with col1:
            n_samples = st.number_input("Number of samples:", min_value=100, max_value=5000, value=1000)
            n_features = st.number_input("Number of features:", min_value=5, max_value=50, value=10)
        
        with col2:
            n_classes = st.number_input("Number of classes:", min_value=2, max_value=10, value=3)
            n_informative = st.number_input("Informative features:", min_value=1, max_value=n_features, value=5)
        
        if st.button("Generate Dataset"):
            set_seed(42)
            
            pipeline = DataPipeline()
            train_dataset, val_dataset, test_dataset, metadata = pipeline.create_synthetic_dataset(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                n_informative=n_informative
            )
            
            # Store in session state
            st.session_state.dataset_info = {
                'train_dataset': train_dataset,
                'val_dataset': val_dataset,
                'test_dataset': test_dataset,
                'metadata': metadata,
                'dataset_name': 'Synthetic'
            }
            
            st.success(f"Generated synthetic dataset with {n_samples} samples!")
            
            # Display dataset information
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Dataset Overview")
                st.write(f"**Name**: {metadata.name}")
                st.write(f"**Description**: {metadata.description}")
                st.write(f"**Features**: {len(metadata.features)}")
                st.write(f"**Classes**: {metadata.target_info['num_classes']}")
                st.write(f"**Train Samples**: {len(train_dataset)}")
                st.write(f"**Val Samples**: {len(val_dataset)}")
                st.write(f"**Test Samples**: {len(test_dataset)}")
            
            with col2:
                st.markdown("### Class Distribution")
                train_targets = train_dataset.targets.numpy()
                unique_classes, counts = np.unique(train_targets, return_counts=True)
                
                fig = px.bar(
                    x=unique_classes, 
                    y=counts,
                    title="Class Distribution",
                    labels={'x': 'Class', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif dataset_choice == "Custom Upload":
        st.info("Custom dataset upload functionality coming soon!")
        st.markdown("""
        For now, please use the Iris dataset or generate a synthetic dataset.
        Future versions will support CSV file uploads with automatic preprocessing.
        """)

with tab3:
    st.markdown('<h2 class="section-header">Model Training</h2>', unsafe_allow_html=True)
    
    if 'dataset_info' not in st.session_state or not st.session_state.dataset_info:
        st.warning("Please load a dataset first in the Dataset tab.")
    else:
        dataset_info = st.session_state.dataset_info
        train_dataset = dataset_info['train_dataset']
        val_dataset = dataset_info['val_dataset']
        test_dataset = dataset_info['test_dataset']
        metadata = dataset_info['metadata']
        
        # Training configuration summary
        st.markdown("### Training Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            **Method**: {method}  
            **Hidden Dims**: {hidden_dims}  
            **Dropout Rate**: {dropout_rate}
            """)
        
        with col2:
            st.markdown(f"""
            **Epochs**: {num_epochs}  
            **Learning Rate**: {learning_rate}  
            **Batch Size**: {batch_size}
            """)
        
        with col3:
            st.markdown(f"""
            **MC Samples**: {num_samples}  
            **Dataset**: {dataset_info['dataset_name']}  
            **Classes**: {metadata.target_info['num_classes']}
            """)
        
        # Train button
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model..."):
                # Set random seed
                set_seed(42)
                
                # Get device
                device = get_device()
                st.info(f"Using device: {device}")
                
                # Create model
                if method == "deep_ensemble":
                    model = DeepEnsemble(
                        input_dim=train_dataset.features.shape[1],
                        hidden_dims=hidden_dims,
                        output_dim=metadata.target_info['num_classes'],
                        num_models=5,
                        dropout_rate=dropout_rate
                    )
                else:
                    model = BayesianNeuralNetwork(
                        input_dim=train_dataset.features.shape[1],
                        hidden_dims=hidden_dims,
                        output_dim=metadata.target_info['num_classes'],
                        method=method,
                        dropout_rate=dropout_rate,
                        num_samples=num_samples
                    )
                
                # Create data loaders
                from torch.utils.data import DataLoader
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                
                # Training progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Train model
                history = train_bayesian_model(
                    model, train_loader, val_loader,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    device=device
                )
                
                # Evaluate model
                status_text.text("Evaluating model...")
                eval_results = evaluate_uncertainty(model, test_loader, device, num_samples)
                
                # Store results
                st.session_state.model_results[method] = {
                    'model': model,
                    'history': history,
                    'eval_results': eval_results,
                    'config': {
                        'method': method,
                        'hidden_dims': hidden_dims,
                        'dropout_rate': dropout_rate,
                        'num_epochs': num_epochs,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'num_samples': num_samples
                    }
                }
                
                st.session_state.model_trained = True
                progress_bar.progress(100)
                status_text.text("Training completed!")
                
                st.success(f"Model trained successfully! Accuracy: {eval_results['accuracy']:.4f}")
        
        # Display training results if available
        if st.session_state.model_trained and method in st.session_state.model_results:
            results = st.session_state.model_results[method]
            history = results['history']
            eval_results = results['eval_results']
            
            st.markdown("### Training Results")
            
            # Training curves
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=history['train_losses'],
                    mode='lines',
                    name='Train Loss',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    y=history['val_losses'],
                    mode='lines',
                    name='Val Loss',
                    line=dict(color='red')
                ))
                fig.update_layout(
                    title="Training and Validation Loss",
                    xaxis_title="Epoch",
                    yaxis_title="Loss"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=history['val_accuracies'],
                    mode='lines',
                    name='Validation Accuracy',
                    line=dict(color='green')
                ))
                fig.update_layout(
                    title="Validation Accuracy",
                    xaxis_title="Epoch",
                    yaxis_title="Accuracy"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics
            st.markdown("### Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{eval_results['accuracy']:.4f}")
            
            with col2:
                st.metric("Mean Uncertainty", f"{eval_results['mean_uncertainty']:.4f}")
            
            with col3:
                st.metric("Max Uncertainty", f"{eval_results['max_uncertainty']:.4f}")
            
            with col4:
                predictions = eval_results['predictions']
                targets = eval_results['targets']
                predicted_classes = torch.argmax(predictions, dim=1)
                correct_predictions = (predicted_classes == targets)
                st.metric("Correct Predictions", f"{correct_predictions.sum().item()}/{len(targets)}")

with tab4:
    st.markdown('<h2 class="section-header">Uncertainty Analysis</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first in the Model Training tab.")
    else:
        # Method selection for analysis
        available_methods = list(st.session_state.model_results.keys())
        selected_method = st.selectbox("Select method for analysis:", available_methods)
        
        if selected_method:
            results = st.session_state.model_results[selected_method]
            eval_results = results['eval_results']
            
            # Analysis options
            analysis_type = st.selectbox(
                "Analysis Type:",
                ["Uncertainty Distribution", "Calibration Analysis", "Model Comparison"]
            )
            
            if analysis_type == "Uncertainty Distribution":
                st.markdown("### Uncertainty Distribution Analysis")
                
                predictions = eval_results['predictions']
                uncertainties = eval_results['uncertainties']
                targets = eval_results['targets']
                
                # Convert to numpy for plotting
                pred_np = predictions.detach().cpu().numpy()
                unc_np = uncertainties.detach().cpu().numpy()
                target_np = targets.detach().cpu().numpy()
                
                predicted_classes = np.argmax(pred_np, axis=1)
                correct_predictions = (predicted_classes == target_np)
                max_uncertainties = np.max(unc_np, axis=1)
                max_confidences = np.max(pred_np, axis=1)
                
                # Create visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Uncertainty distribution
                    fig = px.histogram(
                        x=unc_np.flatten(),
                        title="Uncertainty Distribution",
                        labels={'x': 'Uncertainty (Std Dev)', 'y': 'Frequency'},
                        nbins=50
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Uncertainty vs correctness
                    correct_unc = max_uncertainties[correct_predictions]
                    incorrect_unc = max_uncertainties[~correct_predictions]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=correct_unc, name='Correct', opacity=0.7, nbinsx=30))
                    fig.add_trace(go.Histogram(x=incorrect_unc, name='Incorrect', opacity=0.7, nbinsx=30))
                    fig.update_layout(
                        title="Uncertainty by Correctness",
                        xaxis_title="Max Uncertainty",
                        yaxis_title="Frequency"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Confidence vs uncertainty scatter
                fig = px.scatter(
                    x=max_confidences,
                    y=max_uncertainties,
                    title="Confidence vs Uncertainty",
                    labels={'x': 'Max Confidence', 'y': 'Max Uncertainty'},
                    opacity=0.6
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Uncertainty heatmap
                if len(unc_np) > 50:
                    sample_unc = unc_np[:50]
                else:
                    sample_unc = unc_np
                
                fig = go.Figure(data=go.Heatmap(
                    z=sample_unc.T,
                    colorscale='Viridis',
                    title="Uncertainty Heatmap (First 50 Samples)"
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "Calibration Analysis":
                st.markdown("### Calibration Analysis")
                
                predictions = eval_results['predictions']
                targets = eval_results['targets']
                
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
                
                # Create bins for reliability diagram
                num_bins = 10
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
                
                # Reliability diagram
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=bin_confidences,
                    y=bin_accuracies,
                    name='Accuracy',
                    marker_color='skyblue'
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    name='Perfect Calibration',
                    line=dict(dash='dash', color='red')
                ))
                fig.update_layout(
                    title="Reliability Diagram",
                    xaxis_title="Confidence",
                    yaxis_title="Accuracy"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence distribution
                fig = px.histogram(
                    x=confidences,
                    title="Confidence Distribution",
                    labels={'x': 'Confidence', 'y': 'Frequency'},
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Accuracy vs confidence scatter
                fig = px.scatter(
                    x=confidences,
                    y=correct,
                    title="Accuracy vs Confidence",
                    labels={'x': 'Confidence', 'y': 'Accuracy'},
                    opacity=0.5
                )
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    name='Perfect Calibration',
                    line=dict(dash='dash', color='red')
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "Model Comparison":
                st.markdown("### Model Comparison")
                
                if len(st.session_state.model_results) > 1:
                    # Compare all trained models
                    model_names = list(st.session_state.model_results.keys())
                    
                    # Extract metrics
                    accuracies = []
                    mean_uncertainties = []
                    
                    for name in model_names:
                        results = st.session_state.model_results[name]
                        eval_results = results['eval_results']
                        accuracies.append(eval_results['accuracy'].item())
                        mean_uncertainties.append(eval_results['mean_uncertainty'].item())
                    
                    # Comparison plots
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.bar(
                            x=model_names,
                            y=accuracies,
                            title="Accuracy Comparison",
                            labels={'x': 'Method', 'y': 'Accuracy'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.bar(
                            x=model_names,
                            y=mean_uncertainties,
                            title="Mean Uncertainty Comparison",
                            labels={'x': 'Method', 'y': 'Mean Uncertainty'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Uncertainty vs accuracy scatter
                    fig = px.scatter(
                        x=mean_uncertainties,
                        y=accuracies,
                        text=model_names,
                        title="Uncertainty vs Accuracy",
                        labels={'x': 'Mean Uncertainty', 'y': 'Accuracy'}
                    )
                    fig.update_traces(textposition="top center")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary table
                    comparison_df = pd.DataFrame({
                        'Method': model_names,
                        'Accuracy': accuracies,
                        'Mean Uncertainty': mean_uncertainties
                    })
                    st.dataframe(comparison_df, use_container_width=True)
                
                else:
                    st.info("Train multiple models to enable comparison analysis.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Bayesian Neural Networks Demo | Research and Educational Use Only</p>
    <p>⚠️ Not for regulated decisions without human review</p>
</div>
""", unsafe_allow_html=True)
