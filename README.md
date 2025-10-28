# GTSRB Traffic Sign Classification with PyTorch and Convolutional Neural Networks

This repository contains a complete, production-ready pipeline for training and evaluating a Convolutional Neural Network (CNN) on the German Traffic Sign Recognition Benchmark (GTSRB) dataset using PyTorch. The project emphasizes a modular, reusable codebase, rigorous evaluation, and methodologically sound practices to prevent data leakage.

## Features

This project is a comprehensive showcase of a professional machine learning workflow, including:

  * **Modular Architecture**: A clean separation of concerns with dedicated modules for data loading, model definition, training, evaluation, and robustness testing.
  * **Rigorous Evaluation**: Implements **5-fold stratified cross-validation** to provide a robust measure of the model's performance.
  * **Hyperparameter Tuning**: Uses **Optuna** to systematically search for the optimal learning rate, ensuring peak model performance.
  * **Comprehensive Metrics**: Generates a full suite of performance metrics, including **Top-K accuracy**, **balanced accuracy**, per-class **Precision/Recall/F1 scores**, and visualizations for **ROC/AUC** and **Precision-Recall curves**.
  * **In-depth Robustness Analysis**: Tests the final model's resilience against a variety of data corruptions (Gaussian noise, salt & pepper, brightness, contrast, rotation, and occlusion) at multiple severity levels.
  * **Performance Benchmarking**: Measures the computational performance of the final model, including inference **latency**, **throughput**, and **GPU memory usage**.
  * **Leakage-Free Methodology**: Adopts a gold-standard approach by calculating normalization statistics independently for each training fold, ensuring that no information from the validation or test sets leaks into the training process.

-----

## Project Structure

The codebase is organized into a `src` directory for reusable modules and a set of top-level scripts for running experiments.

```
gtsrb_classification/
├── data/                       # Dataset storage (auto-downloaded)
├── models/                     # Saved model checkpoints (e.g., best_model.pth)
├── notebooks/
│   └── 01_data_exploration.ipynb # Visual showcase of data and transforms
├── results/                    # Output from evaluate_model.py
├── results_cv/                 # Output from run_cross_validation.py
├── results_robustness/         # Output from test_robustness.py
├── src/
│   ├── data_loader.py          # Data loading, transforms, and splitting
│   ├── evaluate.py             # Performance metrics and plotting functions
│   ├── experiments.py          # Shared utilities for CV and tuning
│   ├── model.py                # CNN architecture definition
│   ├── robustness.py           # Data corruption toolkit
│   └── trainer.py              # Core training and validation loop
├── .gitignore                  # Git ignore file
├── evaluate_model.py           # Entry point for final model evaluation
├── run_cross_validation.py     # Entry point for 5-fold cross-validation
├── run_hyperparameter_tuning.py# Entry point for Optuna hyperparameter search
├── test_robustness.py          # Entry point for robustness & performance tests
├── train.py                    # Entry point for a single training run
└── README.md                   # This file
```

-----

## Setup and Installation

To get started, clone the repository and set up the Python environment.

**1. Clone the repository:**

```bash
git clone https://github.com/emilianodesu/IPPRA2.git
```

**2. Create a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install dependencies:**
The required packages are listed in `requirements.txt`.

```bash
pip install torch torchvision scikit-learn pandas seaborn matplotlib optuna tqdm
```

-----

## Usage: The Experimental Workflow

The project is designed to be run in a specific sequence to ensure methodologically sound results.

### Step 1: Discover Optimal Hyperparameters

Run the Optuna search to find the best learning rate for the Adam optimizer. This is the most computationally intensive step.

```bash
python run_hyperparameter_tuning.py --n_trials 25
```

### Step 2: Confirm Performance with Cross-Validation

After finding the best learning rate, run the full 5-fold cross-validation to get a robust statistical measure of your model's performance.

```bash
python run_cross_validation.py --epochs 20
```

*(Note: The best learning rate from Step 1 is already hardcoded in this script for convenience.)*

### Step 3: Train the Final Model Asset

Train one final model using the best learning rate on the largest training split. This creates the `best_model.pth` asset for final analysis.

```bash
python train.py --epochs 30 --lr <best-learning-rate-from-step-1>
```

### Step 4: Run Final Analysis

Use your final model to generate all the plots, reports, and benchmarks.

**A. Generate evaluation metrics and plots:**

```bash
python evaluate_model.py --model_path models/best_model.pth --save_dir final_results
```

This will produce a detailed classification report, confusion matrices, ROC curves, and PR curves in the `final_results/` directory.

**B. Run robustness and performance tests:**

```bash
python test_robustness.py --model_path models/best_model.pth --save_dir final_results
```

This will generate the robustness degradation plot and print the inference speed and memory usage to the console.
