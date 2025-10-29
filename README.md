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
IPPRA2/
├── data/                        # Dataset cache (auto-downloaded by torchvision)
├── models/                      # Saved model checkpoints
├── notebooks/
│   └── 01_data_exploration.ipynb
├── final_results_official_test/ # Evaluation on official test set
├── final_results_split_test/    # Evaluation on project split test set
├── results_cv/                  # Cross-validation outputs
├── src/
│   ├── data_loader.py           # Splits, transforms, (per-fold) normalization
│   ├── evaluate.py              # Predictions, metrics, CM/ROC/PR plotting
│   ├── experiments.py           # Shared builders: seeds, folds, model components
│   ├── model.py                 # GTSRB_CNN architecture
│   ├── robustness.py            # Corruption transforms (severity-aware)
│   └── trainer.py               # Training/validation loop with tqdm
├── evaluate_model.py            # Final evaluation entry point
├── run_cross_validation.py      # K-fold CV (stratified, leakage-safe)
├── run_hyperparameter_tuning.py # Optuna tuning (per-fold stats)
├── test_robustness.py           # Robustness + performance tests
├── train.py                     # Single training run
└── README.md
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

## Sanity and correctness guarantees

* No data leakage
  * create_dataloaders computes mean/std on the training split only.
  * Cross-validation and tuning recompute mean/std per fold using TRAIN indices only.
* Consistent transforms
  * Corruptions applied after ToTensor, before Normalize.
  * Occlusion severity controls square size (pixels), making effects predictable.
* Determinism
  * set_seeds(seed) called in scripts; reproducible runs by default.
* Accurate evaluation
  * Confusion matrix plotting tuned for 43 classes (larger canvas, sparse ticks, adjustable on-cell value font).
  * get_predictions uses inference_mode + model.eval for speed and correctness.

## Notebooks

notebooks/01_data_exploration.ipynb

* Class distribution (seaborn-styled, readable ticks, annotated counts).
* One-sample-per-class ordered grid (0..42).
* Visualizing training augmentations (stochastic).
* Visualizing robustness corruptions at severities 1, 3, 5.

## Acknowledgements

* GTSRB dataset via torchvision.datasets.GTSRB
* PyTorch, scikit-learn, seaborn, matplotlib, Optuna, tqdm
