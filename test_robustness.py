"""Robustness and performance evaluation entry point for GTSRB models.

This script evaluates a trained :class:`src.model.GTSRB_CNN` on corrupted
versions of the GTSRB test data across multiple severity levels and measures
inference performance (throughput, latency, GPU memory). Results are saved to
disk for analysis and reporting.

Overview
--------
- Robustness tests iterate over corruption types (gaussian_noise, salt_pepper,
    brightness, contrast, rotation, occlusion) and severities 1..5 using
    :func:`src.robustness.get_robustness_transform`.
- Performance tests run batched forward passes with CUDA synchronization to
    collect stable latency/throughput and peak GPU memory metrics.
- When `--official_test` is used, the script loads the official GTSRB test set
    and applies normalization with per-training-split statistics (to avoid
    leakage). Otherwise, it uses the project split test loader without reapplying
    normalization.

Outputs
-------
- ``<save_dir>/robustness_curves.png``: Line plot of accuracy vs. severity for
    each corruption.
- ``<save_dir>/performance_results.json``: Throughput, latency, and model size.
- ``<save_dir>/all_test_results.json``: Combined robustness and performance.

Command-line arguments
----------------------
--model_path: Path to a trained ``.pth`` file (state dict).
--save_dir:   Directory to store JSONs and plots (created if missing).
--batch_size: Batch size for evaluation and performance tests.
--official_test: Use the official GTSRB test set instead of the project split.

Notes
-----
- CUDA timing: We call ``torch.cuda.synchronize()`` before timing boundaries to
    obtain accurate latency/throughput measurements on GPU.
- Occlusion severity: In ``src.robustness``, occlusion severity controls the
    side length (pixels) of a square mask for 32×32 inputs, ensuring predictable
    effect strength.
- Reproducibility: A fixed torch seed is set; see training and CV scripts for a
    comprehensive seeding strategy across libraries.
"""

import argparse
import os
import time
import json
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import GTSRB_CNN
from src.robustness import get_robustness_transform
from src.data_loader import get_gtsrb_datasets, get_mean_std, create_dataloaders


# ------------------------------------------------------------
# PERFORMANCE TESTS
# ------------------------------------------------------------
def run_performance_tests(model, device, args):
    """Measures inference latency, throughput, and GPU memory usage, and saves results."""
    print("\n--- Running Performance Tests ---")
    model.eval()
    results = {}

    dummy_input = torch.randn(args.batch_size, 3, 32, 32, device=device)

    # Warm-up
    for _ in range(10):
        _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    total_images = 100 * args.batch_size
    results['throughput_images_per_sec'] = total_images / total_time
    results['latency_ms_per_image'] = (total_time / 100) / args.batch_size * 1000

    print(f"Inference Throughput: {results['throughput_images_per_sec']:.2f} images/sec")
    print(f"Inference Latency: {results['latency_ms_per_image']:.4f} ms/image")

    # --- GPU Memory ---
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(dummy_input)
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        results['peak_gpu_memory_mb'] = peak_memory
        print(f"Peak GPU Memory Usage: {peak_memory:.2f} MB")

    # --- Model size ---
    model_size = os.path.getsize(args.model_path) / 1024 / 1024
    results['model_size_mb'] = model_size
    print(f"Model Size on Disk: {model_size:.2f} MB")

    # --- Save ---
    save_path = os.path.join(args.save_dir, 'performance_results.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"Performance results saved to {save_path}")

    return results


# ------------------------------------------------------------
# ROBUSTNESS TESTS
# ------------------------------------------------------------
def run_robustness_tests(model, device, args):
    """Evaluates the model on various corruptions and severities."""
    print("\n--- Running Robustness Tests ---")

    # --- 1. Load Base Test Dataset ---
    if args.official_test:
        print("Using official GTSRB test set for robustness evaluation.")
        test_dataset = datasets.GTSRB(root='./data', split='test', download=True)
        _, train_dataset_no_norm = get_gtsrb_datasets()
        mean, std = get_mean_std(train_dataset_no_norm)
        needs_normalization = True
    else:
        print("Using project split test set for robustness evaluation.")
        _, _, test_loader_split, _, mean, std = create_dataloaders(batch_size=args.batch_size)
        # Retrieve the base dataset even if wrapped (e.g., Subset inside DataLoader)
        base_ds = test_loader_split.dataset
        test_dataset = getattr(base_ds, 'dataset', base_ds)  # type: ignore[attr-defined]
        # The base dataset is already normalized in the project split; avoid double normalization
        needs_normalization = False  # Avoid double normalization

    corruption_types = ['gaussian_noise', 'salt_pepper', 'brightness',
                        'contrast', 'rotation', 'occlusion']
    results = {}

    for corruption in corruption_types:
        results[corruption] = []
        for severity in range(1, 6):
            corruption_transform = get_robustness_transform(corruption, severity)

            # Apply normalization only if needed
            if needs_normalization:
                full_transform = transforms.Compose([
                    corruption_transform,
                    transforms.Normalize(mean, std)
                ])
            else:
                full_transform = corruption_transform

            test_dataset.transform = full_transform  # type: ignore[attr-defined]
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

            correct, total = 0, 0
            model.eval()  # ensure eval mode for each corruption test
            with torch.no_grad():  # disable gradient computation
                for inputs, labels in tqdm(test_loader, desc=f"{corruption:<12} sev {severity}", leave=False):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            acc = 100 * correct / total
            results[corruption].append(acc)
            print(f"Corruption: {corruption:>12}, Severity: {severity}, Accuracy: {acc:.2f}%")

    return results


# ------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------
def plot_robustness_results(results, save_dir):
    """Plots and saves the robustness results."""
    df = pd.DataFrame(results, index=range(1, 6))
    df.index.name = 'Severity'

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, markers=True)
    plt.title('Model Robustness to Various Corruptions')
    plt.ylabel('Test Accuracy (%)')
    plt.xlabel('Corruption Severity Level')
    plt.grid(True)
    plt.ylim(0, 100)
    plt.legend(title='Corruption Type')
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'robustness_curves.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Robustness plot saved to {save_path}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main(args):
    """Main function to orchestrate robustness and performance tests."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    model = GTSRB_CNN(num_classes=43).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {args.model_path}")

    with torch.no_grad():
        robustness_results = run_robustness_tests(model, device, args)
        performance_results = run_performance_tests(model, device, args)

    # Combine and save results
    all_results = {"robustness": robustness_results, "performance": performance_results}
    results_path = os.path.join(args.save_dir, 'all_test_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    print(f"All test results saved to {results_path}")

    plot_robustness_results(robustness_results, args.save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run robustness and performance tests on a trained GTSRB model.")
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                        help='Path to the trained model file (.pth)')
    parser.add_argument('--save_dir', type=str, default='results_robustness',
                        help='Directory to save test results and plots')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for testing')
    parser.add_argument('--official_test', action='store_true',
                        help='Use the official GTSRB test set instead of project split')
    args = parser.parse_args()
    main(args)
