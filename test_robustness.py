# test_robustness.py

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
from src.data_loader import get_gtsrb_datasets, get_mean_std


def run_performance_tests(model, device, args):
    """
    Measures inference latency, throughput, and GPU memory usage.
    """
    print("\n--- Running Performance Tests ---")
    model.eval()

    # --- Latency and Throughput ---
    dummy_input = torch.randn(args.batch_size, 3, 32, 32, device=device)

    # Warm-up GPU
    for _ in range(10):
        _ = model(dummy_input)

    # Measure (synchronize CUDA for accurate timing)
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
    images_per_second = total_images / total_time
    latency_per_image = (total_time / 100) / args.batch_size * 1000  # in ms

    print(f"Inference Throughput: {images_per_second:.2f} images/sec")
    print(f"Inference Latency: {latency_per_image:.4f} ms/image")

    # --- GPU Memory Usage ---
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(dummy_input)
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # in MB
        print(f"Peak GPU Memory Usage: {peak_memory:.2f} MB")

    # --- Model Size ---
    model_size = os.path.getsize(args.model_path) / 1024 / 1024  # in MB
    print(f"Model Size on Disk: {model_size:.2f} MB")


def run_robustness_tests(model, device, args):
    """
    Evaluates the model on various corruptions and severities.
    """
    print("\n--- Running Robustness Tests ---")

    # Load the raw test dataset (without any transforms yet)
    raw_test_dataset = datasets.GTSRB(
        root='./data', split='test', download=True)

    # Load datasets WITHOUT normalization to compute mean/std
    _, train_dataset_no_aug_no_norm = get_gtsrb_datasets()

    # Calculate mean and std on the full training dataset (no normalization)
    mean, std = get_mean_std(train_dataset_no_aug_no_norm)

    corruption_types = [
        'gaussian_noise', 'salt_pepper', 'brightness',
        'contrast', 'rotation', 'occlusion'
    ]
    results = {}

    for corruption in corruption_types:
        results[corruption] = []
        for severity in range(1, 6):
            # 1. Get the specific corruption transform
            corruption_transform = get_robustness_transform(
                corruption, severity)

            # 2. Create the full pipeline with normalization
            full_transform = transforms.Compose([
                corruption_transform,
                transforms.Normalize(mean, std)
            ])

            # 3. Apply transform to the dataset and create a loader
            raw_test_dataset.transform = full_transform
            test_loader = DataLoader(
                raw_test_dataset, batch_size=args.batch_size, shuffle=False)

            # 4. Evaluate accuracy
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in tqdm(test_loader, desc=f"{corruption} sev {severity}", leave=False):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            results[corruption].append(accuracy)
            print(
                f"Corruption: {corruption}, Severity: {severity}, Accuracy: {accuracy:.2f}%")

    return results


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

    save_path = os.path.join(save_dir, 'robustness_curves.png')
    plt.savefig(save_path)
    plt.close()
    print(f"\nRobustness plot saved to {save_path}")


def main(args):
    """ Main function to orchestrate the tests. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # Load Model
    model = GTSRB_CNN(num_classes=43).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded from {args.model_path}")

    # Run tests
    robustness_results = run_robustness_tests(model, device, args)
    run_performance_tests(model, device, args)

    # Save and plot robustness results
    results_path = os.path.join(args.save_dir, 'robustness_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(robustness_results, f, indent=4)
    print(f"Robustness results saved to {results_path}")

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
    args = parser.parse_args()
    main(args)
