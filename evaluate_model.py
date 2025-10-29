"""Model evaluation entry point for GTSRB.

This script loads a trained ``GTSRB_CNN`` checkpoint, runs batched inference on the
project's split test set (returned by ``create_dataloaders``), computes metrics
(accuracy, balanced accuracy, top-k), and saves plots (confusion matrix, ROC, PR).

Usage (PowerShell):
    python evaluate_model.py --model_path models/best_model.pth --save_dir results --batch_size 128
"""

import argparse
import os
import json
import torch

from src.data_loader import create_dataloaders
from src.model import GTSRB_CNN
from src import evaluate

def main(args):
    """Run evaluation for a trained model on the project split test set.

    This uses the test split created in ``create_dataloaders``, which is normalized
    with statistics computed from the training split only (no leakage).
    """
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # --- 2. Load Data ---
    if args.official_test:
        print("Using official GTSRB test set for evaluation.")
        _, _, _, test_loader, _, _ = create_dataloaders(batch_size=args.batch_size)
    else:
        # We use our 'test_loader_split' (70/15/15 split) for this comprehensive evaluation
        print("Using project split test set for evaluation.")
        _, _, test_loader, _, _, _ = create_dataloaders(batch_size=args.batch_size)

    # --- 3. Load Model ---
    model = GTSRB_CNN(num_classes=43).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval() # Set model to evaluation mode
    print(f"Model loaded from {args.model_path}")

    # --- 4. Get Predictions ---
    with torch.no_grad():
        y_true, y_pred, y_scores = evaluate.get_predictions(model, test_loader, device)

    # --- 5. Calculate and Display Metrics ---
    # Classification Report (Precision, Recall, F1, Balanced Accuracy)
    report_dict = evaluate.get_classification_report_dict(y_true, y_pred)
    print("\n--- Classification Report ---")
    print(f"Overall Accuracy: {report_dict['accuracy']:.4f}")
    print(f"Balanced Accuracy (Macro Recall): {report_dict['balanced_accuracy']:.4f}")
    print(f"Macro Avg F1-score: {report_dict['macro avg']['f1-score']:.4f}")
    print(f"Weighted Avg F1-score: {report_dict['weighted avg']['f1-score']:.4f}")

    # Save full report to a file
    with open(os.path.join(args.save_dir, 'classification_report.json'),'w',encoding='utf-8') as f:
        json.dump(report_dict, f, indent=4)
    print("Full classification report saved to classification_report.json")

    # Top-K Accuracy
    top_3_acc = evaluate.calculate_top_k_accuracy(y_true, y_scores, k=3)
    top_5_acc = evaluate.calculate_top_k_accuracy(y_true, y_scores, k=5)
    print(f"\nTop-3 Accuracy: {top_3_acc:.4f}")
    print(f"Top-5 Accuracy: {top_5_acc:.4f}")

    # --- 6. Generate and Save Plots ---
    print("\n--- Generating Plots ---")
    evaluate.plot_confusion_matrix(y_true, y_pred, args.save_dir)
    evaluate.plot_roc_curves(y_true, y_scores, args.save_dir)
    evaluate.plot_pr_curves(y_true, y_scores, args.save_dir)

    print(f"\nEvaluation complete. All results saved in '{args.save_dir}' directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained GTSRB CNN model.")
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                        help='Path to the trained model file (.pth)')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save evaluation reports and plots')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--official_test', action='store_true',
                        help='Use the official GTSRB test set for evaluation')

    args = parser.parse_args()
    main(args)
