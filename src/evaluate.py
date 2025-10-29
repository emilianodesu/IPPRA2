"""Evaluation utilities: prediction, metrics, and plots for GTSRB models.

This module provides helper functions to:
- Run batched inference and collect predictions/probabilities
- Compute classification metrics (including balanced accuracy and top-k)
- Plot confusion matrix, multi-class ROC, and PR curves (micro-average)

Conventions
----------
- Inputs to the model are assumed to be RGB 32x32 tensors normalized as in training.
- ``get_predictions`` returns NumPy arrays for easy interoperability with sklearn.
- For multi-class curves, we use one-vs-rest micro-averaging.
"""

import os
import torch
import numpy as np
from sklearn.metrics import (classification_report, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def get_predictions(model, data_loader, device):
    """Run inference and collect labels, argmax predictions, and probabilities.

    Args:
        model (torch.nn.Module): Trained model returning class logits.
        data_loader (torch.utils.data.DataLoader): Batches to evaluate.
        device (torch.device): Computation device.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            (y_true, y_pred, y_scores) where y_scores are softmax probabilities.
    """
    model.eval()
    all_labels = []
    all_predictions = []
    all_scores = []

    # inference_mode is slightly faster and enforces inference-only behavior
    with torch.inference_mode():
        for inputs, labels in tqdm(
            data_loader, desc="Evaluating", total=len(data_loader), leave=False
        ):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            scores = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    return np.array(all_labels), np.array(all_predictions), np.array(all_scores)


def get_classification_report_dict(y_true, y_pred):
    """Compute sklearn classification report plus balanced accuracy.

    Balanced accuracy equals the macro-averaged recall and is often more
    informative under class imbalance.

    Args:
        y_true (np.ndarray): Ground-truth integer labels.
        y_pred (np.ndarray): Predicted integer labels.

    Returns:
        dict: Report dictionary from sklearn with an extra 'balanced_accuracy' key.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    balanced_accuracy = report['macro avg']['recall']
    report['balanced_accuracy'] = balanced_accuracy
    return report


def calculate_top_k_accuracy(y_true, y_scores, k=3):
    """Calculate top-K accuracy from probability scores.

    Args:
        y_true (np.ndarray): Ground-truth integer labels.
        y_scores (np.ndarray): Class probabilities of shape [N, C].
        k (int): Number of top predictions to consider.

    Returns:
        float: Proportion of samples whose true class is among the top-K.
    """
    top_k_preds = np.argsort(y_scores, axis=1)[:, -k:]
    correct = sum(y_true[i] in top_k_preds[i] for i in range(len(y_true)))
    return correct / len(y_true)


def plot_confusion_matrix(y_true, y_pred, save_dir, num_classes=43):
    """
    Generate and save professional confusion matrix visualizations using sklearn.

    This function saves four figures:
    - confusion_matrix_counts.png: absolute counts
    - confusion_matrix_normalized.png: row-normalized percentages per true class
    - errors_normalized_row.png: misclassification-weighted and row-normalized
    - errors_normalized_column.png: misclassification-weighted and column-normalized

    Args:
        y_true (np.ndarray): Ground-truth integer labels.
        y_pred (np.ndarray): Predicted integer labels.
        save_dir (str): Directory path to save the PNG image.
        num_classes (int): Expected number of classes (for axis tick labels).
    """
    os.makedirs(save_dir, exist_ok=True)
    class_labels = [str(i) for i in range(num_classes)]

    # --- Dynamic sizing for large label sets (e.g., GTSRB 43 classes) ---
    # scales with number of classes: ~0.35 in per class, capped for readability
    base_width = max(12, min(25, 0.35 * num_classes))
    base_height = base_width * 0.85
    figsize = (base_width, base_height)

    # tick and text scaling
    tick_fontsize = 6 if num_classes > 30 else 8
    title_fontsize = 14
    rotation = 90 if num_classes > 20 else 45

    def save_disp(title, filename, cmap, normalize=None, sample_weight=None):
        plt.figure(figsize=figsize)
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred,
            display_labels=class_labels,
            cmap=cmap,
            normalize=normalize,
            values_format=".0%" if normalize else None,
            sample_weight=sample_weight,
            xticks_rotation=rotation,
            colorbar=True
        )
        plt.title(title, fontsize=title_fontsize, pad=12)
        plt.xlabel("Predicted label", fontsize=10)
        plt.ylabel("True label", fontsize=10)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.tight_layout(pad=2.0)
        out_path = os.path.join(save_dir, filename)
        plt.savefig(out_path, dpi=250)
        plt.close()
        return out_path

    # --- 1) Absolute counts ---
    counts_path = save_disp(
        title="Confusion Matrix — Counts",
        filename="confusion_matrix_counts.png",
        cmap="mako",
        normalize=None
    )

    # --- 2) Row-normalized ---
    norm_path = save_disp(
        title="Confusion Matrix — Row-normalized (%)",
        filename="confusion_matrix_normalized.png",
        cmap="mako",
        normalize="true"
    )

    # --- 3) Weighted by misclassification (row-normalized) ---
    sample_weight = (y_pred != y_true)
    err_row = save_disp(
        title="Errors Normalized by Row (%)",
        filename="errors_normalized_row.png",
        cmap="rocket",
        normalize="true",
        sample_weight=sample_weight
    )

    # --- 4) Weighted by misclassification (column-normalized) ---
    err_col = save_disp(
        title="Errors Normalized by Column (%)",
        filename="errors_normalized_column.png",
        cmap="rocket",
        normalize="pred",
        sample_weight=sample_weight
    )

    print(f"✅ Confusion matrices saved:\n"
          f" - {counts_path}\n"
          f" - {norm_path}\n"
          f" - {err_row}\n"
          f" - {err_col}")


def plot_roc_curves(y_true, y_scores, save_dir, num_classes=43):
    """Generate and save micro-averaged ROC curve for multi-class (OvR).

    Args:
        y_true (np.ndarray): Ground-truth integer labels.
        y_scores (np.ndarray): Class probabilities of shape [N, C].
        save_dir (str): Directory path to save the PNG image.
        num_classes (int): Number of classes for binarization.
    """
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    y_true_bin = np.asarray(y_true_bin)

    plt.figure(figsize=(10, 8))

    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Micro-average ROC curve (area = {roc_auc:0.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'))
    plt.close()
    print("ROC curve plot saved.")


def plot_pr_curves(y_true, y_scores, save_dir, num_classes=43):
    """Generate and save micro-averaged Precision-Recall curve (OvR).

    Args:
        y_true (np.ndarray): Ground-truth integer labels.
        y_scores (np.ndarray): Class probabilities of shape [N, C].
        save_dir (str): Directory path to save the PNG image.
        num_classes (int): Number of classes for binarization.
    """
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    y_true_bin = np.asarray(y_true_bin)

    plt.figure(figsize=(10, 8))

    # Compute micro-average PR curve and AP
    precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_scores.ravel())
    ap = average_precision_score(y_true_bin, y_scores, average="micro")
    plt.plot(recall, precision, label=f'Micro-average PR curve (AP = {ap:0.2f})',
             color='navy', linestyle=':', linewidth=4)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Multi-class Precision-Recall Curve (One-vs-Rest)')
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pr_curves.png'))
    plt.close()
    print("Precision-Recall curve plot saved.")
