"""Evaluation script.

If you can run this, so can we during final evaluation!

Usage:
  Option 1 (Auto-find best model):
    ./submit-job.sh "python evaluate.py" --gpu

  Option 2 (Specify model):
    ./submit-job.sh "python evaluate.py --model /home/team02/models/your_model.pt" --gpu

  Option 3 (Use specific checkpoint):
    ./submit-job.sh "python evaluate.py --checkpoint job_name" --gpu
"""

import os
import sys
import glob
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from cold_start_hackathon.task import Net, load_data, test

# Auto-detect best model if not specified
def find_best_model(models_dir="/home/team02/models"):
    """Find the model with highest AUROC from filename."""
    if not os.path.exists(models_dir):
        return None

    # Find all .pt model files
    model_files = glob.glob(f"{models_dir}/*_auroc*.pt")
    if not model_files:
        return None

    # Extract AUROC from filename (format: *_auroc8234.pt = 0.8234)
    best_model = None
    best_auroc = 0
    for model_file in model_files:
        try:
            # Extract AUROC value from filename
            auroc_str = model_file.split("_auroc")[-1].replace(".pt", "")
            auroc = int(auroc_str) / 10000.0
            if auroc > best_auroc:
                best_auroc = auroc
                best_model = model_file
        except (ValueError, IndexError):
            continue

    return best_model

# Parse command line arguments
MODEL_PATH = None
if "--model" in sys.argv:
    idx = sys.argv.index("--model")
    if idx + 1 < len(sys.argv):
        MODEL_PATH = sys.argv[idx + 1]
elif "--checkpoint" in sys.argv:
    # Use checkpoint from /home/team02/checkpoints/
    idx = sys.argv.index("--checkpoint")
    if idx + 1 < len(sys.argv):
        checkpoint_name = sys.argv[idx + 1]
        MODEL_PATH = f"/home/team02/checkpoints/{checkpoint_name}_latest.pt"

# Auto-detect if not specified
if MODEL_PATH is None:
    print("No model specified, auto-detecting best model...")
    MODEL_PATH = find_best_model()
    if MODEL_PATH:
        print(f"Found best model: {MODEL_PATH}")

# Fallback to old hardcoded path if auto-detection fails
if MODEL_PATH is None:
    MODEL_PATH = f"/home/team02/models/job127_145241_round6_auroc7389.pt"
    print(f"Warning: Using fallback model path: {MODEL_PATH}")

DATASET_DIR = os.environ.get("DATASET_DIR", "/home/team02/xray-data")


def evaluate_split(model, dataset_name, split_name, device):
    """Evaluate on any dataset split and return predictions."""
    loader = load_data(dataset_name, split_name, batch_size=32)
    _, _, _, _, _, probs, labels = test(model, loader, device)
    return probs, labels


def main():
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    print(f"\nLoading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please update MODEL_PATH in evaluate.py with your model filename.")
        return 1

    # Load model
    model = Net()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}.")

    # Evaluate
    print("\nEvaluating...")
    datasets_to_test = [
        ("Hospital A", "HospitalA", "eval"),
        ("Hospital B", "HospitalB", "eval"),
        ("Hospital C", "HospitalC", "eval"),
        ("Test A", "Test", "test_A"),
        ("Test B", "Test", "test_B"),
        ("Test C", "Test", "test_C"),
        ("Test D (OOD)", "Test", "test_D"),
    ]

    # Collect all predictions
    hospital_predictions = {}
    test_predictions = {}
    for display_name, dataset_name, split_name in datasets_to_test:
        try:
            probs, labels = evaluate_split(model, dataset_name, split_name, device)
            n = len(labels)

            # Compute per-dataset AUROC for display
            auroc = roc_auc_score(labels, probs)
            print(f"  {display_name:<15} AUROC: {auroc:.4f} (n={n})")

            # Store predictions for aggregated AUROC calculation
            if display_name.startswith("Hospital"):
                hospital_predictions[display_name] = (probs, labels)
            elif display_name.startswith("Test"):
                test_predictions[display_name] = (probs, labels)
        except FileNotFoundError:
            # Test dataset doesn't exist for participants - skip silently
            pass

    # Eval Average: pool all hospital eval predictions, then compute AUROC
    if hospital_predictions:
        all_probs = np.concatenate([probs for probs, _ in hospital_predictions.values()])
        all_labels = np.concatenate([labels for _, labels in hospital_predictions.values()])
        eval_auroc = roc_auc_score(all_labels, all_probs)
        print(f"  {'Eval Avg':<15} AUROC: {eval_auroc:.4f}")

    # Test Average: pool all test predictions, then compute AUROC
    if test_predictions:
        all_probs = np.concatenate([probs for probs, _ in test_predictions.values()])
        all_labels = np.concatenate([labels for _, labels in test_predictions.values()])
        test_auroc = roc_auc_score(all_labels, all_probs)
        print(f"  {'Test Avg':<15} AUROC: {test_auroc:.4f}")

    print("\n" + "=" * 80)
    return 0


if __name__ == "__main__":
    exit(main())
