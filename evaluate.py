import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from datetime import datetime
from tqdm import tqdm

from src.models.resnet1d import resnet50_1d
from src.processing.data_extract import Preprocess, create_dataset_from_paths
from src.processing.pytorch_dataloader import ECGDataset

# ---------------- CONFIG ----------------
SEED = 42
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------------- FUNCTIONS ----------------

def load_model(model_path: str, device: torch.device):
    model = resnet50_1d(in_channels=2, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def prepare_test_data(data_dir: str, preprocessor, test_split=None, seed=42):
    paths = [os.path.join(data_dir, f[:-4]) for f in os.listdir(data_dir) if f.endswith('.atr')]
    np.random.seed(seed)
    np.random.shuffle(paths)
    
    if test_split and 0.0 < test_split < 1.0:
        _, test_paths = np.split(paths, [int((1 - test_split) * len(paths))])
    else:
        print("No valid test split provided, using all data for testing.")
        test_paths = paths  # Use all paths if no valid split is provided

    print(f"Test data paths: {test_paths}")
    print(f"Test set size: {len(test_paths)}")
    X_test, y_test = create_dataset_from_paths(test_paths, preprocessor)
    return X_test, y_test

def evaluate(model, data_loader, device):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in tqdm(data_loader):
            x = x.to(device)
            output = model(x)
            preds = output.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    return all_labels, all_preds

def print_metrics(y_true, y_pred):
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def save_run_outputs(y_true, y_pred, args, save_dir="runs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    data_name = os.path.basename(os.path.normpath(args.data_dir))
    
    run_name = (
        f"{data_name}_fs{args.fs}_split{int(args.split*100)}pct_"
        f"{model_name}_{timestamp}"
    )

    run_path = os.path.join(save_dir, run_name)
    os.makedirs(run_path, exist_ok=True)

    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    with open(os.path.join(run_path, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(os.path.join(run_path, "confusion_matrix.txt"), cm, fmt='%d')

    with open(os.path.join(run_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    print(f"\nSaved evaluation results to: {run_path}")


def run_evaluation(args):
    print(f"\nEvaluating on dataset: {args.data_dir}")

    preprocessor = Preprocess(fs=args.fs, target_fs=128)
    X_test, y_test = prepare_test_data(args.data_dir, preprocessor, test_split=args.split, seed=SEED)
    print(f"Test set size: {len(X_test)}, Class distribution: {Counter(y_test)}")

    test_ds = ECGDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = load_model(args.model_path, DEVICE)
    y_true, y_pred = evaluate(model, test_loader, DEVICE)

    print_metrics(y_true, y_pred)
    save_run_outputs(y_true, y_pred, args)

# ---------------- MAIN ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ResNet1D ECG model on dataset")

    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory containing ECG data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model file (e.g. best_model.pt)")
    parser.add_argument("--fs", type=int, default=250, help="Original sampling frequency of data")
    parser.add_argument("--split", type=float, default=0.2, help="Fraction of data to use as test set (0 = use all)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")

    args = parser.parse_args()

    run_evaluation(args)
    print("Evaluation complete.")
