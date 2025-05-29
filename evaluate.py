import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

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
        test_paths = paths  # Use all paths if no valid split is provided

    print(f"Test set size: {len(test_paths)}")
    X_test, y_test = create_dataset_from_paths(test_paths, preprocessor)
    return X_test, y_test


def evaluate(model, data_loader, device):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in data_loader:
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


def run_evaluation(data_dir: str, model_path: str, fs=250, test_split=0.2, batch_size=32):
    print(f"\nEvaluating on dataset: {data_dir}")

    preprocessor = Preprocess(fs=fs, target_fs=128)
    X_test, y_test = prepare_test_data(data_dir, preprocessor, test_split=test_split, seed=SEED)
    print(f"Test set size: {len(X_test)}, Class distribution: {Counter(y_test)}")

    test_ds = ECGDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = load_model(model_path, DEVICE)
    y_true, y_pred = evaluate(model, test_loader, DEVICE)

    print_metrics(y_true, y_pred)


# ---------------- MAIN ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ResNet1D ECG model on dataset")

    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory containing ECG data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model file (e.g. best_model.pt)")
    parser.add_argument("--fs", type=int, default=250, help="Original sampling frequency of data")
    parser.add_argument("--split", type=float, default=0.2, help="Fraction of data to use as test set (0 = use all)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")

    args = parser.parse_args()

    run_evaluation(
        data_dir=args.data_dir,
        model_path=args.model_path,
        fs=args.fs,
        test_split=args.split,
        batch_size=args.batch_size
    )
    print("Evaluation complete.")