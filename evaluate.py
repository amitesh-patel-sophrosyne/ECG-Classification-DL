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
import wfdb
import pandas as pd
from sklearn.utils import shuffle

from src.processing.data_extract import Preprocess, create_dataset_from_paths
from src.processing.pytorch_dataloader import ECGDataset

# ---------------- CONFIG ----------------
SEED = 42
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------------- FUNCTIONS ----------------

def model_fn(model_name):
    if model_name == "resnet18_1d":
        from src.models.resnet1d import resnet18_1d
        return resnet18_1d
    elif model_name == "resnet34_1d":
        from src.models.resnet1d import resnet34_1d
        return resnet34_1d
    elif model_name == "resnet50_1d":
        from src.models.resnet1d import resnet50_1d
        return resnet50_1d
    elif model_name == "resnet152_1d":
        from src.models.resnet1d import resnet152_1d
        return resnet152_1d
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def load_model(model_name: str, model_path: str, device: torch.device):
    resnet_model = model_fn(model_name)
    model = resnet_model(in_channels=2, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def prepare_test_data(data_dir: str, preprocessor, test_split=None, seed=42, data_used='afdb'):
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
    X_test, y_test = create_dataset_from_paths(test_paths, preprocessor, data_used)
    print(f"Test set size: {len(X_test)}, Class distribution: {Counter(y_test)}")
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

    dataset_name = os.path.basename(args.data_dir.rstrip("/"))
    run_path = os.path.join(save_dir, dataset_name, args.model,
                            f"fs{args.fs}_split{args.split}_batch{args.batch_size}")
    os.makedirs(run_path, exist_ok=True)

    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    with open(os.path.join(run_path, "classification_report.txt"), "w") as f:
        f.write(json.dumps(report, indent=4))

    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(os.path.join(run_path, "confusion_matrix.txt"), cm, fmt='%d')

    with open(os.path.join(run_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    print(f"\nSaved evaluation results to: {run_path}")

def load_large_scale_data(large_scale_df, directory, preprocess, fs=500):

    SEED = 42 

    # --- filter the AFib records from the large scale dataset ---
    def check_afib(row):
        # afib - SNOMED code: 164889003
        return '164889003' in row['labels']

    # --- take normal records also ---
    def check_normal(row):
        # normal - SNOMED code: 426783006
        return '426783006' in row['labels']

    afib_df = large_scale_df[large_scale_df.apply(check_afib, axis=1)]
    print(f"Found {len(afib_df)} AFib records in large scale dataset")

    normal_df = large_scale_df[large_scale_df.apply(check_normal, axis=1)]
    print(f"Found {len(normal_df)} Normal records in large scale dataset")

    # --- Take a subset of normal records to balance the dataset ---
    normal_df = normal_df.sample(n=len(afib_df), random_state=SEED)
    large_scale_subset_df = pd.concat([afib_df, normal_df])
    large_scale_subset_df = large_scale_subset_df.reset_index(drop=True)
    print(f"Total records in large scale dataset: {len(large_scale_subset_df)}")

    # --- Extract both the signal and the labels ---
    X = []
    y = []

    for _, row in large_scale_subset_df.iterrows():
        full_path = os.path.join(directory, row['path'])
        
        try:
            record = wfdb.rdsamp(full_path)
        except Exception as e:
            print(f"Error reading {row['path']}: {e}")
            continue

        signals = record[0]
        metadata = record[1]

        # --- Get index for leads 'II' and 'V1' ---
        try:
            sig_names = metadata['sig_name']
            ii_index = sig_names.index('II')
            v1_index = sig_names.index('V1')
            leads = signals[:, [ii_index, v1_index]]
        except Exception as e:
            print(f"Skipping {row['path']} due to missing leads: {e}")
            continue

        if len(leads.shape) == 1:
            leads = leads.reshape(-1, 1)

        # --- Preprocess the signal ---
        processed_signal = preprocess.process(leads, fs=fs)
        if processed_signal is None:
            print(f"Skipping record {row['path']} due to preprocessing error. Signal shape: {leads.shape}")
            continue

        X.append(processed_signal)

        # --- Convert labels to binary (0 for normal, 1 for AFib) ---
        labels = row['labels']
        if '164889003' in labels:
            y.append(1)
        elif '426783006' in labels:
            y.append(0)
        else:
            print(f"Skipping record {row['path']} due to unknown label: {labels}")
            continue

    X = np.array(X)
    y = np.array(y)

    # --- Shuffle the dataset to avoid ordered bias ---
    X, y = shuffle(X, y, random_state=SEED)

    print(f"Loaded {len(X)} records from large scale dataset")

    return X, y


def large_scale_data_evaluation(large_scale_df, large_scale_directory, model_path, fs=250, split=0.2, batch_size=32):

    preprocess = Preprocess(500,128)
    X_test, y_test = load_large_scale_data(large_scale_df, large_scale_directory, preprocess)
    print(f"Test set size: {len(X_test)}, Class distribution: {Counter(y_test)}")

    test_ds = ECGDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = load_model(model_path, DEVICE)
    y_true, y_pred = evaluate(model, test_loader, DEVICE)

    print_metrics(y_true, y_pred)
    save_run_outputs(y_true, y_pred, args)


def run_evaluation(args):
    print(f"\nEvaluating on dataset: {args.data_dir}")

    preprocessor = Preprocess(fs=args.fs, target_fs=128)
    X_test, y_test = prepare_test_data(args.data_dir, preprocessor, test_split=args.split, seed=SEED, data_used=args.data_used)
    print(f"Test set size: {len(X_test)}, Class distribution: {Counter(y_test)}")

    test_ds = ECGDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = load_model(args.model, args.model_path, DEVICE)
    y_true, y_pred = evaluate(model, test_loader, DEVICE)

    print_metrics(y_true, y_pred)
    save_run_outputs(y_true, y_pred, args)

# ---------------- MAIN ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ResNet1D ECG model on dataset")

    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory containing ECG data")
    parser.add_argument("--model", type=str, default="resnet50_1d",
                        choices=['resnet18_1d', 'resnet34_1d', 'resnet50_1d', 'resnet150_1d'], help="Model architecture to use (e.g. resnet50_1d)")
    parser.add_argument("--data_used", type=str, default="afdb",
                        choices=['afdb', 'ltafdb', 'large_scale', 'mitdb'], help="Dataset type to use (e.g. afdb, ltafdb, large_scale)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model file (e.g. best_model.pt)")
    parser.add_argument("--fs", type=int, default=250, help="Original sampling frequency of data")
    parser.add_argument("--split", type=float, default=0.2, help="Fraction of data to use as test set (0 = use all)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")

    args = parser.parse_args()

    run_evaluation(args)
    print("Evaluation complete.")
