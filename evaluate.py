import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from collections import Counter

from src.models.resnet1d import resnet50_1d
from src.processing.data_extract import Preprocess, create_dataset_from_paths
from src.processing.pytorch_dataloader import ECGDataset

import os

# ---------------- CONFIG ----------------
DATA_DIR = "/Users/amiteshpatel/Desktop/Sophro/data/data/afdb_data"
BATCH_SIZE = 32
SEED = 42
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------------- SPLIT TEST PATHS ----------------
paths = [os.path.join(DATA_DIR, f[:-4]) for f in os.listdir(DATA_DIR) if f.endswith('.atr')]
np.random.seed(SEED)
np.random.shuffle(paths)
_, test_paths = np.split(paths, [int(0.8 * len(paths))])  # 20% test split

# ---------------- PREPROCESS TEST SET ----------------
preprocessor = Preprocess(fs=250, target_fs=128)
print("Creating test set...")
X_test, y_test = create_dataset_from_paths(test_paths, preprocessor)
print(f"Test set size: {len(X_test)}, Class distribution: {Counter(y_test)}")

test_ds = ECGDataset(X_test, y_test)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ---------------- LOAD MODEL ----------------
model = resnet50_1d(in_channels=2, num_classes=2).to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

# ---------------- EVALUATION ----------------
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        output = model(x)
        preds = output.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

# ---------------- METRICS ----------------
print("Classification Report:")
print(classification_report(all_labels, all_preds, digits=4))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
