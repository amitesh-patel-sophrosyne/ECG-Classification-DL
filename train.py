import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter

from src.models.resnet1d import resnet50_1d
from src.processing.data_extract import Preprocess, create_dataset_from_paths
from src.processing.pytorch_dataloader import ECGDataset

# ---------------- CONFIG ----------------
DATA_DIR = "/Users/amiteshpatel/Desktop/Sophro/data/data/afdb_data"
BATCH_SIZE = 32
EPOCHS = 20
PATIENCE = 10
LEARNING_RATE = 0.001
SEED = 42
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------- SPLIT PATHS ----------------
paths = [os.path.join(DATA_DIR, f[:-4]) for f in os.listdir(DATA_DIR) if f.endswith('.atr')]
np.random.seed(SEED)
np.random.shuffle(paths)
train_val_paths, test_paths = train_test_split(paths, test_size=0.2, random_state=SEED)
train_paths, val_paths = train_test_split(train_val_paths, test_size=0.2, random_state=SEED)

# ---------------- PREPROCESSING ----------------
preprocessor = Preprocess(fs=250, target_fs=128)

print("Creating training set...")
X_train, y_train = create_dataset_from_paths(train_paths, preprocessor)
print("Creating validation set...")
X_val, y_val = create_dataset_from_paths(val_paths, preprocessor)
print("Creating test set...")
X_test, y_test = create_dataset_from_paths(test_paths, preprocessor)

print(f"Training set size: {len(X_train)}, Class distribution: {Counter(y_train)}")
print(f"Validation set size: {len(X_val)}, Class distribution: {Counter(y_val)}")
print(f"Test set size: {len(X_test)}, Class distribution: {Counter(y_test)}")

train_ds = ECGDataset(X_train, y_train)
val_ds = ECGDataset(X_val, y_val)
test_ds = ECGDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ---------------- MODEL & TRAINING SETUP ----------------
model = resnet50_1d(in_channels=2, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# ---------------- TRAINING LOOP ----------------
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    model.train()
    train_loss, correct = 0, 0

    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)
        correct += (output.argmax(1) == y).sum().item()

    train_loss /= len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)

    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            val_loss += loss.item() * x.size(0)
            correct += (output.argmax(1) == y).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = correct / len(val_loader.dataset)
    scheduler.step(val_loss)

    print(f"[Epoch {epoch+1:02d}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break
