import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
from datetime import datetime

import wandb

from src.models.resnet1d import resnet50_1d
from src.processing.data_extract import Preprocess, create_dataset_from_paths
from src.processing.pytorch_dataloader import ECGDataset


# ---------------- TRAINING FUNCTIONS ----------------


def get_model_path(config, base_dir="model_files"):
    from datetime import datetime

    # Folder for the model name
    model_folder = os.path.join(base_dir, config.model)
    os.makedirs(model_folder, exist_ok=True)

    # Filename includes config details
    dataset_name = os.path.basename(config.data_dir.rstrip("/"))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = (
        f"{dataset_name}_fs{config.fs}_bs{config.batch_size}_"
        f"lr{config.learning_rate}_ep{config.epochs}_{timestamp}.pt"
    )
    model_path = os.path.join(model_folder, filename)
    return model_path


def prepare_dataloaders(data_dir, fs, batch_size, seed):
    paths = [os.path.join(data_dir, f[:-4]) for f in os.listdir(data_dir) if f.endswith('.atr')]
    np.random.seed(seed)
    np.random.shuffle(paths)

    train_val_paths, test_paths = train_test_split(paths, test_size=0.2, random_state=seed)
    train_paths, val_paths = train_test_split(train_val_paths, test_size=0.2, random_state=seed)

    preprocessor = Preprocess(fs=fs, target_fs=128)

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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def train_model(config):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    wandb.init(project="ecg-afib-detection", config=config)
    config = wandb.config

    train_loader, val_loader, _ = prepare_dataloaders(
        config.data_dir, config.fs, config.batch_size, config.seed
    )

    model = resnet50_1d(in_channels=2, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3) # this patientce is for the scheduler, not early stopping

    wandb.watch(model, log="all")

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")
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

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]['lr']
        })

        print(f"[Epoch {epoch + 1:02d}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_path = get_model_path(config)
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")
            wandb.run.summary["best_val_loss"] = best_val_loss
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("Early stopping triggered.")
                break

    wandb.finish()


# ---------------- MAIN (ARGPARSE ENTRY) ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet1D ECG model")

    parser.add_argument("--data_dir", type=str, required=True, help="Path to ECG dataset directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--fs", type=int, default=250, help="Original sampling frequency")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    CONFIG = {
        "data_dir": args.data_dir,
        "epochs": args.epochs,
        "fs": args.fs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "seed": args.seed,
        "model": "resnet50_1d",
        "optimizer": "SGD",
        "scheduler": "ReduceLROnPlateau"
    }

    train_model(CONFIG)
