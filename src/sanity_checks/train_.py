import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from collections import Counter

# Assuming these are in your project structure.
# You might need to adjust import paths if they are not directly in the same directory as this script.
try:
    from models.resnet1d import resnet50_1d
    from processing.data_extract import Preprocess, create_dataset_from_paths
    from processing.pytorch_dataloader import ECGDataset
except ImportError as e:
    print(f"ERROR: Could not import necessary modules. Make sure 'models', 'processing' directories and their contents are correctly placed.")
    print(f"Error details: {e}")
    exit() # Exit if core modules can't be imported

print("--- Starting Sanity Checks ---")
print(f"Current working directory: {os.getcwd()}")

# ---------------- CONFIG SANITY CHECKS ----------------
print("\n--- [1/7] Configuration Checks ---")
DATA_DIR = "/Users/amiteshpatel/Desktop/Sophro/data/data/afdb_data"
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 0.001
SEED = 42

print(f"DATA_DIR: {DATA_DIR}")
if os.path.exists(DATA_DIR):
    print(f"STATUS: DATA_DIR '{DATA_DIR}' exists.")
    if len(os.listdir(DATA_DIR)) > 0:
        print(f"STATUS: DATA_DIR is not empty. Contains {len(os.listdir(DATA_DIR))} items.")
    else:
        print("WARNING: DATA_DIR is empty.")
else:
    print(f"ERROR: DATA_DIR '{DATA_DIR}' does NOT exist. Please check the path.")
    exit() # Exit if data directory is not found

print(f"BATCH_SIZE: {BATCH_SIZE}")
print(f"EPOCHS: {EPOCHS}")
print(f"LEARNING_RATE: {LEARNING_RATE}")
print(f"SEED: {SEED}")

# Device check
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Detected device: {device}")
if device.type == "mps":
    print("STATUS: PyTorch is configured to use Apple Silicon (MPS).")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"STATUS: CUDA is available. Using device: {device}")
else:
    print("WARNING: Neither MPS nor CUDA is available. Using CPU. Training might be slow.")
print(f"Is MPS enabled? {torch.backends.mps.is_available()}")
print(f"Is CUDA enabled? {torch.cuda.is_available()}")


# ---------------- PATHS & SPLIT SANITY CHECKS ----------------
print("\n--- [2/7] Data Path and Split Checks ---")
paths = [os.path.join(DATA_DIR, f[:-4]) for f in os.listdir(DATA_DIR) if f.endswith('.atr')]
print(f"Found {len(paths)} unique .atr files in DATA_DIR.")
if len(paths) == 0:
    print("ERROR: No .atr files found. Ensure data is correctly placed.")
    exit()

np.random.seed(SEED)
np.random.shuffle(paths)

train_val_paths, test_paths = train_test_split(paths, test_size=0.2, random_state=SEED)
train_paths, val_paths = train_test_split(train_val_paths, test_size=0.2, random_state=SEED)

print(f"Total paths: {len(paths)}")
print(f"Train+Val paths: {len(train_val_paths)} ({len(train_val_paths)/len(paths):.2%})")
print(f"Test paths: {len(test_paths)} ({len(test_paths)/len(paths):.2%})")
print(f"Train paths: {len(train_paths)} ({len(train_paths)/len(paths):.2%})")
print(f"Validation paths: {len(val_paths)} ({len(val_paths)/len(paths):.2%})")

# Check for overlaps (should be none)
print("Checking for path overlaps between sets (should be 0):")
print(f"Train & Val overlap: {len(set(train_paths).intersection(set(val_paths)))}")
print(f"Train & Test overlap: {len(set(train_paths).intersection(set(test_paths)))}")
print(f"Val & Test overlap: {len(set(val_paths).intersection(set(test_paths)))}")


# ---------------- PREPROCESSING SANITY CHECKS ----------------
print("\n--- [3/7] Preprocessing Checks ---")
try:
    preprocessor = Preprocess(fs=250, target_fs=128)
    print(f"Preprocessor initialized. Original FS: {preprocessor.fs}, Target FS: {preprocessor.target_fs}")
except Exception as e:
    print(f"ERROR: Failed to initialize Preprocess: {e}")
    exit()

# Try processing a single file to check
if len(train_paths) > 0:
    print("Attempting to preprocess a sample file from training set...")
    try:
        sample_path = train_paths[0]
        # create_dataset_from_paths expects a list of paths
        X_sample, y_sample = create_dataset_from_paths([sample_path], preprocessor)
        print(f"Successfully processed a sample file: {sample_path}")
        print(f"Sample X shape: {X_sample.shape}")
        print(f"Sample y shape: {y_sample.shape}")
        if X_sample.shape[0] > 0:
            print(f"Sample preprocessed data min: {X_sample.min():.4f}, max: {X_sample.max():.4f}")
            print(f"Sample preprocessed data mean: {X_sample.mean():.4f}, std: {X_sample.std():.4f}")
        else:
            print("WARNING: Sample preprocessing resulted in empty data.")
    except Exception as e:
        print(f"ERROR: Failed to preprocess a sample file: {e}")
        # Don't exit here, as full dataset creation might still work if it's an edge case
else:
    print("WARNING: No training paths available to test preprocessing.")

print("Creating full datasets (this might take a while)...")
try:
    X_train, y_train = create_dataset_from_paths(train_paths, preprocessor)
    X_val, y_val = create_dataset_from_paths(val_paths, preprocessor)
    X_test, y_test = create_dataset_from_paths(test_paths, preprocessor)

    print("\n--- [4/7] Dataset Sizes and Class Distribution Checks ---")
    print(f"Training set size: {len(X_train)}, Class distribution: {Counter(y_train)}")
    print(f"Validation set size: {len(X_val)}, Class distribution: {Counter(y_val)}")
    print(f"Test set size: {len(X_test)}, Class distribution: {Counter(y_test)}")

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print("ERROR: One or more datasets are empty after creation. Check `create_dataset_from_paths` logic.")
        exit()

    # Check data shapes and types after creation
    print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
    if X_train.ndim != 3 or X_train.shape[1] != 2: # Expected (num_samples, channels, length)
        print(f"WARNING: X_train has unexpected dimensions. Expected (N, 2, L), got {X_train.shape}")
    if y_train.ndim != 1: # Expected (num_samples,)
        print(f"WARNING: y_train has unexpected dimensions. Expected (N,), got {y_train.shape}")

    # Plot sample data
    print("\n--- [PLOT] Sample Preprocessed ECG Data (First channel of first sample) ---")
    if len(X_train) > 0:
        plt.figure(figsize=(12, 4))
        plt.plot(X_train[0, 0, :]) # Plot first channel of first sample
        plt.title(f"Sample Preprocessed ECG (Class: {y_train[0]})")
        plt.xlabel("Time points")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()
    else:
        print("No training data to plot.")

except Exception as e:
    print(f"ERROR: Failed to create datasets: {e}")
    exit()

# ---------------- DATALOADER SANITY CHECKS ----------------
print("\n--- [5/7] DataLoader Checks ---")
try:
    train_ds = ECGDataset(X_train, y_train)
    val_ds = ECGDataset(X_val, y_val)
    test_ds = ECGDataset(X_test, y_test)

    print(f"ECGDataset created. Train len: {len(train_ds)}, Val len: {len(val_ds)}, Test len: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # num_workers=0 for better debugging
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

    print(f"DataLoaders initialized with batch_size={BATCH_SIZE}.")
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Validation loader batches: {len(val_loader)}")
    print(f"Test loader batches: {len(test_loader)}")

    # Test iterating through one batch
    print("Attempting to load one batch from train_loader...")
    try:
        data_batch, labels_batch = next(iter(train_loader))
        print(f"Successfully loaded a batch.")
        print(f"Batch data shape: {data_batch.shape} (Expected: [BATCH_SIZE, Channels, Length])")
        print(f"Batch labels shape: {labels_batch.shape} (Expected: [BATCH_SIZE])")
        print(f"Batch data type: {data_batch.dtype}, Batch labels type: {labels_batch.dtype}")
        if data_batch.shape[0] != BATCH_SIZE and data_batch.shape[0] != len(train_ds) % BATCH_SIZE:
             print(f"WARNING: Batch size mismatch. Expected {BATCH_SIZE}, got {data_batch.shape[0]} for a full batch.")
        if data_batch.ndim != 3 or data_batch.shape[1] != 2:
            print(f"WARNING: Data batch has unexpected dimensions. Expected (B, 2, L), got {data_batch.shape}")
        if not torch.is_floating_point(data_batch):
            print("WARNING: Data batch is not float. Model might expect float input.")
        if not torch.is_tensor(labels_batch) or labels_batch.dtype != torch.long:
            print("WARNING: Labels batch is not a long tensor. CrossEntropyLoss expects long type.")

        # Check if samples are being read correctly (e.g., not all zeros)
        if data_batch.sum() == 0:
            print("CRITICAL WARNING: All values in the first batch are zero! Data loading issue suspected.")

    except Exception as e:
        print(f"ERROR: Failed to load a batch from train_loader: {e}")
        exit()

except Exception as e:
    print(f"ERROR: Failed to initialize DataLoaders: {e}")
    exit()

# ---------------- MODEL & OPTIMIZER SANITY CHECKS ----------------
print("\n--- [6/7] Model and Optimizer Checks ---")
try:
    model = resnet50_1d(in_channels=2, num_classes=2).to(device)
    print(f"Model '{model.__class__.__name__}' initialized successfully.")
    print(f"Model on device: {next(model.parameters()).device}")

    criterion = nn.CrossEntropyLoss()
    print(f"Criterion (Loss Function): {criterion.__class__.__name__} initialized.")

    optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    print(f"Optimizer: {optimizer.__class__.__name__} initialized with LR={LEARNING_RATE}.")

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    print(f"Scheduler: {scheduler.__class__.__name__} initialized.")

    # Check number of model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters in model: {total_params:,}")

    # Test a forward pass with a dummy tensor
    print("Attempting a dummy forward pass...")
    try:
        dummy_input = torch.randn(BATCH_SIZE, 2, X_train.shape[2]).to(device) # Adjust 2nd dim for channels, 3rd for sequence length
        dummy_output = model(dummy_input)
        print(f"Dummy forward pass successful. Output shape: {dummy_output.shape}")
        if dummy_output.shape[0] != BATCH_SIZE or dummy_output.shape[1] != 2: # Expected (BATCH_SIZE, num_classes)
            print(f"WARNING: Model output shape mismatch. Expected ({BATCH_SIZE}, 2), got {dummy_output.shape}")
        # Check if output contains NaNs or Infs
        if torch.isnan(dummy_output).any() or torch.isinf(dummy_output).any():
            print("CRITICAL WARNING: Model output contains NaNs or Infs in dummy forward pass. Check model architecture/initialization.")
    except Exception as e:
        print(f"ERROR: Dummy forward pass failed: {e}")
        exit()

except Exception as e:
    print(f"ERROR: Failed to initialize Model/Optimizer/Criterion/Scheduler: {e}")
    exit()

# ---------------- FINAL CHECK (Ready for Training) ----------------
print("\n--- [7/7] Pre-training Readiness Check ---")
print("All major components appear to be initialized and functioning as expected.")
print("This script has performed sanity checks on configuration, data paths, preprocessing, datasets, dataloaders, model, optimizer, and scheduler.")
print("You are now ready to proceed with the main training loop.")
print("--- Sanity Checks Complete ---")