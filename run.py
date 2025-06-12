import os
import subprocess

# Mapping from folder names to dataset tags
data_dirs = {
    "afdb_data": "afdb",
    "mitbih_data": "mitdb",
    "ltafdb_data": "ltafdb"
}

# Sampling frequency for each dataset
fs_map = {
    "afdb": 250,
    "mitdb": 360,
    "ltafdb": 128
}

# List of model architectures
models = ['resnet18_1d', 'resnet34_1d', 'resnet50_1d']

# Base directory for model files
model_base_dir = "model_files"

# Corrected model path finder
def find_model_path(dataset, model):
    model_root_dir = os.path.join(model_base_dir, dataset, model)
    if not os.path.exists(model_root_dir):
        return None
    for root, _, files in os.walk(model_root_dir):
        if "best_model.pt" in files:
            return os.path.join(root, "best_model.pt")
    return None

# Main loop
for data_dir, data_used in data_dirs.items():
    for model in models:
        for test_data_dir, test_data_used in data_dirs.items():
            model_path = find_model_path(data_dir, model)
            if model_path is None:
                print(f"Skipping: Model not found for {data_dir}, {model}")
                continue

            split = 0.2 if data_used == test_data_used else 0
            fs = fs_map[test_data_used]

            cmd = [
                "python", "evaluate.py",
                "--data_dir", os.path.join("data",test_data_dir),
                "--model", model,
                "--data_used", test_data_used,
                "--model_path", model_path,
                "--fs", str(fs),
                "--split", str(split),
                "--batch_size", "32"
            ]

            print(f"\nRunning: {' '.join(cmd)}")
            subprocess.run(cmd)
