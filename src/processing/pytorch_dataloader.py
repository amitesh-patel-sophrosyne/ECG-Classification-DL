import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std


class ECGDataset(Dataset):
    def __init__(self, signals, labels, transform=None):
        """
        :param signals: numpy array of shape (N, 2, 1280)
        :param labels: numpy array of shape (N,)
        :param transform: Optional PyTorch-style transform to apply
        """
        self.signals = signals.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = self.signals[idx]
        y = self.labels[idx]

        x = torch.from_numpy(x)
        y = torch.tensor(y)

        # Optional transform (e.g., augmentations)
        if self.transform:
            x = self.transform(x)

        return x, y
    

if __name__ == "__main__":

    # X, y = np.random.rand(100, 2, 1280), np.random.randint(0, 2, size=100) 

    # transform = AddGaussianNoise(std=0.02)

    dataset = ECGDataset(signals=X, labels=y, transform=None) # in the paper it is none 

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch_x, batch_y in dataloader:
        print(batch_x.shape)  # torch.Size([32, 2, 1280])
        print(batch_y.shape)  # torch.Size([32])
        break
