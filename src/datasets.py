import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from glob import glob

# すべての変換を定義
transform = transforms.Compose([
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomRotation(degrees=(-180, 180)),
    transforms.RandomCrop(32, padding=(4, 4, 4, 4), padding_mode='constant'),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.8, scale=(0.02, 0.33), ratio=(0.3, 3.3))
])


class ThingsMEGDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "data-omni", transform=None) -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"

        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))
        self.transform = transform  # 変換を追加

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        # データをロード
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        X = torch.from_numpy(np.load(X_path))

        # 変換を適用
        if self.transform:
            X = self.transform(X)

        # subject_idxをロード
        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = torch.from_numpy(np.load(subject_idx_path))

        if self.split in ["train", "val"]:
            # ラベルをロード
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = torch.from_numpy(np.load(y_path))

            return X, y, subject_idx
        else:
            return X, subject_idx

    @property
    def num_channels(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]

    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]