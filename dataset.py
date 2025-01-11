import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from PIL import Image
from autoaugment import SyncTransform

class MyDataset(Dataset):
    def __init__(self, cover_path, stego_path, mode, transform):
        self.cover_path = cover_path
        self.cover = os.listdir(self.cover_path)
        self.cover.sort(key=lambda x: int(x[: -4]))

        self.stego_path = stego_path
        self.stego = os.listdir(stego_path)
        self.stego.sort(key=lambda x: int(x[: -4]))

        self.total_size_cover = len(self.cover)
        self.total_size_stego = len(self.stego)

        self.mode = mode
        if self.mode == "train":
            self.cover = self.cover[: int(0.8 * self.total_size_cover)]
            self.stego = self.stego[: int(0.8 * self.total_size_stego)]
        else:
            self.cover = self.cover[int(0.8 * self.total_size_cover):]
            self.stego = self.stego[int(0.8 * self.total_size_stego):]

        self.sync_transform = SyncTransform(transform)
        self.data_size = len(self.cover)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        cover_img = Image.open(os.path.join(self.cover_path, self.cover[index])).convert("L")
        stego_img = Image.open(os.path.join(self.stego_path, self.stego[index])).convert("L")
        label1 = torch.tensor(0, dtype=torch.long)
        label2 = torch.tensor(1, dtype=torch.long)
        if self.sync_transform:
            cover_img = self.sync_transform(cover_img)
            self.sync_transform.random_state = None
            stego_img = self.sync_transform(stego_img)
            self.sync_transform.random_state = None
            sample = {
                "cover": cover_img,
                "stego": stego_img
            }
        sample["label"] = [label1, label2]
        return sample
