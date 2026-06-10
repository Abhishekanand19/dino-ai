"""
Model & Dataset definitions
============================
Shared between training and inference.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import os


# ─── DATASET ──────────────────────────────────────────────────────────────────

class DinoDataset(Dataset):
    """
    Loads (frame_image, action_label) pairs from the CSV logged during collection.
    Each frame is an 80x80 grayscale image.
    Labels: 0=nothing, 1=jump, 2=duck
    """

    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # Print class balance so you can see if data is skewed
        counts = self.df["action"].value_counts().sort_index()
        total = len(self.df)
        print(f"\n📊  Dataset loaded: {total} frames")
        labels = {0: "nothing", 1: "jump", 2: "duck"}
        for action, count in counts.items():
            pct = 100 * count / total
            print(f"     {labels[action]:>8}: {count:>6} frames ({pct:.1f}%)")
        print()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["frame_path"]
        action = int(row["action"])

        # Load grayscale frame
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Return zeros if file missing — shouldn't happen normally
            img = np.zeros((80, 80), dtype=np.float32)
        else:
            img = img.astype(np.float32) / 255.0

        # Shape: (1, 80, 80) — single channel
        img = np.expand_dims(img, axis=0)
        img_tensor = torch.tensor(img, dtype=torch.float32)

        return img_tensor, action


# ─── MODEL ────────────────────────────────────────────────────────────────────

class DinoNet(nn.Module):
    """
    Lightweight CNN for classifying dino game frames.
    Input:  (batch, 1, 80, 80) grayscale image
    Output: (batch, 3) logits for [nothing, jump, duck]

    Designed to run fast on CPU — no fancy layers.
    """

    def __init__(self, num_classes=3):
        super(DinoNet, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 80x80 → 40x40
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 40x40 → 20x20
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 20x20 → 10x10
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 10x10 → 5x5
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(64 * 5 * 5, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_device():
    """Returns best available device. Falls back to CPU gracefully."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("💻  Using CPU (this is fine — model is lightweight)")
    return device
