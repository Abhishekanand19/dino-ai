"""
STEP 2 — Training
==================
Train the DinoNet model on your collected gameplay data.

HOW TO USE:
  python training/train.py

  After training, the model is saved to: models/dino_model.pth
  Training plots are saved to:           models/training_plot.png
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.model import DinoNet, DinoDataset, get_device

# ─── CONFIG ───────────────────────────────────────────────────────────────────

LABELS_FILE  = "data/labels.csv"
MODEL_DIR    = "models"
MODEL_PATH   = "models/dino_model.pth"
PLOT_PATH    = "models/training_plot.png"

EPOCHS       = 30
BATCH_SIZE   = 64
LEARNING_RATE = 0.001
TRAIN_SPLIT  = 0.85   # 85% train, 15% validation

# ──────────────────────────────────────────────────────────────────────────────


def make_weighted_sampler(dataset):
    """
    Since most frames are 'do nothing', we oversample jump/duck
    so the model doesn't just learn to never act.
    """
    df = dataset.df
    class_counts = df["action"].value_counts().to_dict()
    total = len(df)

    # Weight = inverse frequency
    weights = []
    for _, row in df.iterrows():
        action = int(row["action"])
        w = total / (len(class_counts) * class_counts[action])
        weights.append(w)

    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = get_device()

    # ── Load & split data ──────────────────────────────────────────────────────
    print(f"\nLoading data from {LABELS_FILE}...")
    full_df = pd.read_csv(LABELS_FILE)

    # Remove frames where the image file doesn't actually exist
    full_df = full_df[full_df["frame_path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"Valid frames: {len(full_df)}")

    if len(full_df) < 500:
        print("\n⚠️  Warning: Less than 500 frames detected.")
        print("    For best results, collect at least 2000 frames (3-4 play sessions).")
        print("    Continuing anyway...\n")

    # Split by index to keep temporal order within sessions
    train_idx, val_idx = train_test_split(
        full_df.index, test_size=1 - TRAIN_SPLIT, shuffle=True, random_state=42
    )

    train_df = full_df.loc[train_idx].reset_index(drop=True)
    val_df   = full_df.loc[val_idx].reset_index(drop=True)

    # Save splits temporarily
    train_df.to_csv("data/train_split.csv", index=False)
    val_df.to_csv("data/val_split.csv", index=False)

    # ── Datasets & DataLoaders ─────────────────────────────────────────────────
    print("\n── Training set ──")
    train_dataset = DinoDataset("data/train_split.csv")
    print("── Validation set ──")
    val_dataset   = DinoDataset("data/val_split.csv")

    sampler = make_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)

    # ── Model, loss, optimizer ─────────────────────────────────────────────────
    model = DinoNet(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠  Model parameters: {total_params:,}")

    # ── Training loop ──────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\n🏋️  Training for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [train]", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += images.size(0)

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item() * images.size(0)
                preds        = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += images.size(0)

        scheduler.step()

        # ── Metrics ──
        t_loss = train_loss / train_total
        t_acc  = 100 * train_correct / train_total
        v_loss = val_loss / val_total
        v_acc  = 100 * val_correct / val_total

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        print(f"Epoch {epoch+1:02d}/{EPOCHS}  |  "
              f"Train: loss={t_loss:.4f} acc={t_acc:.1f}%  |  "
              f"Val: loss={v_loss:.4f} acc={v_acc:.1f}%")

        # Save best model
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"             💾  New best saved! ({v_acc:.1f}%)")

    print(f"\n✅  Training complete. Best validation accuracy: {best_val_acc:.1f}%")
    print(f"    Model saved to: {MODEL_PATH}\n")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("DinoNet Training", fontsize=14)

    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"],   label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"],   label="Val")
    ax2.set_title("Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"📈  Training plot saved to: {PLOT_PATH}")


if __name__ == "__main__":
    train()
