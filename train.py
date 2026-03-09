# =============================================================================
#  Malaria Cell Classifier — MLR_DTC Custom CNN
#  Run this ONCE to train and save the model.
#  Output: model/malaria_model.pt
# =============================================================================

import os
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.nn import LeakyReLU
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "data/cell_images"
MODEL_OUT  = "model/malaria_model.pt"
IMG_SIZE   = 128
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 0.01
VAL_SPLIT  = 0.2
SEED       = 42

# ── Model definition ──────────────────────────────────────────────────────────
class MLR_DTC(nn.Module):
    def __init__(self, input: int, hidden: int, output: int) -> None:
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input, hidden, 3),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(),
            nn.Conv2d(hidden, hidden, 3),
            nn.BatchNorm2d(hidden),
            LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(),
            nn.Conv2d(hidden, hidden, 3),
            nn.BatchNorm2d(hidden),
            LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.Classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 29 * 29, output),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.Classifier(x)
        return x


# ── Main — required on Windows to avoid multiprocessing errors ────────────────
if __name__ == '__main__':

    torch.manual_seed(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device : {DEVICE}")

    # ── Transforms ────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # ── Dataset ───────────────────────────────────────────────────────────────
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    class_names  = full_dataset.classes
    n_total      = len(full_dataset)
    n_val        = int(n_total * VAL_SPLIT)
    n_train      = n_total - n_val

    print(f"[INFO] Classes : {class_names}")
    print(f"[INFO] Total   : {n_total}  |  Train: {n_train}  |  Val: {n_val}")

    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    # num_workers=0 is required on Windows
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = MLR_DTC(3, 20, 1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # ── Training loop ─────────────────────────────────────────────────────────
    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc     = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*50}\n  Epoch {epoch}/{EPOCHS}\n{'='*50}")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss    = 0.0
            running_correct = 0
            total           = 0

            for inputs, labels in loader:
                inputs = inputs.to(DEVICE)
                labels = labels.float().unsqueeze(1).to(DEVICE)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                    preds   = torch.round(torch.sigmoid(outputs))

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss    += loss.item() * inputs.size(0)
                running_correct += (preds == labels).sum().item()
                total           += inputs.size(0)

            epoch_loss = running_loss / total
            epoch_acc  = running_correct / total

            print(f"  {phase.upper():5s} — Loss: {epoch_loss:.4f}  Acc: {epoch_acc*100:.2f}%")

            if phase == "train":
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc)
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc)
                scheduler.step(epoch_loss)

                if epoch_acc > best_acc:
                    best_acc     = epoch_acc
                    best_weights = copy.deepcopy(model.state_dict())
                    print(f"  ✓ New best val acc: {best_acc*100:.2f}%")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("model", exist_ok=True)
    torch.save({
        "model_state_dict": best_weights,
        "class_names":      class_names,
        "img_size":         IMG_SIZE,
        "val_accuracy":     best_acc,
    }, MODEL_OUT)

    print(f"\n✅  Model saved → {MODEL_OUT}")
    print(f"    Best val accuracy: {best_acc*100:.2f}%")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, EPOCHS + 1)

    axes[0].plot(epochs_range, history["train_loss"], label="Train", marker='o')
    axes[0].plot(epochs_range, history["val_loss"],   label="Val",   marker='o')
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(epochs_range, [a*100 for a in history["train_acc"]], label="Train", marker='o')
    axes[1].plot(epochs_range, [a*100 for a in history["val_acc"]],   label="Val",   marker='o')
    axes[1].set_title("Accuracy (%)"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    plt.tight_layout()
    plt.savefig("model/training_curves.png", dpi=120)
    print("📊  Training curves → model/training_curves.png")
    plt.show()
