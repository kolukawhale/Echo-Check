"""
train_conv2d.py — Training script for the Conv2D Autoencoder.

Uses the Echo-Check train/test splits (data/splits/pump_id_XX_train.npy).
Trains exclusively on normal samples. Evaluates on the test set after training.
Saves the full autoencoder and encoder-only checkpoints for Phase 3 ONNX export.

Usage:
    cd src
    python train_conv2d.py
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from pathlib import Path

# ── Config
_ROOT = Path(__file__).parent.parent

TRAIN_NPYS = [
    _ROOT / "data/splits/pump_id_00_train.npy",
    _ROOT / "data/splits/pump_id_02_train.npy",
    _ROOT / "data/splits/pump_id_04_train.npy",
    _ROOT / "data/splits/pump_id_06_train.npy",
]
TEST_NPYS = [
    _ROOT / "data/splits/pump_id_00_test.npy",
    _ROOT / "data/splits/pump_id_02_test.npy",
    _ROOT / "data/splits/pump_id_04_test.npy",
    _ROOT / "data/splits/pump_id_06_test.npy",
]
LABEL_NPYS = [
    _ROOT / "data/splits/pump_id_00_test_labels.npy",
    _ROOT / "data/splits/pump_id_02_test_labels.npy",
    _ROOT / "data/splits/pump_id_04_test_labels.npy",
    _ROOT / "data/splits/pump_id_06_test_labels.npy",
]

OUTPUT_DIR    = Path(__file__).parent.parent / "models" / "conv2d"
EMBEDDING_DIM = 128
TARGET_FREQ   = 128
TARGET_TIME   = 432

EPOCHS        = 100
BATCH_SIZE    = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 1e-5
LR_PATIENCE   = 5
LR_FACTOR     = 0.5
THRESHOLD_PCT = 95
# ────────────────────────────


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Dataset
class SpectrogramDataset(Dataset):
    """
    Loads .npy spectrograms of shape [N, 128, 431].
    Pads time dim to 432 (divisible by 16 for 4 stride-2 conv blocks).
    Adds channel dim: [N, 128, 431] → [N, 1, 128, 432].
    """
    def __init__(self, spectrograms: np.ndarray, target_time: int = TARGET_TIME):
        n, freq, t = spectrograms.shape
        if t < target_time:
            spectrograms = np.pad(
                spectrograms,
                ((0, 0), (0, 0), (0, target_time - t)),
                mode="constant", constant_values=0,
            )
        elif t > target_time:
            spectrograms = spectrograms[:, :, :target_time]

        self.data = torch.tensor(spectrograms, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]   # [1, 128, 432]


# ── AutoEncoder Architecture
class Encoder(nn.Module):
    """
    4 Conv2D blocks with stride=2 compress spatial dimensions:
        [1, 128, 432] → [32, 64, 216] → [64, 32, 108] → [128, 16, 54] → [256, 8, 27]
    AdaptiveAvgPool collapses to [256, 1, 1].
    Linear bottleneck: 256 → embedding_dim.
    """
    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1,   32,  3, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32,  64,  3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64,  128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)           # [B, embedding_dim]


class Decoder(nn.Module):
    """
    Linear expands embedding to spatial map.
    4 ConvTranspose2D blocks upsample back to [1, 128, 432].
    Bilinear interpolation ensures exact output size.
    Sigmoid constrains output to [0, 1].
    """
    _H = 8    # TARGET_FREQ // 16
    _W = 27   # TARGET_TIME  // 16

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, 256 * self._H * self._W)
        self.deconv_blocks = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,  1,   4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z).view(z.size(0), 256, self._H, self._W)
        x = self.deconv_blocks(x)
        return nn.functional.interpolate(
            x, size=(TARGET_FREQ, TARGET_TIME), mode="bilinear", align_corners=False
        )


class CNNAutoencoder(nn.Module):
    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(embedding_dim)

    def forward(self, x):
        z    = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def get_embedding(self, x):
        return self.encoder(x)


# ── Training logic
def load_train_data(npy_paths: list) -> np.ndarray:
    arrays = [np.load(p) for p in npy_paths if Path(p).exists()]
    data   = np.concatenate(arrays, axis=0)
    print(f"Loaded {len(arrays)} train file(s) — {len(data)} normal spectrograms")
    return data


def train(model, dataloader, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE, factor=LR_FACTOR
    )

    losses = []
    print(f"\n{'Epoch':>6} | {'Loss':>12} | {'LR':>10} | {'Time':>7}")
    print("-" * 45)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, epoch_loss, n_batches = time.time(), 0.0, 0

        for batch in dataloader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = criterion(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        scheduler.step(avg_loss)
        lr = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6} | {avg_loss:>12.6f} | {lr:>10.6f} | {time.time()-t0:>5.1f}s")

    print(f"\nTraining complete. Final loss: {losses[-1]:.6f}")
    return losses


# ── Evaluation
@torch.no_grad()
def compute_errors(model, spectrograms: np.ndarray, device) -> np.ndarray:
    """Per-sample MSE reconstruction error."""
    model.eval()
    dataset    = SpectrogramDataset(spectrograms)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    errors = []
    for batch in dataloader:
        batch = batch.to(device)
        recon, _ = model(batch)
        mse = ((recon - batch) ** 2).mean(dim=(1, 2, 3))
        errors.append(mse.cpu().numpy())
    return np.concatenate(errors)


def evaluate_all_ids(model, device):
    """Evaluates per machine ID and prints a summary."""
    print(f"\n{'='*55}")
    print("  EVALUATION ACROSS ALL MACHINE IDs")
    print(f"{'='*55}")

    all_aucs = []

    for test_npy, label_npy in zip(TEST_NPYS, LABEL_NPYS):
        if not Path(test_npy).exists() or not Path(label_npy).exists():
            continue

        machine_id = Path(test_npy).name.replace("pump_", "").replace("_test.npy", "")
        specs  = np.load(test_npy)
        labels = np.load(label_npy)

        errors = compute_errors(model, specs, device)

        normal_errors = errors[labels == 0]
        threshold     = np.percentile(normal_errors, THRESHOLD_PCT)
        predictions   = (errors > threshold).astype(int)

        auc  = roc_auc_score(labels, errors)
        f1   = f1_score(labels, predictions, zero_division=0)
        prec = precision_score(labels, predictions, zero_division=0)
        rec  = recall_score(labels, predictions, zero_division=0)
        all_aucs.append(auc)

        print(f"\n── {machine_id} ──────────────────────────────────────")
        print(f"  Normal mean   : {normal_errors.mean():.6f}")
        print(f"  Abnormal mean : {errors[labels==1].mean():.6f}")
        print(f"  Threshold     : {threshold:.6f}  ({THRESHOLD_PCT}th pct)")
        print(f"  AUC-ROC       : {auc:.4f}")
        print(f"  Precision     : {prec:.4f}")
        print(f"  Recall        : {rec:.4f}")
        print(f"  F1 Score      : {f1:.4f}")

    print(f"\n{'='*55}")
    print(f"  Average AUC : {np.mean(all_aucs):.4f}")
    print(f"{'='*55}")
    return all_aucs


# ── Main
def main():
    device = get_device()
    print(f"Device : {device}")

    # Output directory
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    train_specs = load_train_data(TRAIN_NPYS)
    train_ds    = SpectrogramDataset(train_specs)
    train_dl    = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device.type != "cpu"),
    )

    print(f"Training samples : {len(train_ds)}")
    print(f"Batches/epoch    : {len(train_dl)}")
    print(f"Embedding dim    : {EMBEDDING_DIM}")
    print(f"Epochs           : {EPOCHS}\n")

    # Build model
    model = CNNAutoencoder(embedding_dim=EMBEDDING_DIM).to(device)
    total  = sum(p.numel() for p in model.parameters())
    enc_p  = sum(p.numel() for p in model.encoder.parameters())
    dec_p  = sum(p.numel() for p in model.decoder.parameters())
    print(f"Parameters — Total: {total:,}  Encoder: {enc_p:,}  Decoder: {dec_p:,}\n")

    # Train
    losses = train(model, train_dl, device)

    # Save checkpoints
    ae_path  = out_dir / "autoencoder.pth"
    enc_path = out_dir / "encoder.pth"

    torch.save({
        "model_state_dict": model.state_dict(),
        "embedding_dim":    EMBEDDING_DIM,
        "target_freq":      TARGET_FREQ,
        "target_time":      TARGET_TIME,
        "final_loss":       losses[-1],
        "train_losses":     losses,
    }, ae_path)

    torch.save({
        "model_state_dict": model.encoder.state_dict(),
        "embedding_dim":    EMBEDDING_DIM,
    }, enc_path)

    print(f"\nSaved: {ae_path}")
    print(f"Saved: {enc_path}")

    # Save training losses
    np.save(out_dir / "train_losses.npy", np.array(losses))

    # Evaluate
    evaluate_all_ids(model, device)

    # Save deployment config
    config_out = {
        "embedding_dim":   EMBEDDING_DIM,
        "target_freq":     TARGET_FREQ,
        "target_time":     TARGET_TIME,
        "final_loss":      float(losses[-1]),
        "epochs_trained":  EPOCHS,
        "threshold_pct":   THRESHOLD_PCT,
    }
    with open(out_dir / "deployment_config.json", "w") as f:
        json.dump(config_out, f, indent=2)
    print(f"\nSaved: {out_dir / 'deployment_config.json'}")


if __name__ == "__main__":
    main()