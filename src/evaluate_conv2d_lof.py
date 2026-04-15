"""
evaluate_conv2d_lof.py — Fit and save the LOF model for Echo-Check.

Loads the trained Conv2D autoencoder, extracts 128-dim encoder embeddings
for all normal training spectrograms across all machine IDs, fits a single
LOF model on those embeddings, and saves it to disk.

No test evaluation is performed here. Use test_performance.py for that.

Pipeline order:
    1. training.py          -> models/conv2d/autoencoder.pth
    2. evaluate_conv2d_lof.py (this file) -> models/conv2d/lof_model.pkl
    3. test_performance.py  -> evaluation report

Usage:
    python src/evaluate_conv2d_lof.py
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import LocalOutlierFactor
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent

CHECKPOINT = _ROOT / "models" / "conv2d" / "autoencoder.pth"
LOF_OUT    = _ROOT / "models" / "conv2d" / "lof_model.pkl"

TRAIN_NPYS = [
    _ROOT / "data/splits/pump_id_00_train.npy",
    _ROOT / "data/splits/pump_id_02_train.npy",
    _ROOT / "data/splits/pump_id_04_train.npy",
    _ROOT / "data/splits/pump_id_06_train.npy",
]

EMBEDDING_DIM = 128
TARGET_FREQ   = 128
TARGET_TIME   = 432
BATCH_SIZE    = 64
N_NEIGHBORS   = 20   # LOF neighbourhood size
# ─────────────────────────────────────────────────────────────────────────────


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Architecture (must match training.py exactly) ─────────────────────────────
class Encoder(nn.Module):
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
        return self.fc(x)


class Decoder(nn.Module):
    _H, _W = 8, 27

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
        z = self.encoder(x)
        return self.decoder(z), z


# ── Dataset ───────────────────────────────────────────────────────────────────
class SpectrogramDataset(Dataset):
    """Pads/trims [N, 128, T] -> [N, 1, 128, 432] float32 tensors."""
    def __init__(self, spectrograms: np.ndarray):
        _, _, t = spectrograms.shape
        if t < TARGET_TIME:
            spectrograms = np.pad(
                spectrograms,
                ((0, 0), (0, 0), (0, TARGET_TIME - t)),
                mode="constant", constant_values=0,
            )
        elif t > TARGET_TIME:
            spectrograms = spectrograms[:, :, :TARGET_TIME]
        self.data = torch.tensor(spectrograms, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── Embedding extraction ──────────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(model: CNNAutoencoder,
                       spectrograms: np.ndarray,
                       device: torch.device) -> np.ndarray:
    """
    Runs spectrograms through the encoder only.
    Decoder is never called — only the latent vector matters.
    Returns: np.ndarray of shape [N, embedding_dim]
    """
    model.eval()
    loader     = DataLoader(SpectrogramDataset(spectrograms),
                            batch_size=BATCH_SIZE, shuffle=False)
    embeddings = []
    for batch in loader:
        batch = batch.to(device)
        emb   = model.encoder(batch)           # [B, embedding_dim]
        embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)  # [N, embedding_dim]


# ── LOF fitting ───────────────────────────────────────────────────────────────
def fit_lof(train_embeddings: np.ndarray) -> LocalOutlierFactor:
    """
    Fits LOF on normal training embeddings only.

    novelty=True is required so the fitted model can score new unseen
    samples at inference time. Without it, LOF can only re-score the
    points it was trained on.

    n_neighbors=20: LOF compares each point's local density to its 20
    nearest neighbours. Controls the trade-off between stability and
    sensitivity to local anomalies.
    """
    print(f"Fitting LOF on {len(train_embeddings)} embeddings "
          f"of dim {train_embeddings.shape[1]}  (n_neighbors={N_NEIGHBORS})...")
    lof = LocalOutlierFactor(
        n_neighbors=N_NEIGHBORS,
        novelty=True,
        metric="euclidean",
    )
    lof.fit(train_embeddings)
    print("LOF fitted.")
    return lof


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = get_device()
    print(f"Device : {device}")

    # Load autoencoder checkpoint
    print(f"\nLoading checkpoint : {CHECKPOINT}")
    ckpt  = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = CNNAutoencoder(embedding_dim=EMBEDDING_DIM).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Final training loss from checkpoint : {ckpt['final_loss']:.6f}")

    # Load all normal training spectrograms
    print("\nLoading normal training spectrograms...")
    available = [p for p in TRAIN_NPYS if Path(p).exists()]
    if not available:
        raise FileNotFoundError(f"No training .npy files found. "
                                f"Expected them in {_ROOT / 'data/splits/'}")
    train_specs = np.concatenate([np.load(p) for p in available], axis=0)
    print(f"Loaded {len(train_specs)} normal spectrograms "
          f"from {len(available)} machine ID(s)")

    # Extract encoder embeddings for all training normals
    print("\nExtracting training embeddings...")
    train_embeddings = extract_embeddings(model, train_specs, device)
    print(f"Embeddings shape : {train_embeddings.shape}")

    # Fit LOF on training embeddings
    print()
    lof = fit_lof(train_embeddings)

    # Save LOF to disk
    LOF_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(LOF_OUT, "wb") as f:
        pickle.dump(lof, f)
    print(f"\nLOF model saved : {LOF_OUT}")
    print("\nDone. Run test_performance.py to evaluate.")


if __name__ == "__main__":
    main()