"""
evaluate_conv2d_lof.py — LOF-based anomaly scoring for the Conv2D Autoencoder.

Instead of using MSE reconstruction error as the anomaly score, this script:
  1. Extracts 128-dim encoder embeddings for all normal training spectrograms
  2. Fits LOF on those embeddings to model the normal cluster
  3. Scores each test spectrogram by how isolated its embedding is
  4. Reports AUC, F1, Precision, Recall per machine ID

No retraining required — loads the existing autoencoder.pth checkpoint.

Usage:
    python src/evaluate_conv2d_lof.py
"""

import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
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

EMBEDDING_DIM    = 128
TARGET_FREQ      = 128
TARGET_TIME      = 432
BATCH_SIZE       = 64
N_NEIGHBORS      = 20     # LOF neighbourhood size
THRESHOLD_PCT    = 95     # percentile of normal LOF scores for threshold
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
    """Pads [N, 128, 431] → [N, 1, 128, 432] and returns float32 tensors."""
    def __init__(self, spectrograms: np.ndarray):
        n, freq, t = spectrograms.shape
        if t < TARGET_TIME:
            spectrograms = np.pad(
                spectrograms,
                ((0, 0), (0, 0), (0, TARGET_TIME - t)),
                mode="constant", constant_values=0,
            )
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
    Runs spectrograms through the encoder only and returns
    the 128-dim embedding for each sample.

    We call model.encoder(x) directly — the decoder never runs.
    This is the latent vector the encoder has learned to represent
    each spectrogram with. Normal sounds cluster tightly here.
    Abnormal sounds land in sparse isolated regions.

    Returns: np.ndarray of shape [N, 128]
    """
    model.eval()
    dataset    = SpectrogramDataset(spectrograms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    embeddings = []

    for batch in dataloader:
        batch = batch.to(device)
        emb   = model.encoder(batch)          # encoder only — [B, 128]
        embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0)  # [N, 128]


# ── LOF fitting ───────────────────────────────────────────────────────────────
def fit_lof(train_embeddings: np.ndarray) -> LocalOutlierFactor:
    """
    Fits LOF on normal training embeddings.

    novelty=True allows LOF to score new unseen samples at inference.
    Without this, LOF can only score the points it was trained on.

    n_neighbors=20 means LOF compares each point's local density
    to its 20 nearest neighbours. Higher values = more stable but
    slower. Lower values = more sensitive to local anomalies.

    Returns: fitted LOF model
    """
    print(f"Fitting LOF on {len(train_embeddings)} normal embeddings "
          f"of dim {train_embeddings.shape[1]}...")
    lof = LocalOutlierFactor(
        n_neighbors=N_NEIGHBORS,
        novelty=True,
        metric="euclidean",
    )
    lof.fit(train_embeddings)
    print("LOF fitted.\n")
    return lof


# ── LOF scoring ───────────────────────────────────────────────────────────────
def lof_score(lof: LocalOutlierFactor,
              embedding: np.ndarray) -> float:
    """
    Scores a single embedding using LOF.

    lof.score_samples() returns negative scores where more negative
    means more anomalous. We negate so higher = more anomalous,
    consistent with how we think about anomaly scores.

    Returns: float — anomaly score (higher = more anomalous)
    """
    return float(-lof.score_samples(embedding.reshape(1, -1))[0])


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_all_ids(model, lof, device):
    """
    For each machine ID:
      1. Extract embeddings for all test spectrograms
      2. Score each embedding with LOF
      3. Set threshold at THRESHOLD_PCT of normal LOF scores
      4. Report AUC, F1, Precision, Recall
    """
    print(f"{'='*55}")
    print("  LOF EVALUATION ACROSS ALL MACHINE IDs")
    print(f"{'='*55}")

    all_aucs = []

    for test_npy, label_npy in zip(TEST_NPYS, LABEL_NPYS):
        if not Path(test_npy).exists() or not Path(label_npy).exists():
            continue

        machine_id = Path(test_npy).name.replace("pump_", "").replace("_test.npy", "")
        specs  = np.load(test_npy)
        labels = np.load(label_npy)

        # Extract embeddings for all test spectrograms
        embeddings = extract_embeddings(model, specs, device)  # [N, 128]

        # Score each embedding with LOF
        scores = np.array([lof_score(lof, emb) for emb in embeddings])

        # Separate normal and abnormal scores
        normal_scores   = scores[labels == 0]
        abnormal_scores = scores[labels == 1]

        # Set threshold at THRESHOLD_PCT of normal LOF scores
        # This means at most (100 - THRESHOLD_PCT)% of normal samples
        # will be false-alarmed by design
        threshold   = np.percentile(normal_scores, THRESHOLD_PCT)
        predictions = (scores > threshold).astype(int)

        auc  = roc_auc_score(labels, scores)
        f1   = f1_score(labels, predictions, zero_division=0)
        prec = precision_score(labels, predictions, zero_division=0)
        rec  = recall_score(labels, predictions, zero_division=0)
        all_aucs.append(auc)

        print(f"\n── {machine_id} ──────────────────────────────────────")
        print(f"  Normal mean   : {normal_scores.mean():.4f}")
        print(f"  Abnormal mean : {abnormal_scores.mean():.4f}")
        print(f"  Threshold     : {threshold:.4f}  ({THRESHOLD_PCT}th pct)")
        print(f"  AUC-ROC       : {auc:.4f}")
        print(f"  Precision     : {prec:.4f}")
        print(f"  Recall        : {rec:.4f}")
        print(f"  F1 Score      : {f1:.4f}")

    print(f"\n{'='*55}")
    print(f"  Average AUC : {np.mean(all_aucs):.4f}")
    print(f"{'='*55}")

    # ── Combined evaluation across all IDs ────────────────────────────────────
    print(f"\n{'='*55}")
    print("  COMBINED EVALUATION (all machine IDs pooled)")
    print(f"{'='*55}")

    all_specs  = np.concatenate([np.load(p) for p in TEST_NPYS  if Path(p).exists()])
    all_labels = np.concatenate([np.load(p) for p in LABEL_NPYS if Path(p).exists()])

    all_embeddings = extract_embeddings(model, all_specs, device)
    all_scores     = np.array([lof_score(lof, emb) for emb in all_embeddings])

    normal_scores   = all_scores[all_labels == 0]
    abnormal_scores = all_scores[all_labels == 1]
    threshold       = np.percentile(normal_scores, THRESHOLD_PCT)
    predictions     = (all_scores > threshold).astype(int)

    auc  = roc_auc_score(all_labels, all_scores)
    f1   = f1_score(all_labels, predictions, zero_division=0)
    prec = precision_score(all_labels, predictions, zero_division=0)
    rec  = recall_score(all_labels, predictions, zero_division=0)

    tp = int(((predictions == 1) & (all_labels == 1)).sum())
    tn = int(((predictions == 0) & (all_labels == 0)).sum())
    fp = int(((predictions == 1) & (all_labels == 0)).sum())
    fn = int(((predictions == 0) & (all_labels == 1)).sum())

    print(f"\n  Total samples : {len(all_labels)}  "
          f"(normal={int((all_labels==0).sum())}, abnormal={int((all_labels==1).sum())})")
    print(f"  Normal mean   : {normal_scores.mean():.4f}")
    print(f"  Abnormal mean : {abnormal_scores.mean():.4f}")
    print(f"  Threshold     : {threshold:.4f}  ({THRESHOLD_PCT}th pct)")
    print(f"\n  AUC-ROC       : {auc:.4f}")
    print(f"  Precision     : {prec:.4f}")
    print(f"  Recall        : {rec:.4f}")
    print(f"  F1 Score      : {f1:.4f}")
    print(f"\n  Confusion Matrix")
    print(f"  TP: {tp:>4}  FP: {fp:>4}")
    print(f"  FN: {fn:>4}  TN: {tn:>4}")
    print(f"{'='*55}")

    return all_aucs


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = get_device()
    print(f"Device : {device}\n")

    # ── Load trained autoencoder ──────────────────────────────────────────────
    # We load the full checkpoint but only use the encoder for embeddings
    ckpt  = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = CNNAutoencoder(embedding_dim=EMBEDDING_DIM).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT}")
    print(f"Final training loss: {ckpt['final_loss']:.6f}\n")

    # ── Extract normal training embeddings ────────────────────────────────────
    # Load all normal training spectrograms and run them through the encoder.
    # These embeddings define what "normal" looks like in the latent space.
    # LOF will use these to build its model of the normal cluster.
    print("Extracting normal training embeddings...")
    train_arrays = [np.load(p) for p in TRAIN_NPYS if Path(p).exists()]
    train_specs  = np.concatenate(train_arrays, axis=0)
    print(f"Loaded {len(train_specs)} normal training spectrograms")

    train_embeddings = extract_embeddings(model, train_specs, device)
    print(f"Extracted embeddings: {train_embeddings.shape}\n")

    # ── Fit LOF on normal embeddings ──────────────────────────────────────────
    lof = fit_lof(train_embeddings)

    # ── Save LOF model ────────────────────────────────────────────────────────
    with open(LOF_OUT, "wb") as f:
        pickle.dump(lof, f)
    print(f"LOF model saved: {LOF_OUT}\n")

    # ── Evaluate across all machine IDs ───────────────────────────────────────
    evaluate_all_ids(model, lof, device)


if __name__ == "__main__":
    main()