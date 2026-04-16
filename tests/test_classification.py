"""
test_classification.py — Evaluation report for the Echo-Check Conv2D + LOF pipeline.

Loads the trained autoencoder (autoencoder.pth) and the fitted LOF model
(lof_model.pkl) and evaluates anomaly detection performance on the test splits.

Threshold method: 95th percentile of LOF scores on the training normals,
computed per machine ID. This mirrors real deployment — the threshold is
set from known-normal data only, with no access to test labels.

Reports per machine ID and overall:
    - AUC-ROC
    - Accuracy, Precision, Recall, F1
    - Confusion matrix (TP, TN, FP, FN)
    - Full misclassification table

Pipeline order:
    1. training.py             -> models/conv2d/autoencoder.pth
    2. evaluate_conv2d_lof.py  -> models/conv2d/lof_model.pkl
    3. test_performance.py (this file)

Usage:
    python src/test_performance.py
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent

CHECKPOINT = _ROOT / "models" / "conv2d" / "autoencoder.pth"
LOF_PATH   = _ROOT / "models" / "conv2d" / "lof_model.pkl"

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

EMBEDDING_DIM = 128
TARGET_FREQ   = 128
TARGET_TIME   = 432
BATCH_SIZE    = 64
THRESHOLD_PCT = 95
LABEL_NAMES   = {0: "NORMAL", 1: "ANOMALY"}
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
    """Run spectrograms through the encoder only. Returns [N, embedding_dim]."""
    model.eval()
    loader     = DataLoader(SpectrogramDataset(spectrograms),
                            batch_size=BATCH_SIZE, shuffle=False)
    embeddings = []
    for batch in loader:
        emb = model.encoder(batch.to(device))
        embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


# ── LOF scoring ───────────────────────────────────────────────────────────────
def score_embeddings(lof, embeddings: np.ndarray) -> np.ndarray:
    """
    Score an array of embeddings with LOF.
    lof.score_samples() returns negative values — more negative = more anomalous.
    We negate so higher score = more anomalous.
    Returns: np.ndarray of shape [N]
    """
    return -lof.score_samples(embeddings)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp)                   if (tp + fp) > 0            else 0.0
    recall    = tp / (tp + fn)                   if (tp + fn) > 0            else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {"accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1}


# ── Per-machine evaluation ────────────────────────────────────────────────────
def evaluate_machine(model, lof, device,
                     train_npy: Path, test_npy: Path,
                     labels_npy: Path, machine_id: str) -> dict:
    """
    Evaluate one machine ID.

    Threshold is set at THRESHOLD_PCT of LOF scores on the training normals —
    the same data the LOF was fitted on, but now scored to find the boundary.
    This is the correct deployment-safe method: no test labels used.
    """
    # Compute threshold from training normals
    train_specs      = np.load(train_npy)
    train_embeddings = extract_embeddings(model, train_specs, device)
    train_scores     = score_embeddings(lof, train_embeddings)
    threshold        = float(np.percentile(train_scores, THRESHOLD_PCT))

    # Score test set
    test_specs       = np.load(test_npy)
    y_true           = np.load(labels_npy)
    test_embeddings  = extract_embeddings(model, test_specs, device)
    test_scores      = score_embeddings(lof, test_embeddings)

    predictions = (test_scores > threshold).astype(int)

    # Confusion matrix
    tp = int(((predictions == 1) & (y_true == 1)).sum())
    tn = int(((predictions == 0) & (y_true == 0)).sum())
    fp = int(((predictions == 1) & (y_true == 0)).sum())
    fn = int(((predictions == 0) & (y_true == 1)).sum())

    # AUC — uses raw scores, not thresholded predictions
    try:
        auc = float(roc_auc_score(y_true, test_scores))
    except ValueError:
        auc = float("nan")

    # Misclassified entries
    misclassified = []
    for idx in range(len(test_scores)):
        if predictions[idx] != y_true[idx]:
            misclassified.append({
                "index":     idx,
                "true":      LABEL_NAMES[int(y_true[idx])],
                "predicted": LABEL_NAMES[int(predictions[idx])],
                "score":     round(float(test_scores[idx]), 6),
                "threshold": round(threshold, 6),
            })

    metrics = compute_metrics(tp, fp, fn, tn)

    return {
        "machine_id":       machine_id,
        "threshold":        threshold,
        "total":            len(y_true),
        "n_normal":         int((y_true == 0).sum()),
        "n_anomaly":        int((y_true == 1).sum()),
        "normal_score_mean":  float(train_scores.mean()),
        "anomaly_score_mean": float(test_scores[y_true == 1].mean())
                              if (y_true == 1).any() else float("nan"),
        "auc":              auc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "misclassified":    misclassified,
        **metrics,
    }


# ── Print helpers ─────────────────────────────────────────────────────────────
def print_machine_report(r: dict):
    print(f"\n{'=' * 62}")
    print(f"  Machine ID  : {r['machine_id']}")
    print(f"  Threshold   : {r['threshold']:.6f}  ({THRESHOLD_PCT}th pct of train normals)")
    print(f"{'─' * 62}")
    print(f"  Total       : {r['total']}  "
          f"(normal={r['n_normal']}, anomaly={r['n_anomaly']})")
    print(f"  Train score mean (normal)  : {r['normal_score_mean']:.4f}")
    print(f"  Test score mean  (anomaly) : {r['anomaly_score_mean']:.4f}")
    print(f"{'─' * 62}")
    print(f"  AUC-ROC     : {r['auc']:.4f}")
    print(f"{'─' * 62}")
    print(f"  TP : {r['tp']:>4}   TN : {r['tn']:>4}")
    print(f"  FP : {r['fp']:>4}   FN : {r['fn']:>4}")
    print(f"{'─' * 62}")
    print(f"  Accuracy    : {r['accuracy']  * 100:.1f}%")
    print(f"  Precision   : {r['precision'] * 100:.1f}%")
    print(f"  Recall      : {r['recall']    * 100:.1f}%")
    print(f"  F1 Score    : {r['f1']        * 100:.1f}%")
    print(f"{'=' * 62}")

    if r["misclassified"]:
        print(f"\n  Misclassified samples ({len(r['misclassified'])}):")
        print(f"  {'Index':>6}  {'True':>8}  {'Predicted':>10}  "
              f"{'Score':>10}  {'Threshold':>10}")
        print(f"  {'-' * 52}")
        for e in r["misclassified"]:
            print(f"  {e['index']:>6}  {e['true']:>8}  {e['predicted']:>10}  "
                  f"{e['score']:>10.6f}  {e['threshold']:>10.6f}")
    else:
        print("\n  No misclassifications — all samples correctly classified.")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    device = get_device()
    print(f"Device : {device}")

    # Load autoencoder
    print(f"\nLoading checkpoint : {CHECKPOINT}")
    ckpt  = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = CNNAutoencoder(embedding_dim=EMBEDDING_DIM).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Final training loss : {ckpt['final_loss']:.6f}")

    # Load LOF
    print(f"\nLoading LOF model  : {LOF_PATH}")
    with open(LOF_PATH, "rb") as f:
        lof = pickle.load(f)

    # Gather available machine IDs
    triplets = []
    for train_npy, test_npy, labels_npy in zip(TRAIN_NPYS, TEST_NPYS, LABEL_NPYS):
        if not all(Path(p).exists() for p in [train_npy, test_npy, labels_npy]):
            mid = Path(test_npy).name.replace("pump_", "").replace("_test.npy", "")
            print(f"[{mid}] SKIPPED — missing files.")
            continue
        triplets.append((train_npy, test_npy, labels_npy))

    if not triplets:
        raise FileNotFoundError("No complete machine ID triplets found in data/splits/.")

    print(f"\nEvaluating {len(triplets)} machine ID(s). "
          f"Threshold: {THRESHOLD_PCT}th percentile of training normals.\n")

    # Per-machine evaluation
    all_results = []
    for train_npy, test_npy, labels_npy in triplets:
        machine_id = Path(test_npy).name.replace("pump_", "").replace("_test.npy", "")
        print(f"Scoring {machine_id}...")
        r = evaluate_machine(model, lof, device,
                             Path(train_npy), Path(test_npy),
                             Path(labels_npy), machine_id)
        print_machine_report(r)
        all_results.append(r)

    # Overall summary — aggregate confusion matrix then compute metrics
    total_tp = sum(r["tp"] for r in all_results)
    total_tn = sum(r["tn"] for r in all_results)
    total_fp = sum(r["fp"] for r in all_results)
    total_fn = sum(r["fn"] for r in all_results)
    total_n  = sum(r["total"] for r in all_results)
    avg_auc  = float(np.mean([r["auc"] for r in all_results
                               if not np.isnan(r["auc"])]))

    overall = compute_metrics(total_tp, total_fp, total_fn, total_tn)

    print(f"\n{'=' * 62}")
    print(f"  OVERALL SUMMARY  ({total_n} samples, {len(all_results)} machine IDs)")
    print(f"{'─' * 62}")
    print(f"  Average AUC-ROC : {avg_auc:.4f}")
    print(f"{'─' * 62}")
    print(f"  TP : {total_tp:>4}   TN : {total_tn:>4}")
    print(f"  FP : {total_fp:>4}   FN : {total_fn:>4}")
    print(f"{'─' * 62}")
    print(f"  Accuracy    : {overall['accuracy']  * 100:.1f}%")
    print(f"  Precision   : {overall['precision'] * 100:.1f}%")
    print(f"  Recall      : {overall['recall']    * 100:.1f}%")
    print(f"  F1 Score    : {overall['f1']        * 100:.1f}%")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()