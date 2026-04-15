"""
test_classification.py — Per-file classification report for Echo-Check.

Uses the PyTorch Conv2D encoder + LOF model (no ONNX, no quantization).

Runs every test spectrogram through the full inference pipeline and reports:
- Per machine ID: accuracy, precision, recall, F1, TP, TN, FP, FN
- Full misclassification table with index, true label, predicted label, score
- Overall summary across all machine IDs

Usage:
    cd tests
    python test_classification.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from conv2d_model import Encoder, EMBEDDING_DIM

# ── Config ────────────────────────────────────────────────────────────────────
ENCODER_PATH  = _ROOT / "models" / "conv2d" / "encoder.pth"
LOF_PATH      = _ROOT / "models" / "conv2d" / "lof_model.pkl"
SPLITS_DIR    = _ROOT / "data" / "splits"

TARGET_FREQ   = 128
TARGET_TIME   = 432
THRESHOLD_PCT = 95
LABEL_NAMES   = {0: "NORMAL", 1: "ANOMALY"}
DEVICE        = torch.device("cpu")
# ─────────────────────────────────────────────────────────────────────────────


# ── Model loading ─────────────────────────────────────────────────────────────
def load_models():
    """Load PyTorch encoder and LOF model."""
    encoder = Encoder(embedding_dim=EMBEDDING_DIM)
    checkpoint = torch.load(ENCODER_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    encoder.load_state_dict(state_dict)
    encoder.eval()

    with open(LOF_PATH, "rb") as f:
        lof = pickle.load(f)

    return encoder, lof


# ── Inference helpers ─────────────────────────────────────────────────────────
def pad_spectrogram(spec: np.ndarray) -> np.ndarray:
    """Pad/trim [128, T] -> [1, 1, 128, 432] float32."""
    t = spec.shape[1]
    if t < TARGET_TIME:
        spec = np.pad(spec, ((0, 0), (0, TARGET_TIME - t)), mode="constant")
    else:
        spec = spec[:, :TARGET_TIME]
    return spec[np.newaxis, np.newaxis, :, :].astype(np.float32)


def get_embedding(encoder: Encoder, spec: np.ndarray) -> np.ndarray:
    """Run PyTorch encoder on a single spectrogram. Returns (embedding_dim,) array."""
    inp = pad_spectrogram(spec)
    with torch.no_grad():
        emb = encoder(torch.from_numpy(inp).to(DEVICE))
    return emb[0].cpu().numpy()


def compute_threshold(encoder: Encoder, lof, train_npy: Path) -> float:
    """Compute LOF score threshold at THRESHOLD_PCT percentile over training normals."""
    specs  = np.load(train_npy)
    scores = [-lof.score_samples(get_embedding(encoder, s).reshape(1, -1))[0]
              for s in specs]
    return float(np.percentile(scores, THRESHOLD_PCT))


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Compute precision, recall, F1, and accuracy from confusion matrix counts."""
    accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp)                   if (tp + fp) > 0            else 0.0
    recall    = tp / (tp + fn)                   if (tp + fn) > 0            else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


# ── Classification report ─────────────────────────────────────────────────────
def run_classification_report(encoder: Encoder, lof,
                               test_npy: Path, labels_npy: Path,
                               train_npy: Path, machine_id: str) -> dict:
    spectrograms = np.load(test_npy)
    y_true       = np.load(labels_npy)
    threshold    = compute_threshold(encoder, lof, train_npy)

    tp = fp = fn = tn = 0
    misclassified = []

    for idx, (spec, true_label) in enumerate(zip(spectrograms, y_true)):
        emb        = get_embedding(encoder, spec)
        score      = float(-lof.score_samples(emb.reshape(1, -1))[0])
        pred_label = 1 if score > threshold else 0
        correct    = pred_label == int(true_label)

        if correct:
            if pred_label == 1:
                tp += 1
            else:
                tn += 1
        else:
            misclassified.append({
                "index":     idx,
                "true":      LABEL_NAMES[int(true_label)],
                "predicted": LABEL_NAMES[pred_label],
                "score":     round(score, 6),
                "threshold": round(threshold, 6),
            })
            if int(true_label) == 1 and pred_label == 0:
                fn += 1
            else:
                fp += 1

    metrics = compute_metrics(tp, fp, fn, tn)

    return {
        "machine_id":    machine_id,
        "threshold":     threshold,
        "total":         len(spectrograms),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "misclassified": misclassified,
        **metrics,
    }


# ── Print report ──────────────────────────────────────────────────────────────
def print_report(r: dict):
    print(f"\n{'=' * 60}")
    print(f"  Machine ID  : {r['machine_id']}")
    print(f"  Threshold   : {r['threshold']:.6f}  ({THRESHOLD_PCT}th percentile)")
    print(f"{'─' * 60}")
    print(f"  Total       : {r['total']}")
    print(f"  TP          : {r['tp']}   TN : {r['tn']}")
    print(f"  FP          : {r['fp']}   FN : {r['fn']}")
    print(f"{'─' * 60}")
    print(f"  Accuracy    : {r['accuracy']  * 100:.1f}%")
    print(f"  Precision   : {r['precision'] * 100:.1f}%")
    print(f"  Recall      : {r['recall']    * 100:.1f}%")
    print(f"  F1 Score    : {r['f1']        * 100:.1f}%")
    print(f"{'=' * 60}")

    if r["misclassified"]:
        print(f"\n  Misclassified samples ({len(r['misclassified'])}):")
        print(f"  {'Index':>6}  {'True':>8}  {'Predicted':>10}  {'Score':>10}  {'Threshold':>10}")
        print(f"  {'-' * 52}")
        for e in r["misclassified"]:
            print(f"  {e['index']:>6}  {e['true']:>8}  {e['predicted']:>10}  "
                  f"{e['score']:>10.6f}  {e['threshold']:>10.6f}")
    else:
        print("\n  No misclassifications — all samples correctly classified.")


# ── Entry point ───────────────────────────────────────────────────────────────
def run():
    print(f"\nLoading PyTorch encoder : {ENCODER_PATH}")
    print(f"Loading LOF model       : {LOF_PATH}\n")
    encoder, lof = load_models()

    test_files = sorted(SPLITS_DIR.glob("pump_id_*_test.npy"))
    if not test_files:
        raise FileNotFoundError(f"No test .npy files found in '{SPLITS_DIR}'.")

    print(f"Found {len(test_files)} machine ID(s). "
          f"Threshold: {THRESHOLD_PCT}th percentile\n")

    total_tp = total_tn = total_fp = total_fn = total_files = 0

    for test_file in test_files:
        prefix      = test_file.name.replace("_test.npy", "")
        machine_id  = prefix.replace("pump_", "")
        labels_file = SPLITS_DIR / f"{prefix}_test_labels.npy"
        train_file  = SPLITS_DIR / f"{prefix}_train.npy"

        if not labels_file.exists() or not train_file.exists():
            print(f"[{machine_id}] SKIPPED — missing label or train file.")
            continue

        print(f"Scoring {machine_id}...")
        r = run_classification_report(
            encoder, lof, test_file, labels_file, train_file, machine_id
        )
        print_report(r)

        total_tp    += r["tp"]
        total_tn    += r["tn"]
        total_fp    += r["fp"]
        total_fn    += r["fn"]
        total_files += r["total"]

    # Overall metrics from aggregated confusion matrix counts
    overall = compute_metrics(total_tp, total_fp, total_fn, total_tn)

    print(f"\n{'=' * 60}")
    print(f"  OVERALL SUMMARY  ({total_files} samples, {len(test_files)} machine IDs)")
    print(f"{'─' * 60}")
    print(f"  TP : {total_tp}   TN : {total_tn}")
    print(f"  FP : {total_fp}   FN : {total_fn}")
    print(f"{'─' * 60}")
    print(f"  Accuracy    : {overall['accuracy']  * 100:.1f}%")
    print(f"  Precision   : {overall['precision'] * 100:.1f}%")
    print(f"  Recall      : {overall['recall']    * 100:.1f}%")
    print(f"  F1 Score    : {overall['f1']        * 100:.1f}%")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    run()