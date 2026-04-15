"""
test_classification.py — Per-file classification report for Echo-Check.

Uses the INT8 ONNX encoder + LOF model (Conv2D + LOF pipeline).

Runs every test spectrogram through the full inference pipeline and reports:
- Which files are correctly classified
- Which files are misclassified and why
- A summary count per machine ID and overall

Usage:
    cd tests
    python test_classification.py
"""

import numpy as np
import pickle
import onnxruntime as ort
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
_ROOT        = Path(__file__).parent.parent

ONNX_PATH    = _ROOT / "models" / "phase3_outputs_lof" / "encoder_int8.onnx"
LOF_PATH     = _ROOT / "models" / "conv2d" / "lof_model.pkl"
SPLITS_DIR   = _ROOT / "data" / "splits"

TARGET_FREQ      = 128
TARGET_TIME      = 432
THRESHOLD_PCT    = 95
LABEL_NAMES      = {0: "NORMAL", 1: "ANOMALY"}
# ─────────────────────────────────────────────────────────────────────────────


def load_models():
    so = SessionOptions()
    so.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    session = InferenceSession(
        str(ONNX_PATH), sess_options=so, providers=["CPUExecutionProvider"]
    )
    with open(LOF_PATH, "rb") as f:
        lof = pickle.load(f)
    return session, lof


def pad_spectrogram(spec: np.ndarray) -> np.ndarray:
    """Pad/trim [128, T] → [1, 1, 128, 432] float32."""
    t = spec.shape[1]
    if t < TARGET_TIME:
        spec = np.pad(spec, ((0, 0), (0, TARGET_TIME - t)), mode="constant")
    else:
        spec = spec[:, :TARGET_TIME]
    return spec[np.newaxis, np.newaxis, :, :].astype(np.float32)


def get_embedding(session, spec: np.ndarray) -> np.ndarray:
    """Run INT8 ONNX encoder on a single spectrogram. Returns [128] embedding."""
    inp   = pad_spectrogram(spec)
    iname = session.get_inputs()[0].name
    emb   = session.run(None, {iname: inp})[0]
    return emb[0]


def compute_threshold(session, lof, train_npy: str) -> float:
    """Compute LOF threshold from normal training data at THRESHOLD_PCT percentile."""
    specs  = np.load(train_npy)
    scores = [-lof.score_samples(get_embedding(session, s).reshape(1, -1))[0]
              for s in specs]
    return float(np.percentile(scores, THRESHOLD_PCT))


def run_classification_report(session, lof, test_npy, labels_npy,
                               train_npy, machine_id) -> dict:
    spectrograms = np.load(test_npy)
    y_true       = np.load(labels_npy)
    threshold    = compute_threshold(session, lof, train_npy)

    results = {
        "machine_id":    machine_id,
        "threshold":     threshold,
        "correct":       [],
        "misclassified": [],
        "total":         len(spectrograms),
        "n_correct":     0,
        "n_wrong":       0,
        "n_fn":          0,
        "n_fp":          0,
    }

    for idx, (spec, true_label) in enumerate(zip(spectrograms, y_true)):
        emb        = get_embedding(session, spec)
        score      = float(-lof.score_samples(emb.reshape(1, -1))[0])
        pred_label = 1 if score > threshold else 0
        correct    = pred_label == true_label

        entry = {
            "index":     idx,
            "true":      LABEL_NAMES[int(true_label)],
            "predicted": LABEL_NAMES[pred_label],
            "score":     round(score, 6),
            "threshold": round(threshold, 6),
        }

        if correct:
            results["correct"].append(entry)
            results["n_correct"] += 1
        else:
            results["misclassified"].append(entry)
            results["n_wrong"] += 1
            if true_label == 1 and pred_label == 0:
                results["n_fn"] += 1
            elif true_label == 0 and pred_label == 1:
                results["n_fp"] += 1

    return results


def print_report(r: dict):
    total    = r["total"]
    accuracy = r["n_correct"] / total * 100

    print(f"\n{'=' * 55}")
    print(f"  Machine ID : {r['machine_id']}")
    print(f"  Threshold  : {r['threshold']:.6f}  ({THRESHOLD_PCT}th pct)")
    print(f"  Total      : {total}  |  Correct: {r['n_correct']}  |  Wrong: {r['n_wrong']}")
    print(f"  Accuracy   : {accuracy:.1f}%")
    print(f"  Missed anomalies (FN) : {r['n_fn']}")
    print(f"  False alarms     (FP) : {r['n_fp']}")
    print(f"{'=' * 55}")

    if r["misclassified"]:
        print(f"\n  Misclassified files ({r['n_wrong']}):")
        print(f"  {'Index':>6}  {'True':>8}  {'Predicted':>10}  {'Score':>10}  {'Threshold':>10}")
        print(f"  {'-'*52}")
        for e in r["misclassified"]:
            print(f"  {e['index']:>6}  {e['true']:>8}  {e['predicted']:>10}  "
                  f"{e['score']:>10.6f}  {e['threshold']:>10.6f}")
    else:
        print("\n  No misclassifications — all files correctly classified.")


def run():
    print(f"\nLoading INT8 ONNX encoder: {ONNX_PATH}")
    print(f"Loading LOF model:         {LOF_PATH}\n")
    session, lof = load_models()

    test_files = sorted(SPLITS_DIR.glob("pump_id_*_test.npy"))
    if not test_files:
        raise FileNotFoundError(f"No test files found in '{SPLITS_DIR}'.")

    print(f"Found {len(test_files)} machine ID(s). Threshold: {THRESHOLD_PCT}th percentile\n")

    total_correct = total_wrong = total_fn = total_fp = total_files = 0

    for test_file in test_files:
        prefix      = test_file.name.replace("_test.npy", "")
        machine_id  = prefix.replace("pump_", "")
        labels_file = SPLITS_DIR / f"{prefix}_test_labels.npy"
        train_file  = SPLITS_DIR / f"{prefix}_train.npy"

        if not labels_file.exists() or not train_file.exists():
            print(f"[{machine_id}] SKIPPED — missing files.")
            continue

        print(f"Scoring {machine_id}...")
        r = run_classification_report(
            session, lof, test_file, labels_file, train_file, machine_id
        )
        print_report(r)

        total_correct += r["n_correct"]
        total_wrong   += r["n_wrong"]
        total_fn      += r["n_fn"]
        total_fp      += r["n_fp"]
        total_files   += r["total"]

    print(f"\n{'=' * 55}")
    print(f"  OVERALL SUMMARY")
    print(f"{'=' * 55}")
    print(f"  Total files : {total_files}")
    print(f"  Correct     : {total_correct}  ({total_correct/total_files*100:.1f}%)")
    print(f"  Wrong       : {total_wrong}  ({total_wrong/total_files*100:.1f}%)")
    print(f"  Missed anomalies (FN) : {total_fn}")
    print(f"  False alarms     (FP) : {total_fp}")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    run()