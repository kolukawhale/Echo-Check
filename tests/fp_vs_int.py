"""
test_fp32_vs_int8.py — FP32 vs INT8 quantized encoder comparison for Echo-Check.

Loads both ONNX encoder variants and the fitted LOF model, scores the full
test set with each, and produces:
    - Per machine ID side-by-side comparison (FP32 vs INT8)
    - Overall summary across all machine IDs

Metrics reported per machine ID and overall:
    TN, TP, FP, FN, Precision, Recall, F1, Accuracy, AUC-ROC

Threshold method: 95th percentile of LOF scores on training normals,
computed per machine ID — identical to test_performance.py.

Usage:
    python tests/test_fp32_vs_int8.py
"""

import pickle
from pathlib import Path

import numpy as np
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from sklearn.metrics import roc_auc_score

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent

# ── Config ────────────────────────────────────────────────────────────────────
FP32_ONNX = _ROOT / "models" / "phase3_outputs_lof" / "encoder_simplified.onnx"
INT8_ONNX = _ROOT / "models" / "phase3_outputs_lof" / "encoder_int8.onnx"
LOF_PATH  = _ROOT / "models" / "conv2d" / "lof_model.pkl"

MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]

TRAIN_NPYS = {m: _ROOT / f"data/splits/pump_{m}_train.npy"       for m in MACHINE_IDS}
TEST_NPYS  = {m: _ROOT / f"data/splits/pump_{m}_test.npy"        for m in MACHINE_IDS}
LABEL_NPYS = {m: _ROOT / f"data/splits/pump_{m}_test_labels.npy" for m in MACHINE_IDS}

TARGET_TIME   = 432
THRESHOLD_PCT = 95
# ─────────────────────────────────────────────────────────────────────────────


# ── Model loading ─────────────────────────────────────────────────────────────
def load_session(onnx_path: Path) -> InferenceSession:
    so = SessionOptions()
    so.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    return InferenceSession(
        str(onnx_path),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )


# ── Inference helpers ─────────────────────────────────────────────────────────
def pad(spec: np.ndarray) -> np.ndarray:
    """Pad/trim [128, T] -> [1, 1, 128, 432] float32."""
    t = spec.shape[1]
    if t < TARGET_TIME:
        spec = np.pad(spec, ((0, 0), (0, TARGET_TIME - t)), mode="constant")
    else:
        spec = spec[:, :TARGET_TIME]
    return spec[np.newaxis, np.newaxis, :, :].astype(np.float32)


def extract_embeddings(session: InferenceSession,
                       spectrograms: np.ndarray) -> np.ndarray:
    """Run ONNX session on spectrograms. Returns (N, embedding_dim)."""
    iname = session.get_inputs()[0].name
    embs  = []
    for spec in spectrograms:
        emb = session.run(None, {iname: pad(spec)})[0]
        embs.append(emb[0])
    return np.array(embs)


def lof_scores(lof, embeddings: np.ndarray) -> np.ndarray:
    """Negate score_samples so higher = more anomalous."""
    return -lof.score_samples(embeddings)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(tp, fp, fn, tn) -> dict:
    accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp)                   if (tp + fp) > 0            else 0.0
    recall    = tp / (tp + fn)                   if (tp + fn) > 0            else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_session(session: InferenceSession, lof) -> dict:
    """
    Evaluate per machine ID and return both per-ID results and overall totals.

    Returns:
        per_machine : dict[machine_id -> result dict]
        overall     : aggregated result dict
    """
    per_machine = {}
    total_tp = total_tn = total_fp = total_fn = 0
    all_scores, all_labels = [], []

    for machine_id in MACHINE_IDS:
        train_specs = np.load(TRAIN_NPYS[machine_id])
        test_specs  = np.load(TEST_NPYS[machine_id])
        y_true      = np.load(LABEL_NPYS[machine_id])

        # Threshold from training normals only — no data leakage
        train_embs = extract_embeddings(session, train_specs)
        train_sc   = lof_scores(lof, train_embs)
        threshold  = float(np.percentile(train_sc, THRESHOLD_PCT))

        # Score test set
        test_embs   = extract_embeddings(session, test_specs)
        test_sc     = lof_scores(lof, test_embs)
        predictions = (test_sc > threshold).astype(int)

        tp = int(((predictions == 1) & (y_true == 1)).sum())
        tn = int(((predictions == 0) & (y_true == 0)).sum())
        fp = int(((predictions == 1) & (y_true == 0)).sum())
        fn = int(((predictions == 0) & (y_true == 1)).sum())

        try:
            auc = float(roc_auc_score(y_true, test_sc))
        except ValueError:
            auc = float("nan")

        per_machine[machine_id] = {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "total": len(y_true),
            "n_normal": int((y_true == 0).sum()),
            "n_anomaly": int((y_true == 1).sum()),
            "threshold": threshold,
            "auc": auc,
            **compute_metrics(tp, fp, fn, tn),
        }

        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn
        all_scores.append(test_sc)
        all_labels.append(y_true)

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    overall_auc = float(roc_auc_score(all_labels, all_scores))
    overall = {
        "tp": total_tp, "tn": total_tn,
        "fp": total_fp, "fn": total_fn,
        "total": len(all_labels),
        "auc": overall_auc,
        **compute_metrics(total_tp, total_fp, total_fn, total_tn),
    }

    return per_machine, overall


# ── Print helpers ─────────────────────────────────────────────────────────────
def print_machine_comparison(machine_id: str, fp32: dict, int8: dict):
    """Print side-by-side FP32 vs INT8 for one machine ID."""
    print(f"\n{'=' * 68}")
    print(f"  Machine ID : {machine_id}  "
          f"(total={fp32['total']}, "
          f"normal={fp32['n_normal']}, "
          f"anomaly={fp32['n_anomaly']})")
    print(f"  Threshold  FP32: {fp32['threshold']:.6f}   "
          f"INT8: {int8['threshold']:.6f}")
    print(f"{'─' * 68}")
    print(f"  {'Metric':<28}  {'FP32':>10}  {'INT8':>10}  {'Delta':>10}")
    print(f"  {'-' * 62}")

    for key, label in [("tn", "True Normals  (TN)"),
                        ("tp", "True Anomalies (TP)"),
                        ("fp", "False Positives (FP)"),
                        ("fn", "False Negatives (FN)")]:
        d = int8[key] - fp32[key]
        print(f"  {label:<28}  {fp32[key]:>10d}  {int8[key]:>10d}  {d:>+10d}")

    print(f"  {'-' * 62}")

    for key, label in [("precision", "Precision"),
                        ("recall",    "Recall"),
                        ("f1",        "F1 Score"),
                        ("accuracy",  "Accuracy")]:
        fv = fp32[key] * 100
        iv = int8[key] * 100
        print(f"  {label:<28}  {fv:>9.1f}%  {iv:>9.1f}%  {iv-fv:>+9.1f}pp")

    print(f"  {'-' * 62}")
    d_auc = int8["auc"] - fp32["auc"]
    print(f"  {'AUC-ROC':<28}  {fp32['auc']:>10.4f}  {int8['auc']:>10.4f}  {d_auc:>+10.4f}")
    print(f"{'=' * 68}")


def print_overall_comparison(fp32: dict, int8: dict):
    """Print side-by-side FP32 vs INT8 overall summary."""
    print(f"\n{'=' * 68}")
    print(f"  OVERALL SUMMARY  ({fp32['total']} samples, {len(MACHINE_IDS)} machine IDs)")
    print(f"{'─' * 68}")
    print(f"  {'Metric':<28}  {'FP32':>10}  {'INT8':>10}  {'Delta':>10}")
    print(f"  {'-' * 62}")

    for key, label in [("tn", "True Normals  (TN)"),
                        ("tp", "True Anomalies (TP)"),
                        ("fp", "False Positives (FP)"),
                        ("fn", "False Negatives (FN)")]:
        d = int8[key] - fp32[key]
        print(f"  {label:<28}  {fp32[key]:>10d}  {int8[key]:>10d}  {d:>+10d}")

    print(f"  {'-' * 62}")

    for key, label in [("precision", "Precision"),
                        ("recall",    "Recall"),
                        ("f1",        "F1 Score"),
                        ("accuracy",  "Accuracy")]:
        fv = fp32[key] * 100
        iv = int8[key] * 100
        print(f"  {label:<28}  {fv:>9.1f}%  {iv:>9.1f}%  {iv-fv:>+9.1f}pp")

    print(f"  {'-' * 62}")
    d_auc = int8["auc"] - fp32["auc"]
    print(f"  {'AUC-ROC':<28}  {fp32['auc']:>10.4f}  {int8['auc']:>10.4f}  {d_auc:>+10.4f}")
    print(f"{'=' * 68}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    for path, name in [(FP32_ONNX, "FP32 ONNX"), (INT8_ONNX, "INT8 ONNX"),
                       (LOF_PATH,  "LOF model")]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{name} not found at {path}. "
                f"Run phase3_optimize.py first."
            )

    print(f"\nLoading FP32 session : {FP32_ONNX.name}")
    sess_fp32 = load_session(FP32_ONNX)

    print(f"Loading INT8 session : {INT8_ONNX.name}")
    sess_int8 = load_session(INT8_ONNX)

    print(f"Loading LOF model    : {LOF_PATH.name}\n")
    with open(LOF_PATH, "rb") as f:
        lof = pickle.load(f)

    print("Evaluating FP32 encoder...")
    fp32_per_machine, fp32_overall = evaluate_session(sess_fp32, lof)

    print("Evaluating INT8 encoder...")
    int8_per_machine, int8_overall = evaluate_session(sess_int8, lof)

    print("\n\n── Per Machine ID Results ───────────────────────────────────────")
    for machine_id in MACHINE_IDS:
        print_machine_comparison(
            machine_id,
            fp32_per_machine[machine_id],
            int8_per_machine[machine_id],
        )

    print("\n\n── Overall Summary ──────────────────────────────────────────────")
    print_overall_comparison(fp32_overall, int8_overall)


if __name__ == "__main__":
    main()