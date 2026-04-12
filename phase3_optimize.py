"""
phase3_optimize.py — Echo-Check Phase 3 (LOF Encoder Pipeline)

Optimises the Conv2D encoder for edge deployment:
  3.1  Export encoder to ONNX
  3.2  Graph optimisation (onnxsim)
  3.4  INT8 static quantisation (QDQ, QUInt8, Conv+Gemm only)
  3.5  Validate: AUC using LOF scores from INT8 encoder embeddings
  3.6  CPU latency benchmark

Usage:
    python phase3_optimize.py

Inputs  (from Phase 2):
    models/conv2d/autoencoder.pth      — trained weights
    models/conv2d/lof_model.pkl        — fitted LOF model
    models/conv2d/deployment_config.json

Outputs:
    phase3_outputs_lof/encoder_full.onnx
    phase3_outputs_lof/encoder_simplified.onnx
    phase3_outputs_lof/encoder_int8.onnx
"""

import io
import json
import os
import pickle
import sys
import time
import warnings

# Force UTF-8 output so torch.onnx emoji logs don't crash on Windows cp1252
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxsim import simplify
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models", "conv2d")
DATA_DIR   = os.path.join(ROOT, "data", "splits")
OUTPUT_DIR = os.path.join(ROOT, "phase3_outputs_lof")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PTH_AUTOENCODER    = os.path.join(MODELS_DIR, "autoencoder.pth")
LOF_MODEL_PATH     = os.path.join(MODELS_DIR, "lof_model.pkl")
DEPLOY_CONFIG_PATH = os.path.join(MODELS_DIR, "deployment_config.json")

ONNX_FULL_PATH = os.path.join(OUTPUT_DIR, "encoder_full.onnx")
ONNX_SIM_PATH  = os.path.join(OUTPUT_DIR, "encoder_simplified.onnx")
ONNX_PREP_PATH = os.path.join(OUTPUT_DIR, "encoder_prep.onnx")
ONNX_INT8_PATH = os.path.join(OUTPUT_DIR, "encoder_int8.onnx")

OPSET      = 17
CALIB_PER_ID = 32
N_WARMUP   = 20
N_RUNS     = 200
PUMP_IDS   = ["00", "02", "04", "06"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def total_size_kb(path):
    total = os.path.getsize(path)
    if os.path.exists(path + ".data"):
        total += os.path.getsize(path + ".data")
    return total / 1024


def load_split(pump_id, split):
    """Load data/splits/pump_id_XX_<split>.npy → (N, 1, 128, 432) float32."""
    path = os.path.join(DATA_DIR, f"pump_id_{pump_id}_{split}.npy")
    arr  = np.load(path)                          # (N, 128, 431)
    arr  = arr[:, np.newaxis, :, :]               # (N, 1, 128, 431)
    if arr.shape[-1] < 432:
        arr = np.pad(arr, ((0,0),(0,0),(0,0),(0, 432 - arr.shape[-1])))
    return arr.astype(np.float32)


# ── Step 0: Load config + state dict ─────────────────────────────────────────
print("=" * 65)
print("  PHASE 3 — LOF Encoder Optimisation & Export")
print("=" * 65)

with open(DEPLOY_CONFIG_PATH) as f:
    config = json.load(f)

EMBEDDING_DIM = config["embedding_dim"]
TARGET_FREQ   = config["target_freq"]
TARGET_TIME   = config["target_time"]
THRESHOLD_PCT = config["threshold_pct"]

print(f"\n[0] Config loaded from deployment_config.json")
print(f"    Embedding dim  : {EMBEDDING_DIM}")
print(f"    Input shape    : (1, {TARGET_FREQ}, {TARGET_TIME})")
print(f"    LOF threshold  : {THRESHOLD_PCT}th percentile of normal scores")

# ── Auto-detect architecture from state dict ──────────────────────────────────
ckpt = torch.load(PTH_AUTOENCODER, map_location="cpu", weights_only=False)
sd   = ckpt["model_state_dict"]

# Encoder conv channel widths — indices 0,3,6,9 are the Conv2d weight tensors
ENC_CH = [
    sd["encoder.conv_blocks.0.weight"].shape[0],
    sd["encoder.conv_blocks.3.weight"].shape[0],
    sd["encoder.conv_blocks.6.weight"].shape[0],
    sd["encoder.conv_blocks.9.weight"].shape[0],
]
# Last conv out channels → fc input size
FC_IN = ENC_CH[-1]

print(f"\n[0] Architecture detected from state dict")
print(f"    Encoder channels : {ENC_CH}")
print(f"    FC in -> out     : {FC_IN} -> {EMBEDDING_DIM}")


# ── Architecture definition ───────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, channels, embedding_dim):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1,          channels[0], 3, stride=2, padding=1), nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True),
            nn.Conv2d(channels[0],channels[1], 3, stride=2, padding=1), nn.BatchNorm2d(channels[1]), nn.ReLU(inplace=True),
            nn.Conv2d(channels[1],channels[2], 3, stride=2, padding=1), nn.BatchNorm2d(channels[2]), nn.ReLU(inplace=True),
            nn.Conv2d(channels[2],channels[3], 3, stride=2, padding=1), nn.BatchNorm2d(channels[3]), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc   = nn.Linear(channels[3], embedding_dim)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ── Load encoder weights ──────────────────────────────────────────────────────
encoder = Encoder(ENC_CH, EMBEDDING_DIM)

# Strip "encoder." prefix from state dict keys
enc_sd = {k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}
encoder.load_state_dict(enc_sd)
encoder.eval()

n_params = sum(p.numel() for p in encoder.parameters())
print(f"\n[0] Encoder loaded  ({n_params:,} parameters)")


# ── Step 3.1: Export encoder to ONNX ─────────────────────────────────────────
print(f"\n[3.1] Exporting encoder to ONNX (opset {OPSET})...")

dummy = torch.zeros(1, 1, TARGET_FREQ, TARGET_TIME)

torch.onnx.export(
    encoder,
    dummy,
    ONNX_FULL_PATH,
    input_names=["mel_spectrogram"],
    output_names=["embedding"],
    dynamic_axes={
        "mel_spectrogram": {0: "batch_size"},
        "embedding":       {0: "batch_size"},
    },
    opset_version=OPSET,
    do_constant_folding=True,
)

onnx_full    = onnx.load(ONNX_FULL_PATH)
onnx.checker.check_model(onnx_full)
full_size_kb = total_size_kb(ONNX_FULL_PATH)

print(f"    Saved  : {ONNX_FULL_PATH}")
print(f"    Nodes  : {len(onnx_full.graph.node)}")
print(f"    Size   : {full_size_kb:.1f} KB")


# ── Step 3.2: Graph optimisation (onnxsim) ────────────────────────────────────
print(f"\n[3.2] Simplifying with onnxsim...")

sim_model, ok = simplify(onnx_full)
assert ok, "onnxsim simplification failed"
onnx.save(sim_model, ONNX_SIM_PATH)

sim_size_kb = total_size_kb(ONNX_SIM_PATH)
print(f"    Nodes  : {len(onnx_full.graph.node)} → {len(sim_model.graph.node)}")
print(f"    Size   : {full_size_kb:.1f} KB → {sim_size_kb:.1f} KB")
print(f"    Saved  : {ONNX_SIM_PATH}")


# ── Step 3.4a: Calibration data ───────────────────────────────────────────────
print(f"\n[3.4a] Building calibration dataset ({CALIB_PER_ID} samples × {len(PUMP_IDS)} pump IDs)...")

calib_data = np.concatenate([
    load_split(pid, "train")[:CALIB_PER_ID] for pid in PUMP_IDS
], axis=0)

print(f"    Shape : {calib_data.shape}  dtype={calib_data.dtype}")
print(f"    Range : [{calib_data.min():.4f}, {calib_data.max():.4f}]")


class CalibReader(CalibrationDataReader):
    def __init__(self, data, input_name="mel_spectrogram"):
        self._data  = data
        self._iname = input_name
        self._idx   = 0

    def get_next(self):
        if self._idx >= len(self._data):
            return None
        sample     = {self._iname: self._data[self._idx: self._idx + 1]}
        self._idx += 1
        return sample

    def rewind(self):
        self._idx = 0


# ── Step 3.4: INT8 static quantisation ───────────────────────────────────────
print(f"\n[3.4] Running INT8 static quantisation...")
print(f"    Format     : QDQ")
print(f"    Activations: QUInt8  (ORT AVX2 optimised Conv kernels)")
print(f"    Ops        : Conv + Gemm only  (no ConvTranspose in encoder)")

try:
    from onnxruntime.quantization import quant_pre_process
    quant_pre_process(ONNX_SIM_PATH, ONNX_PREP_PATH, skip_optimization=False)
    quant_input = ONNX_PREP_PATH
    print(f"    Pre-process: done → {ONNX_PREP_PATH}")
except Exception as e:
    print(f"    Pre-process skipped ({e})")
    quant_input = ONNX_SIM_PATH

quantize_static(
    model_input             = quant_input,
    model_output            = ONNX_INT8_PATH,
    calibration_data_reader = CalibReader(calib_data),
    quant_format            = QuantFormat.QDQ,
    per_channel             = False,
    activation_type         = QuantType.QUInt8,
    weight_type             = QuantType.QInt8,
    op_types_to_quantize    = ["Conv", "Gemm"],
)

onnx_int8    = onnx.load(ONNX_INT8_PATH)
int8_size_kb = total_size_kb(ONNX_INT8_PATH)

print(f"\n    FP32 size : {sim_size_kb:.1f} KB  ({len(sim_model.graph.node)} nodes)")
print(f"    INT8 size : {int8_size_kb:.1f} KB  ({len(onnx_int8.graph.node)} nodes)  "
      f"{(1 - int8_size_kb/sim_size_kb)*100:.1f}% smaller")
print(f"    Saved     : {ONNX_INT8_PATH}")


# ── Step 3.5: Validate AUC with LOF scores ────────────────────────────────────
print(f"\n[3.5] Validating AUC using LOF scores...")

with open(LOF_MODEL_PATH, "rb") as f:
    lof = pickle.load(f)

so = SessionOptions()
so.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

sess_fp32 = InferenceSession(ONNX_SIM_PATH,  sess_options=so, providers=["CPUExecutionProvider"])
sess_int8 = InferenceSession(ONNX_INT8_PATH, sess_options=so, providers=["CPUExecutionProvider"])


def get_embeddings_ort(session, data):
    """Run encoder ONNX session → (N, embedding_dim) numpy array."""
    iname = session.get_inputs()[0].name
    embs  = []
    for sample in data:
        emb = session.run(None, {iname: sample[np.newaxis]})[0]  # (1, 128)
        embs.append(emb[0])
    return np.array(embs)


def lof_scores(lof_model, embeddings):
    """LOF anomaly scores — higher = more anomalous."""
    return -lof_model.score_samples(embeddings)


# Load validation data — use test splits (unseen during training)
test_normal   = np.concatenate([load_split(pid, "test")   for pid in PUMP_IDS])
test_labels   = np.concatenate([
    np.load(os.path.join(DATA_DIR, f"pump_id_{pid}_test_labels.npy")) for pid in PUMP_IDS
])

print(f"    Test set : {len(test_normal)} samples  "
      f"(normal={int((test_labels==0).sum())}, abnormal={int((test_labels==1).sum())})")

print(f"    Extracting FP32 embeddings...")
fp32_embs   = get_embeddings_ort(sess_fp32, test_normal)
fp32_scores = lof_scores(lof, fp32_embs)
fp32_auc    = roc_auc_score(test_labels, fp32_scores)

print(f"    Extracting INT8 embeddings...")
int8_embs   = get_embeddings_ort(sess_int8, test_normal)
int8_scores = lof_scores(lof, int8_embs)
int8_auc    = roc_auc_score(test_labels, int8_scores)

print(f"\n    FP32 AUC : {fp32_auc:.4f}")
print(f"    INT8 AUC : {int8_auc:.4f}   (delta: {(int8_auc-fp32_auc)*100:+.2f}%)")


# ── Step 3.6: CPU latency benchmark ──────────────────────────────────────────
print(f"\n[3.6] CPU latency benchmark ({N_RUNS} runs, {N_WARMUP} warm-up)...")

dummy_np = dummy.numpy()


def bench(session, label):
    iname = session.get_inputs()[0].name
    for _ in range(N_WARMUP):
        session.run(None, {iname: dummy_np})
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        session.run(None, {iname: dummy_np})
    ms = (time.perf_counter() - t0) / N_RUNS * 1000
    print(f"    {label:<40s} : {ms:.3f} ms")
    return ms


print(f"    {'-'*55}")
fp32_ms = bench(sess_fp32, "FP32  encoder_simplified.onnx")
int8_ms = bench(sess_int8, "INT8  encoder_int8.onnx")
speedup = fp32_ms / int8_ms
print(f"    {'-'*55}")
print(f"    INT8 speedup : {speedup:.2f}x")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  PHASE 3 SUMMARY — LOF Encoder Optimisation")
print(f"{'='*65}")
print(f"  Weights  : {PTH_AUTOENCODER}")
print(f"  Input    : (1, {TARGET_FREQ}, {TARGET_TIME})  [C, Mel-bins, Time-frames]")
print(f"  Latent   : {EMBEDDING_DIM} dimensions")
print(f"  Scoring  : LOF (n_neighbors={lof.n_neighbors_}, novelty=True)")
print()
print(f"  Optimisation pipeline:")
print(f"  3.1  ONNX export     (encoder)    : {full_size_kb:>8.1f} KB   ({len(onnx_full.graph.node)} nodes)")
print(f"  3.2  onnxsim         (simplified) : {sim_size_kb:>8.1f} KB   ({len(sim_model.graph.node)} nodes)")
print(f"  3.4  INT8 quantised  (deploy)     : {int8_size_kb:>8.1f} KB   ({len(onnx_int8.graph.node)} nodes)  "
      f"{(1-int8_size_kb/sim_size_kb)*100:.1f}% smaller")
print()
print(f"  Accuracy (LOF scoring):")
print(f"    FP32 AUC : {fp32_auc:.4f}")
print(f"    INT8 AUC : {int8_auc:.4f}   (delta: {(int8_auc-fp32_auc)*100:+.2f}%)")
print()
print(f"  Latency  (CPU — ORT CPUExecutionProvider):")
print(f"    FP32 : {fp32_ms:.3f} ms   |   INT8 : {int8_ms:.3f} ms   |   Speedup : {speedup:.2f}x")
print()
print(f"  Deploy package:")
print(f"    Encoder : {ONNX_INT8_PATH}")
print(f"    LOF     : {LOF_MODEL_PATH}")
print(f"    Threshold: {THRESHOLD_PCT}th percentile of normal LOF scores")
print(f"{'='*65}")
