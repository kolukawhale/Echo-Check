"""
test_lof.py — Unit tests for LOF scoring on encoder embeddings.

Tests:
  - Embedding extraction shape
  - LOF fitting and scoring
  - Anomaly scores are higher for outliers than inliers
  - Threshold derived from normal scores
  - Checkpoint loading

Usage:
    cd tests
    python test_lof.py
"""

import sys
import numpy as np
import torch
from pathlib import Path
from sklearn.neighbors import LocalOutlierFactor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from conv2d_model import CNNAutoencoder, EMBEDDING_DIM, TARGET_FREQ, TARGET_TIME

_ROOT      = Path(__file__).parent.parent
CHECKPOINT = _ROOT / "models" / "conv2d" / "autoencoder.pth"

all_passed = True


def check(name, condition, detail=""):
    global all_passed
    if condition:
        print(f"  [PASSED] {name}")
    else:
        print(f"  [FAILED] {name}{' — ' + detail if detail else ''}")
        all_passed = False


print(f"\nLOF SCORING TESTS")
print("-" * 50)

# ── Setup: random model and embeddings ────────────────────────────────────────
model = CNNAutoencoder(embedding_dim=EMBEDDING_DIM)
model.eval()

# Simulate normal training embeddings — tight cluster around origin
np.random.seed(42)
normal_embeddings = np.random.randn(100, EMBEDDING_DIM).astype(np.float32) * 0.5

# Simulate abnormal embeddings — far from the cluster
outlier_embeddings = (np.random.randn(10, EMBEDDING_DIM).astype(np.float32) * 0.5
                      + 10.0)   # shifted far away

# ── Test 1: LOF fits without error ────────────────────────────────────────────
try:
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, metric="euclidean")
    lof.fit(normal_embeddings)
    check("LOF fits on normal embeddings", True)
except Exception as e:
    check("LOF fits on normal embeddings", False, str(e))

# ── Test 2: Normal scores are lower than outlier scores ───────────────────────
normal_scores  = -lof.score_samples(normal_embeddings)
outlier_scores = -lof.score_samples(outlier_embeddings)

check(
    "Outlier scores higher than normal scores",
    outlier_scores.mean() > normal_scores.mean(),
    f"normal mean={normal_scores.mean():.4f}, outlier mean={outlier_scores.mean():.4f}"
)

# ── Test 3: Threshold at 95th percentile ──────────────────────────────────────
threshold = float(np.percentile(normal_scores, 95))
check(
    "95th percentile threshold > normal mean",
    threshold > normal_scores.mean(),
    f"threshold={threshold:.4f}, normal mean={normal_scores.mean():.4f}"
)

# ── Test 4: At least 95% of normal scores below threshold ─────────────────────
pct_below = (normal_scores < threshold).mean()
check(
    "~95% of normal scores below threshold",
    abs(pct_below - 0.95) < 0.02,
    f"got {pct_below:.2%}"
)

# ── Test 5: Outliers mostly flagged above threshold ───────────────────────────
flagged = (outlier_scores > threshold).mean()
check(
    "Most outliers flagged above threshold",
    flagged >= 0.8,
    f"only {flagged:.0%} flagged"
)

# ── Test 6: Embedding extraction from model ───────────────────────────────────
dummy = torch.randn(8, 1, TARGET_FREQ, TARGET_TIME)
with torch.no_grad():
    emb = model.encoder(dummy).numpy()

check(
    "Embedding extraction shape [8, 128]",
    emb.shape == (8, EMBEDDING_DIM),
    f"got {emb.shape}"
)
check(
    "Embeddings are finite",
    np.isfinite(emb).all(),
)

# ── Test 7: Checkpoint loads correctly (if available) ─────────────────────────
if CHECKPOINT.exists():
    try:
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        model_loaded = CNNAutoencoder(embedding_dim=EMBEDDING_DIM)
        model_loaded.load_state_dict(ckpt["model_state_dict"])
        model_loaded.eval()
        check("Checkpoint loads without error", True)

        # Verify loaded model produces correct embedding shape
        with torch.no_grad():
            emb_loaded = model_loaded.encoder(dummy).numpy()
        check(
            "Loaded model embedding shape correct",
            emb_loaded.shape == (8, EMBEDDING_DIM),
        )
    except Exception as e:
        check("Checkpoint loads without error", False, str(e))
else:
    print(f"  [SKIPPED] Checkpoint not found: {CHECKPOINT}")

print("-" * 50)
if all_passed:
    print("SUMMARY: All tests passed.")
else:
    print("SUMMARY: Some tests failed.")