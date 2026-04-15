"""
test_training.py — Unit tests for the training pipeline.

Tests:
  - SpectrogramDataset shape and dtype
  - Padding from 431 to 432
  - DataLoader batch shape
  - Training data loads correctly from splits

Usage:
    cd tests
    python test_training.py
"""

import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from conv2d_model import TARGET_FREQ, TARGET_TIME

_ROOT      = Path(__file__).parent.parent
SPLITS_DIR = _ROOT / "data" / "splits"

all_passed = True


def check(name, condition, detail=""):
    global all_passed
    if condition:
        print(f"  [PASSED] {name}")
    else:
        print(f"  [FAILED] {name}{' — ' + detail if detail else ''}")
        all_passed = False


# ── Inline dataset class (mirrors training.py) ────────────────────────────────
class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, spectrograms, target_time=TARGET_TIME):
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
        return self.data[idx]


print(f"\nTRAINING PIPELINE TESTS")
print("-" * 50)

# ── Test 1: Dataset shape with 431 time frames ────────────────────────────────
dummy_specs = np.random.rand(10, TARGET_FREQ, 431).astype(np.float32)
ds = SpectrogramDataset(dummy_specs)

check(
    "Dataset length correct",
    len(ds) == 10,
    f"got {len(ds)}"
)
check(
    "Sample shape after padding [1, 128, 432]",
    ds[0].shape == (1, TARGET_FREQ, TARGET_TIME),
    f"got {tuple(ds[0].shape)}"
)
check(
    "Sample dtype is float32",
    ds[0].dtype == torch.float32,
)

# ── Test 2: Padding behaviour ─────────────────────────────────────────────────
check(
    "Padded column is zero",
    float(ds[0][0, :, -1].sum()) == 0.0,
    "last time frame should be zero-padded"
)

# ── Test 3: Truncation behaviour ──────────────────────────────────────────────
long_specs = np.random.rand(5, TARGET_FREQ, 500).astype(np.float32)
ds_long = SpectrogramDataset(long_specs)
check(
    "Long spectrogram truncated to TARGET_TIME",
    ds_long[0].shape[-1] == TARGET_TIME,
    f"got {ds_long[0].shape[-1]}"
)

# ── Test 4: DataLoader batch shape ────────────────────────────────────────────
dl     = DataLoader(ds, batch_size=4, shuffle=True)
batch  = next(iter(dl))
check(
    "DataLoader batch shape [4, 1, 128, 432]",
    batch.shape == (4, 1, TARGET_FREQ, TARGET_TIME),
    f"got {tuple(batch.shape)}"
)

# ── Test 5: Values normalised ─────────────────────────────────────────────────
normed = np.random.rand(5, TARGET_FREQ, 431).astype(np.float32)
ds_n   = SpectrogramDataset(normed)
check(
    "Values in [0, 1] preserved",
    float(ds_n.data.min()) >= 0.0 and float(ds_n.data.max()) <= 1.0,
    f"range [{ds_n.data.min():.4f}, {ds_n.data.max():.4f}]"
)

# ── Test 6: Real split files load correctly ───────────────────────────────────
train_file = SPLITS_DIR / "pump_id_00_train.npy"
if train_file.exists():
    specs = np.load(train_file)
    ds_real = SpectrogramDataset(specs)
    check(
        "Real train file loads — correct shape",
        ds_real[0].shape == (1, TARGET_FREQ, TARGET_TIME),
        f"got {tuple(ds_real[0].shape)}"
    )
    check(
        "Real train file — no NaNs",
        not torch.isnan(ds_real.data).any(),
    )
else:
    print(f"  [SKIPPED] Real split file not found: {train_file}")

print("-" * 50)
if all_passed:
    print("SUMMARY: All tests passed.")
else:
    print("SUMMARY: Some tests failed.")