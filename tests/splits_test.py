"""
splits_test.py — Integrity checks for the train/test split output.

Verifies shape, label correctness, ratio, and data integrity for every
machine ID found in the splits directory.

Usage:
    python tests/test_train_test_split.py
"""

import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
_ROOT         = Path(__file__).parent.parent
PROCESSED_DIR = _ROOT / "data" / "processed"
SPLITS_DIR    = _ROOT / "data" / "splits"
TRAIN_PERCENT = 0.8
TOLERANCE     = 0.05   # acceptable deviation from expected train/test ratio
# ─────────────────────────────────────────────────────────────────────────────


def test_split_files(splits_dir=SPLITS_DIR, processed_dir=PROCESSED_DIR):
    splits = Path(splits_dir)
    proc   = Path(processed_dir)

    train_files = sorted(splits.glob("pump_id_*_train.npy"))
    if not train_files:
        print(f"ERROR: No train .npy files found in '{splits_dir}'.")
        return

    print(f"ANALYSIS: Found {len(train_files)} machine ID(s) to check.")
    print("-" * 60)

    all_passed = True

    for train_file in train_files:
        prefix     = train_file.name.replace("_train.npy", "")   # e.g. pump_id_00
        machine_id = prefix.replace("pump_", "")
        print(f"\n[{machine_id}]")

        # ── Load split files ──────────────────────────────────────────────────
        test_file   = splits / f"{prefix}_test.npy"
        labels_file = splits / f"{prefix}_test_labels.npy"

        try:
            X_train = np.load(train_file)
            X_test  = np.load(test_file)
            y_test  = np.load(labels_file)
        except FileNotFoundError as e:
            print(f"  FAILED — Missing file: {e}")
            all_passed = False
            continue

        # ── Load original processed files for total count reference ───────────
        normal_path   = proc / f"{prefix}_normal.npy"
        abnormal_path = proc / f"{prefix}_abnormal.npy"

        try:
            X_normal = np.load(normal_path)
            n_normal = len(X_normal)
        except FileNotFoundError:
            print(f"  FAILED — Original normal file not found: {normal_path}")
            all_passed = False
            continue

        n_abnormal = len(np.load(abnormal_path)) if abnormal_path.exists() else 0

        # ── Tests ─────────────────────────────────────────────────────────────
        results = {}

        # 1. No overlap — train + test_normal should equal total normal samples
        n_test_normal = int((y_test == 0).sum())
        results["No data loss"] = (len(X_train) + n_test_normal) == n_normal

        # 2. Train/test ratio is approximately correct
        actual_train_pct = len(X_train) / n_normal
        results["Train ratio (~80%)"] = abs(actual_train_pct - TRAIN_PERCENT) <= TOLERANCE

        # 3. Train size is non-empty
        results["Train size > 0"] = len(X_train) > 0

        # 4. Test set size = held-out normal + all abnormal
        results["Test size correct"] = len(X_test) == (n_test_normal + n_abnormal)

        # 5. Labels length matches test set
        results["Labels match test size"] = len(y_test) == len(X_test)

        # 6. Labels are only 0s and 1s
        results["Labels are binary"] = set(np.unique(y_test)).issubset({0, 1})

        # 7. Abnormal count in labels matches source
        results["Abnormal label count correct"] = int((y_test == 1).sum()) == n_abnormal

        # 8. Correct spectrogram shape [N, 128, 431]
        results["Train shape valid"] = (X_train.ndim == 3 and X_train.shape[1] == 128
                                        and X_train.shape[2] == 431)
        results["Test shape valid"]  = (X_test.ndim  == 3 and X_test.shape[1]  == 128
                                        and X_test.shape[2]  == 431)

        # 9. No NaNs
        results["No NaNs in train"] = not np.isnan(X_train).any()
        results["No NaNs in test"]  = not np.isnan(X_test).any()

        # 10. Data is normalized [0, 1]
        results["Train normalized"] = (X_train.min() >= -1e-7
                                       and X_train.max() <= 1.0000001)
        results["Test normalized"]  = (X_test.min()  >= -1e-7
                                       and X_test.max()  <= 1.0000001)

        # ── Print results ─────────────────────────────────────────────────────
        for check, passed in results.items():
            status = "PASSED" if passed else "FAILED"
            if not passed:
                all_passed = False
            print(f"  [{status}] {check}")

        print(f"\n  Summary — Normal: {n_normal} | Abnormal: {n_abnormal}")
        print(f"  Train: {len(X_train)} ({actual_train_pct*100:.1f}%)  |  "
              f"Test: {len(X_test)} (normal={n_test_normal}, abnormal={n_abnormal})")

    print("\n" + "-" * 60)
    if all_passed:
        print("SUMMARY: All tests passed. Split is verified.")
    else:
        print("SUMMARY: Some tests failed. Review the errors above.")


if __name__ == "__main__":
    test_split_files()