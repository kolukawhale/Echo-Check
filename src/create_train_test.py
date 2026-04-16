"""
create_train_test.py — Train/Test split for Echo-Check (MIMII unsupervised protocol).

Train : normal samples only  (80%)
Test  : remaining normal (20%) + all abnormal samples
Labels: 0 = normal, 1 = abnormal

Saved files (per machine ID):
    pump_id_XX_train.npy   -> normal spectrograms for training
    pump_id_XX_test.npy    -> mixed normal + abnormal spectrograms
    pump_id_XX_test_labels.npy -> corresponding labels (0s and 1s)
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# GLOBAL VARIABLES
TRAIN_PERCENT = 0.8
RANDOM_SEED = 42

def create_split(train_percent = TRAIN_PERCENT, proc_path = "../data/processed",
                 out_path = "../data/splits", random_seed = RANDOM_SEED):

    if train_percent > 1 or train_percent < 0:
        raise Exception("Improper value of variable train percent has been passed")
    proc = Path(proc_path)
    out_dir = Path(out_path)
    try:
        #confirming out_dir location
        if not out_dir.exists():
            out_dir.mkdir(parents = True, exist_ok = True)
    except Exception as e:
        print(f"{e}")
        print("File loaction error")
        return
    
    # Find normal .npy processed files
    normal = sorted(list(proc.glob("pump_id_*_normal.npy")))
    num_normal = len(normal)

    # confirm files have been extracted
    print(f"Number of normal files found = {num_normal}\n")
    if num_normal == 0:
        raise FileNotFoundError(f"No normal .npy files found in '{proc}'.")

    for f in normal:
        # machine id
        machine_id = f.name.replace("pump_id", "").replace("_normal.npy", "")
        print(f"-------- id{machine_id} --------")
        print(f"Processing machine_id : {machine_id}")

        # access the correct .npy file and laod into variable 
        try:
            X_normal = np.load(f)
            print(f"Success loading file : {f}")
            print(f"Sample loaded from {f} = {len(X_normal)}")
        except Exception as e:
            print(f"Problem occured while loading file : {f}")
            return
        
        # Load the correspodning abnormal file
        abnormal_path = proc / f.name.replace("normal", "abnormal")
        if abnormal_path.exists():
            try:
                X_abnormal = np.load(abnormal_path)
                print(f"Success loading file : {abnormal_path}")
                print(f"Sample loaded from {abnormal_path} = {len(X_abnormal)}")
            except Exception as e:
                print(f"Problem loading {abnormal_path}\n{e}")
                return
        else:
            X_abnormal = np.empty((0, *X_normal.shape[1:]), dtype=X_normal.dtype)
            print(f"{abnormal_path}: 0 samples (file not found)")
        
        #  Split normal → train samples percent = train_percent, test_normal = 1 - train_percent
        X_train, X_test_normal = train_test_split(
            X_normal, test_size = 1 - train_percent, random_state=random_seed
        )

        # Test set = held-out normal + all abnormal
        X_test = np.concatenate([X_test_normal, X_abnormal], axis=0)
        y_test = np.concatenate([
            np.zeros(len(X_test_normal), dtype=np.int8),
            np.ones(len(X_abnormal),     dtype=np.int8),
        ])

        # Updates
        prefix = f.name.replace("_normal.npy", "")
        np.save(out_dir / f"{prefix}_train.npy",        X_train)
        np.save(out_dir / f"{prefix}_test.npy",         X_test)
        np.save(out_dir / f"{prefix}_test_labels.npy",  y_test)

        print(f"  Train    : {len(X_train)} samples  -> {prefix}_train.npy")
        print(f"  Test     : {len(X_test)} samples (normal={len(X_test_normal)}, abnormal={len(X_abnormal)})  -> {prefix}_test.npy")
        print(f"  Labels   : {y_test.shape}  -> {prefix}_test_labels.npy\n")

    print("Split complete. Files saved to:", out_dir.resolve())
    


if __name__ == "__main__":
    create_split(TRAIN_PERCENT, "../data/processed", "../data/splits", RANDOM_SEED)