import numpy as np
from pathlib import Path

def test_processed_files(processed_dir):
    proc_path = Path(processed_dir)
    npy_files = list(proc_path.glob("*.npy"))
    
    if not npy_files:
        print("ERROR: No .npy files found! Check your paths.")
        return

    print(f"ANALYSIS: Starting integrity check on {len(npy_files)} files...")
    print("-" * 50)
    
    all_passed = True

    for file in npy_files:
        try:
            data = np.load(file)
            
            # 1. Check Dimensions (Should be 3D: Batch x Mels x Time)
            is_3d = len(data.shape) == 3
            
            # 2. Check Mel Bins (Should be 128)
            correct_mels = data.shape[1] == 128
            
            # 3. Check Normalization (Should be between 0 and 1)
            # We allow a tiny margin for floating point precision
            is_normalized = np.min(data) >= -1e-7 and np.max(data) <= 1.0000001
            
            # 4. Check for NaNs
            no_nans = not np.isnan(data).any()

            if is_3d and correct_mels and is_normalized and no_nans:
                status = "PASSED"
            else:
                status = "FAILED"
                all_passed = False
            
            print(f"[{status}] {file.name}")
            print(f"    Shape: {data.shape}")
            print(f"    Range: [{np.min(data):.4f}, {np.max(data):.4f}]")
            
            if status == "FAILED":
                if not is_3d: print("    -> REASON: Data is not 3D.")
                if not correct_mels: print(f"    -> REASON: Expected 128 mels, got {data.shape[1]}.")
                if not is_normalized: print("    -> REASON: Data out of [0, 1] range.")
                if not no_nans: print("    -> REASON: Found NaN values.")

        except Exception as e:
            print(f"CRITICAL ERROR loading {file.name}: {e}")
            all_passed = False

    print("-" * 50)
    if all_passed:
        print("SUMMARY: All tests passed successfully. Data is verified.")
    else:
        print("SUMMARY: Some tests failed. Please review the errors logged above.")

if __name__ == "__main__":
    # Ensure this path matches your structure
    PROCESSED_DIR = "../data/processed"
    test_processed_files(PROCESSED_DIR)