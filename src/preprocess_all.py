import numpy as np
from pathlib import Path
from ingestion import Wav_to_mel 

def automate_ingestion(root_raw_path, output_path):
    ingestor = Wav_to_mel()
    root = Path(root_raw_path)
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # locate all the id_ folders (id_00, id_02, etc.)
    id_folders = [f for f in root.iterdir() if f.is_dir() and f.name.startswith("id_")]

    for id_folder in id_folders:
        machine_id = id_folder.name
        
        # 'normal' and 'abnormal' subfolders
        for category in ["normal", "abnormal"]:
            category_path = id_folder / category
            
            if category_path.exists():
                files = list(category_path.glob("*.wav"))
                if not files:
                    continue
                
                print(f"Processing {len(files)} files for {machine_id} ({category})...")
                
                all_specs = []
                for f in files:
                    audio = ingestor.load_audio(f)
                    spec = ingestor.mel_spectogram(audio)
                    all_specs.append(spec)
                
                # Save as pump_id_XX_category.npy
                save_name = f"pump_{machine_id}_{category}.npy"
                np.save(out_dir / save_name, np.array(all_specs))
                print(f"Successfully saved {save_name}")

if __name__ == "__main__":
    # Adjust these paths based on your Echo-Check structure
    RAW_DATA_PATH = "../data/raw/pump"
    PROCESSED_DATA_PATH = "../data/processed"
    
    automate_ingestion(RAW_DATA_PATH, PROCESSED_DATA_PATH)