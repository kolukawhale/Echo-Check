# Echo-Check

Modular, ONNX-optimized pipeline for industrial machinery anomaly detection using spectral audio analysis.

## Project Overview

Echo-Check is a system designed to detect anomalies in industrial machinery by analyzing acoustic emissions. It utilizes Mel-spectrogram conversion and unsupervised learning techniques to provide a robust framework for equipment health monitoring. The pipeline is optimized for edge deployment using ONNX.

## Project Structure

| Directory/File | Description |
| :--- | :--- |
| `data/` | Raw and processed audio files (excluded from version control) |
| `docs/` | Project documentation and architecture diagrams |
| `models/` | Trained weights and exported .onnx files |
| `notebooks/` | Jupyter notebooks for EDA and experiments |
| `src/` | Core source code for data processing and inference |
| `src/ingestion.py` | Mel-spectrogram extraction using the `Wav_to_mel` class |
| `src/preprocess_all.py` | Batch processing automation for dataset preparation |
| `src/create_train_test.py` | Train/Test split creation |
| `src/features.py` | Spectrogram and MFCC extraction logic |
| `src/inference.py` | ONNX runtime wrapper for anomaly detection |
| `src/evaluation.py` | Metrics calculation and baseline comparisons |
| `tests/` | Unit tests for processing modules |
| `tests/test_processed_data.py` | Data integrity and normalization verification |
| `tests/splits_test.py` | Train/Test split verification |
| `requirements.txt` | Python package dependencies |
| `environment.yml` | Conda environment specification |

## Getting Started

### Prerequisites

* Python 3.9
* Conda (recommended for environment management)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kolukawhale/Echo-Check.git
   cd Echo-Check
   ```

2. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate echo-check
   ```

3. Alternatively, install dependencies via pip:
   ```bash
   pip install -r requirements.txt
   ```

## Data Procurement

Echo-Check utilizes the MIMII Dataset. Follow these steps to set up the data pipeline:

1. **Download**: Obtain the -6 dB files from the [MIMII Zenodo Page](https://zenodo.org/record/3384388). It is recommended to start with the pump dataset.
2. **Integrity Check**: Verify the integrity of the downloaded `.zip` files via MD5 hash comparison before proceeding.
3. **Organization**: Unzip the files into `data/raw/pump/` maintaining the following structure:
   ```text
   data/raw/pump/
   └── id_00/
       ├── normal/
       └── abnormal/
   ```

## Usage

### 1. Data Preprocessing

Convert raw `.wav` audio into normalized 3D NumPy arrays (Mel-spectrograms). This script processes all machine ID folders and saves optimized `.npy` files to `data/processed/`.

```bash
cd src
python preprocess_all.py
```

### 2. Verify Data Integrity

Ensure all processed data cubes meet the required specifications (Shape: [Batch, 128, 431], Range: [0, 1]).

```bash
cd ../tests
python test_processed_data.py
```

### 3. Create Train/Test Split

Generate train and test sets from the processed data. This script splits normal data and combines all abnormal data into the test set.

```bash
cd ../src
python create_train_test.py
```

### 4. Verify Splits

Verify the integrity of the train/test splits, ensuring no data leakage and correct shapes.

```bash
cd ../tests
python splits_test.py
```

## License

This project is licensed under the MIT License.
