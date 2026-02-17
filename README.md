Markdown

# Echo-Check

A modular, ONNX-optimized pipeline for industrial machinery anomaly detection using spectral audio analysis.

## Project Overview

Echo-Check is designed to detect anomalies in industrial machinery by analyzing audio signals. It leverages spectral analysis and machine learning to provide robust, real-time monitoring.

## Project Structure

- `data/`: Raw and processed audio files (gitignored).
- `docs/`: Project documentation and architecture diagrams.
- `models/`: Trained model weights and exported .onnx files.
- `notebooks/`: Jupyter notebooks for EDA and experiments.
- `src/`: Core engine scripts:
  - `ingestion.py`: Contains the `Wav_to_mel` class for audio loading, trimming, and Mel-spectrogram conversion.
  - `preprocess_all.py`: Automation script to batch-process all machine IDs into normalized NumPy arrays.
  - `test_data.py`: Integrity suite to verify data shapes and normalization ranges.
- `tests/`: Unit tests for processing modules.

## Getting Started

### Prerequisites

- Python 3.10+ (Current environment uses 3.10.16)
- [Conda](https://docs.conda.io/en/latest/) (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/kolukawhale/Echo-Check.git](https://github.com/kolukawhale/Echo-Check.git)
   cd Echo-Check
   Create and activate the Conda environment:
   ```

Bash
conda env create -f environment.yml
conda activate echo-check
Data Procurement
To get started with Echo-Check, you'll need the MIMII dataset. Follow these steps to procure and organize the data:

Download the Dataset:

Download the -6_dB files from the MIMII Zenodo Page.

Recommendation: Download pump.zip.

Structure Your Data:

Unzip the files into data/raw/pump/. Ensure the directory structure follows this pattern:

Plaintext
data/raw/pump/
├── id_00/ # Machine ID
│ ├── normal/ # Normal .wav files
│ └── abnormal/ # Anomalous .wav files
├── id_02/
└── ...
Usage

1. Data Preprocessing

To convert raw .wav files into normalized Mel-spectrograms stored as .npy arrays, run the batch processor from the src directory:

Bash
cd src
python preprocess_all.py 2. Verify Data Integrity

After preprocessing, run the test suite to ensure all generated data cubes have a consistent shape (Batch, 128, 431) and are normalized within the [0, 1] range:

Bash
python test_data.py
License
This project is licensed under the MIT License - see the LICENSE file for details.
