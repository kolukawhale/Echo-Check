# Echo-Check

A modular, ONNX-optimized pipeline for industrial machinery anomaly detection using spectral audio analysis.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview

**Echo-Check** is designed to detect anomalies in industrial machinery by analyzing acoustic emissions. By leveraging Mel-spectrogram conversion and unsupervised learning, the system provides a robust framework for real-time equipment health monitoring.

## 📂 Project Structure

```text
Echo-Check/
├── data/               # Raw and processed audio files (gitignored)
├── docs/               # Project documentation and architecture diagrams
├── models/             # Trained weights and exported .onnx files
├── notebooks/          # Jupyter notebooks for EDA and experiments
├── src/                # Core engine scripts
│   ├── ingestion.py      # Mel-spectrogram extraction (Wav_to_mel class)
│   ├── preprocess_all.py # Batch processing automation
│   └── test_data.py      # Data integrity and normalization verification
└── tests/              # Unit tests for processing modules

🛠️ Getting Started
Prerequisites

Python 3.10+ (Developed on 3.10.16)

Conda (Recommended for environment management)

Installation

Clone the repository:

Bash
git clone [https://github.com/kolukawhale/Echo-Check.git](https://github.com/kolukawhale/Echo-Check.git)
cd Echo-Check
Set up the environment:

Bash
conda env create -f environment.yml
conda activate echo-check
📊 Data Procurement
Echo-Check utilizes the MIMII Dataset. To set up the data pipeline:

Download: Obtain the -6_dB files from the MIMII Zenodo Page. We recommend starting with pump.zip.

Organize: Unzip the files into data/raw/pump/ following the structure below:

Plaintext
data/raw/pump/
├── id_00/          # Machine ID
│   ├── normal/     # Normal .wav files
│   └── abnormal/   # Anomalous .wav files
└── id_02/          # Additional Machine IDs...
🚀 Usage
1. Data Preprocessing

Convert raw .wav audio into normalized 3D NumPy arrays (Mel-spectrograms). This script crawls all machine ID folders and saves optimized .npy files to data/processed/.

Bash
cd src
python preprocess_all.py
2. Verify Data Integrity

Before training, run the test suite to ensure all data cubes meet the required specifications (Shape: [Batch, 128, 431], Range: [0, 1]).

Bash
python test_data.py
📜 License
This project is licensed under the MIT License - see the LICENSE file for details.


Would you like me to generate the **`environment.yml`** file now so your teammate
```
