# Echo-Check

A modular, ONNX-optimized pipeline for industrial machinery anomaly detection using spectral audio analysis.

## Project Overview

Echo-Check is designed to detect anomalies in industrial machinery by analyzing audio signals. It leverages spectral analysis and machine learning to provide robust, real-time monitoring.

## Project Structure

- `data/`: Raw and processed audio files (gitignored).
- `docs/`: Project documentation and architecture diagrams.
- `models/`: Trained model weights and exported .onnx files.
- `notebooks/`: Jupyter notebooks for EDA and experiments.
- `src/`: Core engine scripts for ingestion, feature extraction, inference, and evaluation.
- `tests/`: Unit tests for processing modules.

## Getting Started

### Prerequisites

- Python 3.9+
- [Conda](https://docs.conda.io/en/latest/) (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/echo-check.git
   cd echo-check
   ```

2. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate echo-check
   ```

   *Alternatively, use pip:*
   ```bash
   pip install -r requirements.txt
   ```

## Data Procurement

To get started with Echo-Check, you'll need the MIMII dataset. Follow these steps to procure and organize the data:

1. **Download the Dataset**:
   - Go to the [MIMII Zenodo Page](https://zenodo.org/record/3384388).
   - Scroll down to the **Files** section.
   - Look for the **-6_dB** files (these contain a realistic amount of background noise).
   - **Recommendation**: Download `pump.zip` (approx. 8GB) or `fan.zip` (approx. 10GB). The "Pump" dataset is preferred for its rich spectral features.

2. **Verification**:
   - Move the downloaded `.zip` file into the `data/raw` folder.
   - Verify the file integrity using MD5 hash:
     - **Windows (PowerShell)**: `Get-FileHash data/raw/pump.zip -Algorithm MD5`
     - **Mac/Linux**: `md5sum data/raw/pump.zip`
   - Compare the resulting string to the MD5 hash listed on the Zenodo page.

3. **Structure Your Data**:
   - Unzip the file into `data/raw`. The structure should look like this:
     ```text
     data/raw/pump/
     ├── 00/                  # Machine ID
     │   ├── normal/          # Normal .wav files
     │   └── anomaly/         # Anomalous .wav files
     └── 02/                  # Machine ID
     ```

4. **Sanity Check**:
   - Listen to a few files. A **normal** file should sound like a steady hum, while an **anomaly** file might contain clicks, grinds, or pitch changes.

5. **Note on Git**:
   - The `data/` directory is gitignored to prevent large files from being pushed to GitHub. Ensure your `.gitignore` is correctly configured.

## Usage

(Detailed usage instructions will be added as the project develops.)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
