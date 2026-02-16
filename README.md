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

## Usage

(Detailed usage instructions will be added as the project develops.)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
