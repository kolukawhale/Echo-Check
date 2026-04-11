"""
app.py — Echo-Check Streamlit frontend.

Runs the Conv2D + LOF anomaly detection pipeline on an uploaded .wav file.

Usage:
    streamlit run app.py
"""

import sys
import streamlit as st
import numpy as np
import pickle
import torch
import tempfile
import os
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent / "src"))
from conv2d_model import CNNAutoencoder, EMBEDDING_DIM, TARGET_FREQ, TARGET_TIME

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Echo-Check",
    page_icon="🔊",
    layout="centered",
)

# ── Config ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent

CHECKPOINT    = _ROOT / "models" / "conv2d" / "autoencoder.pth"
LOF_MODEL     = _ROOT / "models" / "conv2d" / "lof_model.pkl"
TRAIN_NPYS    = {
    "id_00": _ROOT / "data/splits/pump_id_00_train.npy",
    "id_02": _ROOT / "data/splits/pump_id_02_train.npy",
    "id_04": _ROOT / "data/splits/pump_id_04_train.npy",
    "id_06": _ROOT / "data/splits/pump_id_06_train.npy",
}
THRESHOLD_PCT = 95


# ── Cached model loading ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    ckpt   = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model  = CNNAutoencoder(embedding_dim=EMBEDDING_DIM).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with open(LOF_MODEL, "rb") as f:
        lof = pickle.load(f)
    return model, lof, device


@st.cache_resource
def load_thresholds():
    model, lof, device = load_model()
    thresholds = {}
    for machine_id, npy_path in TRAIN_NPYS.items():
        if not npy_path.exists():
            continue
        specs      = np.load(npy_path)
        embeddings = extract_embeddings(model, specs, device)
        scores     = np.array([-lof.score_samples(e.reshape(1, -1))[0] for e in embeddings])
        thresholds[machine_id] = float(np.percentile(scores, THRESHOLD_PCT))
    return thresholds


# ── Inference helpers ─────────────────────────────────────────────────────────
def wav_to_spectrogram(wav_path: str) -> np.ndarray:
    import librosa
    audio, _ = librosa.load(wav_path, sr=22050, mono=True)
    audio, _ = librosa.effects.trim(audio)
    spec     = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128)
    mel_db   = librosa.power_to_db(spec, ref=np.max)
    return (mel_db + 80) / 80


def pad_spectrogram(spec: np.ndarray) -> torch.Tensor:
    t = spec.shape[1]
    if t < TARGET_TIME:
        spec = np.pad(spec, ((0, 0), (0, TARGET_TIME - t)), mode="constant")
    elif t > TARGET_TIME:
        spec = spec[:, :TARGET_TIME]
    return torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def extract_embeddings(model, spectrograms, device):
    results = []
    for i in range(0, len(spectrograms), 64):
        batch = spectrograms[i:i+64]
        batch = torch.tensor(batch, dtype=torch.float32).unsqueeze(1)
        if batch.shape[-1] < TARGET_TIME:
            batch = torch.nn.functional.pad(batch, (0, TARGET_TIME - batch.shape[-1]))
        emb = model.encoder(batch.to(device))
        results.append(emb.cpu().numpy())
    return np.concatenate(results, axis=0)


@torch.no_grad()
def predict(wav_path: str, machine_id: str) -> dict:
    model, lof, device = load_model()
    thresholds         = load_thresholds()
    spec    = wav_to_spectrogram(wav_path)
    tensor  = pad_spectrogram(spec).to(device)
    emb     = model.encoder(tensor).cpu().numpy()
    score   = float(-lof.score_samples(emb.reshape(1, -1))[0])
    thresh  = thresholds.get(machine_id, 1.5)
    label   = "ANOMALY" if score > thresh else "NORMAL"
    return {"label": label, "score": score, "threshold": thresh, "spec": spec}


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔊 Echo-Check")
st.caption("Industrial pump anomaly detection — Conv2D Autoencoder + LOF scoring")

st.divider()

col1, col2 = st.columns(2)
with col1:
    machine_id = st.selectbox(
        "Machine ID",
        options=["id_00", "id_02", "id_04", "id_06"],
        help="Select the pump machine ID to calibrate the anomaly threshold against."
    )
with col2:
    st.metric("Threshold percentile", f"{THRESHOLD_PCT}th")

wav_file = st.file_uploader(
    "Upload a pump recording (.wav)",
    type=["wav"],
    help="Upload a 10-second .wav file from the MIMII dataset or a real pump recording."
)

if wav_file is not None:
    st.audio(wav_file, format="audio/wav")

    with st.spinner("Running anomaly detection..."):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_file.read())
            tmp_path = tmp.name
        try:
            result = predict(tmp_path, machine_id)
        finally:
            os.unlink(tmp_path)

    st.divider()

    label = result["label"]
    if label == "ANOMALY":
        st.error(f"⚠️  {label}", icon="🚨")
    else:
        st.success(f"✅  {label}", icon="✅")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Anomaly Score",  f"{result['score']:.4f}")
    col_b.metric("Threshold",      f"{result['threshold']:.4f}")
    col_c.metric("Machine ID",     machine_id)

    st.subheader("Mel-spectrogram")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(result["spec"], aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel bins")
    ax.set_title(f"{wav_file.name} — {label}")
    plt.colorbar(ax.images[0], ax=ax, label="Normalised amplitude")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

else:
    st.info("Upload a .wav file to run anomaly detection.", icon="ℹ️")

st.divider()
st.caption("Echo-Check — CS5130 Applied AI | Northeastern University")