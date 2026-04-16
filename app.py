"""
app.py — Echo-Check Streamlit frontend.

Usage:
    streamlit run app.py
"""

import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Echo-Check",
    layout="centered",
)

# ── Config ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent

ONNX_ENCODER = _ROOT / "models" / "phase3_outputs_lof" / "encoder_int8.onnx"
LOF_MODEL     = _ROOT / "models" / "conv2d" / "lof_model.pkl"
TRAIN_NPYS    = {
    "id_00": _ROOT / "data/splits/pump_id_00_train.npy",
    "id_02": _ROOT / "data/splits/pump_id_02_train.npy",
    "id_04": _ROOT / "data/splits/pump_id_04_train.npy",
    "id_06": _ROOT / "data/splits/pump_id_06_train.npy",
}
TARGET_FREQ   = 128
TARGET_TIME   = 432
THRESHOLD_PCT = 95


# ── Cached model loading ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load ONNX encoder session and LOF model."""
    so = SessionOptions()
    so.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    session = InferenceSession(
        str(ONNX_ENCODER),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    with open(LOF_MODEL, "rb") as f:
        lof = pickle.load(f)
    return session, lof


@st.cache_resource
def load_thresholds_and_embeddings():
    """
    Pre-compute LOF score thresholds and cache training embeddings per machine ID.
    Returns:
        thresholds   : dict[machine_id -> float]
        train_embeddings : dict[machine_id -> np.ndarray (N, embedding_dim)]
    """
    session, lof = load_model()
    thresholds       = {}
    train_embeddings = {}
    for machine_id, npy_path in TRAIN_NPYS.items():
        if not npy_path.exists():
            continue
        specs      = np.load(npy_path)                          # (N, 128, T)
        embeddings = extract_embeddings(session, specs)         # (N, embedding_dim)
        scores     = np.array([-lof.score_samples(e.reshape(1, -1))[0]
                                for e in embeddings])
        thresholds[machine_id]       = float(np.percentile(scores, THRESHOLD_PCT))
        train_embeddings[machine_id] = embeddings
    return thresholds, train_embeddings


# ── Inference helpers ─────────────────────────────────────────────────────────
def wav_to_spectrogram(wav_path: str) -> np.ndarray:
    """Load WAV and convert to normalised mel spectrogram."""
    import librosa
    audio, _ = librosa.load(wav_path, sr=22050, mono=True)
    audio, _ = librosa.effects.trim(audio)
    spec     = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=TARGET_FREQ)
    mel_db   = librosa.power_to_db(spec, ref=np.max)
    return (mel_db + 80) / 80


def pad_spectrogram(spec: np.ndarray) -> np.ndarray:
    """Pad or trim spectrogram to TARGET_TIME frames, return (1,1,128,432) float32."""
    t = spec.shape[1]
    if t < TARGET_TIME:
        spec = np.pad(spec, ((0, 0), (0, TARGET_TIME - t)), mode="constant")
    else:
        spec = spec[:, :TARGET_TIME]
    return spec[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1,1,128,432)


def extract_embeddings(session, spectrograms: np.ndarray) -> np.ndarray:
    """Run ONNX encoder on a batch of spectrograms -> (N, embedding_dim)."""
    iname   = session.get_inputs()[0].name
    results = []
    for i in range(len(spectrograms)):
        spec = spectrograms[i]                              # (128, T)
        inp  = pad_spectrogram(spec)                        # (1,1,128,432)
        emb  = session.run(None, {iname: inp})[0]           # (1, embedding_dim)
        results.append(emb[0])
    return np.array(results)                                # (N, embedding_dim)


def predict(wav_path: str, machine_id: str) -> dict:
    """Full inference pipeline: WAV -> spectrogram -> ONNX embedding -> LOF score."""
    session, lof             = load_model()
    thresholds, _            = load_thresholds_and_embeddings()

    spec  = wav_to_spectrogram(wav_path)
    inp   = pad_spectrogram(spec)

    iname = session.get_inputs()[0].name
    emb   = session.run(None, {iname: inp})[0]              # (1, embedding_dim)

    score  = float(-lof.score_samples(emb.reshape(1, -1))[0])
    thresh = thresholds.get(machine_id, 1.5)
    label  = "ANOMALY" if score > thresh else "NORMAL"

    return {
        "label":     label,
        "score":     score,
        "threshold": thresh,
        "spec":      spec,
        "embedding": emb[0],                                # (embedding_dim,)
    }


# ── Plot helpers ──────────────────────────────────────────────────────────────
def plot_pca_cluster(train_embs: np.ndarray, input_emb: np.ndarray,
                     machine_id: str, label: str):
    """
    Project training embeddings and the input embedding into 2D via PCA.
    Training points: pink circles (#FF69B4).
    Input point:     yellow star (#FFD700).
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    all_embs = np.vstack([train_embs, input_emb.reshape(1, -1)])  # (N+1, D)
    pca      = PCA(n_components=2)
    projected = pca.fit_transform(all_embs)                        # (N+1, 2)

    train_2d = projected[:-1]   # (N, 2)
    input_2d = projected[-1]    # (2,)

    var_explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        train_2d[:, 0], train_2d[:, 1],
        c="#FF69B4", alpha=0.5, s=18, linewidths=0,
        label=f"Training cluster ({machine_id})"
    )
    ax.scatter(
        input_2d[0], input_2d[1],
        c="#FFD700", marker="*", s=220, zorder=5, linewidths=0.8,
        edgecolors="#B8960C",
        label=f"Input file ({label})"
    )
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% variance)")
    ax.set_title(f"LOF embedding space — {machine_id}")
    ax.legend(framealpha=0.9, fontsize=9)
    plt.tight_layout()
    return fig


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Echo-Check")
st.caption("Industrial pump anomaly detection — Conv2D Encoder (ONNX) + LOF scoring")

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
        st.error(label)
    else:
        st.success(label)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Anomaly Score",  f"{result['score']:.4f}")
    col_b.metric("Threshold",      f"{result['threshold']:.4f}")
    col_c.metric("Machine ID",     machine_id)

    st.divider()

    # ── Spectrogram ───────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    st.subheader("Mel-spectrogram")
    fig_spec, ax_spec = plt.subplots(figsize=(10, 3))
    ax_spec.imshow(result["spec"], aspect="auto", origin="lower", cmap="viridis")
    ax_spec.set_xlabel("Time frames")
    ax_spec.set_ylabel("Mel bins")
    ax_spec.set_title(f"{wav_file.name} — {label}")
    plt.colorbar(ax_spec.images[0], ax=ax_spec, label="Normalised amplitude")
    plt.tight_layout()
    st.pyplot(fig_spec)
    plt.close(fig_spec)

    # ── PCA cluster plot ──────────────────────────────────────────────────────
    st.subheader("Embedding space (PCA)")
    st.caption(
        "Training embeddings for the selected machine ID are projected to 2D via PCA. "
        "The input file is projected into the same space. Points far from the training "
        "cluster correspond to higher LOF scores."
    )

    _, train_embeddings = load_thresholds_and_embeddings()
    if machine_id in train_embeddings:
        fig_pca = plot_pca_cluster(
            train_embeddings[machine_id],
            result["embedding"],
            machine_id,
            label,
        )
        st.pyplot(fig_pca)
        plt.close(fig_pca)
    else:
        st.warning(f"Training embeddings not found for {machine_id}.")

else:
    st.info("Upload a .wav file to run anomaly detection.")

st.divider()
st.caption("Echo-Check — CS5130 Applied AI | Northeastern University")