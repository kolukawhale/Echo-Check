import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# How many data points of audio we capture per second, unit is Hz
SAMPLE_RATE = 22050
# 128 rows classfiying sounds that may indicate failure of machine
MEL_BINS = 128

class Wav_to_mel:
    def __init__(self, sample_rate=SAMPLE_RATE, n_mels=MEL_BINS):
        self.sr = sample_rate
        self.n_mels = n_mels

    def load_audio(self, file_path):
        """Loads audio and ensures it is mono and at the correct sample rate."""
        # regenerates audio using the given sample rate, and ensures it is mono
        audio, _ = librosa.load(file_path, sr=self.sr, mono=True)
        # trims silence at start and end
        audio, _ = librosa.effects.trim(audio) 
        return audio
    
    def mel_spectogram(self, audio):
        """Converts raw audio into a log-scaled Mel-Spectrogram."""

        # Make mel-spectogram
        spect = librosa.feature.melspectrogram(
            y = audio, 
            sr = self.sr, 
            n_mels = self.n_mels
        )
        
        # Convert to a log scale: Hz to (Decibels)
        mel_db = librosa.power_to_db(spect, ref=np.max)

        norm_spec = (mel_db + 80) / 80
        
        return norm_spec

    def visualize(self, mel_db, title="Mel-Spectrogram"):
        """Plots the spectrogram for a 'smoke test'."""
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_db, sr=self.sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    pass
    