"""
conv2d_model.py — Conv2D Autoencoder architecture for Echo-Check.

Imported by training.py, evaluate_conv2d_lof.py, and app.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 128
TARGET_FREQ   = 128
TARGET_TIME   = 432


class Encoder(nn.Module):
    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1,   32,  3, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32,  64,  3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64,  128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class Decoder(nn.Module):
    _H, _W = 8, 27

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, 256 * self._H * self._W)
        self.deconv_blocks = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,  1,   4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z).view(z.size(0), 256, self._H, self._W)
        x = self.deconv_blocks(x)
        return F.interpolate(
            x, size=(TARGET_FREQ, TARGET_TIME), mode="bilinear", align_corners=False
        )


class CNNAutoencoder(nn.Module):
    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(embedding_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def get_embedding(self, x):
        return self.encoder(x)