"""
test_conv2d_model.py — Unit tests for the Conv2D Autoencoder architecture.

Tests:
  - Encoder output shape
  - Decoder output shape
  - Full autoencoder input/output shape match
  - Decoder output constrained to [0, 1] by Sigmoid
  - Embedding dimension matches config

Usage:
    cd tests
    python test_conv2d_model.py
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from conv2d_model import CNNAutoencoder, Encoder, Decoder, EMBEDDING_DIM, TARGET_FREQ, TARGET_TIME

BATCH_SIZE = 4
all_passed = True


def check(name, condition, detail=""):
    global all_passed
    if condition:
        print(f"  [PASSED] {name}")
    else:
        print(f"  [FAILED] {name}{' — ' + detail if detail else ''}")
        all_passed = False


print(f"\nARCHITECTURE TESTS")
print("-" * 50)

dummy = torch.randn(BATCH_SIZE, 1, TARGET_FREQ, TARGET_TIME)
model = CNNAutoencoder(embedding_dim=EMBEDDING_DIM)
model.eval()

with torch.no_grad():
    recon, emb = model(dummy)

# 1. Encoder output shape
check(
    "Encoder output shape",
    emb.shape == (BATCH_SIZE, EMBEDDING_DIM),
    f"got {tuple(emb.shape)}"
)

# 2. Full autoencoder output matches input shape
check(
    "Autoencoder output shape matches input",
    recon.shape == dummy.shape,
    f"got {tuple(recon.shape)}, expected {tuple(dummy.shape)}"
)

# 3. Decoder output in [0, 1]
check(
    "Decoder output in [0, 1]",
    float(recon.min()) >= 0.0 and float(recon.max()) <= 1.0,
    f"range [{recon.min():.4f}, {recon.max():.4f}]"
)

# 4. Encoder only call
with torch.no_grad():
    emb_only = model.encoder(dummy)
check(
    "Encoder-only call shape",
    emb_only.shape == (BATCH_SIZE, EMBEDDING_DIM),
    f"got {tuple(emb_only.shape)}"
)

# 5. get_embedding matches encoder direct call
check(
    "get_embedding matches encoder output",
    torch.allclose(emb_only, model.get_embedding(dummy)),
)

# 6. Parameter count sanity
total = sum(p.numel() for p in model.parameters())
check(
    "Parameter count > 1M",
    total > 1_000_000,
    f"got {total:,}"
)

# 7. Different inputs produce different embeddings
dummy2 = torch.randn(BATCH_SIZE, 1, TARGET_FREQ, TARGET_TIME)
with torch.no_grad():
    emb2 = model.encoder(dummy2)
check(
    "Different inputs produce different embeddings",
    not torch.allclose(emb_only, emb2),
)

# 8. Batch size 1 works
dummy_single = torch.randn(1, 1, TARGET_FREQ, TARGET_TIME)
with torch.no_grad():
    recon_single, emb_single = model(dummy_single)
check(
    "Batch size 1 works",
    recon_single.shape == dummy_single.shape,
)

print("-" * 50)
if all_passed:
    print("SUMMARY: All tests passed.")
else:
    print("SUMMARY: Some tests failed.")