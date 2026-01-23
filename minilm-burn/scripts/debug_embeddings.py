#!/usr/bin/env python3
"""Debug embedding differences."""

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

sentences = ["Hello world"]
emb = model.encode(sentences, normalize_embeddings=False)[0]

print(f"First 5 dims: {emb[:5]}")
print(f"L2 norm: {np.linalg.norm(emb):.6f}")
print(f"If normalized, first dim would be: {emb[0] / np.linalg.norm(emb):.6f}")
