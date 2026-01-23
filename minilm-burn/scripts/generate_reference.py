#!/usr/bin/env python3
"""Generate reference outputs for MiniLM integration tests."""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"

# Test sentences
SENTENCES = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "Rust is a systems programming language",
]

def main():
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"\nGenerating embeddings for {len(SENTENCES)} sentences...")
    # normalize_embeddings=False to get raw mean-pooled output
    embeddings = model.encode(SENTENCES, normalize_embeddings=False)

    print("\n// Reference embeddings for integration tests")
    print("// Generated with sentence-transformers Python library")
    print(f"// Model: {MODEL_NAME}\n")

    for i, (sentence, embedding) in enumerate(zip(SENTENCES, embeddings)):
        print(f"// Sentence {i}: \"{sentence}\"")
        # Print first 10 values for verification
        values = ", ".join(f"{v:.8f}" for v in embedding[:10])
        print(f"// First 10 dims: [{values}]")
        # Print full embedding as Rust array
        full_values = ", ".join(f"{v:.8f}" for v in embedding)
        print(f"const EXPECTED_{i}: [f32; 384] = [{full_values}];\n")

    # Also print cosine similarities
    from numpy import dot
    from numpy.linalg import norm

    def cosine_sim(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    print("// Cosine similarities")
    print(f"const SIM_0_1: f32 = {cosine_sim(embeddings[0], embeddings[1]):.8f};")
    print(f"const SIM_0_2: f32 = {cosine_sim(embeddings[0], embeddings[2]):.8f};")
    print(f"const SIM_1_2: f32 = {cosine_sim(embeddings[1], embeddings[2]):.8f};")


if __name__ == "__main__":
    main()
