#!/usr/bin/env python3
"""Generate reference outputs for ALBERT integration tests.

Run with: uv run --with transformers --with torch scripts/generate_reference.py
"""

import torch
from transformers import AlbertForMaskedLM, AlbertTokenizer

MODEL_NAME = "albert/albert-base-v2"

# Test sentence with [MASK]
SENTENCE = "The capital of France is [MASK]."


def main():
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AlbertTokenizer.from_pretrained(MODEL_NAME)
    model = AlbertForMaskedLM.from_pretrained(MODEL_NAME)
    model.eval()

    print(f"\nSentence: \"{SENTENCE}\"")

    inputs = tokenizer(SENTENCE, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Find [MASK] position
    mask_token_id = tokenizer.mask_token_id
    mask_pos = (input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0].item()
    print(f"[MASK] position: {mask_pos}")
    print(f"Input IDs: {input_ids[0].tolist()}")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Get logits at [MASK] position
    mask_logits = logits[0, mask_pos]  # [vocab_size]

    # Top 5 predictions
    top5 = torch.topk(mask_logits, 5)
    print("\n// Top 5 predictions")
    for i, (score, idx) in enumerate(zip(top5.values, top5.indices)):
        token = tokenizer.decode([idx.item()]).strip()
        print(f"//   {i+1}: \"{token}\" (logit: {score.item():.4f})")

    # Print first 10 logit values at mask position for verification
    first10 = mask_logits[:10].tolist()
    values = ", ".join(f"{v:.6f}" for v in first10)
    print(f"\n// First 10 logits at [MASK] position")
    print(f"const EXPECTED_MASK_LOGITS_FIRST_10: [f32; 10] = [{values}];")

    # Print top 5 token IDs and logits
    top5_ids = top5.indices.tolist()
    top5_scores = top5.values.tolist()
    ids_str = ", ".join(str(i) for i in top5_ids)
    scores_str = ", ".join(f"{s:.6f}" for s in top5_scores)
    print(f"\nconst EXPECTED_TOP5_IDS: [i64; 5] = [{ids_str}];")
    print(f"const EXPECTED_TOP5_LOGITS: [f32; 5] = [{scores_str}];")


if __name__ == "__main__":
    main()
