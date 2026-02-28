"""
CUDA Inference Script — Interactive Document Classification Demo

Classifies text input using the trained Mistral 7B + LoRA model.
Supports both pre-defined demo texts and interactive user input.
Long texts are automatically split into chunks and their logits are
averaged for a robust final prediction.

Usage:
    python inference.py                # Run demo texts + interactive mode
    python inference.py --phase 1      # Use Phase 1 (5-class) model
    python inference.py --phase 2      # Use Phase 2 (20-class) model (default)
    python inference.py --demo-only    # Run demo texts only, skip interactive
"""

import argparse
import json
import logging
import re
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

import config
from model import MistralForSequenceClassification, load_trained_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text Processing (mirrors prepare_data.py)
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = text.strip()
    if not text:
        return "[empty post]"
    return re.sub(r"\n{3,}", "\n\n", text)


def chunk_tokens(tokens: list[int], chunk_size: int, stride: int) -> list[list[int]]:
    if len(tokens) <= chunk_size:
        return [tokens]
    chunks = []
    for start in range(0, len(tokens), stride):
        chunk = tokens[start : start + chunk_size]
        chunks.append(chunk)
        if start + chunk_size >= len(tokens):
            break
    return chunks

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_document(
    text: str,
    model: MistralForSequenceClassification,
    tokenizer,
    device: str,
    label_names: list[str],
    chunk_size: int = config.CHUNK_SIZE,
    stride: int = config.STRIDE,
) -> dict:
    """Classify a single document with chunk aggregation.

    Returns a dict with prediction, confidence, top-3 classes, and chunk count.
    """
    model.eval()
    text = clean_text(text)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = chunk_tokens(tokens, chunk_size, stride)

    all_logits = []
    for chunk in chunks:
        # Pad to chunk_size for consistent processing
        pad_len = chunk_size - len(chunk)
        padded = chunk + [tokenizer.pad_token_id] * pad_len
        mask = [1] * len(chunk) + [0] * pad_len

        input_ids = torch.tensor([padded], dtype=torch.long).to(device)
        attention_mask = torch.tensor([mask], dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs["logits"].cpu())

    # Average logits across chunks — more robust than any single chunk
    avg_logits = torch.cat(all_logits, dim=0).mean(dim=0)
    probs = torch.softmax(avg_logits, dim=-1).numpy()
    pred_idx = int(np.argmax(probs))

    # Top-3 predictions for insight into model uncertainty
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [(label_names[i], float(probs[i] * 100)) for i in top3_idx]

    return {
        "prediction": label_names[pred_idx],
        "confidence": float(probs[pred_idx] * 100),
        "top3": top3,
        "num_chunks": len(chunks),
        "num_tokens": len(tokens),
    }

# ---------------------------------------------------------------------------
# Demo Texts
# ---------------------------------------------------------------------------

# These cover a range of scenarios: clear-cut, ambiguous, cross-domain, and
# multi-topic, matching the examples from the solution design
DEMO_TEXTS = [
    "The Hubble telescope captured amazing images of a distant galaxy cluster.",
    "The Penguins dominated the third period with two quick goals.",
    "How do I render 3D graphics using OpenGL on Linux?",
    "The situation in the Middle East continues to escalate with new tensions.",
    "Jesus taught his disciples the importance of forgiveness and love.",
    "I'm selling my old 486 PC with 8MB RAM and a 200MB hard drive.",
    "The new encryption standard should provide better security for communications.",
    "My motorcycle needs new brake pads. Any recommendations for a Honda CB750?",
]

# ---------------------------------------------------------------------------
# Main — load_trained_model is imported from model.py (single source of truth)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Classify documents with trained model")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=2,
                        help="Which model to use (default: 2 for 20-class).")
    parser.add_argument("--demo-only", action="store_true",
                        help="Run demo texts only, skip interactive mode.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("No GPU detected — inference will be slow on CPU.")

    # Load data config for label names
    data_path = config.PHASE1_DATA if args.phase == 1 else config.PHASE2_DATA
    try:
        with open(data_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        log.error(f"Data file not found: {data_path}")
        log.error("Run 'python prepare_data.py' from the project root first.")
        sys.exit(1)

    label_names = data["label_names"]
    num_classes = data["num_classes"]
    checkpoint_dir = config.CHECKPOINT_DIR / f"phase{args.phase}"

    # Load model
    log.info(f"Loading Phase {args.phase} model ({num_classes} classes)...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = load_trained_model(num_classes, checkpoint_dir, device)

    # Run demo texts
    print(f"\n{'=' * 70}")
    print(f"Demo: Phase {args.phase} ({num_classes}-class) Document Classification")
    print(f"{'=' * 70}\n")

    for text in DEMO_TEXTS:
        result = classify_document(text, model, tokenizer, device, label_names)
        display_text = text[:70] + "..." if len(text) > 70 else text
        print(f"Text: {display_text}")
        print(f"  Prediction: {result['prediction']} ({result['confidence']:.1f}%)")
        print(f"  Top 3: {', '.join(f'{n} ({c:.1f}%)' for n, c in result['top3'])}")
        print(f"  [{result['num_tokens']} tokens, {result['num_chunks']} chunk(s)]")
        print()

    # Interactive mode
    if not args.demo_only:
        print("=" * 70)
        print("Interactive Mode — Enter text to classify (Ctrl+C to exit)")
        print("=" * 70)

        try:
            while True:
                text = input("\n> ").strip()
                if not text:
                    continue
                result = classify_document(text, model, tokenizer, device, label_names)
                print(f"  Prediction: {result['prediction']} ({result['confidence']:.1f}%)")
                print(f"  Top 3: {', '.join(f'{n} ({c:.1f}%)' for n, c in result['top3'])}")
                print(f"  [{result['num_tokens']} tokens, {result['num_chunks']} chunk(s)]")
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting. Goodbye!")


if __name__ == "__main__":
    main()
