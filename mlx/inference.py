"""
MLX Inference Script — Interactive Document Classification on Apple Silicon

Classifies text using the trained Mistral 7B + LoRA model via MLX.
Supports pre-defined demo texts and interactive user input.
Long texts are automatically chunked with logit aggregation.

Usage:
    python inference.py                # Demo texts + interactive mode
    python inference.py --phase 1      # Use Phase 1 (5-class) model
    python inference.py --phase 2      # Use Phase 2 (20-class, default)
    python inference.py --demo-only    # Demo texts only
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import config

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
    model,
    tokenizer,
    label_names: list[str],
    chunk_size: int = config.CHUNK_SIZE,
    stride: int = config.STRIDE,
) -> dict:
    """Classify a single document with chunk aggregation."""
    text = clean_text(text)
    # add_special_tokens=False to match prepare_data.py — training data was
    # tokenized without BOS/EOS tokens, so inference must do the same
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = chunk_tokens(tokens, chunk_size, stride)

    all_logits = []
    for chunk in chunks:
        input_ids = mx.array([chunk], dtype=mx.int32)
        lengths = mx.array([len(chunk)], dtype=mx.int32)

        logits = model(input_ids, lengths)
        mx.eval(logits)
        all_logits.append(np.array(logits[0]))

    # Average logits across chunks
    avg_logits = np.mean(all_logits, axis=0)
    # Softmax for probabilities
    exp_logits = np.exp(avg_logits - np.max(avg_logits))
    probs = exp_logits / exp_logits.sum()
    pred_idx = int(np.argmax(probs))

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
# Model Loading (same as evaluate.py)
# ---------------------------------------------------------------------------

def load_trained_model(num_classes: int, checkpoint_dir: Path):
    from model import MistralClassifier
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    log.info(f"Loading base model: {config.MODEL_ID}")
    model, tokenizer = mlx_load(config.MODEL_ID)
    inner_transformer = model.model

    lora_config = {
        "rank": config.LORA_RANK, "scale": config.LORA_SCALE,
        "dropout": config.LORA_DROPOUT, "keys": config.LORA_KEYS,
    }
    linear_to_lora_layers(inner_transformer, config.NUM_LAYERS, lora_config)

    classifier = MistralClassifier(
        transformer=inner_transformer, num_classes=num_classes,
        hidden_size=config.HIDDEN_SIZE, dropout_rate=config.CLASSIFIER_DROPOUT,
    )

    adapter_path = checkpoint_dir / "lora_adapters.safetensors"
    head_path = checkpoint_dir / "classifier_head.safetensors"

    log.info(f"Loading weights from {checkpoint_dir}")
    adapter_weights = mx.load(str(adapter_path))
    head_weights = mx.load(str(head_path))
    # strict=False: we're only loading LoRA adapters + classifier head,
    # not the full base model weights (already loaded by mlx_load above)
    all_weights = {**adapter_weights, **head_weights}
    classifier.load_weights(list(all_weights.items()), strict=False)

    classifier.freeze()
    mx.eval(classifier.parameters())

    return classifier, tokenizer

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Classify documents (MLX)")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=2)
    parser.add_argument("--demo-only", action="store_true")
    args = parser.parse_args()

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

    log.info(f"Loading Phase {args.phase} model ({num_classes} classes)...")
    model, tokenizer = load_trained_model(num_classes, checkpoint_dir)

    # Run demo texts
    print(f"\n{'=' * 70}")
    print(f"Demo: Phase {args.phase} ({num_classes}-class) Document Classification (MLX)")
    print(f"{'=' * 70}\n")

    for text in DEMO_TEXTS:
        result = classify_document(text, model, tokenizer, label_names)
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
                result = classify_document(text, model, tokenizer, label_names)
                print(f"  Prediction: {result['prediction']} ({result['confidence']:.1f}%)")
                print(f"  Top 3: {', '.join(f'{n} ({c:.1f}%)' for n, c in result['top3'])}")
                print(f"  [{result['num_tokens']} tokens, {result['num_chunks']} chunk(s)]")
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting. Goodbye!")


if __name__ == "__main__":
    main()
