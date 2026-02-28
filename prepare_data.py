"""
Shared Data Preprocessing for 20 Newsgroups Classification

This script handles all data loading, cleaning, tokenization, chunking, and
splitting. It produces JSON files consumed by both the CUDA and MLX training
pipelines, ensuring identical data across platforms.

Usage:
    python prepare_data.py

Output:
    data/phase1_data.json  — 5-class subset (diverse categories)
    data/phase2_data.json  — Full 20-class dataset

Requirements: transformers, scikit-learn, numpy (CPU only — no GPU needed)
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "mistralai/Mistral-7B-v0.1"
CHUNK_SIZE = 512
STRIDE = 256
VAL_SPLIT = 0.15
RANDOM_SEED = 42
OUTPUT_DIR = Path("data")

# Phase 1: One category per major topic group for maximum separation
PHASE1_CATEGORIES = [
    "rec.sport.hockey",       # Recreation
    "sci.space",              # Science
    "comp.graphics",          # Computers
    "talk.politics.mideast",  # Politics
    "soc.religion.christian", # Religion
]

# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean a newsgroup post: strip whitespace, collapse excessive newlines.

    Returns '[empty post]' for blank documents so downstream tokenization
    always receives at least one token.
    """
    text = text.strip()
    if not text:
        return "[empty post]"
    # Collapse 3+ consecutive newlines to 2 (preserves paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_tokens(tokens: list[int], chunk_size: int, stride: int) -> list[list[int]]:
    """Split a token sequence into overlapping chunks.

    Short documents (<= chunk_size) pass through as a single chunk.
    Long documents are split with 50% overlap (stride = chunk_size / 2),
    which means every token appears in at least 2 chunks — providing
    redundancy for logit aggregation at inference time.

    Example (chunk_size=512, stride=256, document=1400 tokens):
        Chunk 0: tokens[0:512]
        Chunk 1: tokens[256:768]
        Chunk 2: tokens[512:1024]
        Chunk 3: tokens[768:1280]
        Chunk 4: tokens[1024:1400]  (shorter, final chunk)
    """
    if len(tokens) <= chunk_size:
        return [tokens]

    chunks = []
    for start in range(0, len(tokens), stride):
        chunk = tokens[start : start + chunk_size]
        chunks.append(chunk)
        # Stop once we've captured the tail of the document
        if start + chunk_size >= len(tokens):
            break
    return chunks

# ---------------------------------------------------------------------------
# Dataset Processing
# ---------------------------------------------------------------------------

def process_split(
    texts: list[str],
    labels: np.ndarray,
    tokenizer,
    chunk_size: int,
    stride: int,
    split_name: str,
    track_doc_ids: bool = False,
) -> dict:
    """Tokenize and chunk a data split.

    Args:
        texts: Raw document strings.
        labels: Integer labels per document.
        tokenizer: HuggingFace tokenizer.
        chunk_size: Maximum tokens per chunk.
        stride: Step size between chunk starts.
        split_name: For logging ('train', 'val', 'test').
        track_doc_ids: If True, record which document each chunk came from.
            Needed for val/test splits to aggregate logits per document.

    Returns:
        Dict with 'chunks', 'labels', and optionally 'doc_ids'.
    """
    all_chunks = []
    all_labels = []
    all_doc_ids = [] if track_doc_ids else None
    chunked_count = 0

    for doc_id, (text, label) in enumerate(zip(texts, labels)):
        text = clean_text(text)
        # add_special_tokens=False: we handle tokenization ourselves
        tokens = tokenizer.encode(text, add_special_tokens=False)
        doc_chunks = chunk_tokens(tokens, chunk_size, stride)

        if len(doc_chunks) > 1:
            chunked_count += 1

        for chunk in doc_chunks:
            all_chunks.append(chunk)
            all_labels.append(int(label))
            if track_doc_ids:
                all_doc_ids.append(doc_id)

    chunk_lengths = [len(c) for c in all_chunks]
    print(f"  {split_name}: {len(texts)} docs -> {len(all_chunks)} chunks")
    print(f"    {chunked_count} docs ({100 * chunked_count / len(texts):.1f}%) split into multiple chunks")
    print(f"    Chunk lengths — min: {min(chunk_lengths)}, max: {max(chunk_lengths)}, "
          f"mean: {np.mean(chunk_lengths):.0f}, median: {np.median(chunk_lengths):.0f}")

    result = {"chunks": all_chunks, "labels": all_labels}
    if track_doc_ids:
        result["doc_ids"] = all_doc_ids
    return result

# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def prepare_phase(
    categories: list[str] | None,
    tokenizer,
    phase_name: str,
    output_path: Path,
) -> None:
    """Prepare data for one training phase (5-class or 20-class).

    Args:
        categories: List of category names, or None for all 20.
        tokenizer: HuggingFace tokenizer.
        phase_name: For logging ('Phase 1' or 'Phase 2').
        output_path: Where to write the JSON file.
    """
    print(f"\n{'=' * 60}")
    print(f"Preparing {phase_name}")
    print(f"{'=' * 60}")

    # remove=('headers', 'footers', 'quotes') strips Usenet metadata that
    # directly reveals the newsgroup name — without this, the model would
    # learn to read headers instead of understanding content
    print(f"Fetching data from sklearn (remove headers/footers/quotes)...")
    train_raw = fetch_20newsgroups(
        subset="train",
        categories=categories,
        remove=("headers", "footers", "quotes"),
        random_state=RANDOM_SEED,
    )
    test_raw = fetch_20newsgroups(
        subset="test",
        categories=categories,
        remove=("headers", "footers", "quotes"),
        random_state=RANDOM_SEED,
    )

    label_names = list(train_raw.target_names)
    num_classes = len(label_names)

    print(f"Categories ({num_classes}): {label_names}")
    print(f"Raw train: {len(train_raw.data)}, Raw test: {len(test_raw.data)}")

    # Stratified train/val split ensures each category is proportionally
    # represented in both sets — important for the smaller Phase 1 subset
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_raw.data,
        train_raw.target,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=train_raw.target,
    )
    test_texts = test_raw.data
    test_labels = test_raw.target

    print(f"After split — train: {len(train_texts)}, val: {len(val_texts)}, test: {len(test_texts)}")

    # Tokenize and chunk each split
    print(f"\nTokenizing and chunking (chunk_size={CHUNK_SIZE}, stride={STRIDE})...")
    train_data = process_split(
        train_texts, train_labels, tokenizer,
        CHUNK_SIZE, STRIDE, "train", track_doc_ids=False,
    )
    val_data = process_split(
        val_texts, val_labels, tokenizer,
        CHUNK_SIZE, STRIDE, "val", track_doc_ids=True,
    )
    test_data = process_split(
        test_texts, test_labels, tokenizer,
        CHUNK_SIZE, STRIDE, "test", track_doc_ids=True,
    )

    # Assemble the output structure — both CUDA and MLX scripts load this
    # same JSON, ensuring identical training and evaluation inputs
    output = {
        "train_chunks": train_data["chunks"],
        "train_labels": train_data["labels"],
        "val_chunks": val_data["chunks"],
        "val_labels": val_data["labels"],
        "val_doc_ids": val_data["doc_ids"],
        "val_num_docs": len(val_texts),
        "val_doc_labels": [int(l) for l in val_labels],
        "test_chunks": test_data["chunks"],
        "test_labels": test_data["labels"],
        "test_doc_ids": test_data["doc_ids"],
        "test_num_docs": len(test_texts),
        "test_doc_labels": [int(l) for l in test_labels],
        "label_names": label_names,
        "num_classes": num_classes,
        "chunk_size": CHUNK_SIZE,
        "stride": STRIDE,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved to {output_path} ({file_size_mb:.1f} MB)")


def main():
    np.random.seed(RANDOM_SEED)
    start_time = time.time()

    print("=" * 60)
    print("20 Newsgroups Data Preparation")
    print("=" * 60)

    # Load the tokenizer (CPU only — no GPU or model weights needed)
    print(f"\nLoading tokenizer: {MODEL_ID}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"ERROR: Could not load tokenizer for '{MODEL_ID}'.")
        print(f"  Ensure you have internet access and the model ID is correct.")
        print(f"  Details: {e}")
        sys.exit(1)

    # Mistral's tokenizer has no pad token by default — reuse EOS
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Vocabulary size: {len(tokenizer):,}")
    print(f"Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

    # Phase 1: 5 diverse categories (pipeline validation)
    prepare_phase(
        categories=PHASE1_CATEGORIES,
        tokenizer=tokenizer,
        phase_name="Phase 1 (5 categories)",
        output_path=OUTPUT_DIR / "phase1_data.json",
    )

    # Phase 2: All 20 categories (full classification challenge)
    prepare_phase(
        categories=None,  # None = all 20 categories
        tokenizer=tokenizer,
        phase_name="Phase 2 (20 categories)",
        output_path=OUTPUT_DIR / "phase2_data.json",
    )

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Data preparation complete in {elapsed:.1f}s")
    print(f"Output files:")
    print(f"  {OUTPUT_DIR / 'phase1_data.json'}")
    print(f"  {OUTPUT_DIR / 'phase2_data.json'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
