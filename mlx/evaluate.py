"""
MLX Evaluation Script — Full Test Set Evaluation on Apple Silicon

Loads a trained checkpoint (LoRA adapters + classification head) and runs
comprehensive evaluation with logit aggregation, classification reports,
confusion matrices, and topic group analysis.

Usage:
    python evaluate.py                 # Evaluate Phase 2 (20-class) by default
    python evaluate.py --phase 1       # Evaluate Phase 1 (5-class)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Batching (same as train.py)
# ---------------------------------------------------------------------------

def create_batches(chunks, labels, batch_size, pad_token_id, doc_ids=None):
    n = len(chunks)
    batches = []
    for start in range(0, n, batch_size):
        batch_chunks = chunks[start : start + batch_size]
        batch_labels = labels[start : start + batch_size]
        if not batch_chunks:
            continue

        max_len = max(len(c) for c in batch_chunks)
        padded_ids, lengths = [], []
        for c in batch_chunks:
            pad_len = max_len - len(c)
            padded_ids.append(c + [pad_token_id] * pad_len)
            lengths.append(len(c))

        batch = {
            "input_ids": mx.array(padded_ids, dtype=mx.int32),
            "lengths": mx.array(lengths, dtype=mx.int32),
            "labels": mx.array(batch_labels, dtype=mx.int32),
        }
        if doc_ids is not None:
            batch["doc_ids"] = [int(doc_ids[i]) for i in range(start, min(start + batch_size, n))]
        batches.append(batch)
    return batches

# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_trained_model(num_classes: int, checkpoint_dir: Path):
    """Reconstruct the model and load saved LoRA adapters + classification head."""
    from model import MistralClassifier
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    log.info(f"Loading base model: {config.MODEL_ID}")
    model, tokenizer = mlx_load(config.MODEL_ID)
    inner_transformer = model.model

    # Re-apply LoRA structure (needed before loading weights)
    lora_config = {
        "rank": config.LORA_RANK,
        "scale": config.LORA_SCALE,
        "dropout": config.LORA_DROPOUT,
        "keys": config.LORA_KEYS,
    }
    linear_to_lora_layers(inner_transformer, config.NUM_LAYERS, lora_config)

    classifier = MistralClassifier(
        transformer=inner_transformer,
        num_classes=num_classes,
        hidden_size=config.HIDDEN_SIZE,
        dropout_rate=config.CLASSIFIER_DROPOUT,
    )

    # Load trained weights
    adapter_path = checkpoint_dir / "lora_adapters.safetensors"
    head_path = checkpoint_dir / "classifier_head.safetensors"

    log.info(f"Loading LoRA adapters from {adapter_path}")
    adapter_weights = mx.load(str(adapter_path))

    log.info(f"Loading classification head from {head_path}")
    head_weights = mx.load(str(head_path))

    # Merge trained weights and load into model. strict=False is essential:
    # all_weights contains only the LoRA adapters + classifier head, not the
    # ~7.2B base model parameters (which are already loaded by mlx_load above).
    all_weights = {**adapter_weights, **head_weights}
    classifier.load_weights(list(all_weights.items()), strict=False)

    # Freeze for inference
    classifier.freeze()
    mx.eval(classifier.parameters())

    return classifier, tokenizer

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_with_aggregation(
    model,
    batches: list[dict],
    doc_labels: list[int],
    num_docs: int,
    label_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Full evaluation with per-document logit aggregation."""
    # Disable dropout for deterministic evaluation — without this, dropout
    # adds noise to logits and artificially lowers reported metrics
    model.eval()
    doc_logits: dict[int, list] = {}

    log.info("Running inference on all chunks...")
    for batch in batches:
        logits = model(batch["input_ids"], batch["lengths"])
        mx.eval(logits)
        logits_np = np.array(logits)

        if "doc_ids" in batch:
            for i, doc_id in enumerate(batch["doc_ids"]):
                if doc_id not in doc_logits:
                    doc_logits[doc_id] = []
                doc_logits[doc_id].append(logits_np[i])

    # Aggregate and predict
    all_preds = np.zeros(num_docs, dtype=np.int64)
    for doc_id in range(num_docs):
        if doc_id in doc_logits:
            avg_logits = np.mean(doc_logits[doc_id], axis=0)
            all_preds[doc_id] = int(np.argmax(avg_logits))

    all_labels = np.array(doc_labels)

    # Metrics
    accuracy = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    print("\n" + "=" * 70)
    print("Classification Report")
    print("=" * 70)
    print(classification_report(all_labels, all_preds, target_names=label_names, digits=3))
    print(f"Document-level Accuracy: {accuracy:.4f}")
    print(f"Macro F1:                {macro_f1:.4f}")
    print(f"Weighted F1:             {weighted_f1:.4f}")

    return all_preds, all_labels


def plot_confusion_matrix(all_labels, all_preds, label_names, title, save_path=None):
    cm = confusion_matrix(all_labels, all_preds)
    short_names = [name.split(".")[-1] for name in label_names]
    figsize = (8, 6) if len(label_names) <= 10 else (14, 12)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=short_names, yticklabels=short_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        log.info(f"Confusion matrix saved to {save_path}")
    plt.close(fig)  # Free memory — plt.show() is a no-op with Agg backend


def print_accuracy_by_group(all_labels, all_preds, label_names):
    groups = {}
    for i, name in enumerate(label_names):
        group = name.split(".")[0]
        if group not in groups:
            groups[group] = {"correct": 0, "total": 0}
        mask = all_labels == i
        groups[group]["total"] += int(mask.sum())
        groups[group]["correct"] += int((all_preds[mask] == i).sum())

    print("\nAccuracy by Topic Group:")
    print(f"{'Group':<10} {'Accuracy':>10} {'Correct/Total':>15}")
    print("-" * 40)
    for group, stats in sorted(groups.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"{group:<10} {acc:>9.1f}% {stats['correct']:>6}/{stats['total']:<6}")


def print_top_confused_pairs(all_labels, all_preds, label_names, top_k=10):
    cm = confusion_matrix(all_labels, all_preds)
    cm_off_diag = cm.copy()
    np.fill_diagonal(cm_off_diag, 0)

    pairs = []
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            if i != j and cm_off_diag[i][j] > 0:
                pairs.append((label_names[i], label_names[j], cm_off_diag[i][j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\nTop {top_k} Most Confused Category Pairs:")
    print(f"{'True Category':<35} {'Predicted As':<35} {'Count':>5}")
    print("-" * 80)
    for true_cat, pred_cat, count in pairs[:top_k]:
        print(f"{true_cat:<35} {pred_cat:<35} {count:>5}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained classifier (MLX)")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()

    if args.phase == 1:
        data_path = config.PHASE1_DATA
        checkpoint_dir = config.CHECKPOINT_DIR / "phase1"
        phase_name = "Phase 1 (5-class)"
    else:
        data_path = config.PHASE2_DATA
        checkpoint_dir = config.CHECKPOINT_DIR / "phase2"
        phase_name = "Phase 2 (20-class)"

    log.info(f"Evaluating {phase_name}")
    with open(data_path, "r") as f:
        data = json.load(f)

    label_names = data["label_names"]
    num_classes = data["num_classes"]

    # Load model
    model, tokenizer = load_trained_model(num_classes, checkpoint_dir)
    pad_token_id = tokenizer.eos_token_id

    # Create test batches
    test_batches = create_batches(
        data["test_chunks"], data["test_labels"],
        config.BATCH_SIZE, pad_token_id,
        doc_ids=data["test_doc_ids"],
    )

    # Evaluate
    print(f"\n{'=' * 70}")
    print(f"{phase_name} — Test Set Evaluation (MLX)")
    print(f"{'=' * 70}")

    all_preds, all_labels = evaluate_with_aggregation(
        model, test_batches, data["test_doc_labels"], data["test_num_docs"],
        label_names,
    )

    cm_save_path = str(checkpoint_dir / "confusion_matrix.png")
    plot_confusion_matrix(all_labels, all_preds, label_names,
                          title=f"{phase_name} — Confusion Matrix (MLX)",
                          save_path=cm_save_path)

    if num_classes > 5:
        print_accuracy_by_group(all_labels, all_preds, label_names)
        print_top_confused_pairs(all_labels, all_preds, label_names)


if __name__ == "__main__":
    main()
