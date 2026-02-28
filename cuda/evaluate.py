"""
CUDA Evaluation Script — Full Test Set Evaluation with Confusion Matrices

Loads a trained checkpoint (LoRA adapters + classification head) and runs
comprehensive evaluation on the test set, including:
  - Document-level accuracy with logit aggregation across chunks
  - Classification report (per-class precision, recall, F1)
  - Macro/weighted F1 scores
  - Confusion matrix heatmap
  - Accuracy broken down by top-level topic group

Usage:
    python evaluate.py                 # Evaluate Phase 2 (20-class) by default
    python evaluate.py --phase 1       # Evaluate Phase 1 (5-class)
    python evaluate.py --phase 2       # Evaluate Phase 2 (20-class)
"""

import argparse
import json
import logging
import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

import config
from model import MistralForSequenceClassification, load_trained_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset and Collate (same as train.py — kept here for self-containment)
# ---------------------------------------------------------------------------

class ChunkDataset(Dataset):
    def __init__(self, chunks, labels, doc_ids=None):
        self.chunks = chunks
        self.labels = labels
        self.doc_ids = doc_ids

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        item = {"input_ids": self.chunks[index], "labels": self.labels[index]}
        if self.doc_ids is not None:
            item["doc_id"] = self.doc_ids[index]
        return item


def dynamic_padding_collate(batch, pad_token_id):
    input_ids_list = [item["input_ids"] for item in batch]
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    max_len = max(len(ids) for ids in input_ids_list)

    padded_ids, attention_masks = [], []
    for ids in input_ids_list:
        pad_len = max_len - len(ids)
        padded_ids.append(ids + [pad_token_id] * pad_len)
        attention_masks.append([1] * len(ids) + [0] * pad_len)

    result = {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": labels,
    }
    if "doc_id" in batch[0]:
        result["doc_id"] = torch.tensor([item["doc_id"] for item in batch], dtype=torch.long)
    return result

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_with_aggregation(
    model: MistralForSequenceClassification,
    data_loader: DataLoader,
    doc_labels: list[int],
    num_docs: int,
    label_names: list[str],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run evaluation with per-document logit aggregation.

    For each document, average the logits from all its chunks and take argmax.
    This produces one prediction per document, regardless of how many chunks
    it was split into.
    """
    model.eval()
    doc_logits: dict[int, list[torch.Tensor]] = {}

    log.info("Running inference on all chunks...")
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            doc_ids = batch["doc_id"].numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"].cpu()

            for i, doc_id in enumerate(doc_ids):
                doc_id = int(doc_id)
                if doc_id not in doc_logits:
                    doc_logits[doc_id] = []
                doc_logits[doc_id].append(logits[i])

    # Aggregate: average logits per document, then argmax
    all_preds = np.zeros(num_docs, dtype=np.int64)
    for doc_id in range(num_docs):
        if doc_id in doc_logits:
            avg_logits = torch.stack(doc_logits[doc_id]).mean(dim=0)
            all_preds[doc_id] = torch.argmax(avg_logits).item()

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


def plot_confusion_matrix(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    label_names: list[str],
    title: str,
    save_path: str | None = None,
) -> None:
    """Plot and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(all_labels, all_preds)
    # Use the final segment of each category name for readability
    short_names = [name.split(".")[-1] for name in label_names]

    figsize = (8, 6) if len(label_names) <= 10 else (14, 12)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=short_names, yticklabels=short_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        log.info(f"Confusion matrix saved to {save_path}")
    plt.show()


def print_accuracy_by_group(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    label_names: list[str],
) -> None:
    """Break down accuracy by top-level topic group (comp.*, rec.*, etc.).

    This reveals which topic groups are easy (distinctive vocabulary, e.g., rec.*)
    vs. hard (overlapping sub-topics, e.g., comp.*, talk.*).
    """
    groups: dict[str, dict[str, int]] = {}
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


def print_top_confused_pairs(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    label_names: list[str],
    top_k: int = 10,
) -> None:
    """Show the most commonly confused category pairs.

    Most errors should cluster within topic groups (e.g., comp.* categories
    confusing each other). Cross-group confusion indicates a training problem.
    """
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
# Main — load_trained_model is imported from model.py (single source of truth)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained document classifier")
    parser.add_argument(
        "--phase", type=int, choices=[1, 2], default=2,
        help="Which phase to evaluate (default: 2 for 20-class).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("No GPU detected — evaluation will be slow on CPU.")

    # Select data and checkpoint paths
    if args.phase == 1:
        data_path = config.PHASE1_DATA
        checkpoint_dir = config.CHECKPOINT_DIR / "phase1"
        phase_name = "Phase 1 (5-class)"
    else:
        data_path = config.PHASE2_DATA
        checkpoint_dir = config.CHECKPOINT_DIR / "phase2"
        phase_name = "Phase 2 (20-class)"

    # Load data
    log.info(f"Evaluating {phase_name}")
    with open(data_path, "r") as f:
        data = json.load(f)

    label_names = data["label_names"]
    num_classes = data["num_classes"]

    # Create test DataLoader
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    test_dataset = ChunkDataset(
        data["test_chunks"], data["test_labels"], doc_ids=data["test_doc_ids"]
    )
    collate_fn = partial(dynamic_padding_collate, pad_token_id=tokenizer.pad_token_id)
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=0, drop_last=False, collate_fn=collate_fn,
    )

    # Load model
    model = load_trained_model(num_classes, checkpoint_dir, device)

    # Run evaluation
    print(f"\n{'=' * 70}")
    print(f"{phase_name} — Test Set Evaluation")
    print(f"{'=' * 70}")

    all_preds, all_labels = evaluate_with_aggregation(
        model, test_loader, data["test_doc_labels"], data["test_num_docs"],
        label_names, device,
    )

    # Confusion matrix
    cm_save_path = str(checkpoint_dir / "confusion_matrix.png")
    plot_confusion_matrix(
        all_labels, all_preds, label_names,
        title=f"{phase_name} — Confusion Matrix",
        save_path=cm_save_path,
    )

    # Group-level accuracy and confusion analysis
    if num_classes > 5:
        print_accuracy_by_group(all_labels, all_preds, label_names)
        print_top_confused_pairs(all_labels, all_preds, label_names)


if __name__ == "__main__":
    main()
