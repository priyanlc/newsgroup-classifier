"""
CUDA Training Script — Two-Phase Document Classification with Mistral 7B + LoRA

Phase 1: Train on 5 well-separated categories (3 epochs) to validate the
          pipeline works before committing to the full problem.
Phase 2: Reload a fresh model and train on all 20 categories (5 epochs),
          introducing the harder challenge of within-group disambiguation.

Each phase: load base model in float16 -> apply LoRA -> train with AdamW +
cosine LR -> save LoRA adapters + classification head.

Usage:
    python train.py                  # Run both phases
    python train.py --phase 1        # Run Phase 1 only
    python train.py --phase 2        # Run Phase 2 only

Requires: NVIDIA GPU with >= 24 GB VRAM (e.g., RTX 3090/4090, A5000, A100)
"""

import argparse
import json
import logging
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from model import MistralForSequenceClassification

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic operations trade speed for exact reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChunkDataset(Dataset):
    """Simple dataset wrapping pre-tokenized chunks and labels.

    For training, each chunk is an independent example — a 1400-token document
    that was split into 5 chunks produces 5 training examples, all with the
    same label. This acts as data augmentation for long documents.
    """

    def __init__(self, chunks: list[list[int]], labels: list[int], doc_ids: list[int] | None = None):
        self.chunks = chunks
        self.labels = labels
        self.doc_ids = doc_ids

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> dict:
        item = {
            "input_ids": self.chunks[index],
            "labels": self.labels[index],
        }
        if self.doc_ids is not None:
            item["doc_id"] = self.doc_ids[index]
        return item


def dynamic_padding_collate(batch: list[dict], pad_token_id: int) -> dict:
    """Pad each batch to its longest sequence, not globally to chunk_size.

    This saves significant compute: a batch where all sequences are 150 tokens
    only pads to 150 — not 512. On average, this reduces wasted computation
    by 2-4x since most newsgroup posts are shorter than the chunk size.
    """
    input_ids_list = [item["input_ids"] for item in batch]
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    max_len = max(len(ids) for ids in input_ids_list)

    padded_ids = []
    attention_masks = []
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
        result["doc_id"] = torch.tensor(
            [item["doc_id"] for item in batch], dtype=torch.long
        )

    return result

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_phase_data(data_path: str) -> dict:
    """Load preprocessed JSON data from prepare_data.py."""
    log.info(f"Loading data from {data_path}")
    try:
        with open(data_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        log.error(f"Data file not found: {data_path}")
        log.error("Run 'python prepare_data.py' from the project root first.")
        sys.exit(1)
    log.info(
        f"  Classes: {data['num_classes']}, "
        f"Train chunks: {len(data['train_chunks'])}, "
        f"Val chunks: {len(data['val_chunks'])}, "
        f"Test chunks: {len(data['test_chunks'])}"
    )
    return data


def create_dataloaders(data: dict, pad_token_id: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test DataLoaders from preprocessed data."""
    train_dataset = ChunkDataset(data["train_chunks"], data["train_labels"])
    val_dataset = ChunkDataset(
        data["val_chunks"], data["val_labels"], doc_ids=data["val_doc_ids"]
    )
    test_dataset = ChunkDataset(
        data["test_chunks"], data["test_labels"], doc_ids=data["test_doc_ids"]
    )

    collate_fn = partial(dynamic_padding_collate, pad_token_id=pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,  # Avoids small final batches that could skew gradient estimates
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader

# ---------------------------------------------------------------------------
# Model Construction
# ---------------------------------------------------------------------------

def build_model(num_classes: int) -> tuple[MistralForSequenceClassification, AutoTokenizer]:
    """Load Mistral 7B in float16, apply LoRA, wrap with classification head.

    Returns the classification model and tokenizer.
    """
    log.info(f"Loading tokenizer: {config.MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load in float16 — no quantization. This uses ~14 GB VRAM but preserves
    # full model quality and avoids bitsandbytes dependency complexity.
    log.info(f"Loading base model in float16 (this takes 1-2 minutes)...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        log.error("Ensure you have >= 24 GB GPU VRAM and internet access.")
        sys.exit(1)

    log.info(f"Model loaded. Hidden size: {base_model.config.hidden_size}, "
             f"Layers: {base_model.config.num_hidden_layers}")

    # Apply LoRA to query and value projections — these capture the most
    # task-relevant attention patterns. FEATURE_EXTRACTION task type tells
    # PEFT we're extracting hidden states, not generating text.
    lora_config = LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias=config.LORA_BIAS,
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    # Wrap with classification head
    model = MistralForSequenceClassification(
        base_model,
        num_labels=num_classes,
        dropout_rate=config.CLASSIFIER_DROPOUT,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Classification model ready:")
    log.info(f"  Total params: {total_params:,}")
    log.info(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.3f}%)")
    log.info(f"  Classifier head: Linear({model.hidden_size} -> {num_classes})")

    return model, tokenizer

# ---------------------------------------------------------------------------
# Evaluation Utilities
# ---------------------------------------------------------------------------

def compute_loss(data_loader: DataLoader, model: nn.Module, device: str, num_batches: int) -> float:
    """Compute average loss over a fixed number of batches (for speed)."""
    model.eval()
    total_loss = 0.0
    count = min(num_batches, len(data_loader))
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= count:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs["loss"].item()
    return total_loss / count


def compute_doc_accuracy(
    data_loader: DataLoader,
    model: nn.Module,
    device: str,
    doc_labels: list[int],
    num_batches: int | None = None,
) -> float:
    """Compute document-level accuracy with logit aggregation.

    For each document, average the logits across all its chunks, then argmax.
    This ensemble-like approach is more robust than any single chunk's prediction.
    """
    model.eval()
    doc_logits: dict[int, list[torch.Tensor]] = {}

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if num_batches is not None and i >= num_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            doc_ids = batch["doc_id"].numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"].cpu()

            for j, doc_id in enumerate(doc_ids):
                doc_id = int(doc_id)
                if doc_id not in doc_logits:
                    doc_logits[doc_id] = []
                doc_logits[doc_id].append(logits[j])

    correct = 0
    for doc_id, logits_list in doc_logits.items():
        avg_logits = torch.stack(logits_list).mean(dim=0)
        pred = torch.argmax(avg_logits).item()
        if pred == doc_labels[doc_id]:
            correct += 1

    return correct / len(doc_logits) if doc_logits else 0.0

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_phase(
    model: MistralForSequenceClassification,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_doc_labels: list[int],
    num_epochs: int,
    checkpoint_dir: str,
    phase_name: str,
) -> dict:
    """Train one phase (5-class or 20-class).

    Uses chunk-level CrossEntropyLoss for training but document-level
    aggregated accuracy for validation — matching how the model will be
    evaluated in practice.
    """
    device = "cuda"
    # Only pass trainable parameters (LoRA adapters + classification head) to
    # the optimizer. model.parameters() would include the ~7.2B frozen base
    # model params — AdamW would iterate over them each step with no benefit.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # T_max must equal the number of *optimizer steps* (not batch steps),
    # because scheduler.step() is called once per optimizer step (after
    # gradient accumulation). Using len(train_loader) directly would set
    # T_max ~4x too high, causing the LR to barely decay over training.
    total_steps = num_epochs * (len(train_loader) // config.GRAD_ACCUMULATION_STEPS)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=config.MIN_LR
    )

    log.info(f"\n{'=' * 60}")
    log.info(f"Starting {phase_name}")
    log.info(f"{'=' * 60}")
    log.info(f"Epochs: {num_epochs}, Batch size: {config.BATCH_SIZE}, "
             f"Grad accum: {config.GRAD_ACCUMULATION_STEPS}, "
             f"Effective batch: {config.BATCH_SIZE * config.GRAD_ACCUMULATION_STEPS}")
    log.info(f"LR: {config.LEARNING_RATE} -> {config.MIN_LR} (cosine), "
             f"Total steps: {total_steps}")

    train_losses = []
    val_losses = []
    epoch_accs = []
    global_step = 0
    phase_start = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # Scale loss by accumulation steps so the effective gradient
            # magnitude is independent of the accumulation count
            loss = outputs["loss"] / config.GRAD_ACCUMULATION_STEPS
            loss.backward()

            if (batch_idx + 1) % config.GRAD_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                # Periodic validation (fast, on a subsample)
                if global_step % config.EVAL_FREQ == 0:
                    train_loss = compute_loss(train_loader, model, device, config.EVAL_BATCHES)
                    val_loss = compute_loss(val_loader, model, device, config.EVAL_BATCHES)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    log.info(
                        f"  Epoch {epoch + 1}/{num_epochs} | Step {global_step:05d} | "
                        f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}"
                    )
                    model.train()

            epoch_loss += outputs["loss"].item()
            batch_count += 1

        # End-of-epoch: document-level accuracy on full validation set
        val_acc = compute_doc_accuracy(val_loader, model, device, val_doc_labels)
        epoch_accs.append(val_acc)
        epoch_time = time.time() - epoch_start

        log.info(
            f"  Epoch {epoch + 1} complete — "
            f"Avg loss: {epoch_loss / batch_count:.4f}, "
            f"Val accuracy (doc-level): {val_acc * 100:.1f}%, "
            f"Time: {epoch_time:.0f}s"
        )

    # Save LoRA adapters and classification head separately
    checkpoint_dir = config.CHECKPOINT_DIR / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA weights via PEFT's built-in method
    model.base_model.save_pretrained(str(checkpoint_dir / "lora_adapters"))
    # Save classification head (dropout + linear) as a standard PyTorch state dict
    head_state = {
        "classifier.weight": model.classifier.weight.data.cpu(),
        "classifier.bias": model.classifier.bias.data.cpu(),
        "num_labels": model.num_labels,
    }
    torch.save(head_state, str(checkpoint_dir / "classifier_head.pt"))

    phase_time = (time.time() - phase_start) / 60
    log.info(f"\n{phase_name} complete in {phase_time:.1f} minutes")
    log.info(f"Checkpoints saved to {checkpoint_dir}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epoch_accs": epoch_accs,
        "training_time_min": phase_time,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Mistral 7B + LoRA for document classification")
    parser.add_argument(
        "--phase", type=int, choices=[1, 2], default=None,
        help="Run only Phase 1 (5-class) or Phase 2 (20-class). Default: both.",
    )
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)

    # Log environment for reproducibility
    log.info(f"PyTorch version: {torch.__version__}")
    log.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        log.error("No CUDA GPU detected. This script requires an NVIDIA GPU with >= 24 GB VRAM.")
        sys.exit(1)

    run_phase1 = args.phase is None or args.phase == 1
    run_phase2 = args.phase is None or args.phase == 2

    # ------------------------------------------------------------------
    # Phase 1: 5-class validation
    # ------------------------------------------------------------------
    if run_phase1:
        phase1_data = load_phase_data(config.PHASE1_DATA)
        model, tokenizer = build_model(num_classes=phase1_data["num_classes"])
        train_loader, val_loader, _ = create_dataloaders(phase1_data, tokenizer.pad_token_id)

        phase1_results = train_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            val_doc_labels=phase1_data["val_doc_labels"],
            num_epochs=config.PHASE1_EPOCHS,
            checkpoint_dir="phase1",
            phase_name="Phase 1 (5-class)",
        )

        # Free GPU memory before Phase 2 loads a fresh model
        del model, train_loader, val_loader
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Phase 2: 20-class full training
    # ------------------------------------------------------------------
    if run_phase2:
        phase2_data = load_phase_data(config.PHASE2_DATA)

        # Reload a completely fresh model — Phase 1's LoRA adapters were
        # tuned for 5 classes and would interfere with 20-class learning
        model, tokenizer = build_model(num_classes=phase2_data["num_classes"])
        train_loader, val_loader, _ = create_dataloaders(phase2_data, tokenizer.pad_token_id)

        phase2_results = train_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            val_doc_labels=phase2_data["val_doc_labels"],
            num_epochs=config.PHASE2_EPOCHS,
            checkpoint_dir="phase2",
            phase_name="Phase 2 (20-class)",
        )

    log.info("\nTraining complete. Run evaluate.py to see full test results.")


if __name__ == "__main__":
    main()
