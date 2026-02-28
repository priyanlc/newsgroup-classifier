"""
MLX Training Script — Two-Phase Document Classification on Apple Silicon

Mirrors the CUDA training pipeline but uses Apple's MLX framework:
  - mlx_lm for model loading (auto-converts HuggingFace -> MLX format)
  - mlx.nn for LoRA layers and training
  - nn.value_and_grad + optimizer.update pattern instead of loss.backward()

Phase 1: 5 classes, 3 epochs (pipeline validation)
Phase 2: 20 classes, 5 epochs (full classification challenge)

Usage:
    python train.py                  # Run both phases
    python train.py --phase 1        # Run Phase 1 only
    python train.py --phase 2        # Run Phase 2 only

Requires: Apple Silicon Mac with >= 32 GB unified memory
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np

import config

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
    np.random.seed(seed)
    mx.random.seed(seed)

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
        f"Val chunks: {len(data['val_chunks'])}"
    )
    return data


def create_batches(
    chunks: list[list[int]],
    labels: list[int],
    batch_size: int,
    pad_token_id: int,
    shuffle: bool = False,
    doc_ids: list[int] | None = None,
) -> list[dict]:
    """Create right-padded batches from chunk data.

    Right-padding is safe for causal attention: padding tokens at the end
    cannot attend to anything (they come after real tokens), and real tokens
    cannot attend to future padding (causal mask blocks rightward attention).
    The last real token's hidden state is therefore computed correctly.
    """
    n = len(chunks)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)

    batches = []
    for start in range(0, n, batch_size):
        batch_indices = indices[start : start + batch_size]
        if len(batch_indices) == 0:
            continue

        batch_chunks = [chunks[i] for i in batch_indices]
        batch_labels = [labels[i] for i in batch_indices]
        max_len = max(len(c) for c in batch_chunks)

        # Right-pad each sequence and record actual lengths
        padded_ids = []
        lengths = []
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
            batch["doc_ids"] = [int(doc_ids[i]) for i in batch_indices]

        batches.append(batch)

    return batches

# ---------------------------------------------------------------------------
# Model Construction
# ---------------------------------------------------------------------------

def build_model(num_classes: int) -> tuple:
    """Load Mistral 7B via mlx_lm and wrap with classification head.

    mlx_lm.load() auto-converts the HuggingFace model to MLX format on first
    use (cached for subsequent loads). We access model.model (the inner
    transformer) to get hidden states instead of LM logits.
    """
    from model import MistralClassifier

    log.info(f"Loading model via mlx_lm: {config.MODEL_ID}")
    log.info("(First load converts HF -> MLX format; subsequent loads use cache)")

    try:
        from mlx_lm import load as mlx_load
        model, tokenizer = mlx_load(config.MODEL_ID)
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        log.error("Ensure mlx-lm is installed and you have internet access.")
        sys.exit(1)

    # model.model is the inner transformer that returns hidden states
    # model itself includes the LM head (which we don't need)
    inner_transformer = model.model

    # Apply LoRA to attention Q and V projections
    from mlx_lm.tuner.utils import linear_to_lora_layers

    lora_config = {
        "rank": config.LORA_RANK,
        "scale": config.LORA_SCALE,
        "dropout": config.LORA_DROPOUT,
        "keys": config.LORA_KEYS,
    }
    linear_to_lora_layers(inner_transformer, config.NUM_LAYERS, lora_config)

    # Build classification wrapper
    classifier = MistralClassifier(
        transformer=inner_transformer,
        num_classes=num_classes,
        hidden_size=config.HIDDEN_SIZE,
        dropout_rate=config.CLASSIFIER_DROPOUT,
    )

    # Freeze everything, then selectively unfreeze LoRA + classification head.
    # LoRA parameters (lora_a, lora_b) are new parameters injected by
    # linear_to_lora_layers — they remain trainable after freeze because
    # we explicitly unfreeze them below.
    classifier.freeze()
    classifier.classifier.unfreeze()  # Unfreeze the Linear classification head

    # Unfreeze LoRA parameters in the transformer.
    # unfreeze(keys=...) targets specific parameter names within the module,
    # keeping the base weight frozen while making lora_a/lora_b trainable.
    for layer in classifier.transformer.layers:
        if hasattr(layer.self_attn.q_proj, "lora_a"):
            layer.self_attn.q_proj.unfreeze(keys=["lora_a", "lora_b"])
        if hasattr(layer.self_attn.v_proj, "lora_a"):
            layer.self_attn.v_proj.unfreeze(keys=["lora_a", "lora_b"])

    # Count trainable parameters
    total_params = sum(p.size for _, p in mlx.utils.tree_flatten(classifier.parameters()))
    trainable_params = sum(
        p.size for _, p in mlx.utils.tree_flatten(classifier.trainable_parameters())
    )
    log.info(f"Total params: {total_params:,}")
    log.info(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.3f}%)")
    log.info(f"Classifier head: Linear({config.HIDDEN_SIZE} -> {num_classes})")

    return classifier, tokenizer

# ---------------------------------------------------------------------------
# Cosine Learning Rate Schedule
# ---------------------------------------------------------------------------

def cosine_lr_schedule(step: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """Compute learning rate for a given step using cosine annealing."""
    if step >= total_steps:
        return min_lr
    progress = step / total_steps
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------

def loss_fn(model, input_ids, lengths, labels):
    """Compute cross-entropy loss for a batch.

    This function signature matches what nn.value_and_grad expects:
    the model as first argument, followed by the batch data.
    """
    logits = model(input_ids, lengths)
    return nn.losses.cross_entropy(logits, labels, reduction="mean")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_doc_accuracy(
    model,
    batches: list[dict],
    doc_labels: list[int],
    num_batches: int | None = None,
) -> float:
    """Document-level accuracy with logit aggregation (matches CUDA implementation)."""
    model.eval()  # Disable dropout for deterministic evaluation
    doc_logits: dict[int, list] = {}

    count = len(batches) if num_batches is None else min(num_batches, len(batches))
    for i in range(count):
        batch = batches[i]
        logits = model(batch["input_ids"], batch["lengths"])
        # Materialize the lazy computation
        mx.eval(logits)
        logits_np = np.array(logits)

        if "doc_ids" in batch:
            for j, doc_id in enumerate(batch["doc_ids"]):
                if doc_id not in doc_logits:
                    doc_logits[doc_id] = []
                doc_logits[doc_id].append(logits_np[j])

    correct = 0
    for doc_id, logits_list in doc_logits.items():
        avg_logits = np.mean(logits_list, axis=0)
        pred = int(np.argmax(avg_logits))
        if pred == doc_labels[doc_id]:
            correct += 1

    return correct / len(doc_logits) if doc_logits else 0.0


def compute_loss_batches(model, batches: list[dict], num_batches: int) -> float:
    """Average loss over a fixed number of batches."""
    model.eval()  # Disable dropout for consistent loss measurement
    total_loss = 0.0
    count = min(num_batches, len(batches))
    for i in range(count):
        b = batches[i]
        loss = loss_fn(model, b["input_ids"], b["lengths"], b["labels"])
        mx.eval(loss)
        total_loss += loss.item()
    return total_loss / count

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_phase(
    model,
    val_batches: list[dict],
    val_doc_labels: list[int],
    pad_token_id: int,
    num_epochs: int,
    train_chunks: list,
    train_labels: list,
    checkpoint_dir: Path,
    phase_name: str,
    max_steps: int | None = None,
) -> dict:
    """Train one phase using MLX's value_and_grad pattern.

    MLX uses lazy evaluation — computations are only executed when mx.eval()
    is called. The training loop accumulates gradient updates and calls
    mx.eval() periodically to trigger actual computation.
    """
    # Compute total steps for LR scheduling
    num_train = len(train_chunks)
    batches_per_epoch = num_train // config.BATCH_SIZE
    optimizer_steps_per_epoch = batches_per_epoch // config.GRAD_ACCUMULATION_STEPS
    total_optimizer_steps = num_epochs * optimizer_steps_per_epoch

    # When max_steps is set, cap total steps so the cosine LR schedule
    # decays properly over the shorter run
    if max_steps is not None:
        total_optimizer_steps = min(total_optimizer_steps, max_steps)

    optimizer = optim.AdamW(
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # nn.value_and_grad returns both the loss value and gradients in one call
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    log.info(f"\n{'=' * 60}")
    log.info(f"Starting {phase_name}")
    log.info(f"{'=' * 60}")
    log.info(f"Epochs: {num_epochs}, Batch size: {config.BATCH_SIZE}, "
             f"Grad accum: {config.GRAD_ACCUMULATION_STEPS}, "
             f"Effective batch: {config.BATCH_SIZE * config.GRAD_ACCUMULATION_STEPS}")
    log.info(f"LR: {config.LEARNING_RATE} -> {config.MIN_LR} (cosine), "
             f"Total optimizer steps: {total_optimizer_steps}")
    if max_steps is not None:
        log.info(f"Max steps: {max_steps} (will stop early)")

    train_losses = []
    val_losses = []
    epoch_accs = []
    global_step = 0
    phase_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Re-shuffle training data each epoch
        train_batches = create_batches(
            train_chunks, train_labels, config.BATCH_SIZE,
            pad_token_id, shuffle=True,
        )

        accumulated_grads = None
        accum_count = 0
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(train_batches):
            # Update learning rate via cosine schedule
            lr = cosine_lr_schedule(
                global_step, total_optimizer_steps,
                config.LEARNING_RATE, config.MIN_LR,
            )
            optimizer.learning_rate = lr

            loss, grads = loss_and_grad_fn(
                model, batch["input_ids"], batch["lengths"], batch["labels"]
            )

            # Accumulate gradients (scale by 1/accum_steps for correct averaging)
            if accumulated_grads is None:
                accumulated_grads = mlx.utils.tree_map(
                    lambda g: g / config.GRAD_ACCUMULATION_STEPS, grads
                )
            else:
                accumulated_grads = mlx.utils.tree_map(
                    lambda a, g: a + g / config.GRAD_ACCUMULATION_STEPS,
                    accumulated_grads, grads,
                )
            accum_count += 1

            if accum_count == config.GRAD_ACCUMULATION_STEPS:
                # Apply accumulated gradients
                optimizer.update(model, accumulated_grads)
                # Trigger lazy computation — without this, MLX would just
                # build an ever-growing computation graph
                mx.eval(model.parameters(), optimizer.state)

                global_step += 1
                accumulated_grads = None
                accum_count = 0

                # Check max_steps early-stop
                if max_steps is not None and global_step >= max_steps:
                    log.info(f"  Reached max_steps={max_steps}, stopping training.")
                    break

                # Periodic evaluation
                if global_step % config.EVAL_FREQ == 0:
                    train_loss = compute_loss_batches(model, train_batches[:config.EVAL_BATCHES], config.EVAL_BATCHES)
                    val_loss = compute_loss_batches(model, val_batches, config.EVAL_BATCHES)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    log.info(
                        f"  Epoch {epoch + 1}/{num_epochs} | Step {global_step:05d} | "
                        f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
                        f"LR: {lr:.2e}"
                    )
                    model.train()  # Re-enable dropout after evaluation

            # Track loss for epoch summary
            mx.eval(loss)
            epoch_loss += loss.item()
            batch_count += 1

        # End-of-epoch: document-level accuracy on validation set
        # (also runs when stopped early via max_steps so we get final metrics)
        val_acc = compute_doc_accuracy(model, val_batches, val_doc_labels)
        model.train()  # Re-enable dropout for next epoch
        epoch_accs.append(val_acc)
        epoch_time = time.time() - epoch_start

        log.info(
            f"  Epoch {epoch + 1} complete — "
            f"Avg loss: {epoch_loss / batch_count:.4f}, "
            f"Val accuracy (doc-level): {val_acc * 100:.1f}%, "
            f"Time: {epoch_time:.0f}s"
        )

        # Break out of epoch loop if max_steps reached
        if max_steps is not None and global_step >= max_steps:
            break

    # Save checkpoints
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapter weights only — filter out the classifier head, which
    # is saved separately below. Without this filter, lora_adapters.safetensors
    # would redundantly contain the classifier weights (since
    # trainable_parameters() includes both LoRA params and the head).
    lora_weights = {}
    for name, param in mlx.utils.tree_flatten(model.trainable_parameters()):
        if not name.startswith("classifier."):
            lora_weights[name] = param
    adapter_path = checkpoint_dir / "lora_adapters.safetensors"
    mx.save_safetensors(str(adapter_path), lora_weights)

    # Save classification head separately for clarity
    head_weights = {
        "classifier.weight": model.classifier.weight,
        "classifier.bias": model.classifier.bias,
    }
    head_path = checkpoint_dir / "classifier_head.safetensors"
    mx.save_safetensors(str(head_path), head_weights)

    # Save metadata for loading
    meta = {"num_classes": model.num_classes, "hidden_size": model.hidden_size}
    with open(checkpoint_dir / "metadata.json", "w") as f:
        json.dump(meta, f)

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
    parser = argparse.ArgumentParser(description="Train Mistral 7B + LoRA (MLX) for document classification")
    parser.add_argument(
        "--phase", type=int, choices=[1, 2], default=None,
        help="Run only Phase 1 (5-class) or Phase 2 (20-class). Default: both.",
    )
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)

    log.info(f"MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
    log.info(f"Backend: Apple Silicon (MLX)")

    run_phase1 = args.phase is None or args.phase == 1
    run_phase2 = args.phase is None or args.phase == 2

    # ------------------------------------------------------------------
    # Phase 1: 5-class validation
    # ------------------------------------------------------------------
    if run_phase1:
        phase1_data = load_phase_data(config.PHASE1_DATA)
        model, tokenizer = build_model(num_classes=phase1_data["num_classes"])

        pad_token_id = tokenizer.eos_token_id  # Mistral uses EOS as pad

        val_batches = create_batches(
            phase1_data["val_chunks"], phase1_data["val_labels"],
            config.BATCH_SIZE, pad_token_id,
            shuffle=False, doc_ids=phase1_data["val_doc_ids"],
        )

        train_phase(
            model=model,
            val_batches=val_batches,
            val_doc_labels=phase1_data["val_doc_labels"],
            pad_token_id=pad_token_id,
            num_epochs=config.PHASE1_EPOCHS,
            train_chunks=phase1_data["train_chunks"],
            train_labels=phase1_data["train_labels"],
            checkpoint_dir=config.CHECKPOINT_DIR / "phase1",
            phase_name="Phase 1 (5-class)",
            max_steps=config.PHASE1_MAX_STEPS,
        )

        # Release memory before Phase 2
        del model

    # ------------------------------------------------------------------
    # Phase 2: 20-class full training
    # ------------------------------------------------------------------
    if run_phase2:
        phase2_data = load_phase_data(config.PHASE2_DATA)

        # Fresh model — Phase 1 adapters were tuned for 5 classes
        model, tokenizer = build_model(num_classes=phase2_data["num_classes"])
        pad_token_id = tokenizer.eos_token_id

        val_batches = create_batches(
            phase2_data["val_chunks"], phase2_data["val_labels"],
            config.BATCH_SIZE, pad_token_id,
            shuffle=False, doc_ids=phase2_data["val_doc_ids"],
        )

        train_phase(
            model=model,
            val_batches=val_batches,
            val_doc_labels=phase2_data["val_doc_labels"],
            pad_token_id=pad_token_id,
            num_epochs=config.PHASE2_EPOCHS,
            train_chunks=phase2_data["train_chunks"],
            train_labels=phase2_data["train_labels"],
            checkpoint_dir=config.CHECKPOINT_DIR / "phase2",
            phase_name="Phase 2 (20-class)",
            max_steps=config.PHASE2_MAX_STEPS,
        )

    log.info("\nTraining complete. Run evaluate.py to see full test results.")


if __name__ == "__main__":
    main()
