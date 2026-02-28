"""
CUDA Configuration — Centralized hyperparameters and paths.

All magic numbers live here. Training, evaluation, and inference scripts
import from this module so there's a single source of truth.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_ID = "mistralai/Mistral-7B-v0.1"
HIDDEN_SIZE = 4096  # Mistral 7B's hidden dimension

# ---------------------------------------------------------------------------
# LoRA — applied to attention Q and V projections across all 32 layers
# ---------------------------------------------------------------------------
LORA_RANK = 16           # Low-rank dimension; 16 balances capacity vs. overfitting for 20 classes
LORA_ALPHA = 16          # Scaling = alpha/rank = 1.0 — no amplification of adapter updates
LORA_DROPOUT = 0.05      # Light regularization; chunking already provides data augmentation
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_BIAS = "none"       # No bias adaptation — keeps trainable param count minimal

# ---------------------------------------------------------------------------
# Classification Head
# ---------------------------------------------------------------------------
CLASSIFIER_DROPOUT = 0.1  # Between last-token hidden state and linear head

# ---------------------------------------------------------------------------
# Training — Phase 1 (5 well-separated categories)
# ---------------------------------------------------------------------------
PHASE1_EPOCHS = 3
PHASE1_NUM_CLASSES = 5

# ---------------------------------------------------------------------------
# Training — Phase 2 (all 20 categories)
# ---------------------------------------------------------------------------
PHASE2_EPOCHS = 5
PHASE2_NUM_CLASSES = 20

# ---------------------------------------------------------------------------
# Training — Shared across phases
# ---------------------------------------------------------------------------
LEARNING_RATE = 1e-4      # Validated in reference notebook for Mistral 7B + LoRA
WEIGHT_DECAY = 0.01       # Standard L2 regularization for AdamW
MIN_LR = 1e-6             # Cosine scheduler floor — prevents LR from reaching zero
BATCH_SIZE = 4            # Fits within 24 GB VRAM with float16 model + activations
GRAD_ACCUMULATION_STEPS = 4  # Effective batch size = 4 * 4 = 16
EVAL_FREQ = 50            # Validate every N optimizer steps
EVAL_BATCHES = 5          # Number of batches for mid-training eval (speed vs. accuracy)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
CHUNK_SIZE = 512
STRIDE = 256
DATA_DIR = Path("../data")               # Shared data from prepare_data.py
PHASE1_DATA = DATA_DIR / "phase1_data.json"
PHASE2_DATA = DATA_DIR / "phase2_data.json"

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = Path("checkpoints")
PHASE1_CHECKPOINT = CHECKPOINT_DIR / "phase1"
PHASE2_CHECKPOINT = CHECKPOINT_DIR / "phase2"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
