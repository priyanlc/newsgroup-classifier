"""
MLX Configuration — Centralized hyperparameters and paths for Apple Silicon.

Mirrors cuda/config.py but with MLX-specific values:
  - scale (1.0) instead of alpha/rank
  - Smaller batch size (2) to fit in 32 GB unified memory
  - Higher gradient accumulation (8) to maintain effective batch of 16
  - LoRA target keys use MLX-style dotted paths
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_ID = "mistralai/Mistral-7B-v0.1"
HIDDEN_SIZE = 4096  # Mistral 7B's hidden dimension
NUM_LAYERS = 32     # Number of transformer layers

# ---------------------------------------------------------------------------
# LoRA — MLX uses scale directly instead of alpha/rank
# ---------------------------------------------------------------------------
LORA_RANK = 16
LORA_SCALE = 1.0       # Equivalent to alpha/rank = 16/16 = 1.0 in PEFT
LORA_DROPOUT = 0.05
# MLX uses dotted-path keys to identify which linear layers get LoRA
LORA_KEYS = ["self_attn.q_proj", "self_attn.v_proj"]

# ---------------------------------------------------------------------------
# Classification Head
# ---------------------------------------------------------------------------
CLASSIFIER_DROPOUT = 0.1

# ---------------------------------------------------------------------------
# Training — Phase 1 (5 well-separated categories)
# ---------------------------------------------------------------------------
PHASE1_EPOCHS = 3
PHASE1_NUM_CLASSES = 5
PHASE1_MAX_STEPS = None  # None = full epochs; set to int (e.g. 100) to cap optimizer steps

# ---------------------------------------------------------------------------
# Training — Phase 2 (all 20 categories)
# ---------------------------------------------------------------------------
PHASE2_EPOCHS = 5
PHASE2_NUM_CLASSES = 20
PHASE2_MAX_STEPS = None  # None = full epochs; set to int (e.g. 100) to cap optimizer steps

# ---------------------------------------------------------------------------
# Training — Shared across phases
# ---------------------------------------------------------------------------
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
MIN_LR = 1e-6
BATCH_SIZE = 2            # Conservative for 32 GB unified memory (OS takes ~4-6 GB)
GRAD_ACCUMULATION_STEPS = 8  # Effective batch size = 2 * 8 = 16 (matches CUDA)
EVAL_FREQ = 50            # Validate every N optimizer steps
EVAL_BATCHES = 5

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
CHUNK_SIZE = 512
STRIDE = 256
DATA_DIR = Path("../data")
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
