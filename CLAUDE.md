# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dual-platform (CUDA/MLX) LoRA fine-tuning pipeline for 20-class document classification on the 20 Newsgroups dataset using Mistral 7B. The CUDA and MLX implementations are independent but share identical data preparation, hyperparameters, and training logic.

## Environment & Tooling

- When installing Python packages, always use `uv` (not `pip`). Example: `uv pip install <package>`.
- Python 3.10+ required.
- CUDA path needs >= 24 GB VRAM; MLX path needs >= 32 GB unified memory (Apple Silicon only).

## Commands

All scripts must be run from the **project root** unless noted.

```bash
# Data preparation (shared, CPU-only) — creates data/phase1_data.json and data/phase2_data.json
python prepare_data.py

# Training (run from cuda/ or mlx/ directory)
cd cuda  # or cd mlx
python train.py              # Both phases
python train.py --phase 1    # Phase 1 only (5-class)
python train.py --phase 2    # Phase 2 only (20-class)

# Evaluation
python evaluate.py --phase 1
python evaluate.py --phase 2

# Interactive inference
python inference.py              # Demo texts + interactive mode
python inference.py --demo-only  # Demo texts only
```

There is no test suite (pytest). Verification is manual via VERIFICATION.md checklists and expected accuracy thresholds.

## Architecture

### Pipeline Flow

1. **prepare_data.py** — Downloads 20 Newsgroups, tokenizes with Mistral tokenizer, chunks documents (512 tokens, 256-token stride with overlap), splits train/val/test (70/15/15) at document level, outputs JSON.
2. **train.py** — Two-phase training: Phase 1 validates pipeline on 5 well-separated categories (3 epochs); Phase 2 trains on all 20 categories (5 epochs). Saves LoRA adapters + classification head to `checkpoints/`.
3. **evaluate.py** — Loads checkpoint, runs test set, produces classification report + confusion matrix heatmap.
4. **inference.py** — Loads checkpoint, runs demo texts and/or interactive classification.

### Key Design Decisions

- **Last-token classification**: Causal LM processes left-to-right; the last non-padding token's hidden state is extracted and passed through dropout (0.1) + linear head (4096 → num_classes).
- **Chunk-based logit aggregation**: Long documents are split into overlapping windows. At inference, logits from all chunks are averaged for the final prediction.
- **Dynamic padding**: Each batch is padded to its own max length (not globally to chunk_size), saving 2-4x compute.
- **LoRA targets**: Only `q_proj` and `v_proj` across all 32 layers (~0.2% trainable params). MLX uses full paths (`self_attn.q_proj`, `self_attn.v_proj`).

### Cross-Platform Parity

CUDA and MLX implementations mirror each other but differ in framework details:

| Aspect | CUDA | MLX |
|--------|------|-----|
| Model class | `MistralForSequenceClassification` (PyTorch) | `MistralClassifier` (MLX nn.Module) |
| LoRA | PEFT library | MLX built-in LoRA |
| Batch size / grad accum | 4 / 4 | 2 / 8 |
| Effective batch size | 16 | 16 |
| Checkpoint format | `.pt` + safetensors | `.safetensors` + JSON metadata |

Results should be within ~3% accuracy across platforms.

### Centralized Config

All hyperparameters live in `config.py` (per platform). Scripts import from config — no hardcoded values. Key params: `MODEL_ID = "mistralai/Mistral-7B-v0.1"`, `LORA_RANK = 16`, `LEARNING_RATE = 1e-4`, `CHUNK_SIZE = 512`, `STRIDE = 256`.

## Expected Results

- Phase 1 (5-class): >= 88% doc accuracy, >= 87% macro F1
- Phase 2 (20-class): >= 68% doc accuracy, >= 60% macro F1

## OOM Troubleshooting

Reduce `BATCH_SIZE` in the relevant `config.py` (to 2 for CUDA, 1 for MLX) and proportionally increase `GRAD_ACCUMULATION_STEPS` to keep effective batch size at 16.
