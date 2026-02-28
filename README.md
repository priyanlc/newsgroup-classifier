# Multi-Class Document Classification with Mistral 7B + LoRA

Fine-tune Mistral 7B for 20-class document classification on the 20 Newsgroups dataset using LoRA adapters. Two parallel implementations: **NVIDIA CUDA** (PyTorch + PEFT) and **Apple Silicon** (MLX).

## Overview

This pipeline demonstrates:
- **LoRA fine-tuning** — trains only ~0.2% of Mistral 7B's parameters
- **Chunking with aggregation** — splits long documents into overlapping windows for training and averages logits at inference
- **Two-phase training** — validates on 5 diverse categories before scaling to all 20
- **Last-token classification** — extracts the causal LM's richest representation for classification

Both implementations load Mistral 7B in **float16** (no quantization) and share identical preprocessed data.

## Requirements

### CUDA (NVIDIA GPU)
- Python 3.10+
- NVIDIA GPU with **>= 24 GB VRAM** (RTX 3090, RTX 4090, A5000, or A100)
- CUDA toolkit 11.8+
- ~14 GB disk space for model weights

### MLX (Apple Silicon)
- Python 3.10+
- Apple Silicon Mac with **>= 32 GB unified memory** (M1/M2/M3 Max or Ultra)
- macOS 13.5+
- ~14 GB disk space for model weights

## Setup

### Step 1: Clone and install dependencies

```bash
# Shared (needed for data preparation on both platforms):
pip install -r requirements.txt

# For CUDA:
pip install -r cuda/requirements.txt

# For MLX:
pip install -r mlx/requirements.txt
```

### Step 2: Prepare data (shared, CPU-only)

```bash
# Run from the output/code/ directory
python prepare_data.py
```

This downloads the 20 Newsgroups dataset, tokenizes with the Mistral tokenizer, chunks long documents, and saves:
- `data/phase1_data.json` — 5-class subset
- `data/phase2_data.json` — Full 20-class dataset

**Runtime:** ~2-5 minutes (depends on internet speed for tokenizer download).

### Step 3: Train

```bash
# CUDA — run from cuda/ directory:
cd cuda
python train.py              # Both phases
python train.py --phase 1    # Phase 1 only (5-class, ~10-20 min)
python train.py --phase 2    # Phase 2 only (20-class, ~35-70 min)

# MLX — run from mlx/ directory:
cd mlx
python train.py              # Both phases
python train.py --phase 1    # Phase 1 only (5-class, ~15-30 min)
python train.py --phase 2    # Phase 2 only (20-class, ~50-100 min)
```

Checkpoints are saved to `cuda/checkpoints/` or `mlx/checkpoints/`.

### Step 4: Evaluate

```bash
# CUDA:
cd cuda
python evaluate.py --phase 1    # 5-class test results
python evaluate.py --phase 2    # 20-class test results (default)

# MLX:
cd mlx
python evaluate.py --phase 1
python evaluate.py --phase 2
```

Outputs: classification report, confusion matrix heatmap, accuracy by topic group.

### Step 5: Interactive inference

```bash
# CUDA:
cd cuda
python inference.py              # Demo texts + interactive mode
python inference.py --demo-only  # Demo texts only

# MLX:
cd mlx
python inference.py
python inference.py --demo-only
```

## Project Structure

```
output/code/
├── README.md              # This file
├── VERIFICATION.md        # Manual testing checklist
├── prepare_data.py        # Shared: data loading, tokenization, chunking
├── data/                  # Created by prepare_data.py
│   ├── phase1_data.json   # 5-class preprocessed data
│   └── phase2_data.json   # 20-class preprocessed data
├── cuda/
│   ├── requirements.txt   # PyTorch + PEFT + HuggingFace dependencies
│   ├── config.py          # All hyperparameters and paths
│   ├── model.py           # MistralForSequenceClassification wrapper
│   ├── train.py           # Two-phase training loop
│   ├── evaluate.py        # Test evaluation with confusion matrices
│   ├── inference.py       # Interactive classification demo
│   └── checkpoints/       # Created by train.py
└── mlx/
    ├── requirements.txt   # MLX + mlx-lm dependencies
    ├── config.py          # MLX-specific hyperparameters
    ├── model.py           # MistralClassifier (MLX nn.Module)
    ├── train.py           # Two-phase training (MLX patterns)
    ├── evaluate.py        # Test evaluation
    ├── inference.py       # Interactive classification demo
    └── checkpoints/       # Created by train.py
```

## Key Configuration

| Parameter | CUDA | MLX |
|-----------|------|-----|
| Precision | float16 | float16 |
| LoRA rank | 16 | 16 |
| LoRA targets | q_proj, v_proj | self_attn.q_proj, self_attn.v_proj |
| Batch size | 4 | 2 |
| Grad accumulation | 4 | 8 |
| Effective batch | 16 | 16 |
| Learning rate | 1e-4 | 1e-4 |
| Phase 1 epochs | 3 | 3 |
| Phase 2 epochs | 5 | 5 |
| Chunk size | 512 tokens | 512 tokens |
| Stride | 256 tokens | 256 tokens |

## Expected Results

| Metric | Phase 1 (5-class) | Phase 2 (20-class) |
|--------|-------------------|---------------------|
| Document accuracy | >= 88% | >= 68% |
| Macro F1 | >= 87% | >= 60% |
| Training time (CUDA) | ~10-20 min | ~35-70 min |
| Training time (MLX) | ~15-30 min | ~50-100 min |

## Troubleshooting

- **Out of memory (CUDA):** Reduce `BATCH_SIZE` in `cuda/config.py` to 2 and increase `GRAD_ACCUMULATION_STEPS` to 8.
- **Out of memory (MLX):** Reduce `BATCH_SIZE` in `mlx/config.py` to 1 and increase `GRAD_ACCUMULATION_STEPS` to 16.
- **Tokenizer download fails:** Ensure internet access. The Mistral tokenizer is downloaded from HuggingFace on first use.
- **Model download slow:** The ~14 GB model is downloaded once and cached. Subsequent runs use the cache.
- **CUDA not found:** Verify `nvidia-smi` works and CUDA toolkit is installed.
- **MLX import error:** Ensure you're on Apple Silicon (M1/M2/M3) with macOS 13.5+. MLX does not support Intel Macs.
