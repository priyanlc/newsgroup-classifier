# Verification Checklist

This checklist provides step-by-step instructions to verify each script works correctly. Since the code was written without GPU access, **you must run these checks yourself** to validate correctness.

---

## Environment Setup

### CUDA Platform
- [ ] `pip install -r cuda/requirements.txt` completes without errors
- [ ] GPU detected: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
- [ ] VRAM sufficient: `python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"`
  - **Expect:** >= 24 GB

### MLX Platform
- [ ] `pip install -r mlx/requirements.txt` completes without errors
- [ ] MLX detected: `python -c "import mlx.core as mx; print(mx.default_device())"`
  - **Expect:** `Device(gpu, 0)` on Apple Silicon
- [ ] Memory sufficient: Check Activity Monitor -> Memory tab
  - **Expect:** >= 32 GB total (physical memory)

---

## Script-by-Script Verification

### 1. prepare_data.py (shared, CPU-only)

```bash
# Run from output/code/ directory
python prepare_data.py
```

- [ ] Runs without errors
- [ ] Downloads 20 Newsgroups data successfully
- [ ] Downloads Mistral tokenizer from HuggingFace
- [ ] Creates `data/phase1_data.json` (~5-10 MB)
- [ ] Creates `data/phase2_data.json` (~20-40 MB)
- [ ] Console output shows chunk statistics for both phases:
  - Phase 1: ~2,400 train docs -> ~2,800-3,200 chunks
  - Phase 2: ~9,600 train docs -> ~11,000-13,000 chunks
- [ ] Spot-check JSON structure:
  ```bash
  python -c "
  import json
  with open('data/phase2_data.json') as f:
      d = json.load(f)
  print('Keys:', list(d.keys()))
  print('Num classes:', d['num_classes'])
  print('Label names:', d['label_names'][:5], '...')
  print('Train chunks:', len(d['train_chunks']))
  print('Chunk size sample:', len(d['train_chunks'][0]))
  "
  ```
  - **Expect:** 20 classes, 20 label names, ~11K-13K train chunks, first chunk length <= 512

**Expected runtime:** 2-5 minutes (mostly tokenizer download on first run)

---

### 2. cuda/config.py

```bash
cd cuda
python -c "import config; print(config.MODEL_ID, config.LORA_RANK, config.BATCH_SIZE)"
```

- [ ] Imports without errors
- [ ] Prints: `mistralai/Mistral-7B-v0.1 16 4`
- [ ] Verify model exists: [https://huggingface.co/mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)

### 3. cuda/model.py

```bash
cd cuda
python -c "from model import MistralForSequenceClassification; print('Model class loaded OK')"
```

- [ ] Imports without errors

### 4. cuda/train.py

```bash
cd cuda
python train.py --phase 1
```

- [ ] Model loads in float16 (~1-2 minutes)
- [ ] LoRA applied: look for `trainable params: X / Y` printout (~13.6M trainable)
- [ ] Training starts with loss decreasing over epochs
- [ ] Watch for every-50-steps validation logs:
  ```
  Train loss: X.XXX | Val loss: X.XXX
  ```
- [ ] Epoch-end shows doc-level val accuracy (should reach > 70% by epoch 2 for 5-class)
- [ ] Checkpoints saved to `cuda/checkpoints/phase1/`:
  - `lora_adapters/adapter_model.safetensors`
  - `lora_adapters/adapter_config.json`
  - `classifier_head.pt`
- [ ] **GPU memory usage:** ~16-18 GB peak (check with `nvidia-smi`)

**Expected runtime (Phase 1):** 10-20 minutes on RTX 3090/4090

Then run Phase 2:
```bash
python train.py --phase 2
```

- [ ] Fresh model loaded (not reusing Phase 1 weights)
- [ ] Training runs for 5 epochs
- [ ] Val accuracy should reach >= 55% by epoch 3 for 20-class
- [ ] Checkpoints saved to `cuda/checkpoints/phase2/`

**Expected runtime (Phase 2):** 35-70 minutes on RTX 3090/4090

### 5. cuda/evaluate.py

```bash
cd cuda
python evaluate.py --phase 1
```

- [ ] Loads Phase 1 checkpoint successfully
- [ ] Prints full classification report (5 categories)
- [ ] Document accuracy >= 88%
- [ ] Macro F1 >= 87%
- [ ] Confusion matrix saved to `cuda/checkpoints/phase1/confusion_matrix.png`

```bash
python evaluate.py --phase 2
```

- [ ] Loads Phase 2 checkpoint successfully
- [ ] Prints classification report (20 categories)
- [ ] Document accuracy >= 68%
- [ ] Macro F1 >= 60%
- [ ] Accuracy by topic group shows `rec.*` and `sci.*` highest, `comp.*` and `talk.*` lowest
- [ ] Top confused pairs are semantically related (e.g., comp.sys.* categories)
- [ ] Confusion matrix saved to `cuda/checkpoints/phase2/confusion_matrix.png`

**Expected runtime:** 5-10 minutes per phase (inference only)

### 6. cuda/inference.py

```bash
cd cuda
python inference.py --demo-only
```

- [ ] Loads model and runs 8 demo texts
- [ ] Each prediction shows: category, confidence %, top-3 classes, chunk count
- [ ] Sanity check predictions:
  - "Hubble telescope..." -> `sci.space`
  - "Penguins dominated..." -> `rec.sport.hockey`
  - "3D graphics using OpenGL..." -> `comp.graphics`
  - "Middle East..." -> `talk.politics.mideast`
  - "Jesus taught..." -> `soc.religion.christian`
  - "motorcycle...Honda CB750" -> `rec.motorcycles`

```bash
python inference.py
```

- [ ] Interactive mode starts after demo texts
- [ ] Typing custom text and pressing Enter produces a classification
- [ ] Ctrl+C exits cleanly

---

### 7. mlx/config.py

```bash
cd mlx
python -c "import config; print(config.MODEL_ID, config.LORA_RANK, config.BATCH_SIZE, config.LORA_SCALE)"
```

- [ ] Imports without errors
- [ ] Prints: `mistralai/Mistral-7B-v0.1 16 2 1.0`

### 8. mlx/model.py

```bash
cd mlx
python -c "from model import MistralClassifier; print('MLX model class loaded OK')"
```

- [ ] Imports without errors

### 9. mlx/train.py

```bash
cd mlx
python train.py --phase 1
```

- [ ] Model loads via mlx_lm (first load converts HF -> MLX, ~2-5 minutes)
- [ ] LoRA applied with ~13.6M trainable parameters
- [ ] Training starts with loss decreasing
- [ ] Epoch-end val accuracy > 70% for 5-class
- [ ] Checkpoints saved to `mlx/checkpoints/phase1/`:
  - `lora_adapters.safetensors`
  - `classifier_head.safetensors`
  - `metadata.json`
- [ ] **Memory usage:** ~18-20 GB (check Activity Monitor -> Memory Pressure)

**Expected runtime (Phase 1):** 15-30 minutes on M1/M2/M3 Max

```bash
python train.py --phase 2
```

- [ ] 5-epoch training completes
- [ ] Val accuracy >= 55% by epoch 3
- [ ] Checkpoints saved to `mlx/checkpoints/phase2/`

**Expected runtime (Phase 2):** 50-100 minutes on M1/M2/M3 Max

### 10. mlx/evaluate.py

```bash
cd mlx
python evaluate.py --phase 2
```

- [ ] Loads checkpoint and evaluates
- [ ] Document accuracy >= 68%
- [ ] Macro F1 >= 60%
- [ ] Confusion matrix saved as PNG
- [ ] Results should be within ~3% of CUDA results (cross-platform parity)

### 11. mlx/inference.py

```bash
cd mlx
python inference.py --demo-only
```

- [ ] Runs 8 demo texts with predictions
- [ ] Predictions should match CUDA version's predictions (same model, same data)

---

## Cross-Platform Comparison

After running both CUDA and MLX, compare:

- [ ] Phase 2 accuracy within 3% across platforms
- [ ] Phase 2 macro F1 within 3% across platforms
- [ ] Top confused pairs are the same on both platforms
- [ ] Demo text predictions largely agree

Small differences are expected due to floating-point differences between CUDA and MLX backends.

---

## Common Issues

| Issue | Fix |
|-------|-----|
| `FileNotFoundError: data/phase1_data.json` | Run `python prepare_data.py` from the `output/code/` directory first |
| `torch.cuda.OutOfMemoryError` | Reduce `BATCH_SIZE` in `cuda/config.py` to 2, increase `GRAD_ACCUMULATION_STEPS` to 8 |
| `OSError: We couldn't connect to 'https://huggingface.co'` | Check internet connection; HuggingFace Hub needs to download model/tokenizer |
| CUDA `RuntimeError: CUDA error: out of memory` during model load | Need >= 24 GB VRAM; the float16 model alone uses ~14 GB |
| MLX `MemoryError` or system slowdown | Need >= 32 GB unified memory; close other applications |
| `ModuleNotFoundError: No module named 'mlx'` | MLX requires Apple Silicon (M1+); does not run on Intel Macs or Linux |
| `safetensors` import error in evaluate.py | `pip install safetensors>=0.4.0` |
| Confusion matrix plot doesn't display | Run with `matplotlib` Agg backend (set automatically in MLX evaluate.py) or use `--save-only` |
| Phase 2 accuracy below 60% | Training may not have converged — try increasing `PHASE2_EPOCHS` to 7 in config.py |
| LoRA weights not loading correctly | Ensure the checkpoint directory structure matches what train.py creates |
