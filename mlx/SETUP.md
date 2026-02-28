# MLX Setup: Virtual Environment with uv on Apple Silicon

## Prerequisites

- Apple Silicon Mac (M1/M2/M3) with >= 32 GB unified memory
- macOS 13.5+
- [uv](https://docs.astral.sh/uv/) installed (`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)

## 1. Navigate to the code directory

```bash
cd /Users/priyanchandrapala/codebase/finetuning-blog-pipeline/projects/newsgroup-classifier/output/code
```

## 2. Create a virtual environment

```bash
uv venv --python 3.11 .venv
```

Python 3.11 is recommended — well-tested with MLX and mlx-lm.

## 3. Activate the virtual environment

```bash
source .venv/bin/activate
```

## 4. Install shared dependencies (for data preparation)

```bash
uv pip install transformers scikit-learn numpy
```

## 5. Install MLX dependencies

```bash
uv pip install -r mlx/requirements.txt
```

## 6. Run the pipeline

```bash
# Step 1: Prepare data (CPU-only, runs from output/code/)
python prepare_data.py

# Step 2: Train (run from mlx/ directory)
cd mlx
python train.py --phase 1    # ~15-30 min on M1/M2/M3 Max
python train.py --phase 2    # ~50-100 min on M1/M2/M3 Max

# Step 3: Evaluate
python evaluate.py --phase 2

# Step 4: Interactive inference
python inference.py --demo-only
```

## 7. Deactivate when done

```bash
deactivate
```

## Notes

- The first run of `train.py` downloads Mistral 7B (~14 GB) and converts it to MLX format. This is cached locally so subsequent runs start much faster.
- The float16 model uses ~14 GB of unified memory during training. With 32 GB+ you'll have plenty of headroom.
- If you hit memory pressure, reduce `BATCH_SIZE` to 1 and increase `GRAD_ACCUMULATION_STEPS` to 16 in `config.py`.
