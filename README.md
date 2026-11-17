# HistoPert: Histology-Guided Prediction of Drug-Induced Transcriptional Perturbations

## Overview

**HistoPert predicts patient-specific transcriptional perturbations from H&E, patient context, drug features and dose.**

The pipeline includes molecular alignment, histology alignment and perturbation prediction.

## System Requirements

Python 3.10. Run commands from this directory.

```bash
python -m pip install -r requirements.txt
```

## Dataset Requirements

Prepare precomputed UCE, H&E, drug and patient-context embeddings, plus LINCS treated/control expression. Set file paths in `configs/tcga.json`; see [data formats](data/README.md).

```bash
python -m histopert validate --config configs/tcga.json
```

## Quick Test

Train on synthetic data with small hidden layers and one epoch per stage:

```bash
python examples/make_demo.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m histopert train --config configs/demo.json --device cpu
```

Skip data generation if `examples/demo_data/` already exists. Predict with the trained checkpoint:

```bash
python -m histopert predict \
  --checkpoint outputs/demo/stage3/stage3_text_fusion_unicure_best.pt \
  --pairs examples/demo_data/prediction_pairs.csv \
  --he examples/demo_data/he.parquet \
  --context examples/demo_data/context.npy \
  --context-index examples/demo_data/context_index.csv \
  --drugs examples/demo_data/drugs.parquet \
  --output outputs/demo/predicted_delta.parquet \
  --device cpu
```

The output contains predicted expression changes for 978 genes, indexed by patient, drug and dose.

## Training

Run the full pipeline:

```bash
python -m histopert train --config configs/tcga.json --device cuda:0
```

Or run each stage in order:

```bash
python -m histopert stage1 --config configs/tcga.json --device cuda:0
python -m histopert stage2 --config configs/tcga.json --device cuda:0
python -m histopert stage3 --config configs/tcga.json --device cuda:0
```

Set training options and `run_dir` in the JSON config. Paths are relative to the config file; results go under `outputs/`. Use a new `run_dir` and prediction filename for each run, or add `--resume` to continue Stage 3.
