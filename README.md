# Protein Subcellular Localization

This project predicts the subcellular localization of prokaryotic proteins (6 classes) using
pretrained protein embeddings and a lightweight Transformer+MLP classifier. The pipeline prepares
metadata, builds leakage-safe splits, extracts embeddings, trains the fusion model, and evaluates.
ProstT5 remains optional for GPU setups. By default, the second embedding file is still named
`prostt5.h5` for compatibility.

## Quick Demo

Expected artifacts:
- `data/processed/metadata.csv` and `data/processed/splits.csv`
- `data/processed/embeddings/esmc.h5` and `data/processed/embeddings/prostt5.h5`
- training outputs under `results/`

## Installation

### Local

Prerequisites: Python 3.10+, Git.

```bash
make setup
```

### Docker (optional)

```bash
docker compose build
docker compose run --rm psl
```

The compose file mounts the repo into the container, so code changes do not require a rebuild.
To force a rebuild, run `docker compose build` again or use `docker compose run --rm --build psl`.

## Data

Place DeepLocPro FASTA files under `data/raw/`.
FASTA headers must be in the format:
```
>PROTEIN_ID|LOCATION|GRAM_TYPE|PARTITION
```

Expected layout:
```
data/
  raw/graphpart_set.fasta
  processed/metadata.csv
  processed/splits.csv
  processed/embeddings/
```

## Usage

### 1) Prepare metadata

```bash
python scripts/prepare_metadata.py \
  --fasta data/raw/graphpart_set.fasta \
  --output data/processed/metadata.csv
```

### 2) Create splits

```bash
python scripts/prepare_splits.py \
  --metadata data/processed/metadata.csv \
  --output data/processed/splits.csv \
  --val_size 0.1 \
  --test_size 0.20
```

### 3) Extract embeddings

```bash
make embeddings
```

CPU-only example for an Asus TUF 15 (no suitable GPU). We cap max_len to avoid OOM and use ProtBert:
```bash
python -m src.embeddings.fetch_embeddings \
  --esm_fasta data/raw/graphpart_set.fasta \
  --esm_out data/processed/embeddings/esmc.h5 \
  --prost_out data/processed/embeddings/prostt5.h5 \
  --embed2_backend protbert \
  --esm_batch 32 \
  --prost_batch 32 \
  --max_len 300 \
  --prost_pooling meanpool
```
Note: We do not have a GPU suitable for very long sequences; increase `--max_len` or batch sizes only if you have more RAM/GPU.

Parameters you will likely tune:
- `--embed2_backend`: second embedding model (`protbert` or `prostt5`).
- `--max_len`: maximum sequence length kept (longer sequences are skipped).
- `--subset_frac`: fraction of sequences to keep (0.1 = 10%).
- `--esm_batch`: batch size for ESM-C extraction.
- `--prost_batch`: batch size for the second embedding.
- `--prost_pooling`: pooling for the second embedding (`meanpool`, `cls`, `both`).
- `--prost_offload_dir`: disk offload directory for ProstT5 (CPU only).
- `--prost_max_memory`: memory budget for ProstT5 offload (e.g., `6GB`).

To use ProstT5 instead (GPU recommended):
```bash
python -m src.embeddings.fetch_embeddings \
  --embed2_backend prostt5 \
  --prost_batch 1 \
  --prost_offload_dir data/interim/offload \
  --prost_max_memory 6GB
```

### 4) Train

```bash
make run
```

### 5) Evaluate

```bash
python scripts/evaluate.py \
  --checkpoint results/checkpoints/best_model.pt \
  --config configs/default.yaml
```
Use the same config that was used for training so the model definition matches the checkpoint.

### Results

Evaluation outputs:
- `results/evaluation/metrics.json`: global metrics (Accuracy, Macro-F1, MCC)
- `results/evaluation/per_class_metrics.json`: per-class Precision/Recall/F1

Latest run (from `results/evaluation/metrics.json`):
- Accuracy: 0.8741
- Macro-F1: 0.8005
- MCC: 0.8227

## Report (LaTeX)

Two versions are provided:
- `report/report_fr.tex` (French)
- `report/report_eng.tex` (English)
`report/report.tex` is kept as a copy of the French report for convenience.

To build a PDF (run after `make eda` so figures exist):
```bash
cd report
pdflatex report_fr.tex
pdflatex report_eng.tex
```

## EDA

Generate exploratory figures and a short summary:

```bash
make eda
```

Outputs:
- `reports/figures/*.png`
- `reports/figures/overview.png`
- `reports/eda.md`

## Architecture

```
configs/               # Experiment configs
data/
  raw/                 # Raw FASTA inputs
  processed/           # CSVs + embeddings
  interim/             # Subsets and temporary outputs
  external/            # Optional external assets
notebooks/             # Exploration notebooks
reports/
  figures/             # EDA plots
report/                # LaTeX project report
results/               # Training outputs
scripts/               # CLI entrypoints
src/                   # Library code
tests/                 # Tests
```

## Model

Default model is `transformer_mlp` (see `configs/default.yaml`):
- project ESM-C and ProstT5 embeddings to a shared hidden size
- treat modalities as tokens (2 tokens) and pass through a small Transformer encoder
- classify with an MLP head
For CPU-friendly runs, the second embedding is ProtBert (same 1024-dim).

You can switch to the gated MLP baseline by setting:
```yaml
model:
  type: "gated_mlp"
```

For larger experiments (GPU recommended), use:
```bash
python scripts/train.py --config configs/medium.yaml
```

## Reproducibility

- Random seed is set in `configs/*.yaml`.
- Use `requirements.txt` for pinned dependency versions.

## Troubleshooting

- ESM-C install issues: `pip install git+https://github.com/evolutionaryscale/esm.git`.
- CPU-only runs: use `--embed2_backend protbert` to avoid ProstT5 OOM.
- If you must use ProstT5 on CPU, keep `--prost_batch 1` and enable `--prost_offload_dir`.
- ESM-C is large; start with `--subset_frac 0.1` and `--esm_batch 1-4` on CPU.
- Missing outputs: ensure `data/processed/` exists and paths match the config.

## Roadmap

- Add GPU-optimized embedding extraction path.
- Improve training speed with caching and mixed precision.
- Add hyperparameter sweeps and model selection reports.

## Full Launch Checklist

```bash
make setup
python scripts/prepare_metadata.py --fasta data/raw/graphpart_set.fasta --output data/processed/metadata.csv
python scripts/prepare_splits.py --metadata data/processed/metadata.csv --output data/processed/splits.csv
make embeddings
make run
python scripts/evaluate.py --checkpoint results/checkpoints/best_model.pt --config configs/default.yaml
make test
```

## Container Run (Quick)

```bash
docker compose build
docker compose run --rm psl
```

Inside the container:
```bash
python scripts/prepare_metadata.py --fasta data/raw/graphpart_set.fasta --output data/processed/metadata.csv
python scripts/prepare_splits.py --metadata data/processed/metadata.csv --output data/processed/splits.csv
python -m src.embeddings.fetch_embeddings \
  --esm_fasta data/raw/graphpart_set.fasta \
  --esm_out data/processed/embeddings/esmc.h5 \
  --prost_out data/processed/embeddings/prostt5.h5 \
  --embed2_backend protbert \
  --esm_batch 32 \
  --prost_batch 32 \
  --max_len 300 \
  --prost_pooling meanpool
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --checkpoint results/checkpoints/best_model.pt --config configs/default.yaml
```
