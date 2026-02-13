# Protein Subcellular Localization

This project predicts the subcellular localization of prokaryotic proteins (6 classes) using
pretrained protein embeddings and a custom Transformer+BiLSTM classifier. The pipeline prepares
metadata, builds leakage-safe splits, extracts embeddings, trains the fusion model, and evaluates.
The final setup uses both required embeddings: ESM-C and ProstT5 3Di.

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

On Manjaro/Arch (PEP 668), `make setup` uses a local virtualenv (`.venv`) automatically.
Run commands with:
```bash
PYTHON=.venv/bin/python make <target>
```

### Docker (optional)

```bash
docker compose build
docker compose run --rm psl
```

The compose file mounts the repo into the container, so code changes do not require a rebuild.
To force a rebuild, run `docker compose build` again or use `docker compose run --rm --build psl`.

## Data

Place the subject PDF and FASTA files under `data/`, then run:
```bash
make ingest-data
```
This command renames/moves inputs to canonical paths under `data/raw/` so `data/` stays clean.
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

CPU-only example for an Asus TUF 15 (no suitable GPU). We cap max length at 1000 aa
as a practical compromise (close to full coverage while still feasible with offload):
```bash
python -m src.embeddings.fetch_embeddings \
  --esm_fasta data/raw/graphpart_set.fasta \
  --esm_out data/processed/embeddings/esmc.h5 \
  --prost_out data/processed/embeddings/prostt5.h5 \
  --embed2_backend prostt5 \
  --esm_batch 16 \
  --prost_batch 1 \
  --max_len 1000 \
  --prost_pooling meanpool \
  --prost_offload_dir data/interim/offload \
  --prost_max_memory 6GB
```
Note: if memory is still too tight, reduce `--max_len` (e.g., 800 or 600).

Parameters you will likely tune:
- `--embed2_backend`: second embedding model (`prostt5` required for final submission).
- `--max_len`: maximum sequence length kept (longer sequences are skipped).
- `--subset_frac`: fraction of sequences to keep (0.1 = 10%).
- `--esm_batch`: batch size for ESM-C extraction.
- `--prost_batch`: batch size for ProstT5.
- `--prost_pooling`: pooling for the second embedding (`meanpool`, `cls`, `both`).
- `--prost_offload_dir`: disk offload directory for ProstT5 (CPU only).
- `--prost_max_memory`: memory budget for ProstT5 offload (e.g., `6GB`).

For GPU runs, increase `--prost_batch` and remove offload if memory allows.

### 4) Train

```bash
make run
```

Full chained run (long):
```bash
make project
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

To build a PDF (run after `make eda` and `make results-figures` so all figures exist):
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
- `report/figures/*.png`
- `report/figures/overview.png`
- `report/eda.md`

## Results Figures

Generate figures based on evaluation outputs (`results/evaluation/*.json`):

```bash
make results-figures
```

Outputs:
- `report/figures/results_metrics.png`
- `report/figures/results_per_class_f1.png`

## Architecture

```
configs/               # Experiment configs
data/
  raw/                 # Raw FASTA inputs
  processed/           # CSVs + embeddings
  interim/             # Subsets and temporary outputs
  external/            # Optional external assets
notebooks/             # Exploration notebooks
report/                # LaTeX project report
  figures/             # EDA plots
results/               # Training outputs
scripts/               # CLI entrypoints
src/                   # Library code
tests/                 # Tests
```

## Model

Default model is `transformer_lstm` (see `configs/default.yaml`):
- project ESM-C and ProstT5 embeddings to a shared hidden size
- treat modalities as tokens and pass through a small Transformer encoder
- refine token interactions with a BiLSTM head
- classify with an MLP head

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
- CPU-only runs: keep `--prost_batch 1` and enable `--prost_offload_dir`.
- If needed, reduce `--max_len` below 1000 for faster/safer extraction.
- ESM-C is large; for quick smoke tests, use `--subset_frac 0.1`.
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
  --embed2_backend prostt5 \
  --esm_batch 16 \
  --prost_batch 1 \
  --max_len 1000 \
  --prost_pooling meanpool \
  --prost_offload_dir data/interim/offload \
  --prost_max_memory 6GB
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --checkpoint results/checkpoints/best_model.pt --config configs/default.yaml
```
