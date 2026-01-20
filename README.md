# Protein Subcellular Localization

Prokaryotic protein localization prediction (6 classes) using ESM-C and ProstT5 embeddings.

Dataset: DeepLocPro (11,906 proteins from PSORTdb 4.0 + UniProt)

Model: Cross-attention fusion with gated mechanism.


## Quick Start with Docker Compose

### 1. Build the image

```bash
docker compose build
```

### 2. Run container

```bash
docker compose run --rm psl
```

Stop and clean containers:
```bash
docker compose down
```


## Pipeline (inside container)

Embeddings are stored in HDF5 as pooled vectors (one per protein):
- ESM-C: shape (1280,)
- ProstT5: shape (1024,)

### Step 1: Prepare metadata

If you already have `data/metadata.csv`, you can skip this step.

```bash
python scripts/prepare_metadata.py \
    --fasta data/graphpart_set.fasta \
    --output data/metadata.csv
```

### Step 2: Create train/val/test splits

```bash
python scripts/prepare_splits.py \
    --metadata data/metadata.csv \
    --output data/splits.csv \
    --val_size 0.1 \
    --test_size 0.15
```

### Step 3: Extract embeddings (required once)

Generate pooled ESM-C and ProstT5 embeddings to HDF5:

```bash
python -m src.embeddings.fetch_esmc \
    --fasta data/graphpart_set.fasta \
    --output data/embeddings/esmc.h5

python -m src.embeddings.fetch_prostt5 \
    --fasta data/graphpart_set.fasta \
    --output data/embeddings/prostt5.h5
```

Notes:
- ESM-C requires Git and installs from source: `pip install git+https://github.com/evolutionaryscale/esm.git`.
- If you hit `register_pytree_node` errors, upgrade PyTorch to >= 2.0.

Optional one-shot command (ESM-C + ProstT5 + fusion):

```bash
python -m src.embeddings.fetch_embeddings \
    --esm_fasta data/graphpart_set.fasta \
    --esm_out data/embeddings/esmc.h5 \
    --prost_out data/embeddings/prostt5.h5
```

### Step 4: Train

```bash
python scripts/train.py --config configs/default.yaml
```

### Step 5: Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint results/checkpoints/best_model.pt \
    --config configs/default.yaml
```

## Small dataset (debug run)

If you want to validate the pipeline with limited resources, create a small,
balanced subset from the existing splits, then extract embeddings and train:

```bash
python scripts/make_small_dataset.py \
    --splits data/splits.csv \
    --output_dir data/small \
    --max_per_class 50
```

```bash
python -m src.embeddings.fetch_esmc \
    --fasta data/small/sequences.fasta \
    --output data/small/embeddings/esmc.h5 \
    --batch_size 4

python -m src.embeddings.fetch_prostt5 \
    --fasta data/small/sequences.fasta \
    --output data/small/embeddings/prostt5.h5 \
    --batch_size 2
```

```bash
python scripts/train.py --config configs/small.yaml
```


## Directory structure

```
data/
  embeddings/
    esmc.h5          # pooled embeddings, shape (N, 1280)
    prostt5.h5       # pooled embeddings, shape (N, 1024)
  splits.csv         # protein_id, label, split columns
```


## Configuration

Edit configs/default.yaml:
- batch_size: 32
- learning_rate: 0.0001
- num_epochs: 100
- loss_type: focal or weighted_ce


## Classes

1. Cytoplasmic
2. Cytoplasmic Membrane
3. Periplasmic
4. Outer Membrane
5. Extracellular
6. Cell Wall


## Local install (alternative to Docker)

```bash
pip install -r requirements.txt
pip install -e .
```
