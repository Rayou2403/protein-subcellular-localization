# Data Layout

This project does not version large datasets or generated artifacts. Expected layout:

```
data/
  raw/                 # Original FASTA files
  processed/           # Processed CSVs and embeddings
  interim/             # Temporary subsets or intermediate outputs
  external/            # External assets (optional)
```

Examples:
- `data/raw/graphpart_set.fasta`
- `data/processed/metadata.csv`
- `data/processed/splits.csv`
- `data/processed/embeddings/esmc.h5`
- `data/processed/embeddings/prostt5.h5`

Place the DeepLocPro FASTA files under `data/raw/` and regenerate processed files
with the scripts in `scripts/`.
