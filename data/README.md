# Data Layout

Put the assignment assets directly under `data/`:

- `data/deeplocpro_subject_btae677.pdf` (subject/article PDF)
- `data/graphpart_set.fasta`
- `data/full_dataset.fasta`
- `data/benchmarking_dataset.fasta`

Then run:

```bash
make ingest-data
```

This moves and normalizes inputs under `data/raw/`:

```
data/
  raw/
    graphpart_set.fasta
    full_dataset.fasta
    benchmarking_dataset.fasta
  processed/           # metadata/splits/embeddings
  interim/             # temporary subsets and offload files
  external/            # optional external assets
```

Core generated outputs:
- `data/processed/metadata.csv`
- `data/processed/splits.csv`
- `data/processed/embeddings/esmc.h5`
- `data/processed/embeddings/prostt5.h5`
