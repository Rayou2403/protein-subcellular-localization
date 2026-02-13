# EDA Summary

Rows: 11906
Columns: protein_id, sequence, label, gram_type, partition, split, seq_len, label_display

## Missingness
protein_id       0.0
sequence         0.0
label            0.0
gram_type        0.0
partition        0.0
split            0.0
seq_len          0.0
label_display    0.0

## Label distribution
label_display
Cytoplasmic             6885
Cytoplasmic Membrane    2535
Extracellular           1077
Outer Membrane           756
Periplasmic              566
Cell Wall                 87

## Sequence length stats
count    11906.000000
mean       438.379473
std        289.049394
min          8.000000
10%        146.000000
25%        264.000000
50%        399.000000
75%        557.000000
90%        787.000000
95%        878.000000
99%       1296.000000
max       5627.000000

## Suggested max_len for quick tests
- max_len 146: keeps ~10% of sequences (very small quick test)
- max_len 264: keeps ~25% of sequences (small quick test)
- max_len 150: very fast, may drop longer sequences
- max_len 200: good quick test
- max_len 787: keeps ~90% of sequences
- max_len 878: keeps ~95% of sequences
- max_len 1296: keeps ~99% of sequences

## Gram type distribution
gram_type
negative    8417
positive    3206
archaea      283

## Split distribution
split
train    8333
test     2382
val      1191

## Conclusion (FR)
Les distributions par classe et par split montrent les volumes disponibles pour l'apprentissage. La longueur des sequences varie fortement, ce qui augmente le cout de calcul pendant l'extraction d'embeddings. En pratique, `max_len=1000` est un bon compromis CPU: couverture elevee du dataset tout en restant faisable avec ProstT5 3Di.
Exemple: `python -m src.embeddings.fetch_embeddings --esm_fasta data/raw/graphpart_set.fasta --esm_out data/processed/embeddings/esmc.h5 --prost_out data/processed/embeddings/prostt5.h5 --embed2_backend prostt5 --esm_batch 16 --prost_batch 1 --max_len 1000 --prost_offload_dir data/interim/offload --prost_max_memory 6GB`.