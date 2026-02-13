#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mkdir -p data/raw

old_subject_pdf="data/Predicting the subcellular location of prokaryotic proteins with DeepLocPro - btae677.pdf"
new_subject_pdf="data/deeplocpro_subject_btae677.pdf"
if [[ -f "$old_subject_pdf" && ! -f "$new_subject_pdf" ]]; then
  mv "$old_subject_pdf" "$new_subject_pdf"
  echo "Renamed subject PDF to $new_subject_pdf"
fi

for fasta_name in graphpart_set.fasta full_dataset.fasta benchmarking_dataset.fasta; do
  if [[ -f "data/$fasta_name" ]]; then
    mv -f "data/$fasta_name" "data/raw/$fasta_name"
    echo "Moved data/$fasta_name -> data/raw/$fasta_name"
  fi
done

# Keep the project's canonical input name stable
if [[ ! -f "data/raw/graphpart_set.fasta" ]]; then
  if [[ -f "data/raw/full_dataset.fasta" ]]; then
    cp -f "data/raw/full_dataset.fasta" "data/raw/graphpart_set.fasta"
    echo "Using full_dataset.fasta as data/raw/graphpart_set.fasta"
  elif [[ -f "data/raw/benchmarking_dataset.fasta" ]]; then
    cp -f "data/raw/benchmarking_dataset.fasta" "data/raw/graphpart_set.fasta"
    echo "Using benchmarking_dataset.fasta as data/raw/graphpart_set.fasta"
  else
    echo "ERROR: No FASTA found for canonical input data/raw/graphpart_set.fasta" >&2
    echo "Place one of graphpart_set.fasta, full_dataset.fasta, benchmarking_dataset.fasta under data/." >&2
    exit 1
  fi
fi

echo "Data ingestion completed."
