#!/bin/bash
# Quick test script to verify the setup works

set -e

echo "Testing setup..."
echo ""

echo "1. Testing metadata preparation..."
python scripts/prepare_metadata.py \
    --fasta data/raw/graphpart_set.fasta \
    --output data/processed/metadata_test.csv

if [ -f data/processed/metadata_test.csv ]; then
    echo "   ✓ Metadata CSV created"
    head -5 data/processed/metadata_test.csv
    rm data/processed/metadata_test.csv
else
    echo "   ✗ Failed to create metadata CSV"
    exit 1
fi

echo ""
echo "2. Testing imports..."
python -c "
import torch
import pandas
import numpy
from Bio import SeqIO
from transformers import T5Tokenizer
print('   ✓ All imports successful')
"

echo ""
echo "3. Testing model initialization..."
python -c "
import torch
from src.models.fusion_model import DualEmbeddingFusionModel

config = {
    'esmc_dim': 1280,
    'prostt5_dim': 1024,
    'hidden_dim': 512,
    'num_heads': 8,
    'num_classes': 6,
    'dropout': 0.1,
    'pooling_type': 'attention'
}

model = DualEmbeddingFusionModel(**config)
print('   ✓ Model initialized successfully')
print(f'   Model parameters: {sum(p.numel() for p in model.parameters()):,}')
"

echo ""
echo "Setup test complete! ✓"
