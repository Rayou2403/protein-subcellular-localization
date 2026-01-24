PYTHON ?= python3

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

run:
	$(PYTHON) scripts/train.py --config configs/default.yaml

test:
	$(PYTHON) -m pytest -q

eda:
	$(PYTHON) scripts/eda_report.py --splits data/processed/splits.csv --out_dir reports
	$(PYTHON) scripts/plot_results.py

embeddings:
	$(PYTHON) -m src.embeddings.fetch_embeddings \
		--esm_fasta data/raw/graphpart_set.fasta \
		--esm_out data/processed/embeddings/esmc.h5 \
		--prost_out data/processed/embeddings/prostt5.h5
