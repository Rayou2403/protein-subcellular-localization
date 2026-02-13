VENV ?= .venv
PYTHON ?= python3

venv:
	@if [ ! -x "$(VENV)/bin/python" ]; then python3 -m venv $(VENV); fi
	$(VENV)/bin/python -m pip install --upgrade pip

setup: venv
	$(VENV)/bin/python -m pip install -r requirements.txt
	$(VENV)/bin/python -m pip install -e .
	@echo "Use: PYTHON=$(VENV)/bin/python make <target>"

init-data:
	mkdir -p data/raw data/processed/embeddings data/interim/subsets data/external report/figures
	touch data/raw/.gitkeep data/processed/.gitkeep data/processed/embeddings/.gitkeep
	touch data/interim/.gitkeep data/interim/subsets/.gitkeep data/external/.gitkeep report/figures/.gitkeep

ingest-data:
	bash scripts/ingest_data.sh

prepare: init-data ingest-data
	$(PYTHON) scripts/prepare_metadata.py \
		--fasta data/raw/graphpart_set.fasta \
		--output data/processed/metadata.csv
	$(PYTHON) scripts/prepare_splits.py \
		--metadata data/processed/metadata.csv \
		--output data/processed/splits.csv \
		--val_size 0.1 \
		--test_size 0.20

run:
	$(PYTHON) scripts/train.py --config configs/default.yaml

test:
	$(PYTHON) -m pytest -q

eda:
	$(PYTHON) scripts/eda_report.py --splits data/processed/splits.csv --out_dir report

results-figures:
	$(PYTHON) scripts/plot_results.py

report-figures: eda results-figures

embeddings: init-data ingest-data
	$(PYTHON) -m src.embeddings.fetch_embeddings \
		--esm_fasta data/raw/graphpart_set.fasta \
		--esm_out data/processed/embeddings/esmc.h5 \
		--prost_out data/processed/embeddings/prostt5.h5 \
		--embed2_backend prostt5 \
		--prost_pooling meanpool \
		--esm_batch 16 \
		--prost_batch 1 \
		--max_len 1000 \
		--prost_offload_dir data/interim/offload \
		--prost_max_memory 6GB

run-eval:
	$(PYTHON) scripts/evaluate.py \
		--checkpoint results/checkpoints/best_model.pt \
		--config configs/default.yaml

project: prepare embeddings run run-eval report-figures

status:
	@echo "== Embeddings =="
	@ls -lh data/processed/embeddings/*.h5 2>/dev/null || echo "No embeddings yet."
	@echo ""
	@echo "== Checkpoints =="
	@ls -lh results/checkpoints/*.pt 2>/dev/null || echo "No checkpoints yet."
	@echo ""
	@echo "== Evaluation =="
	@ls -lh results/evaluation/* 2>/dev/null || echo "No evaluation outputs yet."
	@echo ""
	@echo "== Docker running jobs =="
	@if command -v docker >/dev/null 2>&1; then \
		docker ps --format 'table {{.Names}}\t{{.Status}}' | awk 'NR==1 || /psl-run/' ; \
	else \
		echo "docker command not found."; \
	fi
	@echo ""
	@echo "== Live progress =="
	@echo "(progress from logs may lag with tqdm carriage-return updates)"
	@if command -v docker >/dev/null 2>&1; then \
		container=$$(docker ps --format '{{.Names}}' | grep 'psl-run' | head -n 1); \
		if [ -z "$$container" ]; then \
			echo "No running psl-run container."; \
		else \
			progress=$$(docker logs --tail 800 "$$container" 2>&1 | tr '\r' '\n' | \
				grep -E 'ESMC batches:|ProstT5 batches:|Epoch [0-9]+/[0-9]+|Starting training|Starting evaluation|Results saved to' | tail -n 1); \
			if [ -n "$$progress" ]; then \
				echo "$$progress"; \
			else \
				echo "No parsed progress yet (job may still be in setup/download)."; \
			fi; \
		fi; \
	else \
		echo "docker command not found."; \
	fi

rebuild-local: prepare embeddings
