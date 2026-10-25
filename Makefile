.PHONY: install test lint prepare dup-check zeroshot linear finetune compare cam app all clean

# uv runs every command in the project's own environment (CI overrides with PY=python).
PY ?= uv run python
CLI = $(PY) -m damage_triage

install:            ## core + dev + Streamlit
	uv sync --extra dev --extra app

test:               ## offline tests (tiny untrained models, synthetic images)
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests app.py

prepare:            ## labels, same-vehicle groups, train/val/test split, resized images
	$(CLI) prepare

dup-check:          ## save grouped photo pairs to docs/images/grouped_pairs.jpg to check by eye
	$(CLI) dup-check

zeroshot:           ## CLIP baseline, no training (downloads ~600 MB of weights once)
	$(CLI) zeroshot

linear:             ## frozen pretrained model + trained final layer, then predictions
	$(CLI) train --mode linear
	$(CLI) predict results/efficientnet_b0_linear

finetune:           ## fine-tune the whole pretrained model, then predictions (incl. degraded test photos)
	$(CLI) train --mode finetune
	$(CLI) predict results/efficientnet_b0_finetune

compare:            ## score every run on the held-out test split -> results/comparison.md
	$(CLI) compare

cam:                ## Grad-CAM heatmaps for a few test photos -> docs/images/
	$(CLI) cam results/efficientnet_b0_finetune

app:
	$(PY) -m streamlit run app.py

all: prepare zeroshot linear finetune compare cam

clean:
	rm -rf data/processed results/*/model.pt .pytest_cache .ruff_cache
