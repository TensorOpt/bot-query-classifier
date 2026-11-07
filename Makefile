.DEFAULT_GOAL := help

VENV   ?= .venv
PYTHON := $(VENV)/bin/python
BQC    := $(PYTHON) -m bqc

.PHONY: help install fetch generate evaluate evaluate-quick cluster test clean distclean

help:  ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

$(PYTHON):
	python3 -m venv $(VENV)

install: $(PYTHON)  ## Create .venv and install the package with its dependencies
	$(PYTHON) -m pip install --quiet -e .

fetch: install  ## Download the WANDS files into data/raw/ (checksums verified)
	$(BQC) fetch

generate: install  ## Build the training set and write its audit report to results/
	$(BQC) generate

evaluate: install  ## Full run: both arms, leave-one-family-out, catalogue gap (about 80 minutes)
	$(BQC) evaluate --negatives product,real --lofo --catalogue-gap

evaluate-quick: install  ## Primary arm only, no extra sections (about 15 minutes, estimated)
	$(BQC) evaluate

cluster: install  ## Run the label-discovery experiment (seconds)
	$(BQC) cluster

test: install  ## Run the test suite (about 4 minutes)
	$(PYTHON) -m unittest discover -s tests

clean:  ## Remove Python caches
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: clean  ## Also remove .venv and the egg-info directory
	rm -rf $(VENV) *.egg-info
