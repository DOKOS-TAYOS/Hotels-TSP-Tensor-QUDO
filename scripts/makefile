.PHONY: setup lint test app clean analysis-ingest analysis-metrics analysis-plots analysis-all results-web

EXTRAS ?= dev,ui,cudaq
PYTHON := .venv/bin/python
STREAMLIT := .venv/bin/streamlit
# scripts/makefile lives in scripts/; repo root is one level up.
REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)

setup:
	./bin/setup.sh "$(EXTRAS)"

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest

app:
	$(STREAMLIT) run src/streamlit_app/app.py

analysis-ingest:
	$(PYTHON) -m data_analysis.ingest --output-root output

analysis-metrics:
	$(PYTHON) -m data_analysis.metrics --output-root output

analysis-plots:
	$(PYTHON) -m data_analysis.plot --output-root output

analysis-all:
	$(PYTHON) -m data_analysis.pipeline --output-root output

# Serve repo root over HTTP so webpage_results/ can fetch output/processed/*.csv (not for file://).
results-web:
	@echo "Open http://localhost:8765/webpage_results/index.html"
	cd $(REPO_ROOT) && $(PYTHON) -m http.server 8765

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov .tmp tests/.tmp
	find src tests -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
	find src tests -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true
	rm -rf pytest-cache-files-* 2>/dev/null || true
