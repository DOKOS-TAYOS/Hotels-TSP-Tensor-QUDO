.PHONY: setup lint test app clean

EXTRAS ?= dev,ui,cirq
PYTHON := .venv/bin/python
STREAMLIT := .venv/bin/streamlit

setup:
	./bin/setup.sh "$(EXTRAS)"

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest

app:
	$(STREAMLIT) run src/streamlit_app/app.py

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov .tmp tests/.tmp
	find src tests -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
	find src tests -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true
	rm -rf pytest-cache-files-* 2>/dev/null || true
