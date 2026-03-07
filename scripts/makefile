.PHONY: setup lint test app

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
