"""Tests for the optional Streamlit UI package contract."""

from __future__ import annotations

import importlib
import sys


def test_streamlit_app_import_is_lazy() -> None:
    sys.modules.pop("streamlit_app", None)
    sys.modules.pop("streamlit_app.app", None)
    sys.modules.pop("streamlit", None)

    module = importlib.import_module("streamlit_app")

    assert hasattr(module, "main")
    assert "streamlit" not in sys.modules
