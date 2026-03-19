"""Streamlit application package."""


def main() -> None:
    """Lazy wrapper so importing streamlit_app does not require the UI extra."""
    from streamlit_app.app import main as app_main

    app_main()


__all__ = ["main"]
