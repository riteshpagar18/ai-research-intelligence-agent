"""Configuration loading shared by local and hosted Streamlit runs."""

import os


SECRET_KEYS = ("GEMINI_API_KEY", "GEMINI_MODEL", "TAVILY_API_KEY")


def load_streamlit_secrets(streamlit_module, environ=None):
    """Copy Streamlit Cloud secrets into environment variables if not set.

    Environment variables take precedence, which keeps local `.env` use and
    non-Streamlit hosting providers working as before.
    """
    environ = os.environ if environ is None else environ
    try:
        secrets = streamlit_module.secrets
        for key in SECRET_KEYS:
            value = secrets.get(key)
            if value is not None and not environ.get(key):
                environ[key] = str(value)
    except FileNotFoundError:
        return
