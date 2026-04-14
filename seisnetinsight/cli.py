"""Command line entry points."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Dict


def _ensure_runtime_dirs() -> None:
    """Point Matplotlib at a writable config directory before Streamlit imports it."""
    if "MPLCONFIGDIR" in os.environ:
        return
    preferred = Path.home() / ".seisnetinsight" / "matplotlib"
    fallback = Path(tempfile.gettempdir()) / "seisnetinsight-matplotlib"
    for candidate in (preferred, fallback):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        os.environ["MPLCONFIGDIR"] = str(candidate)
        return


def _env_flag(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _streamlit_flag_options() -> Dict[str, Any]:
    options: Dict[str, Any] = {}
    headless = _env_flag("STREAMLIT_SERVER_HEADLESS")
    gather_usage_stats = _env_flag("STREAMLIT_BROWSER_GATHER_USAGE_STATS")
    port = os.environ.get("STREAMLIT_SERVER_PORT")
    address = os.environ.get("STREAMLIT_SERVER_ADDRESS")
    if headless is not None:
        options["server_headless"] = headless
    if gather_usage_stats is not None:
        options["browser_gatherUsageStats"] = gather_usage_stats
    if port:
        try:
            options["server_port"] = int(port)
        except ValueError:
            pass
    if address:
        options["server_address"] = address.strip()
    return options


def run_app() -> None:
    """Launch the Streamlit application."""
    _ensure_runtime_dirs()
    try:
        from streamlit import config as streamlit_config
        from streamlit.web import bootstrap
    except ImportError:  # pragma: no cover
        raise SystemExit("Streamlit is required to run the SeisNetInsight app.")

    script_path = Path(__file__).with_name("streamlit_app.py")
    flag_options = _streamlit_flag_options()
    streamlit_config._main_script_path = str(script_path)
    bootstrap.load_config_options(flag_options=flag_options)
    # Pass False for is_hello to launch a regular Streamlit app session.
    bootstrap.run(str(script_path), False, [], flag_options)
