"""Command line entry points."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
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


class _UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, allow_abbrev=False, **kwargs)

    def error(self, message):
        raise _UsageError(message)


def main(argv=None) -> int:
    """Run a headless command with one JSON response on stdout in JSON mode."""
    argv = list(sys.argv[1:] if argv is None else argv)
    json_mode = "--json" in argv
    parser = _Parser(prog="seisnetinsight", description="Validate, run, or render a seismic monitoring analysis.")
    parser.add_argument("--json", action="store_true", help="Write one JSON result to stdout.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("validate", "Check configuration and local inputs."),
        ("run", "Compute grids and save maps and a run manifest."),
        ("render", "Redraw maps from a previous run manifest."),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("--json", action="store_true", help="Write one JSON result to stdout.")
        if name == "render":
            command.add_argument("--manifest", required=True, type=Path)
        else:
            command.add_argument("--config", required=True, type=Path)
        if name != "validate":
            command.add_argument("--output-dir", required=True, type=Path, help="A new directory for the results.")

    def emit(result, *, error=False):
        if json_mode:
            print(json.dumps({"schema_version": 1, **result}, allow_nan=False))
        elif error:
            print(result["error"]["message"], file=sys.stderr)
        else:
            print(json.dumps(result, indent=2, allow_nan=False))

    def fail(code, message, exit_status):
        emit({"ok": False, "error": {"code": code, "message": message}}, error=True)
        return exit_status

    try:
        args = parser.parse_args(argv)
    except _UsageError as exc:
        return fail("usage_error", str(exc), 2)

    from . import workflow
    from .run_config import WorkflowError

    stage = None

    def progress(_fraction, message):
        nonlocal stage
        if message != stage:
            stage = message
            print(message, file=sys.stderr, flush=True)

    try:
        if args.command == "validate":
            result = workflow.validate(args.config)
        else:
            _ensure_runtime_dirs()
            import matplotlib

            matplotlib.use("Agg")
            if args.command == "run":
                result = workflow.run(args.config, args.output_dir, progress=progress)
            else:
                result = workflow.render(args.manifest, args.output_dir, progress=progress)
        emit({"ok": True, "command": args.command, **result})
        return 0
    except WorkflowError as exc:
        return fail(exc.code, str(exc), 2)
    except (OSError, ValueError, KeyError) as exc:
        return fail("execution_failed", str(exc), 1)
    except KeyboardInterrupt:
        return fail("interrupted", "Analysis interrupted.", 130)
