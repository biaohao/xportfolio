"""
config.py — shared configuration loader.

All scripts import from here instead of opening config.yaml directly.
This ensures the path is resolved once relative to this file, so scripts
remain correct regardless of where they are invoked from.

Usage:
    from config import load_config, ROOT, CONFIG_PATH

    cfg = load_config()          # parsed dict, re-read each call
    tickers = list(cfg["assets"].keys())
"""

from pathlib import Path
import yaml

# Absolute path to the project root (the directory containing this file).
ROOT = Path(__file__).parent

# Absolute path to the config file — single source of truth.
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    """Read and parse config.yaml. Returns a plain dict."""
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)
