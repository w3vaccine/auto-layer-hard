#!/usr/bin/env python3
"""Root entrypoint — delegates to src/run_auto_layer.py."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "src" / "run_auto_layer.py"), run_name="__main__")
