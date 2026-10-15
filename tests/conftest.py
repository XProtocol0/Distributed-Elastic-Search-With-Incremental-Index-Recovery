"""Shared pytest setup: make sure gRPC stubs exist before any test imports them."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUB = ROOT / "src" / "dsearch" / "proto" / "search_pb2.py"


def pytest_configure(config: object) -> None:
    if not STUB.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_proto.py")], check=True)
