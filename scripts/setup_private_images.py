#!/usr/bin/env python3
"""Copy full private-test JPEGs into data/private_test/images/.

Usage:
    python scripts/setup_private_images.py /path/to/private_test
    python scripts/setup_private_images.py ../private_test   # from hackathon monorepo

Or set env and run without args:
    export SMCE_PRIVATE_TEST_DIR=/path/to/private_test
    python scripts/setup_private_images.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smce_baseline_core import setup_full_private_images  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Install full private_test images locally")
    parser.add_argument(
        "source",
        nargs="?",
        default=os.environ.get("SMCE_PRIVATE_TEST_DIR", ""),
        help="Folder containing images/ (e.g. private_test bundle from BTC)",
    )
    args = parser.parse_args()
    if not args.source:
        parser.error("Provide source path or set SMCE_PRIVATE_TEST_DIR")

    source = Path(args.source).expanduser().resolve()
    n = setup_full_private_images(source)
    dest = ROOT / "data" / "private_test" / "images"
    print(f"Copied {n:,} jpg -> {dest}")


if __name__ == "__main__":
    main()
