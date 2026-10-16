"""Load FIRMS CSVs and print the observed data and validation summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from satellite_fire.data import load_firms_data  # noqa: E402
from satellite_fire.data.loader import DEFAULT_DATA_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "location",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="FIRMS CSV or directory containing fire_nrt_*/fire_archive_* CSVs",
    )
    parser.add_argument(
        "--show-rejections",
        action="store_true",
        help="Print rejected row locations and reasons",
    )
    args = parser.parse_args()

    loaded = load_firms_data(args.location)
    print(json.dumps(loaded.summary, indent=2))
    if args.show_rejections:
        for rejected in loaded.rejected_rows:
            print(
                f"{rejected.source_file}:{rejected.source_line}: "
                + "; ".join(rejected.reasons)
            )


if __name__ == "__main__":
    main()

