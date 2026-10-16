"""Load and run basic quality checks on a NASA FIRMS VIIRS CSV.

Usage:
    python scripts/check_firms_data.py
    python scripts/check_firms_data.py path/to/fire_nrt_SV-C2_824593.csv

Requires pandas. This script inspects data only; it does not modify the CSV or
train a model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_CSV = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "DL_FIRE_SV-C2_824593"
    / "fire_nrt_SV-C2_824593.csv"
)

REQUIRED_COLUMNS = {
    "latitude",
    "longitude",
    "brightness",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "bright_t31",
    "frp",
    "daynight",
}
NUMERIC_COLUMNS = [
    "latitude",
    "longitude",
    "brightness",
    "scan",
    "track",
    "bright_t31",
    "frp",
]


def load_firms_csv(csv_path: Path) -> pd.DataFrame:
    """Read a FIRMS CSV while preserving acquisition time's leading zeros."""
    return pd.read_csv(csv_path, dtype={"acq_time": "string"})


def check_quality(data: pd.DataFrame) -> dict[str, object]:
    """Return schema, missingness, ranges, and common integrity checks."""
    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    missing_values = data.isna().sum().astype(int).to_dict()
    blank_strings = {
        column: int(data[column].astype("string").str.strip().eq("").sum())
        for column in data.select_dtypes(include=["object", "string"]).columns
    }

    result: dict[str, object] = {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "column_names": list(data.columns),
        "dtypes": {name: str(dtype) for name, dtype in data.dtypes.items()},
        "missing_columns": missing_columns,
        "missing_values": missing_values,
        "blank_strings": blank_strings,
        "duplicate_rows": int(data.duplicated().sum()),
    }

    if missing_columns:
        result["checks"] = {"schema_ok": False}
        return result

    numeric = data[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    numeric_ranges = {
        column: {
            "min": float(numeric[column].min()),
            "max": float(numeric[column].max()),
        }
        for column in NUMERIC_COLUMNS
    }
    dates = pd.to_datetime(data["acq_date"], format="%Y-%m-%d", errors="coerce")
    times = data["acq_time"].astype("string").str.strip()
    valid_time = times.str.fullmatch(r"(?:[01]\d|2[0-3])[0-5]\d", na=False)

    result["numeric_ranges"] = numeric_ranges
    result["category_counts"] = {
        column: {str(key): int(value) for key, value in data[column].value_counts(dropna=False).items()}
        for column in ["satellite", "instrument", "confidence", "version", "daynight"]
    }
    result["checks"] = {
        "schema_ok": True,
        "invalid_latitude": int((~numeric["latitude"].between(-90, 90)).sum()),
        "invalid_longitude": int((~numeric["longitude"].between(-180, 180)).sum()),
        "invalid_numeric_values": int(numeric.isna().sum().sum()),
        "invalid_dates": int(dates.isna().sum()),
        "invalid_acq_time_hhmm": int((~valid_time).sum()),
        "negative_scan_track_frp": int(
            (numeric[["scan", "track", "frp"]] < 0).sum().sum()
        ),
        "date_range": [str(dates.min().date()), str(dates.max().date())]
        if dates.notna().any()
        else None,
        "confidence_values": sorted(data["confidence"].dropna().astype(str).unique().tolist()),
        "daynight_values": sorted(data["daynight"].dropna().astype(str).unique().tolist()),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=DEFAULT_CSV,
        help=f"FIRMS CSV to inspect (default: {DEFAULT_CSV})",
    )
    args = parser.parse_args()

    if not args.csv_path.is_file():
        parser.error(f"CSV file does not exist: {args.csv_path}")

    report = check_quality(load_firms_csv(args.csv_path))
    print(f"FIRMS data quality report: {args.csv_path}")
    print(f"Rows: {report['rows']:,} | Columns: {report['columns']}")
    print("\nColumn types:")
    for name, dtype in report["dtypes"].items():
        print(f"  {name}: {dtype}")
    print("\nMissing values:")
    for name, count in report["missing_values"].items():
        print(f"  {name}: {count}")
    print("\nChecks and summaries:")
    for name, value in report["checks"].items():
        print(f"  {name}: {value}")
    if "numeric_ranges" in report:
        print("\nNumeric ranges:")
        for name, bounds in report["numeric_ranges"].items():
            print(f"  {name}: {bounds['min']} .. {bounds['max']}")
        print("\nCategory counts:")
        for name, counts in report["category_counts"].items():
            print(f"  {name}: {counts}")


if __name__ == "__main__":
    main()
