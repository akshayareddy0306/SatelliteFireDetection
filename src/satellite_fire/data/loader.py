"""Safe, read-only loading and merging of local FIRMS VIIRS CSV downloads."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .validation import RejectedRow, missing_required_columns, validate_rows


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
FIRMS_FILE_PREFIXES = ("fire_nrt_", "fire_archive_")
DEDUPLICATION_KEY = ("latitude", "longitude", "acq_date", "acq_time")


@dataclass
class LoadResult:
    """Clean observations plus a report of skipped or rejected input records."""

    data: pd.DataFrame
    files: list[Path]
    rejected_rows: list[RejectedRow] = field(default_factory=list)
    schema_errors: dict[str, list[str]] = field(default_factory=dict)
    duplicate_rows_removed: int = 0

    @property
    def summary(self) -> dict[str, object]:
        """Return a serializable summary of the loaded data and validation."""
        if self.data.empty:
            return {
                "files": [str(path) for path in self.files],
                "rows": 0,
                "rejected_rows": len(self.rejected_rows),
                "schema_errors": self.schema_errors,
                "duplicate_rows_removed": self.duplicate_rows_removed,
                "columns": [],
                "satellites": [],
                "date_range": None,
                "bounding_box": None,
            }
        dates = pd.to_datetime(self.data["acq_date"], errors="coerce")
        return {
            "files": [str(path) for path in self.files],
            "rows": int(len(self.data)),
            "rejected_rows": len(self.rejected_rows),
            "schema_errors": self.schema_errors,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "columns": [column for column in self.data.columns if not column.startswith("_")],
            "satellites": sorted(self.data["satellite"].dropna().astype(str).unique().tolist()),
            "date_range": [str(dates.min().date()), str(dates.max().date())],
            "bounding_box": {
                "min_latitude": float(self.data["latitude"].min()),
                "min_longitude": float(self.data["longitude"].min()),
                "max_latitude": float(self.data["latitude"].max()),
                "max_longitude": float(self.data["longitude"].max()),
            },
        }


def discover_firms_files(location: Path) -> list[Path]:
    """Find supported FIRMS data files without mistaking auxiliary CSVs for them."""
    location = Path(location)
    if location.is_file():
        return [location]
    if not location.is_dir():
        return []
    return sorted(
        path
        for path in location.rglob("*.csv")
        if path.name.lower().startswith(FIRMS_FILE_PREFIXES)
    )


def _read_file(path: Path) -> tuple[list[str], list[dict[str, object]], list[RejectedRow]]:
    """Parse records as strings, isolating malformed CSV rows instead of failing."""
    rows: list[dict[str, object]] = []
    rejected: list[RejectedRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, strict=True)
        try:
            header = next(reader)
        except StopIteration:
            return [], rows, rejected
        except csv.Error as error:
            rejected.append(RejectedRow(str(path), reader.line_num or 1, (f"malformed header: {error}",)))
            return [], rows, rejected

        header = [name.strip().lower() for name in header]
        if len(set(header)) != len(header):
            rejected.append(RejectedRow(str(path), 1, ("duplicate column names in header",)))
            return header, rows, rejected

        previous_line = reader.line_num
        while True:
            start_line = previous_line + 1
            try:
                values = next(reader)
                previous_line = reader.line_num
            except StopIteration:
                break
            except csv.Error as error:
                previous_line = reader.line_num
                rejected.append(
                    RejectedRow(str(path), start_line, (f"malformed CSV record: {error}",))
                )
                continue

            if len(values) != len(header):
                rejected.append(
                    RejectedRow(
                        str(path),
                        start_line,
                        (f"wrong field count: expected {len(header)}, got {len(values)}",),
                    )
                )
                continue
            record = dict(zip(header, values))
            record["_source_file"] = path.name
            record["_source_line"] = start_line
            rows.append(record)
    return header, rows, rejected


def load_firms_data(location: Path = DEFAULT_DATA_DIR) -> LoadResult:
    """Load, validate and merge FIRMS CSVs from a file or data directory.

    The source files are only read. Duplicate observations are defined as the
    same latitude, longitude, acquisition date and HHMM time, regardless of
    source filename, and the first valid record is kept.
    """
    files = discover_firms_files(Path(location))
    result = LoadResult(data=pd.DataFrame(), files=files)
    accepted_frames: list[pd.DataFrame] = []

    for path in files:
        header, records, parse_rejections = _read_file(path)
        result.rejected_rows.extend(parse_rejections)
        absent = missing_required_columns(header)
        if absent:
            result.schema_errors[str(path)] = absent
            continue
        raw = pd.DataFrame.from_records(records, columns=header + ["_source_file", "_source_line"])
        clean, row_rejections = validate_rows(raw)
        accepted_frames.append(clean)
        result.rejected_rows.extend(row_rejections)

    if accepted_frames:
        combined = pd.concat(accepted_frames, ignore_index=True)
        duplicate_mask = combined.duplicated(subset=list(DEDUPLICATION_KEY), keep="first")
        result.duplicate_rows_removed = int(duplicate_mask.sum())
        result.data = combined.loc[~duplicate_mask].reset_index(drop=True)
    return result

