"""Validation rules for downloaded NASA FIRMS active-fire CSV files."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


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
NUMERIC_COLUMNS = (
    "latitude",
    "longitude",
    "brightness",
    "scan",
    "track",
    "bright_t31",
    "frp",
)


@dataclass(frozen=True)
class RejectedRow:
    """A source record omitted from the clean observations, with its reasons."""

    source_file: str
    source_line: int
    reasons: tuple[str, ...]
    values: dict[str, Any] = field(default_factory=dict)


def missing_required_columns(columns: list[str]) -> list[str]:
    """Return required header names not present in a CSV."""
    return sorted(REQUIRED_COLUMNS.difference(columns))


def validate_rows(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[RejectedRow]]:
    """Coerce FIRMS fields and split valid observations from rejected records.

    ``_source_file`` and ``_source_line`` are loader metadata and are preserved
    on accepted rows so a reviewer can trace a map point to its input record.
    """
    if raw.empty:
        return raw.copy(), []

    frame = raw.copy()
    reasons_by_row: list[list[str]] = [[] for _ in range(len(frame))]

    for column in REQUIRED_COLUMNS:
        values = frame[column].astype("string").str.strip()
        missing = values.isna() | values.eq("")
        for index in frame.index[missing]:
            reasons_by_row[frame.index.get_loc(index)].append(f"missing required value: {column}")
        frame[column] = values

    numeric = pd.DataFrame(index=frame.index)
    for column in NUMERIC_COLUMNS:
        numeric[column] = pd.to_numeric(frame[column], errors="coerce")
        invalid = numeric[column].isna()
        for index in frame.index[invalid]:
            reasons_by_row[frame.index.get_loc(index)].append(f"invalid numeric value: {column}")

    for column, lower, upper in (
        ("latitude", -90.0, 90.0),
        ("longitude", -180.0, 180.0),
    ):
        invalid = ~numeric[column].between(lower, upper)
        for index in frame.index[invalid.fillna(True)]:
            reasons_by_row[frame.index.get_loc(index)].append(f"out-of-range coordinate: {column}")

    dates = pd.to_datetime(frame["acq_date"], format="%Y-%m-%d", errors="coerce")
    for index in frame.index[dates.isna()]:
        reasons_by_row[frame.index.get_loc(index)].append("invalid acquisition date: acq_date")

    times = frame["acq_time"].astype("string").str.strip()
    valid_time = times.str.fullmatch(r"(?:[01]\d|2[0-3])[0-5]\d", na=False)
    for index in frame.index[~valid_time]:
        reasons_by_row[frame.index.get_loc(index)].append("invalid UTC HHMM time: acq_time")

    rejected: list[RejectedRow] = []
    accepted_positions: list[int] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        reasons = tuple(dict.fromkeys(reasons_by_row[position]))
        if reasons:
            rejected.append(
                RejectedRow(
                    source_file=str(row.get("_source_file", "")),
                    source_line=int(row.get("_source_line", 0)),
                    reasons=reasons,
                    values={key: row[key] for key in frame.columns if not key.startswith("_")},
                )
            )
        else:
            accepted_positions.append(position)

    clean = frame.iloc[accepted_positions].copy()
    if not clean.empty:
        for column in NUMERIC_COLUMNS:
            clean[column] = numeric.loc[clean.index, column].astype("float64")
        clean["acq_date"] = dates.loc[clean.index]
        # Keep HHMM as four-character text; integer conversion would drop zeros.
        clean["acq_time"] = times.loc[clean.index].astype("string")
    return clean.reset_index(drop=True), rejected

