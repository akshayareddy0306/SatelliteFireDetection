"""Regional filtering, spatial-temporal incident clustering and ranking.

Default priority weights are FRP 0.50, confidence 0.10, recency 0.15 and
repeat activity 0.25. Confidence has intentionally low weight because FIRMS
confidence is a three-class algorithm category and this file is mostly nominal.
All ranking features are computed from observations in the selected region and
date range. ``bright_t31`` is retained unchanged and is not used in ranking.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date
from typing import Mapping

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088
DEFAULT_RADIUS_KM = 1.0
DEFAULT_MAX_GAP_HOURS = 72.0
DEFAULT_WEIGHTS = {
    "frp": 0.50,
    "confidence": 0.10,
    "recency": 0.15,
    "repeat": 0.25,
}
REGION_PRESETS: dict[str, tuple[float, float, float, float]] = {
    # Bounds are (min_latitude, min_longitude, max_latitude, max_longitude).
    "India": (6.0, 68.0, 37.0, 98.0),
    "World": (-90.0, -180.0, 90.0, 180.0),
}


@dataclass(frozen=True)
class BoundingBox:
    min_latitude: float
    min_longitude: float
    max_latitude: float
    max_longitude: float

    def __post_init__(self) -> None:
        if not (-90 <= self.min_latitude <= self.max_latitude <= 90):
            raise ValueError("Latitude bounds must be ordered within -90..90")
        if not (-180 <= self.min_longitude <= self.max_longitude <= 180):
            raise ValueError("Longitude bounds must be ordered within -180..180")


def region_bounds(
    region: str | BoundingBox = "India",
    custom_bounds: BoundingBox | None = None,
) -> BoundingBox:
    """Resolve an India/World preset or caller-supplied custom bounding box."""
    if isinstance(region, BoundingBox):
        return region
    if region.lower() == "custom":
        if custom_bounds is None:
            raise ValueError("custom_bounds is required when region='Custom'")
        return custom_bounds
    try:
        return BoundingBox(*REGION_PRESETS[region.title()])
    except KeyError as error:
        raise ValueError("region must be 'India', 'World', or 'Custom'") from error


def filter_observations(
    data: pd.DataFrame,
    region: str | BoundingBox = "India",
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    custom_bounds: BoundingBox | None = None,
) -> pd.DataFrame:
    """Select region/date rows before any clustering or ranking is performed."""
    bounds = region_bounds(region, custom_bounds)
    if data.empty:
        result = data.copy()
        result["observation_time_utc"] = pd.Series(dtype="datetime64[ns]")
        return result

    frame = data.copy()
    required = {"latitude", "longitude", "acq_date", "acq_time"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing fields needed for regional analysis: {', '.join(missing)}")

    dates = pd.to_datetime(frame["acq_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    times = frame["acq_time"].astype("string").str.strip().str.zfill(4)
    frame["observation_time_utc"] = pd.to_datetime(
        dates + " " + times,
        format="%Y-%m-%d %H%M",
        errors="coerce",
        utc=True,
    )
    mask = (
        frame["latitude"].between(bounds.min_latitude, bounds.max_latitude)
        & frame["longitude"].between(bounds.min_longitude, bounds.max_longitude)
        & frame["observation_time_utc"].notna()
    )
    if start_date is not None:
        mask &= frame["observation_time_utc"].dt.date >= pd.Timestamp(start_date).date()
    if end_date is not None:
        mask &= frame["observation_time_utc"].dt.date <= pd.Timestamp(end_date).date()
    return frame.loc[mask].sort_values(
        ["observation_time_utc", "latitude", "longitude"], kind="stable"
    ).reset_index(drop=True)


_CLUSTER_CACHE: OrderedDict[tuple[str, float, float], tuple[str, ...]] = OrderedDict()
_CLUSTER_CACHE_SIZE = 8


def clear_cluster_cache() -> None:
    """Clear cached cluster assignments, for example after a data refresh."""
    _CLUSTER_CACHE.clear()


def _cluster_signature(frame: pd.DataFrame) -> str:
    columns = ["latitude", "longitude", "observation_time_utc"]
    hashes = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy().tobytes()
    return hashlib.sha256(hashes).hexdigest()


def _stable_incident_id(group: pd.DataFrame) -> str:
    centroid_lat = float(group["latitude"].mean())
    centroid_lon = float(group["longitude"].mean())
    first_date = group["observation_time_utc"].min().strftime("%Y%m%d")
    identity = f"{first_date}|{centroid_lat:.3f}|{centroid_lon:.3f}"
    digest = hashlib.blake2s(identity.encode("utf-8"), digest_size=3).hexdigest()
    return f"FD-{first_date}-{digest}"


def _build_incident_ids(
    frame: pd.DataFrame,
    radius_km: float,
    max_gap_hours: float,
) -> tuple[str, ...]:
    if frame.empty:
        return ()
    coordinates_degrees = frame[["latitude", "longitude"]].to_numpy(dtype="float64")
    coordinates = np.radians(coordinates_degrees)
    radius_radians = radius_km / EARTH_RADIUS_KM
    try:
        from sklearn.neighbors import BallTree

        neighbors = BallTree(coordinates, metric="haversine", leaf_size=40).query_radius(
            coordinates, r=radius_radians, return_distance=False
        )
        use_grid_candidates = False
    except ImportError:
        # Grid fallback keeps the analysis runnable in a minimal environment;
        # exact haversine distance is checked below before joining each edge.
        grid_km = 0.5
        cell_range = int(np.ceil(radius_km / grid_km)) + 1
        circumference_km = 2.0 * np.pi * EARTH_RADIUS_KM
        x_cell_count = int(np.ceil(circumference_km / grid_km))
        x_cells = (
            np.floor((coordinates[:, 1] + np.pi) * EARTH_RADIUS_KM / grid_km).astype(np.int64)
            % x_cell_count
        )
        y_cells = np.floor(coordinates[:, 0] * EARTH_RADIUS_KM / grid_km).astype(np.int64)
        buckets: dict[tuple[int, int], list[int]] = {}
        for index, cell in enumerate(zip(x_cells, y_cells)):
            buckets.setdefault((int(cell[0]), int(cell[1])), []).append(index)

        def grid_candidates():
            for index, (x_cell, y_cell) in enumerate(zip(x_cells, y_cells)):
                found: set[int] = set()
                for dx in range(-cell_range, cell_range + 1):
                    wrapped_x = (int(x_cell) + dx) % x_cell_count
                    for dy in range(-cell_range, cell_range + 1):
                        found.update(buckets.get((wrapped_x, int(y_cell) + dy), ()))
                yield np.fromiter(found, dtype=np.int64)

        neighbors = grid_candidates()
        use_grid_candidates = True

    times_ns = frame["observation_time_utc"].astype("int64").to_numpy()
    max_gap_ns = int(max_gap_hours * 60 * 60 * 1_000_000_000)
    parent = np.arange(len(frame), dtype=np.int64)
    size = np.ones(len(frame), dtype=np.int32)

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = int(parent[root])
        while parent[value] != value:
            old_parent = int(parent[value])
            parent[value] = root
            value = old_parent
        return root

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        size[root_left] += size[root_right]

    # BallTree restricts comparisons to spatial neighbors. The timestamp check
    # makes an edge valid only when both the 1 km and 72 hour conditions hold.
    for left, candidates in enumerate(neighbors):
        candidates = candidates[candidates > left]
        if not len(candidates):
            continue
        temporally_close = candidates[np.abs(times_ns[candidates] - times_ns[left]) <= max_gap_ns]
        if use_grid_candidates and len(temporally_close):
            lat1 = coordinates[left, 0]
            lat2 = coordinates[temporally_close, 0]
            delta_lat = lat2 - lat1
            delta_lon = (coordinates[temporally_close, 1] - coordinates[left, 1] + np.pi) % (
                2 * np.pi
            ) - np.pi
            haversine_a = (
                np.sin(delta_lat / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
            )
            distances = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine_a, 0, 1)))
            temporally_close = temporally_close[distances <= radius_km]
        for right in temporally_close:
            union(left, int(right))

    components: dict[int, list[int]] = {}
    for index in range(len(frame)):
        components.setdefault(find(index), []).append(index)

    incident_by_index = [""] * len(frame)
    for indices in components.values():
        group = frame.iloc[indices]
        incident_id = _stable_incident_id(group)
        for index in indices:
            incident_by_index[index] = incident_id
    return tuple(incident_by_index)


def cluster_incidents(
    observations: pd.DataFrame,
    radius_km: float = DEFAULT_RADIUS_KM,
    max_gap_hours: float = DEFAULT_MAX_GAP_HOURS,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Attach stable incident IDs using a cached haversine BallTree clustering.

    Each graph edge joins points no farther than ``radius_km`` apart and no
    more than ``max_gap_hours`` apart. Connected edges form an incident.
    """
    if radius_km <= 0 or max_gap_hours <= 0:
        raise ValueError("radius_km and max_gap_hours must be positive")
    frame = observations.copy().reset_index(drop=True)
    if frame.empty:
        frame["incident_id"] = pd.Series(dtype="string")
        return frame
    if "observation_time_utc" not in frame:
        frame = filter_observations(frame, "World")

    signature = _cluster_signature(frame)
    cache_key = (signature, float(radius_km), float(max_gap_hours))
    incident_ids = _CLUSTER_CACHE.get(cache_key) if use_cache else None
    if incident_ids is None:
        incident_ids = _build_incident_ids(frame, radius_km, max_gap_hours)
        if use_cache:
            _CLUSTER_CACHE[cache_key] = incident_ids
            _CLUSTER_CACHE.move_to_end(cache_key)
            while len(_CLUSTER_CACHE) > _CLUSTER_CACHE_SIZE:
                _CLUSTER_CACHE.popitem(last=False)
    frame["incident_id"] = incident_ids
    return frame


def _confidence_value(value: object) -> float:
    return {"l": 0.25, "n": 0.60, "h": 1.0}.get(str(value).strip().lower(), 0.0)


def summarize_incidents(clustered: pd.DataFrame) -> pd.DataFrame:
    """Aggregate clustered points while retaining brightness fields as raw data."""
    if clustered.empty:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    has_type = "type" in clustered.columns
    for incident_id, group in clustered.groupby("incident_id", sort=False):
        times = group["observation_time_utc"]
        record: dict[str, object] = {
            "incident_id": incident_id,
            "latitude": float(group["latitude"].mean()),
            "longitude": float(group["longitude"].mean()),
            "first_observation_utc": times.min(),
            "last_observation_utc": times.max(),
            "observation_count": int(len(group)),
            "distinct_observation_days": int(times.dt.date.nunique()),
            "max_frp": float(pd.to_numeric(group["frp"], errors="coerce").max())
            if "frp" in group
            else 0.0,
            "confidence_categories": tuple(
                sorted(group["confidence"].dropna().astype(str).str.lower().unique())
            )
            if "confidence" in group
            else (),
            # These are retained as display-only aggregates; neither affects rank.
            "brightness_temperature": float(pd.to_numeric(group["brightness"], errors="coerce").mean())
            if "brightness" in group
            else None,
            "bright_t31_raw": tuple(group["bright_t31"].tolist())
            if "bright_t31" in group
            else (),
            "type_values": tuple(group["type"].dropna().astype(str).unique()) if has_type else None,
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


def _normalize_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    values = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if set(values) != set(DEFAULT_WEIGHTS):
        raise ValueError(f"weights must contain exactly: {', '.join(DEFAULT_WEIGHTS)}")
    if any(value < 0 for value in values.values()) or sum(values.values()) <= 0:
        raise ValueError("weights must be non-negative with a positive total")
    total = float(sum(values.values()))
    return {name: float(value) / total for name, value in values.items()}


def rank_incidents(
    clustered: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Rank incidents 0..100 with documented, region-relative factor scores."""
    incidents = summarize_incidents(clustered)
    if incidents.empty:
        return incidents
    normalized_weights = _normalize_weights(weights)
    selected_frp = pd.to_numeric(clustered.get("frp", 0), errors="coerce")
    p95_frp = float(selected_frp.quantile(0.95)) if len(selected_frp) else 0.0
    if not np.isfinite(p95_frp) or p95_frp <= 0:
        incidents["frp_score"] = 0.0
    else:
        incidents["frp_score"] = incidents["max_frp"].clip(lower=0).div(p95_frp).clip(upper=1.0)

    if "confidence" in clustered:
        confidence_scores = clustered["confidence"].map(_confidence_value)
        means = confidence_scores.groupby(clustered["incident_id"]).mean()
        incidents["confidence_score"] = incidents["incident_id"].map(means).fillna(0.0)
    else:
        incidents["confidence_score"] = 0.0

    newest = clustered["observation_time_utc"].max()
    oldest = clustered["observation_time_utc"].min()
    span_seconds = max((newest - oldest).total_seconds(), 1.0)
    age_seconds = (newest - incidents["last_observation_utc"]).dt.total_seconds()
    incidents["recency_score"] = (1.0 - age_seconds / span_seconds).clip(0.0, 1.0)

    max_count = int(incidents["observation_count"].max())
    max_days = int(incidents["distinct_observation_days"].max())
    count_scores = (
        np.log1p(incidents["observation_count"] - 1) / np.log1p(max_count - 1)
        if max_count > 1
        else pd.Series(0.0, index=incidents.index)
    )
    day_scores = (
        np.log1p(incidents["distinct_observation_days"] - 1) / np.log1p(max_days - 1)
        if max_days > 1
        else pd.Series(0.0, index=incidents.index)
    )
    incidents["repeat_score"] = (count_scores + day_scores) / 2.0

    factor_columns = {
        "frp": "frp_score",
        "confidence": "confidence_score",
        "recency": "recency_score",
        "repeat": "repeat_score",
    }
    contribution_columns: list[str] = []
    for factor, column in factor_columns.items():
        contribution = f"{factor}_contribution"
        incidents[contribution] = incidents[column] * normalized_weights[factor] * 100.0
        contribution_columns.append(contribution)
    incidents["priority_score"] = incidents[contribution_columns].sum(axis=1)
    incidents["factor_contributions"] = incidents.apply(
        lambda row: {
            factor: round(float(row[f"{factor}_contribution"]), 2)
            for factor in factor_columns
        },
        axis=1,
    )
    incidents["priority_explanation"] = incidents["factor_contributions"].map(
        lambda values: "; ".join(
            f"{factor} +{points:.2f} points" for factor, points in values.items()
        )
    )
    incidents.attrs["weights"] = normalized_weights
    incidents.attrs["frp_p95_region"] = p95_frp
    return incidents.sort_values(
        ["priority_score", "max_frp", "last_observation_utc", "incident_id"],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def analyze_region(
    data: pd.DataFrame,
    region: str | BoundingBox = "India",
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    custom_bounds: BoundingBox | None = None,
    weights: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter first, then cluster and rank only the selected observations."""
    selected = filter_observations(data, region, start_date, end_date, custom_bounds)
    clustered = cluster_incidents(selected)
    incidents = rank_incidents(clustered, weights)
    return clustered, incidents
