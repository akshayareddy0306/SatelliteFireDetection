"""Cautious possible non-wildfire-source flags for incident triage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088
NEAR_VOLCANO_KM = 10.0
PERSISTENT_DAYS = 3


def _load_land_geometry(data_dir: Path) -> tuple[Any | None, str]:
    """Load a Natural Earth GeoJSON or shapefile when present and supported."""
    geojson_files = sorted(
        path
        for path in data_dir.rglob("*.geojson")
        if "land" in path.stem.lower() or "naturalearth" in path.stem.lower()
    ) + sorted(
        path
        for path in data_dir.rglob("*.json")
        if "land" in path.stem.lower() or "naturalearth" in path.stem.lower()
    )
    shapefile_files = sorted(path for path in data_dir.rglob("*.shp") if "land" in path.stem.lower())
    if not geojson_files and not shapefile_files:
        return None, "Natural Earth land boundary file missing"
    try:
        from shapely.geometry import shape
        from shapely.ops import unary_union

        geometries = []
        if geojson_files:
            for path in geojson_files:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("type") == "FeatureCollection":
                    geometries.extend(shape(feature["geometry"]) for feature in payload["features"])
                elif payload.get("type") == "Feature":
                    geometries.append(shape(payload["geometry"]))
                else:
                    geometries.append(shape(payload))
        else:
            import shapefile

            for path in shapefile_files:
                reader = shapefile.Reader(str(path))
                geometries.extend(shape(item.__geo_interface__) for item in reader.shapes())
        if not geometries:
            return None, "Natural Earth land boundary contains no geometry"
        return unary_union(geometries), ""
    except (ImportError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        return None, f"Natural Earth boundary unavailable: {error}"


def _load_volcanoes(path: Path) -> tuple[pd.DataFrame | None, str]:
    if not path.is_file():
        return None, "data/volcanoes.csv missing"
    try:
        volcanoes = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeError) as error:
        return None, f"volcano list unavailable: {error}"
    normalized = {str(column).strip().lower(): column for column in volcanoes.columns}
    lat_name = next((normalized[name] for name in ("latitude", "lat", "latitude_dd") if name in normalized), None)
    lon_name = next((normalized[name] for name in ("longitude", "lon", "lng", "longitude_dd") if name in normalized), None)
    name_name = next((normalized[name] for name in ("name", "volcano_name", "volcano") if name in normalized), None)
    if lat_name is None or lon_name is None:
        return None, "volcano list unavailable: expected latitude and longitude columns"
    volcanoes = volcanoes.copy()
    volcanoes["_latitude"] = pd.to_numeric(volcanoes[lat_name], errors="coerce")
    volcanoes["_longitude"] = pd.to_numeric(volcanoes[lon_name], errors="coerce")
    volcanoes = volcanoes.dropna(subset=["_latitude", "_longitude"])
    volcanoes = volcanoes[volcanoes["_latitude"].between(-90, 90) & volcanoes["_longitude"].between(-180, 180)]
    if volcanoes.empty:
        return None, "volcano list unavailable: no valid coordinates"
    volcanoes["_name"] = volcanoes[name_name].astype(str) if name_name else "known volcano"
    return volcanoes.reset_index(drop=True), ""


def add_rule_flags(
    incidents: pd.DataFrame,
    clustered_observations: pd.DataFrame,
    data_dir: Path,
) -> pd.DataFrame:
    """Attach flags and per-source availability notes without asserting cause."""
    result = incidents.copy()
    if result.empty:
        result["flags"] = pd.Series(dtype="object")
        result["possible_non_wildfire_source"] = pd.Series(dtype="bool")
        result["flag_availability"] = pd.Series(dtype="object")
        return result

    land, land_error = _load_land_geometry(Path(data_dir))
    volcanoes, volcano_error = _load_volcanoes(Path(data_dir) / "volcanoes.csv")
    availability: dict[str, str] = {
        "firms_type": "available" if "type" in clustered_observations.columns else "not available",
        "offshore": "available" if land is not None else "not available",
        "near_volcano": "available" if volcanoes is not None else "not available",
        "persistent_location": "available",
    }
    messages_by_id: dict[str, list[str]] = {str(value): [] for value in result["incident_id"]}
    for row in result.itertuples(index=False):
        messages = messages_by_id[str(row.incident_id)]
        types = getattr(row, "type_values", None)
        if "type" not in clustered_observations.columns:
            messages.append("FIRMS type: not available")
        elif not types:
            messages.append("FIRMS type: not available for this incident")
        else:
            for raw_type in types:
                try:
                    type_code = int(float(raw_type))
                except (TypeError, ValueError):
                    messages.append(f"FIRMS type {raw_type}: possible non-wildfire source (unrecognized type)")
                    continue
                labels = {
                    0: "presumed vegetation fire",
                    1: "active volcano",
                    2: "other static land source",
                    3: "offshore detection",
                }
                if type_code != 0:
                    messages.append(
                        f"FIRMS type {type_code} ({labels.get(type_code, 'unrecognized type')}): "
                        "possible non-wildfire source"
                    )

        if land is None:
            messages.append(f"Offshore check: not available ({land_error})")
        else:
            from shapely.geometry import Point

            if not land.covers(Point(float(row.longitude), float(row.latitude))):
                messages.append("Offshore location: possible non-wildfire source; verify against a coastline")

        if volcanoes is None:
            messages.append(f"Near-volcano check: not available ({volcano_error})")

        if int(row.distinct_observation_days) >= PERSISTENT_DAYS:
            messages.append(
                f"Persistent location ({int(row.distinct_observation_days)} observation days): "
                "possible non-wildfire source"
            )
    if volcanoes is not None:
        grid_km = NEAR_VOLCANO_KM
        volcano_lat = np.radians(volcanoes["_latitude"].to_numpy(dtype="float64"))
        volcano_lon = np.radians(volcanoes["_longitude"].to_numpy(dtype="float64"))
        volcano_x = np.floor((volcano_lon + np.pi) * EARTH_RADIUS_KM / grid_km).astype(int)
        volcano_y = np.floor(volcano_lat * EARTH_RADIUS_KM / grid_km).astype(int)
        volcano_grid: dict[tuple[int, int], list[int]] = {}
        for index, cell in enumerate(zip(volcano_x, volcano_y)):
            volcano_grid.setdefault((int(cell[0]), int(cell[1])), []).append(index)
        incident_lat = np.radians(result["latitude"].to_numpy(dtype="float64"))
        incident_lon = np.radians(result["longitude"].to_numpy(dtype="float64"))
        for position, (lat, lon) in enumerate(zip(incident_lat, incident_lon)):
            x_cell = int(np.floor((lon + np.pi) * EARTH_RADIUS_KM / grid_km))
            y_cell = int(np.floor(lat * EARTH_RADIUS_KM / grid_km))
            candidate_indexes: set[int] = set()
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    candidate_indexes.update(volcano_grid.get((x_cell + dx, y_cell + dy), ()))
            if not candidate_indexes:
                continue
            indexes = np.fromiter(candidate_indexes, dtype=int)
            delta_lat = volcano_lat[indexes] - lat
            delta_lon = (volcano_lon[indexes] - lon + np.pi) % (2 * np.pi) - np.pi
            haversine_a = (
                np.sin(delta_lat / 2) ** 2
                + np.cos(lat) * np.cos(volcano_lat[indexes]) * np.sin(delta_lon / 2) ** 2
            )
            distances_km = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine_a, 0, 1)))
            nearest_offset = int(np.argmin(distances_km))
            distance_km = float(distances_km[nearest_offset])
            if distance_km <= NEAR_VOLCANO_KM:
                volcano_name = volcanoes.iloc[int(indexes[nearest_offset])]["_name"]
                incident_id = str(result.iloc[position]["incident_id"])
                messages_by_id[incident_id].append(
                    f"Near known volcano {volcano_name} ({distance_km:.1f} km): possible non-wildfire source"
                )

    result["flags"] = result["incident_id"].map(lambda value: tuple(messages_by_id[str(value)]))
    result["possible_non_wildfire_source"] = result["flags"].map(
        lambda messages: any("possible non-wildfire source" in message.lower() for message in messages)
    )
    result["flag_availability"] = [dict(availability) for _ in range(len(result))]
    return result
