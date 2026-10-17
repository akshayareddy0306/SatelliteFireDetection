"""Run regional clustering, ranking and rule flags for a local FIRMS dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from satellite_fire.analysis import (  # noqa: E402
    BoundingBox,
    cluster_incidents,
    filter_observations,
    rank_incidents,
    region_bounds,
)
from satellite_fire.data import load_firms_data  # noqa: E402
from satellite_fire.data.loader import DEFAULT_DATA_DIR  # noqa: E402
from satellite_fire.flags import add_rule_flags  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR, help="FIRMS CSV or data directory")
    parser.add_argument("--region", choices=["India", "World", "Custom"], default="India")
    parser.add_argument("--start-date", help="Inclusive UTC date, YYYY-MM-DD")
    parser.add_argument("--end-date", help="Inclusive UTC date, YYYY-MM-DD")
    parser.add_argument("--min-lat", type=float)
    parser.add_argument("--min-lon", type=float)
    parser.add_argument("--max-lat", type=float)
    parser.add_argument("--max-lon", type=float)
    args = parser.parse_args()

    custom_bounds = None
    if args.region == "Custom":
        bounds = (args.min_lat, args.min_lon, args.max_lat, args.max_lon)
        if any(value is None for value in bounds):
            parser.error("--region Custom requires --min-lat, --min-lon, --max-lat and --max-lon")
        custom_bounds = BoundingBox(*bounds)

    started = time.perf_counter()
    loaded = load_firms_data(args.data)
    loaded_at = time.perf_counter()
    selected = filter_observations(
        loaded.data, args.region, args.start_date, args.end_date, custom_bounds
    )
    filtered_at = time.perf_counter()
    clustered = cluster_incidents(selected)
    clustered_at = time.perf_counter()
    incidents = rank_incidents(clustered)
    ranked_at = time.perf_counter()
    incidents = add_rule_flags(incidents, clustered, args.data if args.data.is_dir() else args.data.parent)
    completed_at = time.perf_counter()

    top = incidents.head(5)
    top_records = []
    for row in top.to_dict(orient="records"):
        top_records.append(
            {
                "incident_id": row["incident_id"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "first_observation_utc": str(row["first_observation_utc"]),
                "last_observation_utc": str(row["last_observation_utc"]),
                "observation_count": row["observation_count"],
                "distinct_observation_days": row["distinct_observation_days"],
                "confidence_categories": list(row["confidence_categories"]),
                "max_frp": row["max_frp"],
                "priority_score": round(float(row["priority_score"]), 2),
                "factor_contributions": row["factor_contributions"],
                "flags": list(row["flags"]),
            }
        )
    selected_dates = selected["observation_time_utc"].dt.date if len(selected) else []
    output = {
        "region": args.region,
        "region_bounds": vars(region_bounds(args.region, custom_bounds)),
        "date_range": {
            "start": args.start_date or (str(min(selected_dates)) if len(selected) else None),
            "end": args.end_date or (str(max(selected_dates)) if len(selected) else None),
        },
        "source_rows": len(loaded.data),
        "selected_observations": len(selected),
        "incident_count": len(incidents),
        "ranking_weights": incidents.attrs.get("weights", {}),
        "frp_95th_percentile_in_region": incidents.attrs.get("frp_p95_region"),
        "clustering_seconds": round(clustered_at - filtered_at, 3),
        "filter_seconds": round(filtered_at - loaded_at, 3),
        "ranking_seconds": round(ranked_at - clustered_at, 3),
        "flags_seconds": round(completed_at - ranked_at, 3),
        "total_seconds_including_load": round(completed_at - started, 3),
        "available_rule_checks": incidents["flag_availability"].iloc[0] if len(incidents) else {},
        "top_5": top_records,
    }
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
