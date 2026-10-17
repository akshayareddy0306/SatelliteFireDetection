"""Tests for region filtering, clustering, stable IDs and ranking factors."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satellite_fire.analysis import (  # noqa: E402
    BoundingBox,
    cluster_incidents,
    filter_observations,
    rank_incidents,
)
from satellite_fire.flags import add_rule_flags  # noqa: E402


def observations(rows: list[tuple[float, float, str, str, float, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "latitude": lat,
                "longitude": lon,
                "acq_date": day,
                "acq_time": hhmm,
                "frp": frp,
                "confidence": confidence,
                "brightness": 330.0,
                "bright_t31": 290.0,
            }
            for lat, lon, day, hhmm, frp, confidence in rows
        ]
    )


class IncidentAnalysisTests(unittest.TestCase):
    def test_nearby_points_cluster_and_far_point_does_not(self) -> None:
        frame = observations(
            [
                (20.0000, 78.0000, "2026-10-01", "0100", 10, "n"),
                (20.0010, 78.0000, "2026-10-02", "0100", 12, "h"),
                (20.0200, 78.0000, "2026-10-02", "0100", 20, "n"),
            ]
        )
        selected = filter_observations(frame, "India")

        clustered = cluster_incidents(selected, use_cache=False)

        self.assertEqual(clustered["incident_id"].nunique(), 2)
        self.assertEqual(clustered.loc[0, "incident_id"], clustered.loc[1, "incident_id"])
        self.assertNotEqual(clustered.loc[0, "incident_id"], clustered.loc[2, "incident_id"])

    def test_ids_are_stable_across_reload_and_input_order(self) -> None:
        frame = observations(
            [
                (20.0, 78.0, "2026-10-01", "0100", 10, "n"),
                (20.001, 78.0, "2026-10-01", "0200", 12, "h"),
            ]
        )
        first = cluster_incidents(filter_observations(frame, "India"), use_cache=False)
        second = cluster_incidents(filter_observations(frame.iloc[::-1], "India"), use_cache=False)

        self.assertEqual(first["incident_id"].iloc[0], second["incident_id"].iloc[0])
        self.assertRegex(first["incident_id"].iloc[0], r"^FD-20261001-[0-9a-f]{6}$")

    def test_frp_zero_scores_zero_and_does_not_break_ranking(self) -> None:
        frame = observations(
            [
                (20.0, 78.0, "2026-10-01", "0100", 0, "n"),
                (20.02, 78.0, "2026-10-01", "0200", 100, "n"),
            ]
        )
        ranked = rank_incidents(cluster_incidents(filter_observations(frame, "India"), use_cache=False))
        zero_incident = ranked.loc[ranked["max_frp"] == 0].iloc[0]

        self.assertEqual(zero_incident["frp_score"], 0.0)
        self.assertEqual(zero_incident["frp_contribution"], 0.0)

    def test_higher_frp_is_first_when_other_factors_match(self) -> None:
        frame = observations(
            [
                (20.0, 78.0, "2026-10-01", "0100", 10, "n"),
                (20.02, 78.0, "2026-10-01", "0100", 90, "n"),
            ]
        )
        ranked = rank_incidents(cluster_incidents(filter_observations(frame, "India"), use_cache=False))

        self.assertGreater(ranked.iloc[0]["priority_score"], ranked.iloc[1]["priority_score"])
        self.assertGreater(ranked.iloc[0]["frp_contribution"], ranked.iloc[1]["frp_contribution"])

    def test_custom_bounds_are_applied_before_analysis(self) -> None:
        frame = observations(
            [
                (20.0, 78.0, "2026-10-01", "0100", 10, "n"),
                (40.0, 78.0, "2026-10-01", "0100", 90, "n"),
            ]
        )
        selected = filter_observations(frame, BoundingBox(10, 70, 30, 90))
        self.assertEqual(len(selected), 1)

    def test_missing_optional_flag_sources_are_reported(self) -> None:
        frame = observations([(20.0, 78.0, "2026-10-01", "0100", 10, "n")])
        clustered = cluster_incidents(filter_observations(frame, "India"), use_cache=False)
        incidents = rank_incidents(clustered)

        flagged = add_rule_flags(incidents, clustered, Path(__file__).resolve().parents[1] / "missing-data")

        self.assertEqual(flagged.iloc[0]["flag_availability"]["offshore"], "not available")
        self.assertTrue(any("not available" in message.lower() for message in flagged.iloc[0]["flags"]))

    def test_nonzero_firms_type_is_a_cautious_possible_source_flag(self) -> None:
        frame = observations([(20.0, 78.0, "2026-10-01", "0100", 10, "n")])
        frame["type"] = 2
        clustered = cluster_incidents(filter_observations(frame, "India"), use_cache=False)
        incidents = rank_incidents(clustered)

        flagged = add_rule_flags(incidents, clustered, Path(__file__).resolve().parents[1] / "missing-data")

        self.assertTrue(bool(flagged.iloc[0]["possible_non_wildfire_source"]))
        self.assertTrue(any("FIRMS type 2" in message for message in flagged.iloc[0]["flags"]))


if __name__ == "__main__":
    unittest.main()
