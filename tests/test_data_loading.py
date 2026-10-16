"""Basic tests for FIRMS loading and row validation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satellite_fire.data import load_firms_data  # noqa: E402


HEADER = (
    "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,"
    "instrument,confidence,version,bright_t31,frp,daynight\n"
)


class LoadFirmsDataTests(unittest.TestCase):
    def test_preserves_hhmm_and_rejects_invalid_coordinate(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            csv_path = Path(temporary) / "fire_nrt_test.csv"
            csv_path.write_text(
                HEADER
                + "10,20,300,0.4,0.5,2026-10-01,0001,SNPP,SNPP,n,2.0NRT,280,1.2,N\n"
                + "91,20,300,0.4,0.5,2026-10-01,0002,SNPP,SNPP,n,2.0NRT,280,1.2,N\n",
                encoding="utf-8",
            )

            loaded = load_firms_data(csv_path)

            self.assertEqual(len(loaded.data), 1)
            self.assertEqual(loaded.data.loc[0, "acq_time"], "0001")
            self.assertEqual(len(loaded.rejected_rows), 1)
            self.assertIn("out-of-range coordinate: latitude", loaded.rejected_rows[0].reasons)

    def test_merges_files_and_removes_same_pixel_time_duplicate(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            folder = Path(temporary)
            record = "10,20,300,0.4,0.5,2026-10-01,0001,SNPP,SNPP,n,2.0NRT,280,1.2,N\n"
            (folder / "fire_nrt_a.csv").write_text(HEADER + record, encoding="utf-8")
            (folder / "fire_nrt_b.csv").write_text(HEADER + record, encoding="utf-8")

            loaded = load_firms_data(folder)

            self.assertEqual(len(loaded.files), 2)
            self.assertEqual(len(loaded.data), 1)
            self.assertEqual(loaded.duplicate_rows_removed, 1)

    def test_reports_wrong_field_count_without_failing(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            csv_path = Path(temporary) / "fire_nrt_badrow.csv"
            csv_path.write_text(HEADER + "10,20,300\n", encoding="utf-8")

            loaded = load_firms_data(csv_path)

            self.assertTrue(loaded.data.empty)
            self.assertEqual(len(loaded.rejected_rows), 1)
            self.assertIn("wrong field count", loaded.rejected_rows[0].reasons[0])


if __name__ == "__main__":
    unittest.main()

