import unittest
from pathlib import Path

from stdf import parse_stdf
from stdf.analytics import bin_pareto, overview, test_stats, yield_by_lot

DATA = Path(__file__).resolve().parents[1] / "data"


class TestParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = DATA / "LOT-A-2646.stdf"
        cls.data = parse_stdf(cls.path)

    def test_header_and_counts(self):
        self.assertEqual(self.data.meta["lot_id"], "LOT-A-2646")
        self.assertEqual(len(self.data.parts), 1695)
        self.assertEqual(len(self.data.results), 16950)
        self.assertEqual(self.data.wafer_count, 5)
        self.assertEqual(self.data.record_counts[(5, 30)], 16950)

    def test_part_columns(self):
        for col in ("lot_id", "wafer_id", "x", "y", "hard_bin", "soft_bin", "passed"):
            self.assertIn(col, self.data.parts.columns)
        self.assertTrue(self.data.parts["passed"].any())
        self.assertTrue((~self.data.parts["passed"]).any())

    def test_limits_and_units(self):
        r = self.data.results
        self.assertEqual(r.loc[r["test_name"] == "VOUT_10mA", "units"].iloc[0], "V")
        vout = r[r["test_name"] == "VOUT_10mA"]
        self.assertAlmostEqual(vout["lo_limit"].iloc[0], 3.270, places=4)
        self.assertAlmostEqual(vout["hi_limit"].iloc[0], 3.330, places=4)

    def test_yield_and_bins(self):
        o = overview(self.data)
        self.assertGreater(o["yield"], 85)
        self.assertLess(o["yield"], 100)
        lots = yield_by_lot(self.data.parts)
        self.assertEqual(len(lots), 1)
        bins = bin_pareto(self.data.parts, self.data.bin_names)
        self.assertIn("PASS", bins["bin_name"].tolist())

    def test_cpk(self):
        stats = test_stats(self.data.results)
        self.assertEqual(len(stats), 10)
        self.assertTrue(stats["cpk"].notna().all())
        self.assertGreater(stats.loc[stats["test_name"] == "Continuity", "cpk"].iloc[0], 2)


if __name__ == "__main__":
    unittest.main()
