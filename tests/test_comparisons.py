import json
import unittest
import run
from statistics_output import compare_tap


class ComparisonTests(unittest.TestCase):
    def test_all_25_records_and_each_event_are_retained(self):
        records = json.loads((run.ROOT / "experiments/wetlab-25.json").read_text())["records"]
        self.assertEqual(len(records), 25)
        self.assertEqual(len({r["clip_id"] for r in records}), 25)
        movement = [m for r in records for m in r["movement"]]
        taps = [t for r in records for t in r["taps"]]
        self.assertEqual(len(movement), 21)
        self.assertEqual(sum(m["walking_summary_eligible"] for m in movement), 19)
        self.assertEqual(len(taps), 11)
        self.assertEqual(len({t["event_id"] for t in taps}), 11)
        self.assertEqual(sum(t["no_departure_observation_ms"] is not None for t in taps), 5)

    def test_nonresponse_is_censored_and_departure_disagreement_retained(self):
        source = json.loads((run.ROOT / "experiments/wetlab-25.json").read_text())
        wet = next(t for r in source["records"] for t in r["taps"] if t["no_departure_observation_ms"] is not None)
        model = {"eligible": True, "takeoff_latency_ms": 37.8, "gf_latency_ms": 29.5,
                 "successor_retention_pass": True, "inversion_after_input": False}
        comparison = compare_tap(wet, model)
        self.assertIsNone(comparison["wet_departure_lower_ms"])
        self.assertIsNone(comparison["wet_departure_upper_ms"])
        self.assertEqual(comparison["relation"], "DIFFERS: MODEL DEPARTS")
        model["takeoff_latency_ms"] = None
        self.assertIsNone(compare_tap(wet, model)["digital_support_loss_latency_ms"])


if __name__ == "__main__":
    unittest.main()
