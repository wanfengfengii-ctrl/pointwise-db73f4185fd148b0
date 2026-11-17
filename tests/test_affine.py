"""Independent affine-region tests: geometry, measure ranks and fixed chain.

These checks do not modify the original V1 suite; they exercise the new region
intake channel and the conditional-report archive path end to end.
"""
import copy
import math
import sqlite3
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from lab.workflow import Workflow
from lab.numerics import NumericEngine, NumericalFailure
from lab.gaussian_model import GaussianParameters
from lab.observation_likelihood import observation_log_likelihood
from lab.regions import (
    normalize_region, analyze_region, region_measure, row_to_region,
    representative_point,
)
from lab.codecs import IntegrityError
from affine_examples import (
    triangle_region, triangle_reference, triangle_rows, region_row,
    anchors, axis_regions, VARIANCE, LOGL,
)
from samples import mixed_rows, E, L, I

STANDARD = GaussianParameters((0, 0), ((1, 0), (0, 1)))
ENGINE = NumericEngine()


class TestExactGeometry(unittest.TestCase):
    def test_rank_detection(self):
        self.assertEqual(analyze_region(triangle_region()).rank, 0)
        line = analyze_region({"le": [], "eq": [[1, 1, 1]]})
        self.assertEqual(line.rank, 1)
        self.assertIsNone(line.t_lower)
        self.assertIsNone(line.t_upper)
        point = analyze_region({"le": [], "eq": [[1, 1, 0], [1, -1, 0]]})
        self.assertEqual(point.rank, 2)
        self.assertEqual(point.point, (Fraction(0), Fraction(0)))

    def test_exact_rational_widths(self):
        width = math.ulp(0.0)
        # A genuinely positive, sub-float-epsilon strip keeps positive area.
        strip = {"le": [[1, 0, 0], [-1, 0, width]], "eq": []}
        self.assertEqual(analyze_region(strip).rank, 0)
        # The exact zero-width counterpart must be rejected.
        zero = {"le": [[1, 0, 0], [-1, 0, 0]], "eq": []}
        with self.assertRaises(ValueError):
            normalize_region(zero)

    def test_geometric_rejection_is_exact_and_batched(self):
        bad = [
            {"le": [], "eq": [[1, 1, 0], [2, 2, 1]]},
            {"le": [[1, 0, 0], [-1, 0, -math.ulp(0.0)]], "eq": []},
            {"le": [[1, 0, 0], [-1, 0, 0]], "eq": [[0, 1, 0]]},
            {"le": [[0, 1, 0], [0, -1, 0]], "eq": [[1, 0, 0]]},
        ]
        with Workflow() as w:
            before = list(w.storage.connection.iterdump())
            for index, region in enumerate(bad):
                with self.assertRaises(ValueError):
                    w.put_batch("bad%d" % index, [region_row("s", region)])
                self.assertEqual(list(w.storage.connection.iterdump()), before)

    def test_plane_validation(self):
        for region in (
            {"le": [[1, 17, 0]], "eq": []},
            {"le": [[0, 0, 0]], "eq": []},
            {"le": [[True, 1, 0]], "eq": []},
            {"le": [[1, 1]], "eq": []},
            {"le": "nope", "eq": []},
            {"le": [], "eq": [[1, 1, 0], [1, -1, 0], [1, 0, 0]]},
            {"le": [], "eq": [[1, 1, float("inf")]]},
            {"le": [], "eq": [[1, 1, float("nan")]]},
        ):
            with self.assertRaises(ValueError):
                normalize_region(region)

    def test_redundant_and_scaled_equalities(self):
        line = {"le": [], "eq": [[1, 1, 1]]}
        scaled = {"le": [], "eq": [[-2, -2, -2], [1, 1, 1]]}
        a = math.exp(observation_log_likelihood({"region": line}, STANDARD, ENGINE))
        b = math.exp(observation_log_likelihood({"region": scaled}, STANDARD, ENGINE))
        self.assertAlmostEqual(a, b, delta=1e-14)
        # Redundant inactive half planes do not change the measure.
        redundant = {"le": [[1, 0, 100], [1, 0, 50]], "eq": [[1, 1, 1]]}
        c = math.exp(observation_log_likelihood({"region": redundant}, STANDARD, ENGINE))
        self.assertAlmostEqual(a, c, delta=1e-14)

    def test_normalization_preserves_order_and_coefficients(self):
        region = {"le": [[1, 0, 0], [0, -1, -2.5]], "eq": []}
        normalized = normalize_region(copy.deepcopy(region))
        self.assertEqual(normalized["le"], [[1, 0, 0.0], [0, -1, -2.5]])
        line = normalize_region({"le": [[-1, 0, 0]], "eq": [[2, -2, 1]]})
        self.assertEqual([plane[:2] for plane in line["eq"]], [[2, -2]])
        self.assertEqual(normalize_region({"le": [], "eq": [[1, 1, -0.0]]})["eq"][0][2], 0.0)


class TestHausdorffMeasures(unittest.TestCase):
    def test_rank0_triangle_against_decimal_oracle(self):
        measure = math.exp(observation_log_likelihood(
            {"region": triangle_region()}, STANDARD, ENGINE))
        self.assertAlmostEqual(measure, float(triangle_reference()), delta=1e-11)

    def test_rank1_arc_length_and_segment(self):
        full_line = math.exp(observation_log_likelihood(
            {"region": {"le": [], "eq": [[1, 1, 1]]}}, STANDARD, ENGINE))
        self.assertAlmostEqual(full_line, 0.31069656037692774, delta=1e-12)
        segment = math.exp(observation_log_likelihood(
            {"region": {"le": [[-1, 0, 0], [0, -1, 0]], "eq": [[1, 1, 1]]}},
            STANDARD, ENGINE))
        self.assertAlmostEqual(segment, 0.16171752171312473, delta=1e-12)

    def test_rank2_point_density(self):
        region = {"le": [], "eq": [[1, 1, 0], [1, -1, 0]]}
        measure = math.exp(observation_log_likelihood(
            {"region": region}, STANDARD, ENGINE))
        self.assertAlmostEqual(measure, 1 / (2 * math.pi), delta=1e-12)

    def test_equality_scaling_invariance(self):
        a = math.exp(observation_log_likelihood(
            {"region": {"le": [], "eq": [[1, 1, 1]]}}, STANDARD, ENGINE))
        b = math.exp(observation_log_likelihood(
            {"region": {"le": [], "eq": [[3, 3, 3]]}}, STANDARD, ENGINE))
        self.assertAlmostEqual(a, b, delta=1e-14)

    def test_x_region_equivalence(self):
        for old, new in zip(mixed_rows(), axis_regions(mixed_rows())):
            a = observation_log_likelihood(old, STANDARD, ENGINE)
            b = observation_log_likelihood(new, STANDARD, ENGINE)
            self.assertAlmostEqual(a, b, delta=1e-10)

    def test_representative_points_are_feasible(self):
        for region in (
            triangle_region(),
            {"le": [[-1, 0, 0], [0, -1, 0]], "eq": [[1, 1, 1]]},
            {"le": [], "eq": [[1, 1, 0], [1, -1, 0]]},
        ):
            x, y = representative_point(region)
            for a, b, c in region["le"]:
                self.assertLessEqual(a * x + b * y, c + 1e-9)


class TestConditionalReport(unittest.TestCase):
    def _fit(self, w):
        rows = triangle_rows()
        w.put_batch("triangles", rows)
        w.freeze("triangles", ["affine"])
        return w.fit("triangles"), rows

    def test_triangle_conditional_ratio_and_strict_thresholds(self):
        with Workflow() as w:
            fit, rows = self._fit(w)
            report = w.conditional_risk(fit["result_id"], ["affine", "tri11"], [0.4, 0.4])
            expected = float(triangle_reference(VARIANCE, 0.4)
                             / triangle_reference(VARIANCE))
            self.assertAlmostEqual(report["probability"], expected, delta=1e-7)
            self.assertAlmostEqual(
                report["denominator"], float(triangle_reference(VARIANCE)), delta=1e-7)
            self.assertEqual(report["kind"], "conditional_report")
            self.assertEqual(report["sample"], rows[-1])
            # Empty numerator is an explicit zero, not a failure.
            empty = w.conditional_risk(fit["result_id"], ["affine", "tri-11"], [0.4, 0.4])
            self.assertEqual(empty["probability"], 0.0)

    def test_replay_isolation_and_identity(self):
        with Workflow() as w:
            fit, _ = self._fit(w)
            key, thresholds = ["affine", "tri11"], [0.1, 0.2]
            first = w.conditional_risk(fit["result_id"], key, thresholds)
            self.assertEqual(w.conditional_risk(fit["result_id"], key, thresholds), first)
            other = w.conditional_risk(fit["result_id"], ["affine", "tri-11"], thresholds)
            self.assertNotEqual(other["result_id"], first["result_id"])
            # Must not collide with an unconditional risk request.
            unconditional = w.risk(fit["result_id"], thresholds)
            self.assertNotEqual(unconditional["result_id"], first["result_id"])
            # The frozen parent task is the only admissible sample source.
            with self.assertRaises(KeyError):
                w.conditional_risk(fit["result_id"], ["elsewhere", "tri11"], thresholds)
            for bad_key in (None, ["only"], ["a", "b", "c"], [1, "b"]):
                with self.assertRaises(ValueError):
                    w.conditional_risk(fit["result_id"], bad_key, thresholds)

    def test_strict_thresholds_on_line_and_point(self):
        with Workflow() as w:
            rows = anchors("line")
            rows.append(region_row("line", {"le": [], "eq": [[1, 1, 0]]}, "line"))
            w.put_batch("line", rows)
            w.freeze("line", ["line"])
            fit = w.fit("line")
            self.assertEqual(
                w.conditional_risk(fit["result_id"], ["line", "line"], [0, 0])["probability"], 0.0)
            # Exact legacy point (1, 1): strict >1 on the first axis excludes it.
            self.assertEqual(
                w.conditional_risk(fit["result_id"], ["line", "point3"], [1, 0])["probability"], 0.0)
            self.assertEqual(
                w.conditional_risk(fit["result_id"], ["line", "point3"], [0, 0])["probability"], 1.0)

    def test_publication_rollback_and_retry(self):
        with Workflow() as w:
            fit, _ = self._fit(w)
            key = ["affine", "tri11"]
            for stage in ("result.payload", "result.index", "result.parent", "commit"):
                before = {t: w.storage.fetch_all("SELECT * FROM " + t)
                          for t in ("results", "result_index", "parents")}
                with w.storage.failing(stage):
                    with self.assertRaises(sqlite3.OperationalError):
                        w.conditional_risk(fit["result_id"], key, [0.2, 0.3])
                for table, data in before.items():
                    self.assertEqual(w.storage.fetch_all("SELECT * FROM " + table), data)
            published = w.conditional_risk(fit["result_id"], key, [0.2, 0.3])
            with w.engine.failing("risk"):
                with self.assertRaises(NumericalFailure):
                    w.conditional_risk(fit["result_id"], key, [0.25, 0.35])
            # Replay never recomputes, even inside the failure scope.
            with w.engine.failing("risk"):
                self.assertEqual(
                    w.conditional_risk(fit["result_id"], key, [0.2, 0.3]), published)

    def test_evidence_chain_and_reopen_after_correction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "lab.sqlite")
            with Workflow(path) as w:
                fit, rows = self._fit(w)
                report = w.conditional_risk(fit["result_id"], ["affine", "tri11"], [0.4, 0.4])
                evidence = w.export_result(report["result_id"])
                saved = copy.deepcopy(report)
                correction = region_row("tri11", {"le": [], "eq": [[1, 0, 9], [0, 1, 9]]})
                correction["rev"] = 2
                w.put_batch("corrected", [correction])
                rows[-1]["region"]["le"].clear()
                self.assertEqual(w.read_result(saved["result_id"]), saved)
                self.assertEqual(w.export_result(saved["result_id"]), evidence)
            with Workflow(path) as reopened:
                with reopened.engine.failing("fit", "risk", "integrate", "minimize", "linear"):
                    self.assertEqual(reopened.read_result(saved["result_id"]), saved)
                    self.assertEqual(reopened.export_result(saved["result_id"]), evidence)

    def test_damaged_conditional_chain_rejected(self):
        with Workflow() as w:
            fit, _ = self._fit(w)
            report = w.conditional_risk(fit["result_id"], ["affine", "tri11"], [0.4, 0.4])
            payload = dict(report, model_digest="broken")
            from lab.codecs import encode_payload
            w.storage.execute("UPDATE results SET payload=? WHERE result_id=?",
                              (encode_payload(payload), report["result_id"]))
            with self.assertRaises(IntegrityError):
                w.read_result(report["result_id"])
            with self.assertRaises(IntegrityError):
                w.export_result(report["result_id"])


class TestRegionFit(unittest.TestCase):
    def test_symmetric_eight_pattern_reference(self):
        with Workflow() as w:
            rows = triangle_rows()
            w.put_batch("triangles", rows)
            w.freeze("triangles", ["affine"])
            fit = w.fit("triangles")
            self.assertEqual(len(rows), 8)
            for value in fit["mu"]:
                self.assertAlmostEqual(value, 0.0, delta=1e-6)
            for i in range(2):
                for j in range(2):
                    expected = VARIANCE if i == j else 0.0
                    self.assertAlmostEqual(fit["covariance"][i][j], expected, delta=1e-6)
            self.assertAlmostEqual(fit["log_likelihood"], LOGL, delta=1e-7)

    def test_region_fit_matches_censored_x_fit(self):
        with Workflow() as w:
            w.put_batch("old", mixed_rows("axis"))
            w.freeze("old", ["axis"])
            first = w.fit("old")
            w.put_batch("new", axis_regions(mixed_rows("region")))
            w.freeze("new", ["region"])
            second = w.fit("new")
            for a, b in zip(first["mu"], second["mu"]):
                self.assertAlmostEqual(a, b, delta=1e-6)
            for a, b in zip(sum(first["covariance"], []), sum(second["covariance"], [])):
                self.assertAlmostEqual(a, b, delta=1e-6)
            self.assertAlmostEqual(first["log_likelihood"], second["log_likelihood"], delta=1e-7)

    def test_pattern_count_ignores_identity_and_count(self):
        base = triangle_region()
        rows = []
        for k in range(41):
            rows.append(region_row("s%d" % k, base, count=1 + k))
        with Workflow() as w:
            w.put_batch("many", rows)
            w.freeze("many", ["affine"])
            with self.assertRaises(NumericalFailure):
                w.fit("many")


if __name__ == "__main__":
    unittest.main()
