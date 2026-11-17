"""Public inputs and independent fixed-case checks, not a region implementation."""
import copy
import json
import math
import sqlite3
import sys
import tempfile
import unittest
from decimal import localcontext
from pathlib import Path
from tests.test_numerics import D, PI, cdf, integral
from samples import E, mixed_rows
from lab.workflow import Workflow
from lab.numerics import NumericEngine, NumericalFailure
from lab.gaussian_model import GaussianParameters
from lab.observation_likelihood import observation_log_likelihood

VARIANCE = 0.8307377982665323
LOGL = -55.85222847626954

def triangle_region(sx=1, sy=1):
    return {"le": [[-sx, 0, 0], [0, -sy, 0], [sx, sy, 1]], "eq": []}

def region_row(sample, region, cohort="affine", count=1):
    return {"cohort_id": cohort, "sample_id": sample, "rev": 1,
            "count": count, "region": copy.deepcopy(region)}

def anchors(cohort="affine"):
    return [{"cohort_id": cohort, "sample_id": "point%d" % i, "rev": 1,
             "count": 4, "x": [E(x), E(y)]}
            for i, (x, y) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1)))]

def triangle_rows():
    return anchors() + [region_row("tri%d%d" % (sx, sy), triangle_region(sx, sy))
                        for sx in (-1, 1) for sy in (-1, 1)]

def axis_regions(rows):
    """Produce equivalent test inputs; no general geometric processing."""
    result = copy.deepcopy(rows)
    for row in result:
        le, eq = [], []
        for j, c in enumerate(row.pop("x")):
            a, b = (1, 0) if j == 0 else (0, 1)
            if c["kind"] == "exact":
                eq.append([a, b, c["value"]])
            else:
                if c["kind"] in ("right", "interval"):
                    le.append([-a, -b, -c["lower"]])
                if c["kind"] in ("left", "interval"):
                    le.append([a, b, c["upper"]])
        row["region"] = {"le": le, "eq": eq}
    return result

def triangle_reference(variance=1, lower=0, n=80):
    """Only this fixed right triangle; independent Decimal quadrature."""
    with localcontext() as context:
        context.prec = 70
        v, lo = D(str(variance)), D(str(lower))
        sd, root = v.sqrt(), (2 * PI).sqrt()
        def mass_at(x):
            return ((-x*x/(2*v)).exp()/(sd*root) *
                    (cdf((1-x)/sd)-cdf(lo/sd)))
        return +integral(mass_at, lo, 1-lo, n)

def symmetric_reference(n):
    """Scalar score equation for the symmetric supplied fixture only."""
    with localcontext() as context:
        context.prec = 70
        def moments(v):
            sd, root = v.sqrt(), (2*PI).sqrt()
            def at(x, second=False):
                z = (1-x)/sd
                mass = cdf(z)-D(".5")
                px = (-x*x/(2*v)).exp()/(root*sd)
                if second:
                    return px*(x*x*mass+v*(mass-z*(-z*z/2).exp()/root))
                return px*mass
            p = integral(at, D(0), D(1), n)
            m = integral(lambda x: at(x, True), D(0), D(1), n)/p
            return p, m
        lo, hi = D(".8"), D(1)
        for _ in range(50):
            v = (lo+hi)/2
            p, moment = moments(v)
            if 40*v < 32+4*moment:
                lo = v
            else:
                hi = v
        v = (lo+hi)/2
        p, _ = moments(v)
        ll = -16*((2*PI).ln()+v.ln())-16/v+4*p.ln()
        return +v, +p, +ll

def reference():
    p, q = triangle_reference(), triangle_reference(lower=.4)
    assert abs(p-triangle_reference(n=48)) < D("1e-30")
    assert abs(q-triangle_reference(lower=.4, n=48)) < D("1e-30")
    with localcontext() as context:
        context.prec = 70
        ratio = q/p
        box = ((cdf(D(1))-cdf(D(".4")))/(cdf(D(1))-D(".5")))**2
        line = (-D(".25")).exp()/(2*PI).sqrt()
        segment = line*(2*cdf(1/D(2).sqrt())-1)
        assert abs(ratio-D("0.03772562049949979")) < D("1e-16")
        assert box-ratio > D(".25")
    coarse, fine = symmetric_reference(48), symmetric_reference(80)
    assert max(abs(a-b) for a, b in zip(coarse, fine)) < D("1e-30")
    assert abs(fine[0]-D(str(VARIANCE))) < D("1e-14")
    assert abs(fine[2]-D(str(LOGL))) < D("1e-12")
    return {"triangle_denominator": str(p), "triangle_conditional": str(ratio),
            "box_conditional": str(box), "line_full_H1": str(line),
            "line_segment_H1": str(segment),
            "symmetric_fit": [str(x) for x in fine]}

def close(actual, expected, tolerance):
    assert abs(actual-expected) <= tolerance, (actual, expected, tolerance)

def kernel_checks():
    engine = NumericEngine()
    model = GaussianParameters((0, 0), ((1, 0), (0, 1)))
    def contribution(region):
        return math.exp(observation_log_likelihood({"region": region}, model, engine))
    close(contribution(triangle_region()), float(triangle_reference()), 1e-11)
    line = {"le": [], "eq": [[1, 1, 1]]}
    expected = math.exp(-.25)/math.sqrt(2*math.pi)
    close(contribution(line), expected, 1e-11)
    close(contribution({"le": [], "eq": [[-2, -2, -2], [1, 1, 1]]}), expected, 1e-11)
    segment = {"le": [[-1, 0, 0], [0, -1, 0]], "eq": [[1, 1, 1]]}
    close(contribution(segment), expected*math.erf(.5), 1e-11)
    close(contribution({"le": [], "eq": [[1, 1, 0], [1, -1, 0]]}),
          1/(2*math.pi), 1e-11)
    for old, new in zip(mixed_rows(), axis_regions(mixed_rows())):
        close(observation_log_likelihood(old, model, engine),
              observation_log_likelihood(new, model, engine), 1e-10)

def geometry_checks(w):
    bad = [
        {"le": [], "eq": [[1, 1, 0], [2, 2, 1]]},
        {"le": [[1, 0, 0], [-1, 0, 0]], "eq": []},
        {"le": [[1, 0, 0], [-1, 0, -math.ulp(0.)]], "eq": []},
        {"le": [[1, 0, 0], [-1, 0, 0]], "eq": [[0, 1, 0]]},
    ]
    before = list(w.storage.connection.iterdump())
    for i, region in enumerate(bad):
        try:
            w.put_batch("invalid%d" % i, [region_row("bad", region)])
        except ValueError:
            pass
        else:
            raise AssertionError("geometrically invalid region accepted")
        assert list(w.storage.connection.iterdump()) == before

def business(path):
    with Workflow(path) as w:
        geometry_checks(w)
        rows = triangle_rows()
        w.put_batch("triangles", rows)
        w.freeze("triangles", ["affine"])
        fit = w.fit("triangles")
        for value in fit["mu"]:
            close(value, 0, 1e-6)
        for i in range(2):
            for j in range(2):
                close(fit["covariance"][i][j], VARIANCE if i == j else 0, 1e-6)
        close(fit["log_likelihood"], LOGL, 1e-7)
        key, thresholds = ["affine", "tri11"], [.4, .4]
        report = w.conditional_risk(fit["result_id"], key, thresholds)
        assert report["kind"] == "conditional_report"
        expected = float(triangle_reference(VARIANCE, .4)/triangle_reference(VARIANCE))
        close(report["probability"], expected, 1e-7)
        close(report["denominator"], float(triangle_reference(VARIANCE)), 1e-7)
        assert report["sample"] == rows[-1]
        assert w.conditional_risk(fit["result_id"], key, thresholds) == report
        other = w.conditional_risk(fit["result_id"], ["affine", "tri-11"], thresholds)
        assert other["result_id"] != report["result_id"]
        assert other["probability"] == 0
        for stage in ("result.payload", "result.index", "result.parent", "commit"):
            before = {t: w.storage.fetch_all("SELECT * FROM "+t)
                      for t in ("results", "result_index", "parents")}
            with w.storage.failing(stage):
                try:
                    w.conditional_risk(fit["result_id"], key, [.1, .2])
                except sqlite3.OperationalError:
                    pass
                else:
                    raise AssertionError("failure port did not fire")
            for table, data in before.items():
                assert w.storage.fetch_all("SELECT * FROM "+table) == data
        w.conditional_risk(fit["result_id"], key, [.1, .2])
        with w.engine.failing("risk"):
            try:
                w.conditional_risk(fit["result_id"], key, [.2, .3])
            except NumericalFailure:
                pass
            else:
                raise AssertionError("numerical failure port did not fire")
        try:
            w.conditional_risk(fit["result_id"], ["elsewhere", "tri11"], thresholds)
        except KeyError:
            pass
        else:
            raise AssertionError("sample outside parent task accepted")
        evidence = w.export_result(report["result_id"])
        assert evidence["report"] == report
        saved = copy.deepcopy(report)
        correction = region_row("tri11", {"le": [], "eq": [[1, 0, 9], [0, 1, 9]]})
        correction["rev"] = 2
        w.put_batch("corrected", [correction])
        rows[-1]["region"]["le"].clear()
        report["sample"]["region"]["le"].clear()
        key.clear()
        thresholds[0] = 9
        assert w.read_result(saved["result_id"]) == saved
        assert w.export_result(saved["result_id"]) == evidence
    with Workflow(path) as reopened:
        with reopened.engine.failing("fit", "risk", "integrate", "minimize", "linear"):
            assert reopened.read_result(saved["result_id"]) == saved
            assert reopened.export_result(saved["result_id"]) == evidence
    return {"probability": saved["probability"], "denominator": saved["denominator"],
            "sample_rev": saved["sample"]["rev"], "old_evidence_unchanged": True}

def remaining_checks():
    with Workflow() as w:
        old = mixed_rows("axis")
        w.put_batch("old", old)
        w.freeze("old", ["axis"])
        f1 = w.fit("old")
        new = axis_regions(mixed_rows("region"))
        w.put_batch("new", new)
        w.freeze("new", ["region"])
        f2 = w.fit("new")
        for a, b in zip(f1["mu"], f2["mu"]):
            close(a, b, 1e-6)
        for a, b in zip(sum(f1["covariance"], []), sum(f2["covariance"], [])):
            close(a, b, 1e-6)
        close(f1["log_likelihood"], f2["log_likelihood"], 1e-7)
        line_rows = anchors("line")
        line_rows.append(region_row("line", {"le": [], "eq": [[1, 1, 0]]}, "line"))
        w.put_batch("line", line_rows)
        w.freeze("line", ["line"])
        f = w.fit("line")
        for i in range(2):
            close(f["mu"][i], 0, 1e-6)
            for j in range(2):
                close(f["covariance"][i][j], 33/34 if i == j else -1/34, 1e-6)
        r = w.conditional_risk(f["result_id"], ["line", "line"], [0, 0])
        assert r["probability"] == 0
        # Strict thresholds on an exact observed point, also a legacy x row.
        r = w.conditional_risk(f["result_id"], ["line", "point3"], [1, 0])
        assert r["probability"] == 0
        r = w.conditional_risk(f["result_id"], ["line", "point3"], [0, 0])
        assert r["probability"] == 1

def main():
    mode = sys.argv[1] if len(sys.argv) == 2 else "reference"
    if mode == "reference":
        print(json.dumps(reference(), sort_keys=True))
    elif mode in ("verify", "demo"):
        if mode == "verify":
            suite = unittest.defaultTestLoader.discover("tests")
            if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
                return 1
            reference()
            kernel_checks()
            remaining_checks()
        with tempfile.TemporaryDirectory(prefix="affine-example-") as directory:
            observed = business(str(Path(directory)/"lab.sqlite"))
        print("POINTWISE_SCENARIO "+json.dumps({"case_id": "affine.fixed_sample", **observed},
                                               sort_keys=True))
    else:
        raise ValueError("reference, verify or demo required")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
