import copy
import json
import math
import unittest
from lab.workflow import Workflow
from lab.numerics import NumericEngine,NumericalFailure
from lab.gaussian_model import GaussianParameters
from lab.observation_likelihood import observation_log_likelihood
from samples import exact_rows,mixed_rows,E,L,R,I,REFERENCE

class TestAnalysis(unittest.TestCase):
    def _run(self,rows,thresholds):
        with Workflow() as w:
            w.put_batch("batch",rows)
            w.freeze("task",sorted({r["cohort_id"] for r in rows}))
            fit = w.fit("task")
            report = w.risk(fit["result_id"],thresholds)
            return fit,report

    def test_exact_correlated(self):
        fit,report = self._run(exact_rows(),[0,0])
        for mean in fit["mu"]:
            self.assertAlmostEqual(mean,0.,delta=1e-6)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(fit["covariance"][i][j],(2 if i==j else 1)/3,delta=1e-6)
        self.assertAlmostEqual(report["probability"],1/3,delta=1e-7)
        independent = exact_rows()[:4]
        for row,(a,b) in zip(independent,[(1,1),(1,-1),(-1,1),(-1,-1)]):
            row["x"] = [E(a),E(b)]
        f,r = self._run(independent,[0,0])
        self.assertEqual(f["covariance"],[[1.,0.],[0.,1.]])
        self.assertAlmostEqual(r["probability"],.25,delta=1e-7)
        print("POINTWISE_SCENARIO "+json.dumps({
            "case_id":"v4.correlated_tail","input":exact_rows(),
            "expected":{"mu":[0,0],"covariance":[[2/3,1/3],[1/3,2/3]]},
            "actual":{"mu":fit["mu"],"covariance":fit["covariance"]},
            "downstream_expected":1/3,"downstream_actual":report["probability"],
            "code_paths":["lab/fit_solver.py","lab/fitting.py","lab/joint_tail.py","lab/risk.py"],
            "oracle":"weighted empirical MLE; 1/4+asin(1/2)/(2*pi)=1/3"
        },sort_keys=True))

    def test_mixed_censoring(self):
        fit,report = self._run(mixed_rows(),REFERENCE["thresholds"])
        for a,b in zip(fit["mu"],REFERENCE["mu"]):
            self.assertAlmostEqual(a,b,delta=1e-6)
        for actual,expected in zip(fit["covariance"],REFERENCE["covariance"]):
            for a,b in zip(actual,expected):
                self.assertAlmostEqual(a,b,delta=1e-6)
        self.assertAlmostEqual(fit["log_likelihood"],REFERENCE["log_likelihood"],delta=1e-8)
        self.assertAlmostEqual(report["probability"],REFERENCE["probability"],delta=1e-7)
        expanded = []
        for row in mixed_rows("expanded"):
            for k in range(row["count"]):
                item = copy.deepcopy(row)
                item["sample_id"] += ":"+str(k)
                item["count"] = 1
                expanded.append(item)
        f,r = self._run(expanded,REFERENCE["thresholds"])
        self.assertEqual(len(expanded),36)
        for a,b in zip(f["mu"],fit["mu"]):
            self.assertAlmostEqual(a,b,delta=1e-7)
        for arow,brow in zip(f["covariance"],fit["covariance"]):
            for a,b in zip(arow,brow):
                self.assertAlmostEqual(a,b,delta=1e-7)
        self.assertAlmostEqual(r["probability"],report["probability"],delta=1e-8)
        swapped = mixed_rows("swapped")
        for row in swapped:
            row["x"].reverse()
        f,r = self._run(swapped,list(reversed(REFERENCE["thresholds"])))
        for a,b in zip(f["mu"],reversed(fit["mu"])):
            self.assertAlmostEqual(a,b,delta=1e-6)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(f["covariance"][i][j],fit["covariance"][1-i][1-j],delta=1e-6)
        self.assertAlmostEqual(r["probability"],report["probability"],delta=1e-7)

    def test_directional_constraints(self):
        model = GaussianParameters((0,0),((1,.5),(.5,1)))
        engine = NumericEngine()
        density = 1/math.sqrt(2*math.pi)
        def likelihood(first,second):
            return math.exp(observation_log_likelihood({"x":[first,second]},model,engine))
        self.assertAlmostEqual(likelihood(E(0),L(0)),density/2,delta=1e-12)
        self.assertAlmostEqual(likelihood(E(0),R(0)),density/2,delta=1e-12)
        expected = density*math.erf(1/math.sqrt(1.5))
        self.assertAlmostEqual(likelihood(E(0),I(-1,1)),expected,delta=1e-12)
        self.assertAlmostEqual(likelihood(L(0),L(0)),1/3,delta=1e-12)
        self.assertAlmostEqual(likelihood(R(0),L(0)),1/6,delta=1e-12)
        # Conditional mean is .5 at X=1; left and right are intentionally unequal.
        conditional_left = .5*math.erfc(.5/math.sqrt(1.5))
        marginal = math.exp(-.5)/math.sqrt(2*math.pi)
        self.assertAlmostEqual(likelihood(E(1),L(0)),marginal*conditional_left,delta=1e-12)
        self.assertAlmostEqual(likelihood(E(1),L(0))+likelihood(E(1),R(0)),marginal,delta=1e-12)

    def test_numeric_failure(self):
        with Workflow() as w:
            w.put_batch("good",exact_rows())
            w.freeze("good",["exact"])
            old = w.fit("good")
            w.put_batch("mixed",mixed_rows())
            w.freeze("new",["mixed"])
            before = w.storage.fetch_one("SELECT count(*) FROM results")
            for operation in ("fit","minimize","integrate","linear"):
                with w.engine.failing(operation):
                    with self.assertRaises(NumericalFailure):
                        w.fit("new")
                self.assertIsNone(w.archive.find_fit_for_task("new"))
                self.assertEqual(w.storage.fetch_one("SELECT count(*) FROM results"),before)
                self.assertEqual(w.read_result(old["result_id"]),old)
            fit = w.fit("new")
            with w.engine.failing("risk"):
                with self.assertRaises(NumericalFailure):
                    w.risk(fit["result_id"],[.1,.2])
            self.assertIsNone(w.archive.find_report(fit["result_id"],[.1,.2]))
            w.risk(fit["result_id"],[.1,.2])

    def test_input_guarantee_and_thresholds(self):
        with Workflow() as w:
            singular = exact_rows("singular")
            for row in singular:
                row["x"] = [E(0),E(0)]
            w.put_batch("singular",singular)
            w.freeze("singular",["singular"])
            with self.assertRaises(NumericalFailure):
                w.fit("singular")
            self.assertEqual(w.storage.fetch_one("SELECT count(*) FROM results"),(0,))
            w.put_batch("good",exact_rows())
            w.freeze("good",["exact"])
            fit = w.fit("good")
            for thresholds in ([0],[0,1,2],[True,0],[None,0],[float("inf"),0]):
                with self.assertRaises(ValueError):
                    w.risk(fit["result_id"],thresholds)
            self.assertAlmostEqual(w.risk(fit["result_id"],[100,100])["probability"],0.,delta=1e-7)
            self.assertAlmostEqual(w.risk(fit["result_id"],[-100,-100])["probability"],1.,delta=1e-7)
