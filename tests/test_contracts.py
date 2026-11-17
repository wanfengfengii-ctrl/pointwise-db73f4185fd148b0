import json
import math
import unittest
from lab.codecs import encode_payload,decode_payload,canonical_digest,IntegrityError
from lab.observation_rules import normalize_constraint
from lab.gaussian_model import GaussianParameters
from lab.report_rules import validate_probability
from lab.numerics import NumericalFailure
from lab.workflow import Workflow
from samples import E,L,R,I,exact_rows

class TestContracts(unittest.TestCase):
    def test_codec_integrity_and_kinds(self):
        constraints = [E(0),L(0),R(0),I(-1,1)]
        data = {"constraints":[normalize_constraint(c).payload() for c in constraints],
                "numbers":[-0.,.1,math.nextafter(1.,2.)],"version":1}
        encoded = encode_payload(data)
        decoded = decode_payload(encoded)
        self.assertEqual(decoded,data)
        self.assertEqual([c["kind"] for c in decoded["constraints"]],["exact","left","right","interval"])
        decoded["constraints"].clear()
        self.assertEqual(len(decode_payload(encoded)["constraints"]),4)
        self.assertEqual(canonical_digest({"n":-0.}),canonical_digest({"n":0.}))
        changed = json.loads(encoded)
        changed["body"]["constraints"][0]["kind"] = "left"
        with self.assertRaises(IntegrityError):
            decode_payload(json.dumps(changed))
        changed = json.loads(encoded)
        changed["version"] = 2
        with self.assertRaises(IntegrityError):
            decode_payload(json.dumps(changed))
        with self.assertRaises(IntegrityError):
            decode_payload('{"version":1,"version":1,"body":{},"digest":"x"}')
        with self.assertRaises(IntegrityError):
            decode_payload('{"version":1,"body":NaN,"digest":"x"}')
        with self.assertRaises(ValueError):
            encode_payload({"n":math.inf})

    def test_covariance_and_probability(self):
        for c in ([[1,2],[2,1]],[[0,0],[0,1]],[[1,.2],[.3,1]]):
            with self.assertRaises(ValueError):
                GaussianParameters((0,0),c)
        self.assertEqual(validate_probability(-1e-14),0.)
        with self.assertRaises(NumericalFailure):
            validate_probability(-1e-5)
        with Workflow() as w:
            w.put_batch("base",exact_rows())
            w.freeze("t",["exact"])
            fit = w.fit("t")
            report = w.risk(fit["result_id"],[0,0])
            w.storage.execute("UPDATE results SET payload='broken' WHERE result_id=?",(report["result_id"],))
            with self.assertRaises(IntegrityError):
                w.read_result(report["result_id"])
            with self.assertRaises(IntegrityError):
                w.export_result(report["result_id"])

    def test_schema_version_rejection(self):
        with Workflow() as w:
            w.storage.execute("UPDATE meta SET version=99")
            with self.assertRaises(ValueError):
                w.storage.initialize()
            self.assertEqual(w.storage.fetch_one("SELECT version FROM meta"),(99,))
