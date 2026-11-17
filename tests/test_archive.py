import copy
import sqlite3
import tempfile
import unittest
from pathlib import Path
from lab.workflow import Workflow
from lab.codecs import IntegrityError,encode_payload
from samples import exact_rows,mixed_rows,E,REFERENCE

class TestArchive(unittest.TestCase):
    def _fit(self,w):
        w.put_batch("base",exact_rows())
        w.freeze("task",["exact"])
        return w.fit("task")

    def test_publish_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/"lab.sqlite")
            with Workflow(path) as w:
                fit = self._fit(w)
                baseline = w.risk(fit["result_id"],[0,0])
                for stage in ("result.payload","result.index","result.parent","commit"):
                    before = {table:w.storage.fetch_all("SELECT * FROM "+table) for table in ("results","result_index","parents")}
                    with w.storage.failing(stage):
                        with self.assertRaises(sqlite3.OperationalError):
                            w.risk(fit["result_id"],[.15,.25])
                    self.assertIsNone(w.archive.find_report(fit["result_id"],[.15,.25]))
                    for table,rows in before.items():
                        self.assertEqual(w.storage.fetch_all("SELECT * FROM "+table),rows)
                    self.assertEqual(w.read_result(baseline["result_id"]),baseline)
            with Workflow(path) as w:
                report = w.risk(fit["result_id"],[.15,.25])
                self.assertEqual(w.risk(fit["result_id"],[.15,.25]),report)
                self.assertEqual(w.storage.fetch_one("SELECT count(*) FROM results WHERE kind='report'"),(2,))
                w.storage.initialize()
                self.assertEqual(w.read_result(report["result_id"]),report)

    def test_fixed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/"lab.sqlite")
            with Workflow(path) as w:
                w.put_batch("mixed",mixed_rows())
                task = w.freeze("mixed",["mixed"])
                fit = w.fit("mixed")
                thresholds = [.15,.25]
                report = w.risk(fit["result_id"],thresholds)
                saved = w.export_result(report["result_id"])
                expected = copy.deepcopy(saved)
                revision = mixed_rows()[0]
                revision["rev"],revision["x"] = 2,[E(7),E(-2)]
                w.put_batch("correction",[revision])
                thresholds[0] = 9
                fit["mu"][0] = 999
                task["input"]["rows"].clear()
                saved["sources"][0]["x"][0]["kind"] = "left"
                self.assertEqual(w.export_result(report["result_id"]),expected)
            with Workflow(path) as w:
                with w.engine.failing("fit","risk","integrate","minimize","linear"):
                    self.assertEqual(w.export_result(report["result_id"]),expected)
                    self.assertEqual(w.read_result(report["result_id"]),report)
                self.assertEqual(w.observations.revisions.resolve_current("mixed")[0]["rev"],2)
                self.assertEqual(expected["sources"][0]["rev"],1)

    def test_fit_publication_rollback(self):
        with Workflow() as w:
            w.put_batch("base",exact_rows())
            w.freeze("t",["exact"])
            for stage in ("result.payload","result.index","result.parent","commit"):
                with w.storage.failing(stage):
                    with self.assertRaises(sqlite3.OperationalError):
                        w.fit("t")
                for table in ("results","result_index","parents"):
                    self.assertEqual(w.storage.fetch_one("SELECT count(*) FROM "+table),(0,))
            fit = w.fit("t")
            with w.engine.failing("fit"):
                self.assertEqual(w.fit("t"),fit)

    def test_missing_and_mismatched_parent(self):
        for damage in ("link","source","model","task"):
            with self.subTest(damage=damage),Workflow() as w:
                fit = self._fit(w)
                report = w.risk(fit["result_id"],[0,0])
                if damage == "link":
                    w.storage.execute("DELETE FROM parents WHERE result_id=?",(report["result_id"],))
                elif damage == "source":
                    w.storage.execute("DELETE FROM revisions WHERE cohort_id='exact' AND sample_id='0'")
                elif damage == "model":
                    payload = dict(report,model_digest="wrong")
                    w.storage.execute("UPDATE results SET payload=? WHERE result_id=?",
                                      (encode_payload(payload),report["result_id"]))
                else:
                    w.storage.execute("DELETE FROM tasks WHERE task_id='task'")
                with self.assertRaises(IntegrityError):
                    w.export_result(report["result_id"])
