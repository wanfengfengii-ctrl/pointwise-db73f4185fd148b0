import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from lab.workflow import demonstrate,Workflow
from samples import exact_rows,E

class TestWorkflow(unittest.TestCase):
    def test_workflow_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/"lab.sqlite")
            demonstrate(path)
            with Workflow(path) as w:
                before = list(w.storage.connection.iterdump())
                w.storage.initialize()
                self.assertEqual(list(w.storage.connection.iterdump()),before)

    def test_cli_real_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory/"rows.json"
            source.write_text(json.dumps(exact_rows()),encoding="utf-8")
            prefix = [sys.executable,"-B","-m","lab.workflow","--db",str(directory/"lab.sqlite")]
            environment = dict(os.environ,PYTHONDONTWRITEBYTECODE="1")
            def run(*arguments,code=0):
                process = subprocess.run(prefix+list(arguments),capture_output=True,text=True,
                                         env=environment,timeout=90)
                self.assertEqual(process.returncode,code,process.stdout+process.stderr)
                return json.loads(process.stdout)
            run("put","first",str(source))
            run("freeze","task","exact")
            fit = run("fit","task")
            report = run("risk",fit["result_id"],"0","0")
            self.assertAlmostEqual(report["probability"],1/3,delta=1e-7)
            self.assertEqual(run("read",report["result_id"]),report)
            evidence = run("export",report["result_id"])
            self.assertEqual(evidence["task"]["task_id"],"task")
            row = exact_rows()[0]
            row["rev"],row["x"] = 2,[E(2),E(2)]
            source.write_text(json.dumps([row]),encoding="utf-8")
            run("put","corrected",str(source))
            self.assertEqual(run("export",report["result_id"]),evidence)
            run("freeze","task","exact",code=2)
            run("risk",fit["result_id"],"nan","0",code=2)
