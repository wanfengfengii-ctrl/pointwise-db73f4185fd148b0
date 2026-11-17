import argparse
import json
import sqlite3
import tempfile
from pathlib import Path
from .storage import Storage
from .numerics import NumericEngine, NumericalFailure
from .observations import ObservationService
from .tasks import TaskService
from .results import ResultArchive
from .fitting import FitService
from .risk import RiskService
from .conditional_risk import ConditionalRiskService
from .evidence import EvidenceExporter

class Workflow:
    def __init__(self,path=":memory:",engine=None):
        self.storage = Storage(path)
        self.engine = engine if engine is not None else NumericEngine()
        self.observations = ObservationService(self.storage)
        self.tasks = TaskService(self.storage)
        self.archive = ResultArchive(self.storage)
        self.fitting = FitService(self.tasks,self.archive,self.engine)
        self.risks = RiskService(self.archive,self.engine)
        self.conditional = ConditionalRiskService(self.archive,self.tasks,self.engine)
        self.evidence = EvidenceExporter(self.archive,self.tasks,self.storage)

    def put_batch(self,batch_id,rows):
        return self.observations.put_batch(batch_id,rows)

    def freeze(self,task_id,cohort_ids):
        return self.tasks.freeze(task_id,cohort_ids)

    def fit(self,task_id):
        return self.fitting.fit(task_id)

    def risk(self,fit_id,thresholds):
        return self.risks.risk(fit_id,thresholds)

    def conditional_risk(self,fit_id,sample_key,thresholds):
        return self.conditional.conditional_risk(fit_id,sample_key,thresholds)

    def read_result(self,result_id):
        return self.archive.read_result(result_id)

    def export_result(self,report_id):
        return self.evidence.export_result(report_id)

    def close(self):
        self.storage.close()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()

def demonstrate(path):
    from samples import exact_rows,mixed_rows,E,REFERENCE
    with Workflow(path) as workflow:
        rows = exact_rows()
        workflow.put_batch("exact-day",rows)
        workflow.freeze("exact-day",["exact"])
        fit = workflow.fit("exact-day")
        report = workflow.risk(fit["result_id"],[0,0])
        assert abs(report["probability"]-1/3) < 1e-7
        workflow.put_batch("mixed-day",mixed_rows())
        workflow.freeze("mixed-day",["mixed"])
        mixed = workflow.fit("mixed-day")
        joint = workflow.risk(mixed["result_id"],REFERENCE["thresholds"])
        for a,b in zip(mixed["mu"],REFERENCE["mu"]):
            assert abs(a-b) < 1e-6
        assert abs(joint["probability"]-REFERENCE["probability"]) < 1e-7
        evidence = workflow.export_result(joint["result_id"])
        corrected = mixed_rows()[0]
        corrected["rev"],corrected["x"] = 2,[E(2),E(2)]
        workflow.put_batch("corrected-day", [corrected])
        workflow.freeze("corrected-day",["mixed"])
        new_fit = workflow.fit("corrected-day")
        new_report = workflow.risk(new_fit["result_id"],REFERENCE["thresholds"])
        assert workflow.export_result(joint["result_id"]) == evidence
    with Workflow(path) as reopened:
        with reopened.engine.failing("fit","risk","integrate","minimize"):
            assert reopened.export_result(joint["result_id"]) == evidence
        print("POINTWISE_SCENARIO "+json.dumps({
            "case_id":"v8.workflow_reopen","input":{"mixed":mixed_rows(),"thresholds":REFERENCE["thresholds"]},
            "expected":{"mu":REFERENCE["mu"],"covariance":REFERENCE["covariance"]},
            "actual":{"mu":mixed["mu"],"covariance":mixed["covariance"]},
            "downstream_expected":{"old_probability":REFERENCE["probability"],"old_evidence_unchanged":True},
            "downstream_actual":{"old_probability":joint["probability"],"old_evidence_unchanged":True,
                                 "new_probability":new_report["probability"],"exact_probability":report["probability"]},
            "code_paths":["lab/observations.py","lab/tasks.py","lab/fit_solver.py","lab/risk.py","lab/evidence.py"],
            "oracle":"six-point analytic MLE and quadrant formula; independently checked mixed reference; exact evidence equality"
        },sort_keys=True))

def main(argv=None):
    parser = argparse.ArgumentParser(description="Qingxi fixed-cohort analysis")
    parser.add_argument("--db")
    parser.add_argument("command",choices=["put","freeze","fit","risk","read","export","demo"])
    parser.add_argument("arguments",nargs="*")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            if args.arguments:
                raise ValueError("demo takes no positional arguments")
            if args.db:
                demonstrate(args.db)
            else:
                with tempfile.TemporaryDirectory(prefix="qingxi-demo-") as directory:
                    demonstrate(str(Path(directory)/"lab.sqlite"))
            return 0
        if not args.db:
            raise ValueError("--db is required for persistent operations")
        with Workflow(args.db) as workflow:
            a = args.arguments
            if args.command == "put" and len(a) == 2:
                rows = json.loads(Path(a[1]).read_text(encoding="utf-8"))
                result = workflow.put_batch(a[0],rows)
            elif args.command == "freeze" and len(a) >= 2:
                result = workflow.freeze(a[0],a[1:])
            elif args.command == "fit" and len(a) == 1:
                result = workflow.fit(a[0])
            elif args.command == "risk" and len(a) == 3:
                result = workflow.risk(a[0],[float(a[1]),float(a[2])])
            elif args.command == "read" and len(a) == 1:
                result = workflow.read_result(a[0])
            elif args.command == "export" and len(a) == 1:
                result = workflow.export_result(a[0])
            else:
                raise ValueError("wrong command arguments")
        print(json.dumps(result,sort_keys=True,allow_nan=False))
        return 0
    except (ValueError,KeyError,NumericalFailure,sqlite3.Error,OSError) as exc:
        print(json.dumps({"error":str(exc)},ensure_ascii=False))
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
