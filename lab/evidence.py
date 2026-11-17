from .codecs import canonical_digest, clone_payload, IntegrityError
from .source_links import verify_source_links
from .results import CONTRACT

def verify_evidence_chain(report,fit,task,sources):
    try:
        if report["kind"] != "report" or fit["kind"] != "fit":
            raise ValueError("report and fit required")
        if report["fit_id"] != fit["result_id"] or fit["task_id"] != task["task_id"]:
            raise ValueError("parent identity mismatch")
        if report["contract"] != CONTRACT or fit["contract"] != CONTRACT:
            raise ValueError("computation contract mismatch")
        if report["model_digest"] != canonical_digest(fit) or fit["input_digest"] != task["input_digest"]:
            raise ValueError("parent digest mismatch")
        if sources != task["input"]["rows"] or canonical_digest(task["input"]) != task["input_digest"]:
            raise ValueError("source evidence mismatch")
    except (KeyError,TypeError,ValueError) as exc:
        raise IntegrityError("incomplete evidence chain") from exc
    return True

class EvidenceExporter:
    def __init__(self,archive,tasks,storage):
        self.archive,self.tasks,self.storage = archive,tasks,storage

    def export_result(self,report_id):
        try:
            report = self.archive.read_result(report_id)
            if report["kind"] != "report":
                raise ValueError("only risk reports have evidence exports")
            fit = self.archive.read_result(report["fit_id"])
            task = self.tasks.read_frozen(fit["task_id"])
            sources = verify_source_links(self.storage,task)
            verify_evidence_chain(report,fit,task,sources)
        except (KeyError,ValueError) as exc:
            raise IntegrityError("cannot export incomplete report evidence") from exc
        return clone_payload({"report":report,"fit":fit,"task":task,"sources":sources})
