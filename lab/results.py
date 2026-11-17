from .codecs import encode_payload, decode_payload, canonical_digest, clone_payload, IntegrityError
from .gaussian_model import GaussianParameters, finite_number

CONTRACT = "qingxi-bvn-1"

def request_key(kind,parent_id,thresholds=None,sample_key=None):
    request = {"contract":CONTRACT,"kind":kind,"parent_id":parent_id}
    if kind == "report":
        request["thresholds"] = list(thresholds)
    elif kind == "conditional_report":
        request["thresholds"] = list(thresholds)
        request["sample_key"] = list(sample_key)
    return canonical_digest(request)

class ResultArchive:
    def __init__(self,storage):
        self.storage = storage

    def _find(self,key):
        found = self.storage.fetch_one("SELECT result_id FROM result_index WHERE request_key=?",(key,))
        if found is None:
            return None
        result = self.read_result(found[0])
        if self._key(result) != key:
            raise IntegrityError("request index mismatch")
        return result

    def find_fit_for_task(self,task_id):
        return self._find(request_key("fit",task_id))

    def find_report(self,fit_id,thresholds):
        return self._find(request_key("report",fit_id,thresholds))

    def find_conditional_report(self,fit_id,sample_key,thresholds):
        return self._find(request_key("conditional_report",fit_id,thresholds,sample_key))

    def _key(self,payload):
        kind = payload["kind"]
        if kind == "fit":
            return request_key("fit",payload["task_id"])
        if kind == "report":
            return request_key("report",payload["fit_id"],payload["thresholds"])
        sample = payload["sample"]
        return request_key("conditional_report",payload["fit_id"],payload["thresholds"],
                           [sample["cohort_id"],sample["sample_id"]])

    def _publish(self,payload,parent_kind,parent_id):
        key = self._key(payload)
        result_id = payload["kind"]+":"+key
        payload = dict(payload,result_id=result_id,contract=CONTRACT)
        with self.storage.transaction():
            old = self._find(key)
            if old is not None:
                return old
            if parent_kind == "task":
                saved = self.storage.fetch_one("SELECT payload FROM tasks WHERE task_id=?",(parent_id,))
                if not saved or decode_payload(saved[0])["input_digest"] != payload["input_digest"]:
                    raise IntegrityError("fit parent task mismatch")
            else:
                parent = self.read_result(parent_id)
                if parent["kind"] != "fit" or canonical_digest(parent) != payload["model_digest"]:
                    raise IntegrityError("report parent model mismatch")
            self.storage.execute("INSERT INTO results VALUES(?,?,?)",(result_id,payload["kind"],encode_payload(payload)))
            self.storage.checkpoint("result.payload")
            self.storage.execute("INSERT INTO result_index VALUES(?,?)",(key,result_id))
            self.storage.checkpoint("result.index")
            self.storage.execute("INSERT INTO parents VALUES(?,?,?)",(result_id,parent_kind,parent_id))
            self.storage.checkpoint("result.parent")
        return self.read_result(result_id)

    def publish_fit(self,payload):
        if payload.get("kind") != "fit":
            raise ValueError("fit payload required")
        return self._publish(payload,"task",payload["task_id"])

    def publish_report(self,payload):
        if payload.get("kind") not in ("report","conditional_report"):
            raise ValueError("report payload required")
        return self._publish(payload,"fit",payload["fit_id"])

    def read_result(self,result_id):
        stored = self.storage.fetch_one("SELECT kind,payload FROM results WHERE result_id=?",(result_id,))
        if stored is None:
            raise KeyError("unknown result: "+result_id)
        result = decode_payload(stored[1])
        try:
            kind = result["kind"]
            if kind not in ("fit","report","conditional_report") or stored[0] != kind \
                    or result["contract"] != CONTRACT:
                raise ValueError("result kind or contract")
            key = self._key(result)
            if result["result_id"] != result_id or result_id != kind+":"+key:
                raise ValueError("result identity")
            index = self.storage.fetch_one("SELECT request_key FROM result_index WHERE result_id=?",(result_id,))
            if index != (key,):
                raise ValueError("result index")
            if kind == "fit":
                GaussianParameters.from_payload(result)
                finite_number(result["log_likelihood"])
                if result["numeric_status"]["converged"] is not True:
                    raise ValueError("unvalidated fit")
                parent = ("task",result["task_id"])
            else:
                thresholds = result["thresholds"]
                if len(thresholds) != 2:
                    raise ValueError("threshold dimension")
                for value in thresholds:
                    finite_number(value)
                probability = finite_number(result["probability"])
                if not 0 <= probability <= 1:
                    raise ValueError("probability range")
                parent = ("fit",result["fit_id"])
                if kind == "conditional_report":
                    denominator = finite_number(result["denominator"])
                    if denominator <= 0:
                        raise ValueError("conditional denominator must be positive")
                    sample = result["sample"]
                    if not isinstance(sample,dict) or ("x" not in sample and "region" not in sample):
                        raise ValueError("conditional sample row required")
                    for field in ("cohort_id","sample_id","rev","count"):
                        if field not in sample:
                            raise ValueError("conditional sample identity required")
                parent_fit = self.read_result(result["fit_id"])
                if parent_fit["kind"] != "fit" \
                        or canonical_digest(parent_fit) != result["model_digest"]:
                    raise ValueError("report parent model digest mismatch")
            actual = self.storage.fetch_one("SELECT parent_kind,parent_id FROM parents WHERE result_id=?",(result_id,))
            if actual != parent:
                raise ValueError("result parent link")
        except (KeyError,TypeError,ValueError,OverflowError) as exc:
            raise IntegrityError("invalid archived result") from exc
        return result
