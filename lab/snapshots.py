from dataclasses import dataclass
from .observation_rules import normalize_batch
from .codecs import canonical_digest, clone_payload, IntegrityError
from .task_rules import normalize_selection

@dataclass(frozen=True)
class FrozenTask:
    task_id: str
    input_digest: str
    content: dict

    def payload(self):
        return clone_payload({"task_id":self.task_id,"input_digest":self.input_digest,"input":self.content})

def build_snapshot(task_id,selection,rows):
    content = {"representation":1,"selection":normalize_selection(selection),"rows":normalize_batch(rows)}
    return FrozenTask(task_id,canonical_digest(content),content).payload()

def decode_snapshot(payload):
    try:
        if set(payload) != {"task_id","input_digest","input"}:
            raise ValueError("snapshot fields")
        content = payload["input"]
        rebuilt = build_snapshot(payload["task_id"],content["selection"],content["rows"])
        if content.get("representation") != 1 or rebuilt != payload:
            raise ValueError("noncanonical snapshot")
        rows = content["rows"]
        if sorted({r["cohort_id"] for r in rows}) != content["selection"]:
            raise ValueError("incomplete selected cohort identities")
        keys = [(r["cohort_id"],r["sample_id"]) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("multiple revisions for frozen sample")
    except (KeyError,TypeError,ValueError) as exc:
        raise IntegrityError("invalid frozen task") from exc
    return clone_payload(payload)
