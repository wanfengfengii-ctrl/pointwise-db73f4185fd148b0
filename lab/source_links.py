from dataclasses import dataclass
from .codecs import canonical_digest, IntegrityError
from .revisions import RevisionStore

@dataclass(frozen=True)
class SourceLink:
    task_id: str
    ordinal: int
    cohort_id: str
    sample_id: str
    rev: int
    row_digest: str

def bind_source_links(storage,task):
    for ordinal,row in enumerate(task["input"]["rows"]):
        link = SourceLink(task["task_id"],ordinal,row["cohort_id"],row["sample_id"],row["rev"],canonical_digest(row))
        storage.execute("INSERT INTO source_links VALUES(?,?,?,?,?,?)",tuple(vars(link).values()))

def verify_source_links(storage,task):
    links = storage.fetch_all("SELECT ordinal,cohort_id,sample_id,rev,row_digest FROM source_links WHERE task_id=? ORDER BY ordinal",
                              (task["task_id"],))
    rows = task["input"]["rows"]
    if len(links) != len(rows):
        raise IntegrityError("missing task source links")
    evidence = []
    revisions = RevisionStore(storage)
    for ordinal,(link,row) in enumerate(zip(links,rows)):
        expected = (ordinal,row["cohort_id"],row["sample_id"],row["rev"],canonical_digest(row))
        if link != expected:
            raise IntegrityError("source link mismatch")
        original = revisions.read_revision(*link[1:4])
        if original != row:
            raise IntegrityError("frozen row differs from original revision")
        evidence.append(original)
    return evidence
