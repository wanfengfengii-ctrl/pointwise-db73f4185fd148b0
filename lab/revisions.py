from .codecs import encode_payload, decode_payload, IntegrityError
from .observation_rules import normalize_row

class RevisionStore:
    def __init__(self,storage):
        self.storage = storage

    def stage_revisions(self,rows):
        for row in rows:
            key = row["cohort_id"],row["sample_id"],row["rev"]
            existing = self.storage.fetch_one(
                "SELECT payload FROM revisions WHERE cohort_id=? AND sample_id=? AND rev=?",key)
            if existing and decode_payload(existing[0]) != row:
                raise ValueError("source revision identity conflict")
            if not existing:
                self.storage.execute("INSERT INTO revisions VALUES(?,?,?,?)",(*key,encode_payload(row)))
            self.storage.execute(
                "INSERT INTO current VALUES(?,?,?) ON CONFLICT(cohort_id,sample_id) DO UPDATE SET rev=excluded.rev WHERE excluded.rev>current.rev",key)

    def resolve_current(self,cohort_id):
        records = self.storage.fetch_all(
            "SELECT sample_id,rev FROM current WHERE cohort_id=? ORDER BY sample_id",(cohort_id,))
        return [self.read_revision(cohort_id,sample,rev) for sample,rev in records]

    def read_revision(self,cohort_id,sample_id,rev):
        stored = self.storage.fetch_one(
            "SELECT payload FROM revisions WHERE cohort_id=? AND sample_id=? AND rev=?",(cohort_id,sample_id,rev))
        if stored is None:
            raise IntegrityError("missing source revision")
        row = decode_payload(stored[0])
        if (row.get("cohort_id"),row.get("sample_id"),row.get("rev")) != (cohort_id,sample_id,rev):
            raise IntegrityError("source identity mismatch")
        try:
            if normalize_row(row).payload() != row:
                raise ValueError("noncanonical row")
        except (TypeError,ValueError) as exc:
            raise IntegrityError("invalid source revision") from exc
        return row
