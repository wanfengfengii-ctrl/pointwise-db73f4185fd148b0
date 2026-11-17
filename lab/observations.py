from .observation_rules import identity, normalize_batch
from .revisions import RevisionStore
from .batch_ledger import BatchLedger
from .codecs import clone_payload

class ObservationService:
    def __init__(self,storage):
        self.storage = storage
        self.revisions = RevisionStore(storage)
        self.ledger = BatchLedger(storage)

    def put_batch(self,batch_id,rows):
        identity(batch_id)
        normalized = normalize_batch(rows)
        with self.storage.transaction():
            previous = self.ledger.lookup(batch_id,normalized)
            if previous is not None:
                return previous
            self.revisions.stage_revisions(normalized)
            keys = sorted({(r["cohort_id"],r["sample_id"]) for r in normalized})
            current = []
            for cohort,sample in keys:
                rev = self.storage.fetch_one(
                    "SELECT rev FROM current WHERE cohort_id=? AND sample_id=?",(cohort,sample))[0]
                current.append({"cohort_id":cohort,"sample_id":sample,"rev":rev})
            result = {"batch_id":batch_id,"revisions":[{k:r[k] for k in ("cohort_id","sample_id","rev")}
                                                     for r in normalized],"current":current}
            self.ledger.record(batch_id,normalized,result)
            self.storage.checkpoint("batch")
        return clone_payload(result)
