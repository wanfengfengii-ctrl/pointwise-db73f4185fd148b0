from .codecs import canonical_digest, encode_payload, decode_payload

def canonical_batch_digest(rows):
    return canonical_digest({"representation":1,"rows":rows})

class BatchLedger:
    def __init__(self,storage):
        self.storage = storage

    def lookup(self,batch_id,rows):
        saved = self.storage.fetch_one("SELECT payload,result FROM batches WHERE batch_id=?",(batch_id,))
        if saved is None:
            return None
        original = decode_payload(saved[0])
        candidate = {"digest":canonical_batch_digest(rows),"rows":rows}
        if original != candidate:
            raise ValueError("batch identity conflict")
        return decode_payload(saved[1])

    def record(self,batch_id,rows,result):
        payload = {"digest":canonical_batch_digest(rows),"rows":rows}
        self.storage.execute("INSERT INTO batches VALUES(?,?,?)",
                             (batch_id,encode_payload(payload),encode_payload(result)))
