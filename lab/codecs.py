import hashlib
import json
import math

ENCODING_VERSION = 1

class IntegrityError(ValueError):
    pass

def _clean(value):
    if value is None or isinstance(value,(str,bool)):
        return value
    if isinstance(value,int):
        return value
    if isinstance(value,float):
        if not math.isfinite(value):
            raise ValueError("nonfinite payload")
        return 0. if value == 0 else value
    if isinstance(value,(list,tuple)):
        return [_clean(v) for v in value]
    if isinstance(value,dict) and all(isinstance(k,str) for k in value):
        return {k:_clean(v) for k,v in value.items()}
    raise ValueError("JSON payload types required")

def _json(value):
    return json.dumps(_clean(value),sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)

def canonical_digest(payload):
    return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()

def encode_payload(payload):
    body = _clean(payload)
    return _json({"version":ENCODING_VERSION,"body":body,"digest":canonical_digest(body)})

def decode_payload(encoded):
    def unique(pairs):
        result = {}
        for k,v in pairs:
            if k in result:
                raise IntegrityError("duplicate JSON key")
            result[k] = v
        return result
    try:
        envelope = json.loads(encoded,object_pairs_hook=unique,
                              parse_constant=lambda _: (_ for _ in ()).throw(IntegrityError("nonfinite JSON")))
        if set(envelope) != {"version","body","digest"} or type(envelope["version"]) is not int:
            raise IntegrityError("invalid envelope")
        if envelope["version"] != ENCODING_VERSION:
            raise IntegrityError("unsupported encoding version")
        if canonical_digest(envelope["body"]) != envelope["digest"]:
            raise IntegrityError("payload digest mismatch")
        return envelope["body"]
    except (TypeError,KeyError,json.JSONDecodeError,UnicodeError) as exc:
        raise IntegrityError("invalid encoding") from exc

def clone_payload(payload):
    return decode_payload(encode_payload(payload))
