from .observation_rules import identity
from .codecs import IntegrityError, canonical_digest

def normalize_selection(selection):
    if not isinstance(selection,(list,tuple)) or not selection:
        raise ValueError("nonempty cohort selection required")
    return sorted({identity(item) for item in selection})

def validate_task_binding(existing,candidate):
    if existing != candidate:
        raise ValueError("task identity already bound to different frozen input")
    if existing["input_digest"] != canonical_digest(existing["input"]):
        raise IntegrityError("task digest mismatch")
    return existing
