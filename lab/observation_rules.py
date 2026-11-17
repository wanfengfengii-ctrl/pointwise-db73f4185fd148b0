from dataclasses import dataclass
from .gaussian_model import finite_number

def identity(value):
    if not isinstance(value,str) or not value.strip():
        raise ValueError("nonempty string identity required")
    return value

def positive_integer(value):
    if type(value) is not int or value <= 0:
        raise ValueError("positive nonboolean integer required")
    return value

@dataclass(frozen=True)
class Constraint:
    kind: str
    lower: float | None = None
    upper: float | None = None
    value: float | None = None

    def payload(self):
        return {k:v for k,v in vars(self).items() if v is not None}

@dataclass(frozen=True)
class ObservationRow:
    cohort_id: str
    sample_id: str
    rev: int
    x: tuple
    count: int

    def payload(self):
        return {"cohort_id":self.cohort_id,"sample_id":self.sample_id,"rev":self.rev,
                "x":[c.payload() for c in self.x],"count":self.count}

def normalize_constraint(raw):
    if not isinstance(raw,dict):
        raise ValueError("constraint object required")
    fields = {"exact":{"kind","value"},"left":{"kind","upper"},
              "right":{"kind","lower"},"interval":{"kind","lower","upper"}}
    kind = raw.get("kind")
    if not isinstance(kind,str) or kind not in fields or set(raw) != fields[kind]:
        raise ValueError("constraint kind and fields disagree")
    values = {k:finite_number(v) for k,v in raw.items() if k != "kind"}
    if kind == "interval" and values["lower"] >= values["upper"]:
        raise ValueError("interval lower must be less than upper")
    return Constraint(kind,**values)

def normalize_row(raw):
    required = {"cohort_id","sample_id","rev","x","count"}
    if not isinstance(raw,dict) or set(raw) != required:
        raise ValueError("observation fields")
    if not isinstance(raw["x"],(list,tuple)) or len(raw["x"]) != 2:
        raise ValueError("two ordered constraints required")
    return ObservationRow(identity(raw["cohort_id"]),identity(raw["sample_id"]),
                          positive_integer(raw["rev"]),tuple(normalize_constraint(c) for c in raw["x"]),
                          positive_integer(raw["count"]))

def normalize_batch(rows):
    if not isinstance(rows,(list,tuple)) or not rows:
        raise ValueError("nonempty row list required")
    unique = {}
    for raw in rows:
        row = normalize_row(raw)
        key = row.cohort_id,row.sample_id,row.rev
        if key in unique and unique[key] != row:
            raise ValueError("conflicting revisions within batch")
        unique[key] = row
    return [unique[k].payload() for k in sorted(unique)]
