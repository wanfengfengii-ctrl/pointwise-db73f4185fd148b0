from dataclasses import dataclass
import math
from .codecs import canonical_digest, clone_payload
from .report_rules import normalize_thresholds, validate_probability

@dataclass(frozen=True)
class ConditionalReport:
    fit_id: str
    sample: dict
    thresholds: tuple
    denominator: float
    probability: float
    model_digest: str

    def payload(self):
        return {"kind":"conditional_report","fit_id":self.fit_id,
                "sample":clone_payload(self.sample),"thresholds":list(self.thresholds),
                "denominator":self.denominator,"probability":self.probability,
                "model_digest":self.model_digest}

def build_conditional_report(fit,sample,thresholds,denominator,probability):
    request = normalize_thresholds(thresholds)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("positive finite denominator required")
    return ConditionalReport(fit["result_id"],clone_payload(sample),tuple(request),
                             denominator,validate_probability(probability),
                             canonical_digest(clone_payload(fit))).payload()
