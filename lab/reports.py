from dataclasses import dataclass
from .codecs import canonical_digest, clone_payload
from .report_rules import normalize_thresholds, validate_probability

@dataclass(frozen=True)
class RiskReport:
    fit_id: str
    thresholds: tuple
    probability: float
    model_digest: str

    def payload(self):
        return {"kind":"report","fit_id":self.fit_id,"thresholds":list(self.thresholds),
                "probability":self.probability,"model_digest":self.model_digest}

def build_risk_report(fit,thresholds,probability):
    request = normalize_thresholds(thresholds)
    return RiskReport(fit["result_id"],tuple(request),validate_probability(probability),
                      canonical_digest(clone_payload(fit))).payload()
