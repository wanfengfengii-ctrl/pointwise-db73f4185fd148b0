from .report_rules import normalize_thresholds, validate_fit_reference
from .joint_tail import joint_exceedance_probability
from .reports import build_risk_report

class RiskService:
    def __init__(self,archive,engine):
        self.archive,self.engine = archive,engine

    def risk(self,fit_id,thresholds):
        thresholds = normalize_thresholds(thresholds)
        old = self.archive.find_report(fit_id,thresholds)
        if old is not None:
            return old
        fit = self.archive.read_result(fit_id)
        model = validate_fit_reference(fit,fit_id)
        probability = joint_exceedance_probability(model,thresholds,self.engine)
        return self.archive.publish_report(build_risk_report(fit,thresholds,probability))
