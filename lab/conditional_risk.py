import math
from .observation_rules import identity
from .report_rules import normalize_thresholds, validate_fit_reference
from .regions import region_measure, row_to_region
from .conditional_reports import build_conditional_report
from .numerics import NumericalFailure

def normalize_sample_key(sample_key):
    if not isinstance(sample_key,(list,tuple)) or len(sample_key) != 2:
        raise ValueError("sample_key must be [cohort_id, sample_id]")
    return [identity(sample_key[0]),identity(sample_key[1])]

class ConditionalRiskService:
    def __init__(self,archive,tasks,engine):
        self.archive,self.tasks,self.engine = archive,tasks,engine

    def conditional_risk(self,fit_id,sample_key,thresholds):
        sample_key = normalize_sample_key(sample_key)
        thresholds = normalize_thresholds(thresholds)
        old = self.archive.find_conditional_report(fit_id,sample_key,thresholds)
        if old is not None:
            return old
        fit = self.archive.read_result(fit_id)
        model = validate_fit_reference(fit,fit_id)
        task = self.tasks.read_frozen(fit["task_id"])
        sample = None
        for row in task["input"]["rows"]:
            if [row["cohort_id"],row["sample_id"]] == sample_key:
                sample = row
                break
        if sample is None:
            raise KeyError("sample not present in the frozen parent task: "+str(sample_key))
        self.engine.guard("risk")
        region = sample["region"] if "region" in sample else row_to_region(sample)
        numerator,denominator,_full = region_measure(region,model,self.engine,thresholds)
        if not math.isfinite(denominator) or denominator <= 0:
            raise NumericalFailure("conditional denominator is not resolvable")
        probability = 0. if numerator <= 0 else numerator/denominator
        payload = build_conditional_report(fit,sample,thresholds,denominator,probability)
        return self.archive.publish_report(payload)
