from .gaussian_model import finite_number, GaussianParameters
from .codecs import IntegrityError
from .numerics import NumericalFailure

def normalize_thresholds(thresholds):
    if not isinstance(thresholds,(list,tuple)) or len(thresholds) != 2:
        raise ValueError("two ordered finite thresholds required")
    return [finite_number(value) for value in thresholds]

def validate_probability(probability):
    probability = finite_number(probability)
    if probability < -3e-13 or probability > 1+3e-13:
        raise NumericalFailure("joint probability outside rounding budget")
    return max(0.,min(1.,probability))

def validate_fit_reference(fit,fit_id):
    if fit.get("kind") != "fit" or fit.get("result_id") != fit_id:
        raise IntegrityError("published fit identity required")
    return GaussianParameters.from_payload(fit)
