import math
from .gaussian_model import rectangle_probability
from .numerics import NumericalFailure

def joint_exceedance_probability(model, thresholds, engine):
    engine.guard("risk")
    sx,sy = [math.sqrt(model.covariance[j][j]) for j in range(2)]
    a = (thresholds[0]-model.mu[0])/sx
    b = (thresholds[1]-model.mu[1])/sy
    if a >= 12 or b >= 12:
        return 0.
    if a <= -12:
        return .5*math.erfc(b/math.sqrt(2))
    if b <= -12:
        return .5*math.erfc(a/math.sqrt(2))
    result = rectangle_probability(model,[(thresholds[0],math.inf),(thresholds[1],math.inf)],engine)
    if not math.isfinite(result):
        raise NumericalFailure("nonfinite joint tail")
    return result
