import math
from dataclasses import dataclass
from .numerics import NumericalFailure

ROOT2 = math.sqrt(2.)
LOG2PI = math.log(2*math.pi)

def finite_number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)):
        raise ValueError("finite binary64 number required")
    try:
        result = float(value)
    except (OverflowError,ValueError) as exc:
        raise ValueError("finite binary64 number required") from exc
    if not math.isfinite(result):
        raise ValueError("finite binary64 number required")
    return 0. if result == 0 else result

def validate_covariance(covariance):
    if len(covariance) != 2 or any(len(row) != 2 for row in covariance):
        raise ValueError("2 by 2 covariance required")
    a,b,c,d = [finite_number(v) for row in covariance for v in row]
    if b != c or a <= 0 or d <= 0 or a*d-b*b <= 0:
        raise ValueError("symmetric positive definite covariance required")
    return ((a,b),(c,d))

@dataclass(frozen=True)
class GaussianParameters:
    mu: tuple
    covariance: tuple

    def __post_init__(self):
        if len(self.mu) != 2:
            raise ValueError("two means required")
        object.__setattr__(self,"mu",tuple(finite_number(v) for v in self.mu))
        object.__setattr__(self,"covariance",validate_covariance(self.covariance))

    def payload(self):
        return {"mu":list(self.mu),"covariance":[list(row) for row in self.covariance]}

    @classmethod
    def from_payload(cls, payload):
        return cls(payload["mu"],payload["covariance"])

def normal_interval_probability(lower, upper):
    if lower >= upper:
        return 0.
    if lower >= 0:
        return .5*(math.erfc(lower/ROOT2)-math.erfc(upper/ROOT2))
    if upper <= 0:
        return .5*(math.erfc(-upper/ROOT2)-math.erfc(-lower/ROOT2))
    return 1-.5*math.erfc(-lower/ROOT2)-.5*math.erfc(upper/ROOT2)

def conditional_normal(model, exact_axis, value):
    other = 1-exact_axis
    c = model.covariance
    slope = c[other][exact_axis]/c[exact_axis][exact_axis]
    mean = model.mu[other]+slope*(value-model.mu[exact_axis])
    variance = c[other][other]-slope*c[exact_axis][other]
    return mean, math.sqrt(variance)

def joint_log_density(model, x, y):
    a,b = model.covariance[0]
    d = model.covariance[1][1]
    det = a*d-b*b
    u,v = x-model.mu[0],y-model.mu[1]
    return -LOG2PI-.5*math.log(det)-.5*(d*u*u-2*b*u*v+a*v*v)/det

def rectangle_probability(model, bounds, engine):
    sx = math.sqrt(model.covariance[0][0])
    a,b = ((v-model.mu[0])/sx for v in bounds[0])
    a,b = max(-12.,a),min(12.,b)
    if a >= b:
        raise NumericalFailure("rectangle outside resolved probability range")
    def integrand(z):
        mean,sd = conditional_normal(model,0,model.mu[0]+sx*z)
        lo,hi = bounds[1]
        p = normal_interval_probability((lo-mean)/sd,(hi-mean)/sd)
        return math.exp(-z*z/2)/math.sqrt(2*math.pi)*p
    result = engine.integrate_1d(integrand,a,b)
    budget = min(2e-13,max(abs(result.value)*2e-13,1e-28))
    if result.error > budget:
        result = engine.integrate_1d(integrand,a,b,tolerance=budget)
    if not result.converged or result.error > budget:
        raise NumericalFailure("unresolved rectangle probability")
    return result.value
