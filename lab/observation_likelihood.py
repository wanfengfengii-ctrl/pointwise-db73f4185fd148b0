import math
from .gaussian_model import joint_log_density, conditional_normal, normal_interval_probability, rectangle_probability, LOG2PI
from .numerics import NumericalFailure
from .regions import region_log_likelihood

def bounds(constraint):
    kind = constraint["kind"]
    if kind == "left":
        return -math.inf,constraint["upper"]
    if kind == "right":
        return constraint["lower"],math.inf
    if kind == "interval":
        return constraint["lower"],constraint["upper"]
    return constraint["value"],constraint["value"]

def observation_log_likelihood(row, model, engine, enforce_truncation=False):
    if "region" in row:
        return region_log_likelihood(row["region"],model,engine,enforce_truncation)
    constraints = row["x"]
    exact = [j for j,c in enumerate(constraints) if c["kind"] == "exact"]
    if len(exact) == 2:
        return joint_log_density(model,constraints[0]["value"],constraints[1]["value"])
    if len(exact) == 1:
        j = exact[0]
        value = constraints[j]["value"]
        variance = model.covariance[j][j]
        marginal = -.5*(LOG2PI+math.log(variance)+(value-model.mu[j])**2/variance)
        mean,sd = conditional_normal(model,j,value)
        lo,hi = bounds(constraints[1-j])
        probability = normal_interval_probability((lo-mean)/sd,(hi-mean)/sd)
    else:
        marginal = 0.
        probability = rectangle_probability(model,[bounds(c) for c in constraints],engine)
    if probability < 1e-14 or not math.isfinite(probability):
        raise NumericalFailure("observation probability below numerical support")
    return marginal+math.log(probability)

def weighted_log_likelihood(rows, model, engine, enforce_truncation=False):
    return math.fsum(row["count"]*observation_log_likelihood(row,model,engine,enforce_truncation)
                     for row in rows)
