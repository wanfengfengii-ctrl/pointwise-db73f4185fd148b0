import math
from dataclasses import dataclass
from .gaussian_model import GaussianParameters
from .observation_likelihood import weighted_log_likelihood, observation_log_likelihood
from .numerics import NumericalFailure
from .regions import representative_point


def _pattern_key(row):
    if "region" in row:
        return "region:"+str(row["region"])
    return "x:"+str(row["x"])


def _row_pairs(row):
    if "region" in row:
        x, y = representative_point(row["region"])
        return (x, y), row["count"]
    values = []
    for c in row["x"]:
        values.append(c.get("value",c.get("upper",c.get("lower"))))
        if c["kind"] == "interval":
            values[-1] = (c["lower"]+c["upper"])/2
    return tuple(values), row["count"]

@dataclass(frozen=True)
class FitCandidate:
    model: GaussianParameters
    log_likelihood: float
    status: dict

def _encode(model):
    a,b = model.covariance[0]
    d = model.covariance[1][1]
    return [*model.mu,.5*math.log(a),.5*math.log(d),math.atanh(b/math.sqrt(a*d))]

def _decode(point):
    m,n,u,v,z = point
    a,d = math.exp(2*u),math.exp(2*v)
    b = math.tanh(z)*math.sqrt(a*d)
    return GaussianParameters((m,n),((a,b),(b,d)))

def _initial(rows):
    pairs = [_row_pairs(row) for row in rows]
    total = sum(n for _,n in pairs)
    mu = [sum(p[j]*n for p,n in pairs)/total for j in range(2)]
    cov = [[sum(n*(p[i]-mu[i])*(p[j]-mu[j]) for p,n in pairs)/total
            for j in range(2)] for i in range(2)]
    for j in range(2):
        cov[j][j] = max(cov[j][j],.1)
    limit = .8*math.sqrt(cov[0][0]*cov[1][1])
    cov[0][1] = cov[1][0] = max(-limit,min(limit,cov[0][1]))
    return GaussianParameters(mu,cov)

def validate_fit_candidate(candidate):
    model = candidate.model
    if not math.isfinite(candidate.log_likelihood):
        raise NumericalFailure("nonfinite likelihood")
    sd = [math.sqrt(model.covariance[j][j]) for j in range(2)]
    rho = model.covariance[0][1]/(sd[0]*sd[1])
    if max(map(abs,model.mu)) > 8 or min(sd) < .2 or max(sd) > 4 or abs(rho) > .9:
        raise NumericalFailure("outside published numerical guarantee")
    if candidate.status["parameter_error_estimate"] > 2e-7:
        raise NumericalFailure("unresolved parameter accuracy")
    return candidate

def solve_joint_mle(rows, engine):
    engine.guard("fit")
    if not rows:
        raise NumericalFailure("empty analysis")
    patterns = {_pattern_key(r) for r in rows}
    if len(patterns) > 40:
        raise NumericalFailure("at most 40 observation patterns")
    total = sum(r["count"] for r in rows)
    start = _initial(rows)
    if all("x" in r and all(c["kind"] == "exact" for c in r["x"]) for r in rows):
        mu = [sum(r["count"]*r["x"][j]["value"] for r in rows)/total for j in range(2)]
        covariance = [[sum(r["count"]*(r["x"][i]["value"]-mu[i])*(r["x"][j]["value"]-mu[j])
                       for r in rows)/total for j in range(2)] for i in range(2)]
        try:
            model = GaussianParameters(mu,covariance)
        except ValueError as exc:
            raise NumericalFailure("noninternal exact-data MLE") from exc
        return validate_fit_candidate(FitCandidate(model,weighted_log_likelihood(rows,model,engine),
            {"method":"exact_mle","parameter_error_estimate":0.,"converged":True}))
    def objective(point):
        try:
            return -weighted_log_likelihood(rows,_decode(point),engine)/total
        except NumericalFailure:
            return math.inf
    result = engine.minimize(objective,_encode(start))
    if not result.converged:
        raise NumericalFailure("optimizer did not converge")
    point = list(result.value)
    correction = math.inf
    inverse_norm = math.inf
    for _ in range(4):
        gradient = engine.gradient(objective,point)
        hessian = engine.hessian(objective,point)
        inverse = [engine.solve_linear(hessian,[float(i==j) for i in range(5)]) for j in range(5)]
        inverse_norm = max(sum(abs(inverse[j][i]) for j in range(5)) for i in range(5))
        step = engine.solve_linear(hessian,gradient)
        correction = max(map(abs,step))
        if correction < 2e-10:
            break
        point = [a-b for a,b in zip(point,step)]
    model = _decode(point)
    if inverse_norm > 100 or correction > 2e-8:
        raise NumericalFailure("ill-conditioned optimum")
    for i in range(5):
        if hessian[i][i] <= 0:
            raise NumericalFailure("nonminimum candidate")
    # Verify positive definiteness by Cholesky, rather than diagonal signs alone.
    lower = [[0.]*5 for _ in range(5)]
    for i in range(5):
        for j in range(i+1):
            value = hessian[i][j]-sum(lower[i][k]*lower[j][k] for k in range(j))
            if i == j:
                if value <= .002:
                    raise NumericalFailure("curvature outside guarantee")
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = value/lower[j][j]
    model_scale = max(1.,*(abs(v) for row in model.covariance for v in row))
    error = max(correction, inverse_norm*2e-11)*8*model_scale
    # Publication-time check: every affine region must retain at least the
    # guaranteed truncation probability in the optimum neighborhood.
    for row in rows:
        if "region" in row:
            observation_log_likelihood(row,model,engine,enforce_truncation=True)
    candidate = FitCandidate(model,weighted_log_likelihood(rows,model,engine,
                                                           enforce_truncation=True),
        {"method":"joint_likelihood","converged":True,
         "parameter_error_estimate":error,"score_norm":max(map(abs,gradient)),
         "inverse_hessian_norm":inverse_norm,"iterations":result.iterations})
    return validate_fit_candidate(candidate)
