"""Affine detection regions: exact rational geometry and Gaussian Hausdorff integrals.

A region is the intersection of at most six half planes (``le``) and at most two
equality planes (``eq``).  Geometric validity is decided on the exact rational
values of the converted binary64 inputs, never with floating point tolerances.

Observation likelihoods are Gaussian integrals over the region with respect to
its (2-r)-dimensional Euclidean Hausdorff measure:

* rank 0: two dimensional area integral;
* rank 1: Euclidean arc length integral along the equality line;
* rank 2: joint density at the single feasible point.
"""
import math
from dataclasses import dataclass
from fractions import Fraction

from .gaussian_model import (
    finite_number, joint_log_density, conditional_normal,
    normal_interval_probability, LOG2PI,
)
from .numerics import NumericalFailure

TRUNCATION_FLOOR = 1e-8


# ---------------------------------------------------------------------------
# Source normalization and exact geometry
# ---------------------------------------------------------------------------

def _coefficient(value):
    if type(value) is not int or isinstance(value, bool):
        raise ValueError("region coefficients must be nonboolean integers")
    if abs(value) > 16:
        raise ValueError("region coefficients must have absolute value at most 16")
    return value


def normalize_plane(raw):
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError("plane must be a three element list [a,b,c]")
    a = _coefficient(raw[0])
    b = _coefficient(raw[1])
    if a == 0 and b == 0:
        raise ValueError("plane normal must be nonzero")
    c = finite_number(raw[2])
    return [a, b, c]


def normalize_region(raw):
    if not isinstance(raw, dict) or set(raw) != {"le", "eq"}:
        raise ValueError("region requires exactly le and eq")
    le_raw, eq_raw = raw["le"], raw["eq"]
    if not isinstance(le_raw, list) or not isinstance(eq_raw, list):
        raise ValueError("region le and eq must be lists")
    if len(le_raw) > 6 or len(eq_raw) > 2:
        raise ValueError("region allows at most six half planes and two equalities")
    region = {"le": [normalize_plane(plane) for plane in le_raw],
              "eq": [normalize_plane(plane) for plane in eq_raw]}
    analyze_region(region)
    return region


@dataclass(frozen=True)
class RegionGeometry:
    rank: int
    base: tuple | None = None          # rank 1: exact point on the line (Fraction)
    direction: tuple | None = None     # rank 1: primitive integer direction
    t_lower: Fraction | None = None    # rank 1: inclusive arc parameter bounds
    t_upper: Fraction | None = None
    point: tuple | None = None         # rank 2: exact feasible point (Fraction)


def _feasible_1d(bounds, zero_checks):
    """Exact one-dimensional feasibility.

    bounds entries are (value, kind, strict) with kind in {"lo","hi"}.
    zero_checks are (constant, strict) requiring 0 <= constant (or <).
    """
    lower = upper = None
    for value, kind, strict in bounds:
        if kind == "lo":
            if lower is None or value > lower[0] or (value == lower[0] and strict):
                lower = (value, strict)
        else:
            if upper is None or value < upper[0] or (value == upper[0] and strict):
                upper = (value, strict)
    for constant, strict in zero_checks:
        if constant < 0 or (constant == 0 and strict):
            return False
    if lower is None or upper is None:
        return True
    if lower[0] > upper[0]:
        return False
    if lower[0] == upper[0] and (lower[1] or upper[1]):
        return False
    return True


def _strict_feasible_2d(halfplanes):
    """Strict feasibility of a*x+b*y <= c in the rationals (Fourier-Motzkin)."""
    positive, negative, zero = [], [], []
    for a, b, c in halfplanes:
        if a > 0:
            positive.append((a, b, c))
        elif a < 0:
            negative.append((a, b, c))
        else:
            zero.append((a, b, c))
    bounds = []
    zero_checks = []
    for ap, bp, cp in positive:
        for an, bn, cn in negative:
            # (an*bp-ap*bn)*y >= an*cp-ap*cn, strict because every plane is strict
            coeff = an * bp - ap * bn
            bound = an * cp - ap * cn
            if coeff > 0:
                bounds.append((bound / coeff, "lo", True))
            elif coeff < 0:
                bounds.append((bound / coeff, "hi", True))
            else:
                zero_checks.append((-bound, True))
    for a, b, c in zero:
        if b > 0:
            bounds.append((c / b, "hi", True))
        elif b < 0:
            bounds.append((c / b, "lo", True))
        else:
            zero_checks.append((c, True))
    return _feasible_1d(bounds, zero_checks)


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)


def _line_frame(a, b, c):
    """Primitive integer direction and an exact base point of a*x+b*y=c."""
    g = _gcd(a, b)
    dx, dy = -b // g, a // g
    if a != 0:
        base = (Fraction(c, a), Fraction(0))
    else:
        base = (Fraction(0), Fraction(c, b))
    return base, (dx, dy)


def _line_bounds(base, direction, halfplanes, strict_planes=()):
    """Exact arc-parameter interval; None when the strict intersection is empty."""
    x0, y0 = base
    dx, dy = direction
    lower = None  # (Fraction, strict)
    upper = None

    def absorb(q, k, strict):
        nonlocal lower, upper
        if q == 0:
            if k < 0 or (k == 0 and strict):
                return False
            return True
        ratio = k / q
        if q > 0:
            if upper is None or ratio < upper[0] or (ratio == upper[0] and strict):
                upper = (ratio, strict)
        else:
            if lower is None or ratio > lower[0] or (ratio == lower[0] and strict):
                lower = (ratio, strict)
        return True

    for a, b, c in halfplanes:
        q = a * dx + b * dy
        k = c - a * x0 - b * y0
        if not absorb(q, k, False):
            return None
    for a, b, c in strict_planes:
        q = a * dx + b * dy
        k = c - a * x0 - b * y0
        if not absorb(q, k, True):
            return None
    lo = lower[0] if lower is not None else None
    hi = upper[0] if upper is not None else None
    if lo is not None and hi is not None:
        if lo > hi or (lo == hi and (lower[1] or upper[1])):
            return None
    return lo, hi, bool(lower and lower[1]), bool(upper and upper[1])


def analyze_region(region):
    le = [(a, b, Fraction(c)) for a, b, c in region["le"]]
    eq = region["eq"]
    if not eq:
        if not _strict_feasible_2d(le):
            raise ValueError("region has no positive two dimensional measure")
        return RegionGeometry(0)
    if len(eq) == 1:
        a, b, c = eq[0]
        base, direction = _line_frame(a, b, Fraction(c))
        bounds = _line_bounds(base, direction, le)
        if bounds is None or (bounds[0] is not None and bounds[1] is not None
                              and bounds[0] >= bounds[1]):
            raise ValueError("region has no positive one dimensional measure")
        return RegionGeometry(1, base=base, direction=direction,
                              t_lower=bounds[0], t_upper=bounds[1])
    (a1, b1, c1), (a2, b2, c2) = eq
    c1, c2 = Fraction(c1), Fraction(c2)
    determinant = a1 * b2 - a2 * b1
    if determinant != 0:
        x = (c1 * b2 - b1 * c2) / determinant
        y = (a1 * c2 - c1 * a2) / determinant
        for a, b, c in le:
            if a * x + b * y > c:
                raise ValueError("equality point violates a half plane")
        return RegionGeometry(2, point=(x, y))
    if a1 * c2 != a2 * c1 or b1 * c2 != b2 * c1:
        raise ValueError("incompatible parallel equality planes")
    base, direction = _line_frame(a1, b1, c1)
    bounds = _line_bounds(base, direction, le)
    if bounds is None or (bounds[0] is not None and bounds[1] is not None
                          and bounds[0] >= bounds[1]):
        raise ValueError("region has no positive one dimensional measure")
    return RegionGeometry(1, base=base, direction=direction,
                          t_lower=bounds[0], t_upper=bounds[1])


def representative_point(region):
    geometry = analyze_region(region)
    if geometry.rank == 2:
        return float(geometry.point[0]), float(geometry.point[1])
    if geometry.rank == 1:
        candidates = [Fraction(0)]
        if geometry.t_lower is not None:
            candidates.append(geometry.t_lower)
        if geometry.t_upper is not None:
            candidates.append(geometry.t_upper)
        if geometry.t_lower is not None and geometry.t_upper is not None:
            candidates.append((geometry.t_lower + geometry.t_upper) / 2)
        x0, y0 = geometry.base
        dx, dy = geometry.direction
        for t in candidates:
            if ((geometry.t_lower is None or t >= geometry.t_lower)
                    and (geometry.t_upper is None or t <= geometry.t_upper)):
                return float(x0 + dx * t), float(y0 + dy * t)
        t = candidates[-1]
        return float(x0 + dx * t), float(y0 + dy * t)
    bounded = _clip_polygon(region["le"], [])
    bounded = _clip_one(bounded, 1, 0, 8.0)
    bounded = _clip_one(bounded, -1, 0, 8.0)
    bounded = _clip_one(bounded, 0, 1, 8.0)
    bounded = _clip_one(bounded, 0, -1, 8.0)
    if len(bounded) >= 3:
        x = sum(vertex[0] for vertex in bounded) / len(bounded)
        y = sum(vertex[1] for vertex in bounded) / len(bounded)
        return x, y
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Rank 0: polygon clipping and slab integration
# ---------------------------------------------------------------------------

def _clip_polygon(halfplanes, strict_planes):
    """Sutherland-Hodgman clipping on float half planes (a*x+b*y <= c)."""
    polygon = [(-1e4, -1e4), (1e4, -1e4), (1e4, 1e4), (-1e4, 1e4)]
    planes = [(float(a), float(b), float(c), False) for a, b, c in halfplanes]
    planes += [(float(a), float(b), float(c), True) for a, b, c in strict_planes]
    for a, b, c, _strict in planes:
        output = []
        if not polygon:
            return output
        for index, end in enumerate(polygon):
            start = polygon[index - 1]
            start_side = a * start[0] + b * start[1] - c
            end_side = a * end[0] + b * end[1] - c
            start_in = start_side <= 0.0
            end_in = end_side <= 0.0
            if end_in:
                if not start_in:
                    t = start_side / (start_side - end_side)
                    output.append((start[0] + t * (end[0] - start[0]),
                                   start[1] + t * (end[1] - start[1])))
                output.append(end)
            elif start_in:
                t = start_side / (start_side - end_side)
                output.append((start[0] + t * (end[0] - start[0]),
                               start[1] + t * (end[1] - start[1])))
        polygon = output
    return polygon


def _strict_feasible_with(region, strict_planes):
    planes = [(a, b, Fraction(c), False) for a, b, c in region["le"]]
    planes += [(a, b, c, True) for a, b, c in strict_planes]
    positive, negative, zero = [], [], []
    bounds, zero_checks = [], []
    for a, b, c, strict in planes:
        if a > 0:
            positive.append((a, b, c, strict))
        elif a < 0:
            negative.append((a, b, c, strict))
        else:
            zero.append((a, b, c, strict))
    for ap, bp, cp, sp in positive:
        for an, bn, cn, sn in negative:
            coeff = an * bp - ap * bn
            bound = an * cp - ap * cn
            strict = sp or sn
            if coeff > 0:
                bounds.append((bound / coeff, "lo", strict))
            elif coeff < 0:
                bounds.append((bound / coeff, "hi", strict))
            else:
                zero_checks.append((-bound, strict))
    for a, b, c, strict in zero:
        if b > 0:
            bounds.append((c / b, "hi", strict))
        elif b < 0:
            bounds.append((c / b, "lo", strict))
        else:
            zero_checks.append((c, strict))
    return _feasible_1d(bounds, zero_checks)


def _polygon_integral(polygon, model, engine):
    xs = sorted({vertex[0] for vertex in polygon})
    edges = [(polygon[index - 1], vertex) for index, vertex in enumerate(polygon)]
    sx = math.sqrt(model.covariance[0][0])
    total = 0.0
    for x0, x1 in zip(xs, xs[1:]):
        if x1 <= x0:
            continue
        xm = (x0 + x1) / 2.0
        crossings = []
        for (xa, ya), (xb, yb) in edges:
            if xa == xb:
                continue
            if min(xa, xb) <= xm <= max(xa, xb):
                crossings.append((xa, ya, xb, yb))
        if len(crossings) < 2:
            continue

        def boundary(x, crossing):
            xa, ya, xb, yb = crossing
            return ya + (yb - ya) * (x - xa) / (xb - xa)

        def integrand(x):
            ys = [boundary(x, crossing) for crossing in crossings]
            lo, hi = min(ys), max(ys)
            mean, sd = conditional_normal(model, 0, x)
            if sd <= 0 or not math.isfinite(sd):
                raise NumericalFailure("degenerate conditional scale in region integral")
            probability = normal_interval_probability((lo - mean) / sd, (hi - mean) / sd)
            z = (x - model.mu[0]) / sx
            return math.exp(-z * z / 2.0) / (sx * math.sqrt(2.0 * math.pi)) * probability

        result = engine.integrate_1d(integrand, x0, x1)
        budget = min(2e-13, max(abs(result.value) * 2e-13, 1e-28))
        if result.error > budget:
            result = engine.integrate_1d(integrand, x0, x1, tolerance=budget)
        if not result.converged or result.error > budget or not math.isfinite(result.value):
            raise NumericalFailure("unresolved region area integral")
        total += result.value
    if not math.isfinite(total):
        raise NumericalFailure("nonfinite region area integral")
    return total


def _area_measure(region, model, engine, strict_planes=()):
    if strict_planes and not _strict_feasible_with(region, strict_planes):
        return 0.0
    polygon = _clip_polygon(region["le"], strict_planes)
    if len(polygon) < 3:
        return 0.0
    # Fixed raw-coordinate clipping frame: model-independent so that the
    # integral stays smooth in the parameters.  At roughly 50 binary64
    # standard deviations the omitted Gaussian mass is effectively zero.
    box = [(1, 0, 50.0), (-1, 0, 50.0), (0, 1, 50.0), (0, -1, 50.0)]
    bounded = polygon
    for a, b, c in box:
        bounded = _clip_one(bounded, a, b, c)
        if len(bounded) < 3:
            return 0.0
    return _polygon_integral(bounded, model, engine)


def _clip_one(polygon, a, b, c):
    output = []
    for index, end in enumerate(polygon):
        start = polygon[index - 1]
        start_side = a * start[0] + b * start[1] - c
        end_side = a * end[0] + b * end[1] - c
        start_in = start_side <= 0.0
        end_in = end_side <= 0.0
        if end_in:
            if not start_in:
                t = start_side / (start_side - end_side)
                output.append((start[0] + t * (end[0] - start[0]),
                               start[1] + t * (end[1] - start[1])))
            output.append(end)
        elif start_in:
            t = start_side / (start_side - end_side)
            output.append((start[0] + t * (end[0] - start[0]),
                           start[1] + t * (end[1] - start[1])))
    return output


# ---------------------------------------------------------------------------
# Rank 1: Euclidean arc length along the equality line
# ---------------------------------------------------------------------------

def _line_quantities(model, geometry):
    a, b = model.covariance[0]
    d = model.covariance[1][1]
    determinant = a * d - b * b
    inverse = ((d / determinant, -b / determinant), (-b / determinant, a / determinant))
    dx, dy = geometry.direction
    x0, y0 = geometry.base
    ux = float(x0) - model.mu[0]
    uy = float(y0) - model.mu[1]
    q0 = inverse[0][0] * ux * ux + 2 * inverse[0][1] * ux * uy + inverse[1][1] * uy * uy
    h = dx * (inverse[0][0] * ux + inverse[0][1] * uy) \
        + dy * (inverse[1][0] * ux + inverse[1][1] * uy)
    # The exponent is built with Sigma inverse, so the t quadratic coefficient
    # is d' Sigma^-1 d and the tangent mean is m_t = -h / vstar.
    vstar = dx * dx * inverse[0][0] + 2 * dx * dy * inverse[0][1] \
        + dy * dy * inverse[1][1]
    if vstar <= 0 or not math.isfinite(vstar):
        raise NumericalFailure("degenerate line direction variance")
    m_t = -h / vstar
    norm = math.hypot(dx, dy)
    # |d|/(2 pi sqrt(det Sigma)) exp(-.5 q0) = C * phi(t; m_t, 1/sqrt(vstar))
    # with C = |d|/sqrt(2 pi vstar det Sigma) exp(-.5 q0 + .5 h^2/vstar).
    base_log = math.log(norm) - .5 * (LOG2PI + math.log(determinant) + math.log(vstar)) \
        - .5 * q0 + .5 * h * h / vstar
    if not math.isfinite(base_log):
        raise NumericalFailure("nonfinite line measure parameters")
    return m_t, 1.0 / math.sqrt(vstar), base_log


def _line_measure(model, geometry, engine, t_lower, t_upper):
    m_t, sd_t, base_log = _line_quantities(model, geometry)
    lower_z = -math.inf if t_lower is None else (float(t_lower) - m_t) / sd_t
    upper_z = math.inf if t_upper is None else (float(t_upper) - m_t) / sd_t
    probability = normal_interval_probability(lower_z, upper_z)
    if not math.isfinite(probability) or probability < 0:
        raise NumericalFailure("unresolved line segment probability")
    try:
        measure = math.exp(base_log) * probability
    except OverflowError as exc:
        raise NumericalFailure("line measure outside binary64 range") from exc
    if not math.isfinite(measure):
        raise NumericalFailure("nonfinite line measure")
    return measure


def _line_full_measure(model, geometry):
    _m_t, _sd_t, base_log = _line_quantities(model, geometry)
    try:
        measure = math.exp(base_log)
    except OverflowError as exc:
        raise NumericalFailure("line measure outside binary64 range") from exc
    if not math.isfinite(measure):
        raise NumericalFailure("nonfinite line measure")
    return measure


# ---------------------------------------------------------------------------
# Public measure API used by likelihoods and conditional reports
# ---------------------------------------------------------------------------

def _threshold_planes(thresholds):
    t1, t2 = thresholds
    return [(-1, 0, -Fraction(t1)), (0, -1, -Fraction(t2))]


def region_measure(region, model, engine, thresholds=None):
    """Return (numerator, denominator); numerator is L(R) without thresholds."""
    geometry = analyze_region(region)
    strict_planes = _threshold_planes(thresholds) if thresholds is not None else ()
    if geometry.rank == 2:
        x, y = geometry.point
        try:
            denominator = math.exp(joint_log_density(model, float(x), float(y)))
        except OverflowError as exc:
            raise NumericalFailure("point density outside binary64 range") from exc
        full = denominator
        if thresholds is None:
            numerator = denominator
        else:
            t1, t2 = thresholds
            inside = x > Fraction(t1) and y > Fraction(t2)
            numerator = denominator if inside else 0.0
    elif geometry.rank == 1:
        denominator = _line_measure(model, geometry, engine,
                                    geometry.t_lower, geometry.t_upper)
        full = _line_full_measure(model, geometry)
        if thresholds is None:
            numerator = denominator
        else:
            le = [(a, b, Fraction(c)) for a, b, c in region["le"]]
            bounds = _line_bounds(geometry.base, geometry.direction, le, strict_planes)
            if bounds is None:
                numerator = 0.0
            else:
                numerator = _line_measure(model, geometry, engine, bounds[0], bounds[1])
    else:
        denominator = _area_measure(region, model, engine)
        full = 1.0
        if thresholds is None:
            numerator = denominator
        else:
            numerator = _area_measure(region, model, engine, strict_planes)
    for value in (numerator, denominator, full):
        if not math.isfinite(value) or value < 0:
            raise NumericalFailure("nonfinite region measure")
    return numerator, denominator, full


def region_log_likelihood(region, model, engine, enforce_truncation=False):
    try:
        numerator, denominator, full = region_measure(region, model, engine)
    except NumericalFailure:
        if enforce_truncation:
            raise
        return -40.0
    if enforce_truncation:
        if denominator <= 0:
            raise NumericalFailure("region has zero observation measure")
        if full > 0 and denominator / full < TRUNCATION_FLOOR:
            raise NumericalFailure("region truncation probability below numerical support")
        return math.log(denominator)
    # During optimization the trial parameter can sit outside the resolved
    # support.  A smooth quadratic extension keeps the objective finite and
    # the five-point scores well defined; the real floor is enforced on the
    # converged candidate before publication.
    if denominator <= 0:
        return -40.0
    ratio = denominator / full if full > 0 else 1.0
    floor = 1e-12
    if ratio < floor:
        return math.log(denominator) - 0.5 * ((ratio - floor) / floor) ** 2
    return math.log(denominator)

def row_to_region(row):
    """Canonical region equivalent of a legacy x observation row."""
    le, eq = [], []
    for axis, constraint in enumerate(row["x"]):
        a, b = (1, 0) if axis == 0 else (0, 1)
        kind = constraint["kind"]
        if kind == "exact":
            eq.append([a, b, constraint["value"]])
        else:
            if kind in ("right", "interval"):
                le.append([-a, -b, -constraint["lower"]])
            if kind in ("left", "interval"):
                le.append([a, b, constraint["upper"]])
    return {"le": le, "eq": eq}
