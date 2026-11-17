import math
from contextlib import contextmanager
from dataclasses import dataclass

class NumericalFailure(ArithmeticError):
    pass

@dataclass(frozen=True)
class NumericOutcome:
    value: object
    error: float
    converged: bool
    iterations: int

def _rule(n):
    nodes = []
    for k in range(1, n + 1):
        x = math.cos(math.pi * (k - .25) / (n + .5))
        for _ in range(30):
            p, q = 1., 0.
            for j in range(1, n + 1):
                p, q = ((2*j-1)*x*p-(j-1)*q)/j, p
            d = n*(x*p-q)/(x*x-1)
            step = p/d
            x -= step
            if abs(step) < 2e-16:
                break
        nodes.append((x, 2/((1-x*x)*d*d)))
    return tuple(nodes)

_RULES = {n: _rule(n) for n in (12, 24)}

class NumericEngine:
    def __init__(self):
        self.failures = set()

    @contextmanager
    def failing(self, *operations):
        old = self.failures.copy()
        self.failures.update(operations)
        try:
            yield self
        finally:
            self.failures = old

    def guard(self, operation):
        if operation in self.failures:
            raise NumericalFailure("injected numerical failure: " + operation)

    def integrate_1d(self, function, lower, upper, tolerance=2e-13):
        self.guard("integrate")
        if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
            raise NumericalFailure("finite ordered integration bounds required")
        calls = 0
        def rule(a, b, n):
            nonlocal calls
            calls += n
            mid, half = (a+b)/2, (b-a)/2
            return half*math.fsum(w*function(mid+half*x) for x,w in _RULES[n])
        def part(a, b, tol, depth):
            lo, hi = rule(a,b,12), rule(a,b,24)
            error = abs(hi-lo)
            if not math.isfinite(hi):
                raise NumericalFailure("nonfinite integral")
            if error <= tol:
                return hi,error
            if depth == 0:
                raise NumericalFailure("quadrature precision exhausted")
            m = (a+b)/2
            v,e = part(a,m,tol/2,depth-1)
            w,d = part(m,b,tol/2,depth-1)
            return v+w,e+d
        if lower == upper:
            return NumericOutcome(0.,0.,True,0)
        pieces = max(1, math.ceil((upper-lower)/2))
        values, errors = [], []
        for k in range(pieces):
            a = lower+(upper-lower)*k/pieces
            b = lower+(upper-lower)*(k+1)/pieces
            v,e = part(a,b,tolerance/pieces,12)
            values.append(v)
            errors.append(e)
        return NumericOutcome(math.fsum(values), math.fsum(errors), True, calls)

    def solve_linear(self, matrix, vector):
        self.guard("linear")
        n = len(vector)
        a = [list(map(float,row))+[float(v)] for row,v in zip(matrix,vector)]
        if len(a) != n or any(len(row) != n+1 for row in a):
            raise NumericalFailure("matrix dimensions")
        for k in range(n):
            p = max(range(k,n), key=lambda j: abs(a[j][k]))
            a[k],a[p] = a[p],a[k]
            pivot = a[k][k]
            if abs(pivot) < 1e-14:
                raise NumericalFailure("singular linear system")
            a[k] = [v/pivot for v in a[k]]
            for j in range(n):
                if j != k:
                    scale = a[j][k]
                    a[j] = [v-scale*w for v,w in zip(a[j],a[k])]
        return [row[-1] for row in a]

    def gradient(self, function, point):
        result = []
        for j,x in enumerate(point):
            h = 1e-4*(1+abs(x))
            vals = []
            for step in (-2,-1,1,2):
                q = list(point)
                q[j] += step*h
                vals.append(function(q))
            result.append((vals[0]-8*vals[1]+8*vals[2]-vals[3])/(12*h))
        return result

    def hessian(self, function, point):
        n = len(point)
        columns = []
        for j,x in enumerate(point):
            h = 8e-4*(1+abs(x))
            p,m = list(point),list(point)
            p[j] += h
            m[j] -= h
            gp,gm = self.gradient(function,p),self.gradient(function,m)
            columns.append([(a-b)/(2*h) for a,b in zip(gp,gm)])
        return [[(columns[i][j]+columns[j][i])/2 for j in range(n)] for i in range(n)]

    def minimize(self, function, initial, max_iterations=200):
        self.guard("minimize")
        x = list(initial)
        n = len(x)
        inv = [[float(i==j) for j in range(n)] for i in range(n)]
        value = function(x)
        g = self.gradient(function,x)
        for iteration in range(max_iterations):
            if max(map(abs,g)) < 2e-9:
                return NumericOutcome(x,max(map(abs,g)),True,iteration)
            direction = [-sum(inv[i][j]*g[j] for j in range(n)) for i in range(n)]
            slope = sum(a*b for a,b in zip(direction,g))
            if slope >= 0:
                direction = [-v for v in g]
                slope = -sum(v*v for v in g)
                inv = [[float(i==j) for j in range(n)] for i in range(n)]
            alpha = 1.
            accepted = False
            for _ in range(45):
                q = [a+alpha*b for a,b in zip(x,direction)]
                try:
                    new_value = function(q)
                except (NumericalFailure, OverflowError, ValueError):
                    new_value = math.inf
                if new_value <= value+1e-4*alpha*slope+2e-15:
                    accepted = True
                    break
                alpha /= 2
            if not accepted:
                return NumericOutcome(x,max(map(abs,g)),False,iteration)
            ng = self.gradient(function,q)
            s = [a-b for a,b in zip(q,x)]
            y = [a-b for a,b in zip(ng,g)]
            ys = sum(a*b for a,b in zip(y,s))
            if ys > 1e-18:
                hy = [sum(row[j]*y[j] for j in range(n)) for row in inv]
                yhy = sum(a*b for a,b in zip(y,hy))
                inv = [[inv[i][j]+(1+yhy/ys)*s[i]*s[j]/ys
                        -(s[i]*hy[j]+hy[i]*s[j])/ys for j in range(n)] for i in range(n)]
            x,value,g = q,new_value,ng
        return NumericOutcome(x,max(map(abs,g)),False,max_iterations)
