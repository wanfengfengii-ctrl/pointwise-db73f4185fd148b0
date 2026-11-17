import math
import unittest
from decimal import Decimal,localcontext
from functools import lru_cache
from lab.numerics import NumericEngine,NumericalFailure
from lab.gaussian_model import GaussianParameters
from lab.fit_solver import solve_joint_mle
from lab.joint_tail import joint_exceedance_probability
from samples import mixed_rows,REFERENCE

# Independent 70-digit arithmetic, independently generated fixed quadrature,
# and raw covariance coordinates. No production likelihood/CDF/integrator calls.
D = Decimal
PI = D("3.141592653589793238462643383279502884197169399375105820974944592307816406")

def cdf(x):
    if x < 0:
        return 1-cdf(-x)
    if x > 16:
        return D(1)
    term,total = x,x
    n = 0
    while True:
        n += 1
        term *= -x*x/(2*n)
        addition = term/(2*n+1)
        total += addition
        if abs(addition) < D("1e-65"):
            return D(".5")+total/(2*PI).sqrt()
        if n > 1200:
            raise ArithmeticError("independent CDF series")

@lru_cache(None)
def nodes(n):
    result = []
    for k in range(1,n+1):
        z = D(str(math.cos(math.pi*(k-.25)/(n+.5))))
        for _ in range(40):
            p,q = D(1),D(0)
            for j in range(1,n+1):
                p,q = ((2*j-1)*z*p-(j-1)*q)/j,p
            derivative = n*(z*p-q)/(z*z-1)
            step = p/derivative
            z -= step
            if abs(step) < D("1e-65"):
                break
        result.append((z,2/((1-z*z)*derivative*derivative)))
    return tuple(result)

def integral(function,a,b,n):
    if a >= b:
        return D(0)
    midpoint,half = (a+b)/2,(b-a)/2
    return half*sum((weight*function(midpoint+half*x) for x,weight in nodes(n)),D(0))

def oracle(rows,parameters,n=48,thresholds=None):
    u,v,a,b,d = [D(str(x)) for x in parameters]
    determinant = a*d-b*b
    root2pi = (2*PI).sqrt()
    def interval(c):
        if c["kind"] == "left":
            return D("-Infinity"),D(str(c["upper"]))
        if c["kind"] == "right":
            return D(str(c["lower"])),D("Infinity")
        return D(str(c["lower"])),D(str(c["upper"]))
    def rectangle(xbound,ybound):
        lo,hi = xbound
        lo,hi = max(D(-12),(lo-u)/a.sqrt()),min(D(12),(hi-u)/a.sqrt())
        sd = (d-b*b/a).sqrt()
        def fun(z):
            conditional_mean = v+b/a.sqrt()*z
            yl,yh = ybound
            return (-z*z/2).exp()/root2pi*(cdf((yh-conditional_mean)/sd)-cdf((yl-conditional_mean)/sd))
        return integral(fun,lo,hi,n)
    if thresholds is not None:
        return rectangle((D(str(thresholds[0])),D("Infinity")),(D(str(thresholds[1])),D("Infinity")))
    result = D(0)
    for row in rows:
        x,y = row["x"]
        xe,ye = x["kind"]=="exact",y["kind"]=="exact"
        if xe and ye:
            p,q = D(str(x["value"]))-u,D(str(y["value"]))-v
            logp = -(2*PI).ln()-determinant.ln()/2-(d*p*p-2*b*p*q+a*q*q)/(2*determinant)
        elif xe or ye:
            value = D(str(x["value"] if xe else y["value"]))
            mu,var,other_mu,other_var = (u,a,v,d) if xe else (v,d,u,a)
            mean = other_mu+b/var*(value-mu)
            sd = (other_var-b*b/var).sqrt()
            lo,hi = interval(y if xe else x)
            mass = cdf((hi-mean)/sd)-cdf((lo-mean)/sd)
            logp = -((2*PI*var).ln()+(value-mu)**2/var)/2+mass.ln()
        else:
            logp = rectangle(interval(x),interval(y)).ln()
        result += row["count"]*logp
    return result

class TestNumerics(unittest.TestCase):
    def test_generic_engine(self):
        engine = NumericEngine()
        result = engine.integrate_1d(lambda x:x**4,-2,3)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.value,55.,delta=1e-11)
        solution = engine.solve_linear([[3,1],[1,2]],[7,5])
        self.assertAlmostEqual(solution[0],1.8,delta=1e-12)
        self.assertAlmostEqual(solution[1],1.6,delta=1e-12)
        fit = engine.minimize(lambda x:(x[0]-2)**2+3*(x[1]+1)**2,[0,0])
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(fit.value[0],2,delta=1e-7)
        self.assertAlmostEqual(fit.value[1],-1,delta=1e-7)
        with engine.failing("integrate"):
            with self.assertRaises(NumericalFailure):
                engine.integrate_1d(math.sin,0,1)

    def test_independent_decimal_reference(self):
        parameters = [*REFERENCE["mu"],REFERENCE["covariance"][0][0],
                      REFERENCE["covariance"][0][1],REFERENCE["covariance"][1][1]]
        with localcontext() as context:
            context.prec = 70
            coarse = oracle(mixed_rows(),parameters,48)
            fine = oracle(mixed_rows(),parameters,80)
            self.assertLess(abs(coarse-fine),D("2e-12"))
            self.assertLess(abs(fine-D(str(REFERENCE["log_likelihood"]))),D("2e-11"))
            risk = oracle([],parameters,80,REFERENCE["thresholds"])
            self.assertLess(abs(risk-D(str(REFERENCE["probability"]))),D("2e-12"))
            # Five-point independent score at the frozen reference.
            for j in range(5):
                h = D("0.00001")
                values = []
                for multiplier in (-2,-1,1,2):
                    p = [D(str(v)) for v in parameters]
                    p[j] += multiplier*h
                    values.append(oracle(mixed_rows(),p,80))
                score = (values[0]-8*values[1]+8*values[2]-values[3])/(12*h)
                self.assertLess(abs(score),D("2e-8"))
        candidate = solve_joint_mle(mixed_rows(),NumericEngine())
        self.assertLess(candidate.status["parameter_error_estimate"],2e-7)
        self.assertAlmostEqual(candidate.log_likelihood,float(fine),delta=1e-10)
        probability = joint_exceedance_probability(candidate.model,REFERENCE["thresholds"],NumericEngine())
        self.assertAlmostEqual(probability,float(risk),delta=1e-9)
