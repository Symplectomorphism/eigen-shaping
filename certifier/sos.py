"""
Fast hand-rolled sum-of-squares (SOS) layer over CVXPY + MOSEK.

Polynomials are represented as dicts {exponent-tuple: coefficient}, where a
coefficient is either a float (known data) or a CVXPY affine expression
(decision). "p is SOS" becomes p = b^T G b, G PSD, assembled by accumulating
Gram entries into monomial coefficients via exponent-tuple addition (no symbolic
expansion). The polynomial identity "everything cancels" is then one linear
equality per monomial. Each program is a clean SDP (verification mode: the
Lyapunov function and control are fixed data, gamma is bisected).
"""
from __future__ import annotations
import itertools
import os
import sys
import sympy as sp
import cvxpy as cp


def _resolve_mosek_license():
    """Point MOSEKLM_LICENSE_FILE at a valid mosek.lic: an already valid env var,
    else the repo-root mosek.lic, else ~/mosek/mosek.lic.  A stale value is cleared
    rather than left in place, so MOSEK's own search runs and its error names a path
    the user recognises."""
    cur = os.environ.get("MOSEKLM_LICENSE_FILE")
    if cur and os.path.exists(cur):
        return
    if cur:
        del os.environ["MOSEKLM_LICENSE_FILE"]
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "..", "..", "mosek.lic"),      # repo root
                 os.path.expanduser("~/mosek/mosek.lic")):
        if os.path.exists(cand):
            os.environ["MOSEKLM_LICENSE_FILE"] = os.path.abspath(cand)
            return
    print("sos: no mosek.lic found (repo root or ~/mosek/); the SOS programs need "
          "MOSEK.  See the run instructions.", file=sys.stderr)


_resolve_mosek_license()


def monomials(vars_, deg):
    """Sympy monomials in vars_ of total degree <= deg."""
    out, n = [], len(vars_)
    for d in range(deg + 1):
        for e in itertools.combinations_with_replacement(range(n), d):
            m = sp.Integer(1)
            for i in e:
                m *= vars_[i]
            out.append(sp.expand(m))
    seen, uniq = set(), []
    for m in out:
        k = sp.srepr(m)
        if k not in seen:
            seen.add(k); uniq.append(m)
    return uniq


def _exp(monomial, vars_):
    """Exponent tuple of a sympy monomial in vars_."""
    p = sp.Poly(monomial, *vars_)
    return p.monoms()[0]


def to_dict(expr, vars_):
    """Known sympy polynomial -> {exp-tuple: float}."""
    p = sp.Poly(sp.expand(expr), *vars_)
    return {m: float(c) for m, c in zip(p.monoms(), p.coeffs())}


class CvxPoly:
    """Polynomial with float-or-CVXPY coefficients, keyed by exponent tuple."""
    def __init__(self, terms=None):
        self.terms = dict(terms or {})

    def _acc(self, e, c):
        self.terms[e] = (self.terms[e] + c) if e in self.terms else c

    def __add__(self, o):
        r = CvxPoly(self.terms)
        it = o.terms.items() if isinstance(o, CvxPoly) else o.items()
        for e, c in it:
            r._acc(e, c)
        return r

    __radd__ = __add__

    def __neg__(self):
        return CvxPoly({e: -c for e, c in self.terms.items()})

    def __sub__(self, o):
        return self + (-o if isinstance(o, CvxPoly) else CvxPoly({e: -c for e, c in o.items()}))

    def __mul__(self, other):
        """Multiply by a scalar or a KNOWN polynomial (dict {exp: float})."""
        r = CvxPoly()
        if isinstance(other, (int, float)):
            for e, c in self.terms.items():
                r._acc(e, c * other)
            return r
        od = other.terms if isinstance(other, CvxPoly) else other
        for e1, c1 in self.terms.items():
            for e2, c2 in od.items():
                r._acc(tuple(x + y for x, y in zip(e1, e2)), c1 * c2)
        return r

    __rmul__ = __mul__


def deriv(p, i):
    """Partial derivative d/dx_i of a CvxPoly (linear in its coefficients)."""
    out = CvxPoly()
    for e, c in p.terms.items():
        if e[i] > 0:
            e2 = list(e); k = e2[i]; e2[i] -= 1
            out._acc(tuple(e2), c * k)
    return out


def evaluate(p, point):
    """Value of a CvxPoly at a numeric point (CVXPY affine expression / float)."""
    val = 0.0
    for e, c in p.terms.items():
        m = 1.0
        for xi, ei in zip(point, e):
            m *= xi ** ei
        val = val + c * m
    return val


class SOSProblem:
    def __init__(self, vars_):
        self.vars = tuple(vars_)
        self._psd = []
        self._constr = []

    def sos(self, basis):
        """Return a CvxPoly constrained SOS over the given monomial half-basis."""
        n = len(basis)
        Q = cp.Variable((n, n), symmetric=True)
        self._psd.append(Q)
        exps = [_exp(b, self.vars) for b in basis]
        p = CvxPoly()
        for i in range(n):
            ei = exps[i]
            p._acc(tuple(2 * x for x in ei), Q[i, i])
            for j in range(i + 1, n):
                p._acc(tuple(a + b for a, b in zip(ei, exps[j])), 2 * Q[i, j])
        return p

    def free(self, monos):
        """Return a sign-free CvxPoly multiplier sum c_k m_k."""
        v = cp.Variable(len(monos))
        return CvxPoly({_exp(m, self.vars): v[k] for k, m in enumerate(monos)})

    def const(self, expr):
        """Known sympy polynomial as a CvxPoly with float coefficients."""
        return CvxPoly(to_dict(expr, self.vars))

    def zero(self, cpoly):
        """Constrain a CvxPoly to be identically zero (one equality per monomial)."""
        for e, c in cpoly.terms.items():
            if isinstance(c, (int, float)):
                if abs(c) > 1e-12:
                    self._constr.append(cp.Constant(c) == 0)  # infeasible flag
            else:
                self._constr.append(c == 0)

    def solve(self, verbose=False):
        prob = cp.Problem(cp.Minimize(0), [Q >> 0 for Q in self._psd] + self._constr)
        try:
            prob.solve(solver=cp.MOSEK, verbose=verbose)
            self.status = prob.status
        except cp.error.SolverError:
            self.status = "solver_error"
        return prob

    @property
    def feasible(self):
        return self.status in ("optimal", "optimal_inaccurate")
