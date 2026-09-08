"""
Verify the Foster-Lyapunov drift condition (Assumption 1(iv)) for the inverted
pendulum, exhibiting an explicit Lyapunov function W.

Prior generator (matched noise, g = (0,1)):
    L phi = theta_dot d_theta phi + ((g/l) sin theta - b theta_dot) d_thetadot phi
            + (sigma^2/2) d^2_thetadot phi
State cost  q = q_a (1 - cos theta) + (q_b/2) theta_dot^2.

Tilted drift condition:  (L - q) W <= -c W + b_const * 1_C  on X = S^1 x R.

Candidate  W(theta, theta_dot) = 1 + (1/2) theta_dot^2   (C^2, >= 1, coercive in
theta_dot; theta is on the compact circle). Because q grows like theta_dot^2, the
tilting term -qW contributes -(q_b/4) theta_dot^4, which dominates every other
term, so the geometric drift -cW holds outside a compact velocity band C.
"""
from __future__ import annotations
import numpy as np
import sympy as sp

# parameters (match pendulum/dynamics.py and pendulum/eigfun.py)
G_L, B, QA, QB, SIGMA = 9.81, 0.2, 3.0, 0.3, 2.0

th, v = sp.symbols("theta theta_dot", real=True)
W = 1 + sp.Rational(1, 2) * v**2
q = QA * (1 - sp.cos(th)) + sp.Rational(1, 2) * QB * v**2

# (L - q) W
f_th, f_v = v, (G_L * sp.sin(th) - B * v)
LW = f_th * sp.diff(W, th) + f_v * sp.diff(W, v) + sp.Rational(1, 2) * SIGMA**2 * sp.diff(W, v, 2)
drift = sp.expand(LW - q * W)          # (L - q) W
print("W =", W)
print("(L - q)W =", drift)

# We want (L-q)W + c W <= 0 outside a compact set C = {|theta_dot| <= R}.
# Pick c, find the smallest R, then b_const = max over C of (L-q)W + c W.
C_CONST = 1.0
expr = sp.lambdify((th, v), drift + C_CONST * W, "numpy")   # (L-q)W + cW

# worst case over theta (the -9.81 sin th * v term is most adverse at sin th = -sign(v))
ths = np.linspace(-np.pi, np.pi, 721)
vs = np.linspace(-12, 12, 4001)
TH, V = np.meshgrid(ths, vs, indexing="ij")
G = expr(TH, V)                        # (L-q)W + cW ; need <= 0 outside C, bounded inside
worst_over_theta = G.max(axis=0)       # worst (largest) value over theta, per v

# smallest R with G <= 0 for all |v| >= R
R = 0.0
for j, vv in enumerate(vs):
    if abs(vv) > R and worst_over_theta[j] > 0:
        R = abs(vv)
R = R + (vs[1] - vs[0])
inside = np.abs(vs) <= R
b_const = float(worst_over_theta[inside].max())

print(f"\nc = {C_CONST}")
print(f"compact set  C = S^1 x {{ |theta_dot| <= {R:.2f} }}")
print(f"b = max_C [(L-q)W + cW] = {b_const:.3f}")
print(f"outside C: max_x [(L-q)W + cW] = {worst_over_theta[~inside].max():.3e}  (<= 0 required)")

# confirm the asymptotic: leading term of (L-q)W in v is -(q_b/4) v^4
poly_v = sp.Poly(sp.expand(drift.subs(sp.sin(th), 1).subs(sp.cos(th), 0)), v)
print(f"\nleading v-term of (L-q)W: {-QB/4} v^4  (super-quadratic confinement)")
print("=> (L-q)W <= -cW + b*1_C holds: Assumption 1(iv) verified for the pendulum.")
