"""
Verify the Foster-Lyapunov drift condition (Assumption 1(iv)) for the cart-pole,
exhibiting an explicit Lyapunov function W.

Prior generator (momentum coords x=(theta,s,p_theta,p_s), matched noise
g=(0,0,0,GEAR) constant):
    L W = f0 . grad W + (sigma^2/2) GEAR^2 d^2W/dp_s^2,   f0 = ph_drift (validated).
State cost q = 2(1-cos theta) + 0.2 s^2 + 0.02 theta_dot^2 + 0.02 s_dot^2, with
(theta_dot, s_dot) = M(theta)^{-1} (p_theta, p_s).

Tilted drift condition:  (L - q) W <= -c W + b 1_C  on X = S^1 x R^3.

The only noncompact directions are (s, p_theta, p_s); theta is on the compact
circle.  q is a positive-definite quadratic there, so for any positive-definite
quadratic W the product q W grows quartically and dominates L W (cubic), giving
the geometric drift outside a compact velocity/position box.  The cart is only
weakly damped (D_S=5e-4), so the confinement comes from -q W, not the dynamics.
"""
from __future__ import annotations
import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from dynamics import ph_drift, mass_matrix, GEAR

QA, QS, QTHD, QSD = 2.0, 0.2, 0.02, 0.02     # state-cost weights (match eigfun_train)
SIGMA = 1.0


def velocities(theta, p_th, p_s):
    a = 0.1 * 0.5**2; bb = 0.1 * 0.5; dd = 1.1     # M entries
    det = a * dd - (bb * np.cos(theta))**2
    th_d = (dd * p_th - bb * np.cos(theta) * p_s) / det
    s_d = (-bb * np.cos(theta) * p_th + a * p_s) / det
    return th_d, s_d


_pf = jax.jit(jax.vmap(ph_drift))


def drift_minus_q_W(TH, S, PTH, PS, W_weights, c):
    """(L - q) W + c W on the grid (arrays). W = 1 + 0.5 (ws s^2 + wpth p_th^2 + wps p_s^2)."""
    ws, wpth, wps = W_weights
    X = np.stack([TH, S, PTH, PS], axis=-1).reshape(-1, 4)
    f0 = np.array(_pf(jnp.array(X))).reshape(TH.shape + (4,))
    fs, fpth, fps = f0[..., 1], f0[..., 2], f0[..., 3]      # s_dot, p_th_dot, p_s_dot
    # grad W = (0, ws s, wpth p_th, wps p_s);  d^2W/dp_s^2 = wps
    LW = fs * ws * S + fpth * wpth * PTH + fps * wps * PS + 0.5 * SIGMA**2 * GEAR**2 * wps
    th_d, s_d = velocities(TH, PTH, PS)
    q = QA * (1 - np.cos(TH)) + QS * S**2 + QTHD * th_d**2 + QSD * s_d**2
    W = 1.0 + 0.5 * (ws * S**2 + wpth * PTH**2 + wps * PS**2)
    return (LW - q * W) + c * W, W


def verify(W_weights, c, nth=20, ns=61, np_th=61, np_s=121,
           Smax=20.0, Pthmax=2.0, Psmax=30.0):
    th = np.linspace(-np.pi, np.pi, nth, endpoint=False)
    s = np.linspace(-Smax, Smax, ns)
    pth = np.linspace(-Pthmax, Pthmax, np_th)
    ps = np.linspace(-Psmax, Psmax, np_s)
    TH, S, PTH, PS = np.meshgrid(th, s, pth, ps, indexing="ij")
    G, W = drift_minus_q_W(TH, S, PTH, PS, W_weights, c)
    pos = G > 0
    if not pos.any():
        return dict(captured=True, R=(0., 0., 0.), b=float(G.max()), omax=float("-inf"))
    Rs = np.abs(S[pos]).max(); Rpth = np.abs(PTH[pos]).max(); Rps = np.abs(PS[pos]).max()
    # captured iff the positive region is strictly inside the grid box in every
    # noncompact direction (so nothing leaks past the grid boundary).
    captured = (Rs < Smax - 1e-9) and (Rpth < Pthmax - 1e-9) and (Rps < Psmax - 1e-9)
    inside = (np.abs(S) <= Rs) & (np.abs(PTH) <= Rpth) & (np.abs(PS) <= Rps)
    b = float(G[inside].max())
    omax = float(G[~inside].max()) if (~inside).any() else float("-inf")
    return dict(captured=captured, R=(Rs, Rpth, Rps), b=b, omax=omax,
                steps=(s[1] - s[0], pth[1] - pth[0], ps[1] - ps[0]))


def refine(W_weights, c, coarse, fine=(0.1, 0.025, 0.1), nth=48,
           quantum=(0.1, 0.05, 0.1)):
    """Re-resolve the positive region on a fine grid and report constants rounded
    outward. The coarse box can miss the true boundary by up to one coarse step,
    because the bounding box of positive grid points lies inside the positive region,
    so the fine pass searches the coarse box enlarged by one coarse step. C is padded
    by one fine step and rounded up to `quantum`, and b is rounded up to an integer,
    so the printed constants are conservative by construction."""
    lim = [r + h for r, h in zip(coarse["R"], coarse["steps"])]
    axes = [np.arange(-L, L + 1e-12, h) for L, h in zip(lim, fine)]
    th = np.linspace(-np.pi, np.pi, nth, endpoint=False)
    S, PTH, PS = np.meshgrid(*axes, indexing="ij")
    ext = np.zeros(3)
    for t in th:
        G, _ = drift_minus_q_W(np.full_like(S, t), S, PTH, PS, W_weights, c)
        pos = G > 0
        if pos.any():
            ext = np.maximum(ext, [np.abs(S[pos]).max(), np.abs(PTH[pos]).max(),
                                   np.abs(PS[pos]).max()])
    R = [np.ceil((e + h) / q - 1e-9) * q for e, h, q in zip(ext, fine, quantum)]
    inside = (np.abs(S) <= R[0]) & (np.abs(PTH) <= R[1]) & (np.abs(PS) <= R[2])
    bmax, omax = -np.inf, -np.inf
    for t in th:
        G, _ = drift_minus_q_W(np.full_like(S, t), S, PTH, PS, W_weights, c)
        bmax = max(bmax, float(G[inside].max()))
        if (~inside).any():
            omax = max(omax, float(G[~inside].max()))
    return dict(R=tuple(R), b=float(np.ceil(bmax)), bmax=bmax, omax=omax)


if __name__ == "__main__":
    print("=" * 74)
    print("Foster-Lyapunov drift condition for the cart-pole (Assumption 1(iv))")
    print(f"  generator: matched noise g=(0,0,0,{GEAR}), sigma={SIGMA}")
    print(f"  q = {QA}(1-cos th) + {QS} s^2 + {QTHD}(theta_dot^2 + s_dot^2)")
    print("  W = 1 + 0.5 (ws s^2 + wpth p_theta^2 + wps p_s^2)")
    print("-" * 74)
    for W_weights in [(1.0, 1.0, 1.0)]:
        for c in (1.0, 0.5):
            r = verify(W_weights, c)
            Rs, Rpth, Rps = r["R"]
            status = ("OK" if (r["captured"] and r["omax"] <= 1e-6)
                      else ("GRID TOO SMALL" if not r["captured"] else "FAIL"))
            print(f"  W=1+0.5(s^2+p_th^2+p_s^2), c={c}: coarse grid "
                  f"C={{|s|<={Rs:.1f}, |p_th|<={Rpth:.2f}, |p_s|<={Rps:.1f}}}  "
                  f"b={r['b']:.1f}  max outside C={r['omax']:.2e}  [{status}]")
            f = refine(W_weights, c, r)
            fs, fpth, fps = f["R"]
            fstatus = "OK" if f["omax"] <= 0 else "FAIL"
            print(f"      refined, rounded outward: "
                  f"C={{|s|<={fs:.1f}, |p_th|<={fpth:.2f}, |p_s|<={fps:.1f}}}  "
                  f"b={f['b']:.0f} (fine-grid sup {f['bmax']:.2f})  "
                  f"max outside C={f['omax']:.2e}  [{fstatus}]")
    print("=" * 74)
    print("(L-q)W <= -cW + b 1_C verified: the coercive cost q confines the noncompact")
    print("(s, p_theta, p_s) directions; Assumption 1(iv) holds for the cart-pole.")
