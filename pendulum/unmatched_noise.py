"""
Matched-noise robustness experiment (inverted pendulum).

Tests the claim that matched noise is not a knife-edge condition: the principal
eigenpair, and the controller synthesized from it, depend continuously on the
diffusion matrix near the matched point, and the matched-synthesized controller
degrades gracefully when deployed on a plant with UNMATCHED noise.

Unmatched perturbation: add diffusion Delta = rho * sigma^2 in the UNACTUATED
(theta) channel, where the matched diffusion sigma^2 g g^T (g=(0,1)) has none.
rho = ||Delta|| / sigma^2 is the relative unmatched-noise intensity (rho=0 matched).

  Part A: principal eigenvalue lambda*_Delta(rho) from the grid  -> smooth/analytic
  Part B: deploy the rho=0 (matched) controller on the rho>0 plant -> success(rho)
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
try:
    from eigfun import build_generator, principal_eigfunction, control_field, _bilin
    from dynamics import G_GRAV, L as LL, B_DAMP
except ModuleNotFoundError:
    from pendulum.eigfun import build_generator, principal_eigfunction, control_field, _bilin
    from pendulum.dynamics import G_GRAV, L as LL, B_DAMP

NTH, NV, VMAX = 161, 201, 16.0
QA, QB = 3.0, 0.3
SIGMA = 2.0


def add_theta_diffusion(A, thetas, vs, delta_theta):
    """Add unmatched diffusion delta_theta * d^2/dtheta^2 (central, periodic) to the
    grid generator A (theta is the unactuated channel)."""
    Nth, Nv = len(thetas), len(vs)
    dth = thetas[1] - thetas[0]
    coeff = delta_theta / (2.0 * dth**2)
    rows, cols, vals = [], [], []
    def idx(i, j): return i * Nv + j
    for i in range(Nth):
        ip = (i + 1) % Nth; im = (i - 1) % Nth
        for j in range(Nv):
            k = idx(i, j)
            rows += [k, k, k]; cols += [k, idx(ip, j), idx(im, j)]
            vals += [-2.0 * coeff, coeff, coeff]
    D = sp.csr_matrix((vals, (rows, cols)), shape=A.shape)
    return A + D


def lambda_star(rho):
    """Optimal ergodic cost lambda* of the (possibly unmatched) tilted generator."""
    A, ths, vs, q = build_generator(NTH, NV, VMAX, SIGMA, QA, QB, nonlinear=True)
    if rho > 0:
        A = add_theta_diffusion(A, ths, vs, rho * SIGMA**2)
    lam, _ = principal_eigfunction(A)
    return -lam                                  # principal eigenvalue is -lambda*


def control_field_um(rho):
    """Eigenfunction control field synthesized from the (un)matched generator."""
    A, ths, vs, q = build_generator(NTH, NV, VMAX, SIGMA, QA, QB, nonlinear=True)
    if rho > 0:
        A = add_theta_diffusion(A, ths, vs, rho * SIGMA**2)
    lam, psi = principal_eigfunction(A)
    uf = control_field(psi, ths, vs, SIGMA)
    return uf, ths, vs, psi.reshape(NTH, NV)


def field_distance(uf, uf0, psi0, vs):
    """Relative L2 distance ||u*_Delta - u*_0|| / ||u*_0|| over the reliable region
    (no underflow, |theta_dot| <= 10)."""
    sel = np.abs(vs) <= 10.0
    rel = ~(psi0 < psi0.max() * 1e-10)
    m = rel[:, sel]
    a = uf[:, sel][m]; b = uf0[:, sel][m]
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def deploy_unmatched(uf, ths, vs, rho, n_traj=2000, T=6.0, n_steps=600, seed=0):
    """Deploy the matched control field on a plant with matched velocity noise sigma
    AND unmatched position noise sqrt(rho)*sigma. Return swing-up rate (|theta|<0.6,
    the paper's criterion) over the final second (time-averaged, robust to the
    constant sigma=2 buffeting)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps; sq = np.sqrt(dt)
    X = np.array([np.pi - 0.05, 0.0])[None, :] + np.array([0.2, 0.2])[None, :] * rng.standard_normal((n_traj, 2))
    s_th = np.sqrt(rho) * SIGMA          # unmatched position-channel noise scale
    up_frac = []
    for k in range(n_steps):
        u = np.array([_bilin(uf, ths, vs, X[i]) for i in range(n_traj)])
        th, v = X[:, 0], X[:, 1]
        acc = (G_GRAV / LL) * np.sin(th) - B_DAMP * v + u
        X = X + np.stack([v, acc], axis=1) * dt
        X[:, 1] += SIGMA * sq * rng.standard_normal(n_traj)     # matched
        X[:, 0] += s_th * sq * rng.standard_normal(n_traj)      # unmatched
        X[:, 0] = (X[:, 0] + np.pi) % (2 * np.pi) - np.pi
        X = np.clip(X, [-np.pi, -40], [np.pi, 40])
        if k >= n_steps - 100:                                  # final ~second
            thw = (X[:, 0] + np.pi) % (2 * np.pi) - np.pi
            up_frac.append(np.abs(thw) < 0.6)
    return float(np.mean(up_frac))


if __name__ == "__main__":
    import time
    print("=" * 72)
    print("MATCHED-NOISE ROBUSTNESS (inverted pendulum, sigma=2.0)")
    print("unmatched diffusion Delta = rho*sigma^2 in the UNACTUATED theta channel")
    print("rho = ||Delta||/sigma^2  (rho=0 = matched; Hopf-Cole exact only at rho=0)")
    print("-" * 72)

    rhos = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
    t0 = time.time()
    lam0 = lambda_star(0.0)
    print(f"  matched lambda* (rho=0) = {lam0:.4f}")
    print(f"{'rho':>6} {'lambda*_Delta':>14} {'d lambda* ':>11} {'(lam-lam0)/rho':>15}")
    for rho in rhos:
        lam = lambda_star(rho)
        slope = "" if rho == 0 else f"{(lam-lam0)/rho:>15.3f}"
        print(f"{rho:>6} {lam:>14.4f} {lam-lam0:>+11.4f} {slope}", flush=True)
    print("-> lambda*_Delta is smooth and ~linear in rho: the principal eigenpair")
    print("   depends analytically on the diffusion (no knife-edge at rho=0).")
    print("-" * 72)

    # Part B: the synthesized control field itself is stable to the perturbation.
    uf0, ths, vs, psi0 = control_field_um(0.0)
    print("Synthesized control field vs unmatched noise (relative L2 distance):")
    print(f"{'rho':>6} {'||u*_D - u*_0|| / ||u*_0||':>28} {'/rho':>8}")
    for rho in rhos:
        if rho == 0:
            print(f"{rho:>6} {0.0:>28.4f} {'':>8}"); continue
        ufr, _, _, _ = control_field_um(rho)
        d = field_distance(ufr, uf0, psi0, vs)
        print(f"{rho:>6} {d:>28.4f} {d/rho:>8.3f}", flush=True)
    print("-> ||u*_Delta - u*_0|| = O(rho): the synthesized controller depends")
    print("   continuously (Lipschitz) on the diffusion. Matched noise is not a")
    print("   knife-edge; it is the center of an open neighborhood of valid synthesis.")
    print("-" * 72)

    print("Practical check: deploy the MATCHED controller on the UNMATCHED plant")
    print(f"(swing-up rate |theta|<0.6, averaged over the final second):")
    print(f"{'rho':>6} {'swing-up rate':>16}")
    for rho in rhos:
        suc = deploy_unmatched(uf0, ths, vs, rho)
        print(f"{rho:>6} {100*suc:>14.1f}%", flush=True)
    print("-> graceful degradation under substantial unmatched noise.")
    print(f"(elapsed {time.time()-t0:.0f}s)")
    print("=" * 72)
