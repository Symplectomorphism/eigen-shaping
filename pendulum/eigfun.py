"""
Stationary Schrödinger-Bridge controller via the principal eigenfunction
(the paper's Direction 2 / Theorem 2), prototyped on the pendulum.

We solve the Donsker-Varadhan principal-eigenpair problem for the prior's
generator tilted by a state cost q:
    (L - q) psi* = -lambda* psi*,    L = f0 . grad + (sigma^2/2) d^2/dthetadot^2,
with psi* > 0 (Perron-Frobenius / Krein-Rutman, valid for the NON-reversible
cart-pole/pendulum generator). The optimal stationary feedback is
    u*(x) = sigma^2 g^T grad log psi*(x) = sigma^2 d/dthetadot [log psi*],
(g = B = e_thetadot), and V* = -sigma^2 log psi* is a Lyapunov function. As
sigma -> 0 this is the IDA-PBC / quasi-potential gradient feedback (Theorem 3).

NO reference bridge, NO bridge-matching, NO coupling to collapse -> immune to the
DSBM cold-start trap. For the 2D pendulum the eigenpair is an exact sparse solve.

Run: cd sim && uv run python -m pendulum.eigfun
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pendulum.dynamics import G_GRAV, L as L_LEN, B_DAMP, M, U_MAX


def build_generator(Ntheta, Nv, vmax, sigma, q_alpha, q_beta, nonlinear=True):
    """
    Sparse (L - q) on a grid: theta in [-pi,pi) periodic, thetadot in [-vmax,vmax]
    Dirichlet (psi->0 at the velocity boundary). Upwind advection, central
    diffusion in thetadot. Returns (A, thetas, vs, q).
    """
    thetas = np.linspace(-np.pi, np.pi, Ntheta, endpoint=False)
    vs = np.linspace(-vmax, vmax, Nv)
    dth = thetas[1] - thetas[0]
    dv = vs[1] - vs[0]
    TH, V = np.meshgrid(thetas, vs, indexing="ij")        # (Ntheta, Nv)

    # drift components
    a = V                                                  # dtheta/dt = thetadot
    if nonlinear:
        c = (G_GRAV / L_LEN) * np.sin(TH) - B_DAMP * V     # nonlinear gravity
    else:
        c = (G_GRAV / L_LEN) * TH - B_DAMP * V             # linearized
    # state cost: 0 at upright (theta=0, thetadot=0), grows away (periodic in theta)
    q = q_alpha * (1.0 - np.cos(TH)) + 0.5 * q_beta * V**2

    N = Ntheta * Nv
    def idx(i, j):
        return i * Nv + j

    rows, cols, vals = [], [], []
    def add(r, c_, v):
        rows.append(r); cols.append(c_); vals.append(v)

    diff = sigma**2 / (2.0 * dv**2)
    for i in range(Ntheta):
        ip = (i + 1) % Ntheta
        im = (i - 1) % Ntheta
        for j in range(Nv):
            k = idx(i, j)
            aij = a[i, j]; cij = c[i, j]
            # theta advection: upwind giving a Metzler generator (off-diag >= 0).
            # a>0 transports toward +theta -> forward difference (rate to ip).
            if aij >= 0:
                add(k, k, -aij / dth); add(k, idx(ip, j), aij / dth)
            else:
                add(k, k, aij / dth); add(k, idx(im, j), -aij / dth)
            # thetadot advection (Dirichlet outside the velocity window)
            if cij >= 0:
                add(k, k, -cij / dv)
                if j + 1 < Nv:
                    add(k, idx(i, j + 1), cij / dv)
            else:
                add(k, k, cij / dv)
                if j - 1 >= 0:
                    add(k, idx(i, j - 1), -cij / dv)
            # thetadot diffusion (central, Dirichlet outside)
            add(k, k, -2.0 * diff)
            if j - 1 >= 0:
                add(k, idx(i, j - 1), diff)
            if j + 1 < Nv:
                add(k, idx(i, j + 1), diff)
            # tilt by -q
            add(k, k, -q[i, j])

    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    return A, thetas, vs, q


def principal_eigfunction(A):
    """Largest-real-part eigenpair; eigenfunction made positive."""
    # Shift-invert just above 0: the principal (Perron) eigenvalue is in
    # [-max q, 0], so the eigenvalue nearest a small positive shift is it.
    w, V = spla.eigs(A, k=1, sigma=0.05, which="LM", maxiter=20000, tol=1e-9)
    lam = w[0].real
    psi = V[:, 0].real
    if psi.sum() < 0:
        psi = -psi
    psi = np.clip(psi, 1e-300, None)
    return lam, psi


def control_field(psi, thetas, vs, sigma):
    """u*(x) = sigma^2 d/dthetadot log psi*, on the grid (central diff in v)."""
    Ntheta, Nv = len(thetas), len(vs)
    logpsi = np.log(psi).reshape(Ntheta, Nv)
    dv = vs[1] - vs[0]
    dlog = np.gradient(logpsi, dv, axis=1)
    u = sigma**2 * dlog                                     # (Ntheta, Nv)
    # tame the Dirichlet velocity-boundary artifact (psi->0 => log singular)
    return np.clip(u, -40.0, 40.0)


def _bilin(field, thetas, vs, x):
    """Bilinear interp of field at x=(theta,thetadot), theta periodic."""
    th = (x[0] + np.pi) % (2 * np.pi) - np.pi
    v = np.clip(x[1], vs[0], vs[-1])
    Ntheta = len(thetas); dth = thetas[1] - thetas[0]; dv = vs[1] - vs[0]
    fi = (th - thetas[0]) / dth
    i0 = int(np.floor(fi)) % Ntheta; i1 = (i0 + 1) % Ntheta; fx = fi - np.floor(fi)
    fj = np.clip((v - vs[0]) / dv, 0, len(vs) - 1.0001)
    j0 = int(np.floor(fj)); j1 = j0 + 1; fy = fj - j0
    return ((1 - fx) * (1 - fy) * field[i0, j0] + fx * (1 - fy) * field[i1, j0]
            + (1 - fx) * fy * field[i0, j1] + fx * fy * field[i1, j1])


def deploy(ufield, thetas, vs, sigma, mu0, std0, T, n_steps, n_traj, nonlinear,
           do_saturate, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / n_steps; sq = np.sqrt(dt)
    X = np.array(mu0)[None, :] + np.array(std0)[None, :] * rng.standard_normal((n_traj, 2))
    Bv = np.array([0.0, 1.0 / (M * L_LEN**2)])
    for _ in range(n_steps):
        u = np.array([_bilin(ufield, thetas, vs, X[k]) for k in range(n_traj)])
        if do_saturate:
            u = np.clip(u, -U_MAX, U_MAX)
        th = X[:, 0]; v = X[:, 1]
        if nonlinear:
            c = (G_GRAV / L_LEN) * np.sin(th) - B_DAMP * v
        else:
            c = (G_GRAV / L_LEN) * th - B_DAMP * v
        drift = np.stack([v, c + Bv[1] * u], axis=1)
        dW = sq * rng.standard_normal(n_traj)
        X = X + drift * dt + sigma * Bv[None, :] * dW[:, None]
        X[:, 0] = (X[:, 0] + np.pi) % (2 * np.pi) - np.pi
        X = np.clip(X, [-np.pi, -40], [np.pi, 40])
    thw = (X[:, 0] + np.pi) % (2 * np.pi) - np.pi
    dist = np.sqrt(thw**2 + X[:, 1]**2)
    absu_mean = float(np.abs(np.array([_bilin(ufield, thetas, vs, X[k]) for k in range(n_traj)])).mean())
    swung_up = float((np.abs(thw) < 0.6).mean())   # reached the upright neighborhood
    return float((dist <= 0.35).mean()), float(np.abs(thw).mean()), absu_mean, swung_up


def main():
    Ntheta, Nv, vmax = 161, 201, 16.0
    q_alpha, q_beta = 3.0, 0.3
    # Resolve psi*: dynamic range ~ exp(max q / sigma^2). Too small a sigma (or
    # too sharp a cost) underflows psi* far from upright on the grid -> log psi*
    # is flat noise there -> no control where swing-up needs it. This is exactly
    # the "blind in high-cost regions" failure the neural relative-loss fixes.
    print("=== NONLINEAR swing-up from near-hanging (the real test) ===")
    for sigma in (2.0, 1.5, 1.0, 0.7):
        A, thetas, vs, q = build_generator(Ntheta, Nv, vmax, sigma, q_alpha,
                                           q_beta, nonlinear=True)
        lam, psi = principal_eigfunction(A)
        underflow = (psi < psi.max() * 1e-12).mean()
        uf = control_field(psi, thetas, vs, sigma)
        entry, mth, mu, up = deploy(uf, thetas, vs, sigma, mu0=(np.pi - 0.05, 0.0),
                                    std0=(0.2, 0.2), T=6.0, n_steps=600,
                                    n_traj=1000, nonlinear=True, do_saturate=False)
        print(f"  sigma={sigma}: psi*-underflow={underflow:.2f}  swung-up(|th|<0.6) "
              f"{up:.2%}  tight-entry(<0.35) {entry:.2%}  mean|theta| {mth:.3f}")


if __name__ == "__main__":
    main()
