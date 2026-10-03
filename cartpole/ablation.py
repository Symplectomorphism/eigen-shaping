"""
Cart-pole swing-up ablation: isolate the eigenfunction's contribution.

The deployment harness is held FIXED across all arms (pole-energy gate +
cart-centering PD coast + LQR catch + |u|<=U_MAX + identical initial conditions
and integrator).  Only the swing-up pump direction u_nn (applied while the pole is
energy-deficient) is swapped:

  1. eigenfunction (ours)  u_nn = -g^T grad V_theta   (learned quasi-potential)
  2. classical energy pump u_nn = k_E * E_pole * theta_dot * cos(theta)  (Astrom-
     Furuta / Spong collocated energy-shaping law; k_E swept, best reported)
  3. LQR-only              no swing-up (latched from the start)
  4. zero pump             harness only, no active pumping
  5. random direction      u_nn ~ Uniform[-U_MAX, U_MAX] (no structure)

Integration is classical RK4 on the state augmented with the three work integrals
    e1 = int u^2 dt,   e2 = int y u dt  (y = g^T grad H, the collocated output),
    e3 = int (grad H)^T R (grad H) dt   (dissipated energy),
so the quadratures inherit the integrator's order and the port-Hamiltonian power
balance  e2 = Delta H + e3  closes to integration error.  Explicit Euler is
retained behind --integrator euler purely for the refinement study: at dt = 5e-3 it
drifts H by ~0.26 J over the 12 s horizon, against a swing-up budget of
2 m_p g l = 0.98 J.  Within a step the latch state and any stochastic pump value
are held fixed, so u is a deterministic function of the stage state.

Reports the deterministic catch rate, median time-to-upright, and BOTH effort
measures.  Expected (and intended) result: the eigenfunction MATCHES the classical
energy pump, confirming Theorem 3 (the eigenfunction's sigma->0 limit is the
energy-shaping controller, derived with no hand-tuned energy law or matching-PDE
solve), while the degenerate baselines (3,4,5) fail.  The supplied energy int y u dt
is pinned near 2 m_p g l for every successful arm (it is a state function plus
negligible dissipation), so it cannot separate strategies; int u^2 dt can.

Run: cd sim/cartpole && uv run python ablation.py [--integrator rk4] [--dt 0.005]
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from plot_cartpole import (control_vmap, drift_vmap, K_lqr, _velocities, load_model,
                           U_MAX, GATE_WIDTH, KP_CART, KD_CART, CATCH_CONE,
                           CATCH_THDMAX, _I_POLE, _MGL)
from dynamics import G_VEC, GEAR, hamiltonian, D_S, D_H

GVEC = np.array(G_VEC)
MODEL = load_model(1.0)
_H_vmap = jax.jit(jax.vmap(hamiltonian))
_gradH_vmap = jax.jit(jax.vmap(jax.grad(hamiltonian)))


def _harness_u(X, latched, u_nn):
    """Full harness control from the state, the frozen latch mask, and the pump."""
    theta = X[:, 0]
    theta_dot, s_dot = _velocities(X)
    E_pole = 0.5 * _I_POLE * theta_dot**2 + _MGL * (np.cos(theta) - 1.0)
    gate = np.clip(-E_pole / GATE_WIDTH, 0.0, 1.0)
    u_cart = -(KP_CART * X[:, 1] + KD_CART * s_dot)
    u_swing = gate * u_nn + (1.0 - gate) * u_cart
    Xm = X.copy(); Xm[:, 0] = np.sin(theta)
    u_lqr = -(Xm @ K_lqr)
    return np.clip(np.where(latched, u_lqr, u_swing), -U_MAX, U_MAX)


def _rhs(X, latched, u_nn_fn):
    """Augmented right-hand side: (xdot, u^2, y*u, dissipated power) at state X."""
    theta = X[:, 0]
    theta_dot, s_dot = _velocities(X)
    E_pole = 0.5 * _I_POLE * theta_dot**2 + _MGL * (np.cos(theta) - 1.0)
    u_nn = u_nn_fn(X, theta, theta_dot, s_dot, E_pole)
    u = _harness_u(X, latched, u_nn)
    gH = np.array(_gradH_vmap(jnp.array(X)))
    xdot = np.array(drift_vmap(X)) + GVEC[None, :] * u[:, None]
    y = GEAR * gH[:, 3]                       # y = g^T grad H = GEAR * s_dot
    power_dis = D_H * gH[:, 2]**2 + D_S * gH[:, 3]**2
    return xdot, u**2, y * u, power_dis


def rollout(pump, n=1024, steps=2400, dt=0.005, seed=123, noise=0.0,
            lqr_only=False, integrator="rk4"):
    """Fixed harness.  `pump` is a per-step factory: pump(rng, n) returns the
    callable f(X, theta, theta_dot, s_dot, E_pole) -> u_nn used for every stage of
    that step, so a stochastic pump is held zero-order over the step while a
    state-feedback pump is re-evaluated at each RK4 stage."""
    sq = np.sqrt(dt)
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 4)); X[:, 0] = np.pi + 0.1 * rng.standard_normal(n)
    latched = np.ones(n, bool) if lqr_only else np.zeros(n, bool)
    t_catch = np.full(n, np.nan)
    e1 = np.zeros(n); e2 = np.zeros(n); e3 = np.zeros(n)
    H0 = np.array(_H_vmap(jnp.array(X)))

    for k in range(steps):
        theta_dot, _ = _velocities(X)
        newly = ((1.0 - np.cos(X[:, 0])) < CATCH_CONE) & (np.abs(theta_dot) < CATCH_THDMAX)
        t_catch[newly & ~latched] = k * dt
        latched |= newly
        u_nn = pump(rng, n)                      # frozen for this step

        k1 = _rhs(X, latched, u_nn)
        if integrator == "euler":
            X = X + dt * k1[0]
            e1 += dt * k1[1]; e2 += dt * k1[2]; e3 += dt * k1[3]
        else:
            k2 = _rhs(X + dt/2 * k1[0], latched, u_nn)
            k3 = _rhs(X + dt/2 * k2[0], latched, u_nn)
            k4 = _rhs(X + dt * k3[0], latched, u_nn)
            w = lambda i: dt/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
            X = X + w(0); e1 += w(1); e2 += w(2); e3 += w(3)
        if noise > 0:
            X = X + sq * noise * GVEC[None, :] * rng.standard_normal((n, 4))
        X[:, 0] = (X[:, 0] + np.pi) % (2 * np.pi) - np.pi

    dH = np.array(_H_vmap(jnp.array(X))) - H0
    caught = (1.0 - np.cos(X[:, 0])) < 0.2
    return caught, t_catch, e1, e2, e3, dH


# ---- pump-direction arms (each is a per-step factory) -------------------------
def pump_eig(rng, n):
    return lambda X, theta, thd, sd, Ep: np.array(control_vmap(MODEL, X))


def pump_energy(kE):
    def factory(rng, n):
        return lambda X, theta, thd, sd, Ep: kE * Ep * thd * np.cos(theta)
    return factory


def pump_zero(rng, n):
    return lambda X, theta, thd, sd, Ep: np.zeros(len(X))


def pump_random(rng, n):
    """Zero-order hold: one draw per step, shared by every RK4 stage."""
    u = rng.uniform(-U_MAX, U_MAX, n)
    return lambda X, theta, thd, sd, Ep: u


def summary(name, r, n):
    caught, t_catch, e1, e2, e3, dH = r
    pct = 100.0 * caught.sum() / n
    tc = t_catch[caught & ~np.isnan(t_catch)]
    if not caught.any():
        return f"{name:<28} {pct:>6.1f}%   {'n/a':>8}   {'n/a':>8}   {'n/a':>8}   {'n/a':>9}"
    tcs = f"{np.median(tc):.2f}" if len(tc) else "  n/a "
    return (f"{name:<28} {pct:>6.1f}%   {tcs:>8}   {np.mean(e1[caught]):>8.2f}   "
            f"{np.mean(e2[caught]):>8.3f}   "
            f"{np.mean(e2[caught] - dH[caught] - e3[caught]):>9.1e}")


if __name__ == "__main__":
    import time, argparse
    ap = argparse.ArgumentParser(description="Cart-pole swing-up ablation.")
    ap.add_argument("--integrator", choices=("rk4", "euler"), default="rk4")
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("-n", type=int, default=1024)
    ap.add_argument("--horizon", type=float, default=12.0)
    args = ap.parse_args()
    n = args.n
    steps = int(round(args.horizon / args.dt))
    kw = dict(n=n, steps=steps, dt=args.dt, integrator=args.integrator)

    print("=" * 92)
    print("CART-POLE SWING-UP ABLATION  (fixed harness; only the pump direction varies)")
    print(f"  deterministic (sigma=0), |u|<=U_MAX={U_MAX}, integrator={args.integrator}, "
          f"dt={args.dt}, T={args.horizon}s, {n} ICs")
    print(f"  ICs: theta_0 ~ N(pi, 0.1^2) rad, s_0 = 0, p_theta,0 = p_s,0 = 0")
    print(f"  harness: gate(width={GATE_WIDTH}) + cart-PD(kp={KP_CART},kd={KD_CART})"
          f" + LQR catch(cone={CATCH_CONE},thd<{CATCH_THDMAX})")
    print("-" * 92)
    print(f"{'arm':<28} {'catch':>7}   {'t_up(s)':>8}   {'int u^2':>8}   {'int y*u':>8}   {'bal.res':>9}")
    print("-" * 92)

    t0 = time.time()
    print(summary("1. eigenfunction (ours)", rollout(pump_eig, **kw), n), flush=True)

    best = None
    for kE in (-8, -4, -2, -1, -0.5, 0.5, 1, 2, 4, 8):
        r = rollout(pump_energy(kE), **kw)
        pct = 100.0 * r[0].sum() / n
        if best is None or pct > best[1]:
            best = (kE, pct, r)
    print(summary(f"2. energy pump (k_E={best[0]:+g})", best[2], n), flush=True)

    print(summary("3. LQR only (no swing-up)", rollout(pump_zero, lqr_only=True, **kw), n), flush=True)
    print(summary("4. no pump (harness only)", rollout(pump_zero, **kw), n), flush=True)

    print(summary("5. random direction", rollout(pump_random, **kw), n), flush=True)

    print("-" * 92)
    print(f"(elapsed {time.time()-t0:.0f}s)   2 m_p g l = {2*_MGL:.4f} J")
    print("=" * 92)
    print("int u^2 dt and int y*u dt are means over caught trajectories; y = g^T grad H")
    print("is the collocated output, so int y*u dt is the energy supplied through the")
    print("port.  'bal.res' is the power-balance residual int y*u dt - (Delta H + int")
    print("dissip), which vanishes with the integration error and validates the")
    print("energy accounting.  t_up = median time-to-upright over caught trajectories.")
