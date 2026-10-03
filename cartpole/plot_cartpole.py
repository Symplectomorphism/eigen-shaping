import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Uniform on-page text (~7.8pt) across all paper figures: base = 7.8 / scale.
# This 16in figure at \linewidth (scale ~0.41) -> base ~19.
plt.rcParams.update({
    "font.size": 19,
    "axes.titlesize": 19,
    "axes.labelsize": 19,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
})
import jax
import jax.numpy as jnp
import equinox as eqx
from pathlib import Path

# Import dynamics and model architecture
from dynamics import (ph_drift, G_VEC, GEAR, mass_matrix,
                      M_P, M_C, L, G_GRAV, lqr_gains)
from eigfun_train import LogEigenfunction, TrainState

FIGDIR = Path(__file__).resolve().parents[1] / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = Path(__file__).parent.parent / "checkpoints" / "eigfun"

# ── Deployment controller (sigma -> 0 IDA-PBC limit) ────────────────────────────
# The eigenfunction feedback u* = -g^T grad V is the energy-pumping swing-up. Left
# unregulated it over-pumps (converges to an energy orbit through the top, not to
# rest), so it is gated by the pole-energy deficit: full pump while the pole lacks
# the energy to reach upright, fading to zero as that energy is reached, with a cart
# PD keeping the base centred during the coast. Once the pole arrives at the apex
# (small angle and speed) control latches to the certified LQR catch. The diffusion
# sigma is a computational device for the eigenproblem; the control is deployed
# deterministically (Theorem 3 limit).
U_MAX        = 3.0    # control authority |u| <= U_MAX (post-gear force = GEAR*U_MAX)
GATE_WIDTH   = 1.0    # pole-energy deficit (J) over which the pump fades out
KP_CART      = 1.0    # cart-centring proportional gain during coast
KD_CART      = 1.0    # cart-centring derivative gain during coast
CATCH_CONE   = 0.5    # latch to LQR when 1 - cos(theta) < CATCH_CONE
CATCH_THDMAX = 4.0    # ... and |theta_dot| < CATCH_THDMAX

_I_POLE = M_P * L**2          # pole inertia about pivot
_MGL    = M_P * G_GRAV * L    # gravitational scale
_M11    = M_P * L**2
_M22    = M_C + M_P


def load_model(sigma=1.0):
    ckpt_path = CKPT_DIR / f"cartpole_sigma{sigma}.eqx"
    key = jax.random.PRNGKey(0)
    dummy_model = LogEigenfunction(key)
    dummy_state = TrainState(model=dummy_model, lambda_param=jnp.array(0.0))
    state = eqx.tree_deserialise_leaves(ckpt_path, dummy_state)
    return state.model


def get_control(model, x):
    # Eigenfunction feedback u* = sigma^2 g^T grad log psi* = -g^T grad V
    # (G_VEC carries GEAR, so u* is the direct command).
    return -jnp.dot(G_VEC, jax.grad(model)(x))


get_control_vmap = jax.vmap(get_control, in_axes=(None, 0))

P_lqr, K_lqr = lqr_gains()

drift_vmap = jax.jit(jax.vmap(ph_drift))
control_vmap = eqx.filter_jit(get_control_vmap)


def _velocities(X):
    """Vectorized (theta_dot, s_dot) from the momentum-state array X (n, 4)."""
    m12 = M_P * L * np.cos(X[:, 0])
    det = _M11 * _M22 - m12**2
    theta_dot = (_M22 * X[:, 2] - m12 * X[:, 3]) / det
    s_dot = (-m12 * X[:, 2] + _M11 * X[:, 3]) / det
    return theta_dot, s_dot


def simulate(model, noise_sigma, seed=123, n=1024, steps=2000, dt=0.005):
    """Roll out the energy-gated eigenfunction swing-up with LQR catch from the
    hanging state. Returns (trajs (n, steps+1, 4), caught mask (n,))."""
    sq = np.sqrt(dt)
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 4))
    X[:, 0] = np.pi + 0.1 * rng.standard_normal(n)
    latched = np.zeros(n, dtype=bool)
    trajs = [X.copy()]

    for _ in range(steps):
        theta = X[:, 0]
        theta_dot, s_dot = _velocities(X)

        u_nn = np.array(control_vmap(model, X))
        # Pole energy relative to upright-rest: 0 at the target, -2*mgl at hanging.
        E_pole = 0.5 * _I_POLE * theta_dot**2 + _MGL * (np.cos(theta) - 1.0)
        gate = np.clip(-E_pole / GATE_WIDTH, 0.0, 1.0)
        u_cart = -(KP_CART * X[:, 1] + KD_CART * s_dot)
        u_swing = gate * u_nn + (1.0 - gate) * u_cart

        Xm = X.copy(); Xm[:, 0] = np.sin(theta)
        u_lqr = -(Xm @ K_lqr)

        latched |= ((1.0 - np.cos(theta)) < CATCH_CONE) & (np.abs(theta_dot) < CATCH_THDMAX)
        u = np.where(latched, u_lqr, u_swing)
        u = np.clip(u, -U_MAX, U_MAX)

        # RK4 with the control held over the step (zero-order hold). Explicit
        # Euler at dt=5e-3 drifts H by ~0.26 J over the 12 s horizon, against a
        # swing-up budget of 2 m_p g l = 0.98 J.
        def _f(Z):
            return np.array(drift_vmap(Z)) + np.array(G_VEC)[None, :] * u[:, None]
        k1 = _f(X); k2 = _f(X + dt/2*k1); k3 = _f(X + dt/2*k2); k4 = _f(X + dt*k3)
        X = X + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        if noise_sigma > 0:
            X = X + sq * float(noise_sigma) * np.array(G_VEC)[None, :] * rng.standard_normal((n, 4))
        X[:, 0] = (X[:, 0] + np.pi) % (2 * np.pi) - np.pi
        trajs.append(X.copy())

    caught = (1.0 - np.cos(X[:, 0])) < 0.2
    return np.stack(trajs, axis=1), caught


def _rollout_catch(model, noise_sigma, seed=123, n_eval=1024, steps=2000, dt=0.005):
    """Catch count for the energy-gated swing-up + LQR policy.

    dt=0.005: the previous dt=0.02 forward-Euler step was coarse enough that
    discretization error itself dumped energy into the cone, inflating the catch
    rate by ~18 points. Keep the step small so the number reflects the controller."""
    _, caught = simulate(model, noise_sigma, seed=seed, n=n_eval, steps=steps, dt=dt)
    return int(np.sum(caught)), n_eval


def _raw_reach(model, n=1024, steps=2000, dt=0.005, seed=123):
    """Deterministic rollout under the RAW eigenfunction control only (u=-g^T grad V,
    clipped; no gate, cart-PD, or LQR catch). Returns (reach%, hold%): fraction that
    ever enter the upright cone, and fraction in it at the end."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 4)); X[:, 0] = np.pi + 0.1 * rng.standard_normal(n)
    reached = np.zeros(n, bool)
    for _ in range(steps):
        u = np.clip(np.array(control_vmap(model, X)), -U_MAX, U_MAX)
        def _f2(Z):
            return np.array(drift_vmap(Z)) + np.array(G_VEC)[None, :] * u[:, None]
        k1 = _f2(X); k2 = _f2(X + dt/2*k1); k3 = _f2(X + dt/2*k2); k4 = _f2(X + dt*k3)
        X = X + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        X[:, 0] = (X[:, 0] + np.pi) % (2 * np.pi) - np.pi
        reached |= (1.0 - np.cos(X[:, 0])) < 0.2
    held = (1.0 - np.cos(X[:, 0])) < 0.2
    return 100.0 * reached.mean(), 100.0 * held.mean()


def evaluate_lqr_catch(model, sigma=0.0):
    # Headline is the DETERMINISTIC catch: the controller is the sigma->0 limit, so
    # it is deployed without diffusion (added noise perturbs the energy coast/catch).
    # Also report the cart excursion (peak |s| during the swing-up) and the final
    # |s|, so a q_cart sweep shows the catch-vs-excursion trade-off directly.
    trajs, caught = simulate(model, 0.0, n=1024, steps=2000, dt=0.005)
    n = len(caught); succ = int(caught.sum()); rate = succ / n * 100
    peak_s = float(np.abs(trajs[:, :, 1]).max())
    final_s = float(np.abs(trajs[:, -1, 1]).max())
    print(f"Deterministic catch: {succ} / {n} ({rate:.1f}%) | "
          f"cart excursion peak|s|={peak_s:.2f} m | final|s|={final_s:.3f} m", flush=True)

    # Overshoot probe: RAW eigenfunction control only (no gate, no cart-PD, no catch).
    # 'reach' = ever entered the upright cone; 'hold' = in it at the end. A large
    # reach-minus-hold gap is the separatrix overshoot the gate exists to trim; a
    # sharper field (smaller sigma, closer to H_d) should close that gap.
    reach, hold = _raw_reach(model)
    print(f"Raw un-gated control: reach upright {reach:.1f}% | hold {hold:.1f}% | "
          f"overshoot gap {reach-hold:.1f} pts", flush=True)

    # the manuscript \input{}s this value; written beside the figures here
    tex_path = FIGDIR / "catch_rate.tex"
    with open(tex_path, "w") as f:
        f.write(f"{rate:.1f}\\%")
    return rate


def plot_all(sigma=1.0):
    model = load_model(sigma)

    # 1. Lyapunov slice V(theta, theta_dot) at s = s_dot = 0.
    N_th, N_thd = 100, 100
    thetas = np.linspace(-np.pi, np.pi, N_th)
    th_dots = np.linspace(-10, 10, N_thd)
    TH, THD = np.meshgrid(thetas, th_dots, indexing="ij")

    def qdot_to_p(theta, s, thd, sd):
        M = mass_matrix(theta)
        p = M @ jnp.array([thd, sd])
        return jnp.array([theta, s, p[0], p[1]])
    vmap_qdot_to_p = jax.vmap(qdot_to_p)
    X_grid_p = vmap_qdot_to_p(TH.flatten(), np.zeros(TH.size),
                              THD.flatten(), np.zeros(TH.size))
    V_grid = np.array(jax.vmap(model)(X_grid_p)).reshape(N_th, N_thd)
    V_grid -= V_grid.min()
    levels = np.linspace(0, np.percentile(V_grid, 95), 15)

    # 2. Deterministic swing-up trajectories (the deployed sigma->0 controller).
    trajs, caught = simulate(model, 0.0, seed=42, n=8, steps=2400, dt=0.005)
    def p_to_qdot(x):
        theta, s, p_theta, p_s = x
        v = jnp.linalg.solve(mass_matrix(theta), jnp.array([p_theta, p_s]))
        return jnp.array([theta, s, v[0], v[1]])
    trajs_v = np.array(jax.vmap(jax.vmap(p_to_qdot))(trajs))
    n_traj = trajs_v.shape[0]
    t = np.arange(trajs_v.shape[1]) * 0.005

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

    # Left: Lyapunov slice.
    im = ax0.pcolormesh(TH, THD, V_grid, cmap="viridis", shading="auto")
    ax0.contour(TH, THD, V_grid, levels=levels, colors='w', alpha=0.3, linewidths=0.5)
    fig.colorbar(im, ax=ax0)
    ax0.set_title(r"Lyapunov $V_\theta(x)$ slice")
    ax0.set_xlabel(r"$\theta$ (rad)"); ax0.set_ylabel(r"$\dot\theta$ (rad/s)")
    ax0.set_xlim([-np.pi, np.pi]); ax0.set_ylim([-10, 10])

    # Middle: pole phase portrait with explicit start and end markers.
    ax1.pcolormesh(TH, THD, V_grid, cmap="viridis", shading="auto", alpha=0.3)
    ax1.contour(TH, THD, V_grid, levels=levels, colors='0.5', alpha=0.6, linewidths=0.5)
    for i in range(n_traj):
        theta = trajs_v[i, :, 0]; theta_dot = trajs_v[i, :, 2]
        wrap_idx = np.where(np.abs(np.diff(theta)) > np.pi)[0]
        ax1.plot(np.insert(theta, wrap_idx + 1, np.nan),
                 np.insert(theta_dot, wrap_idx + 1, np.nan),
                 lw=1.0, alpha=0.7, color="C1")
    ax1.scatter(trajs_v[:, 0, 0], trajs_v[:, 0, 2], s=40, color="C2",
                edgecolor="k", linewidth=0.5, label=r"Start ($\theta\approx\pi$)", zorder=4)
    ax1.scatter(trajs_v[:, -1, 0], trajs_v[:, -1, 2], s=60, marker="X", color="C3",
                edgecolor="k", linewidth=0.5, label="End", zorder=5)
    ax1.plot(0, 0, "*", ms=18, color="k", label="Target $x^\\star$", zorder=3)
    ax1.set_title("Pole phase portrait")
    ax1.set_xlabel(r"$\theta$ (rad)"); ax1.set_ylabel(r"$\dot\theta$ (rad/s)")
    ax1.set_xlim([-np.pi, np.pi]); ax1.set_ylim([-10, 10])
    # R12: legend was overlapping the trajectories; move it clear of the data.
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)

    # Right: time traces. Pole angle and cart position both converge to the origin;
    # the cart starts at rest at the origin, excurses during the swing-up, and returns.
    ax2b = ax2.twinx()
    for i in range(n_traj):
        ax2.plot(t, 1.0 - np.cos(trajs_v[i, :, 0]), color="C0", lw=1.0, alpha=0.7)
        ax2b.plot(t, trajs_v[i, :, 1], color="C3", lw=1.0, alpha=0.7)
    ax2.axhline(0.0, color="0.5", lw=0.8, ls="--")
    ax2.set_title("Swing-up and cart regulation")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel(r"$1-\cos\theta$  (0 = upright)", color="C0")
    ax2b.set_ylabel(r"cart position $s$ (m)", color="C3")
    ax2.tick_params(axis="y", labelcolor="C0"); ax2b.tick_params(axis="y", labelcolor="C3")
    ax2.set_xlim([0, 8.0]); ax2.set_ylim([-0.1, 2.1])
    ax2.plot([], [], color="C0", label=r"$1-\cos\theta\to0$ (upright)")
    ax2.plot([], [], color="C3", label=r"$s\to0$ (centered)")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)

    fig.savefig(FIGDIR / "cartpole_eigfun_combined.pdf")
    plt.close(fig)
    print(f"Plots saved. Deterministic catch in figure batch: {int(caught.sum())}/{n_traj}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate and Plot Cartpole Eigenfunction Policy")
    parser.add_argument("--sigma", type=float, default=1.0, help="Checkpoint sigma to load")
    args = parser.parse_args()

    model = load_model(args.sigma)
    plot_all(sigma=args.sigma)
    evaluate_lqr_catch(model)
