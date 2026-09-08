import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Uniform on-page text (~7.8pt) across all paper figures: base = 7.8 / scale,
# where scale = display_width / figsize_width. Here 5in panels at 0.49\linewidth
# (scale ~0.64) -> base ~12.
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

from pendulum.eigfun import build_generator, principal_eigfunction, control_field
from pendulum.dynamics import U_MAX, G_GRAV, L as L_LEN, B_DAMP, M
import jax.numpy as jnp
from pathlib import Path

FIGDIR = Path(__file__).resolve().parents[1] / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

def plot_all():
    Ntheta, Nv, vmax = 161, 201, 16.0
    q_alpha, q_beta = 3.0, 0.3
    sigma = 2.0
    
    A, thetas, vs, q = build_generator(Ntheta, Nv, vmax, sigma, q_alpha, q_beta, nonlinear=True)
    lam, psi = principal_eigfunction(A)
    uf = control_field(psi, thetas, vs, sigma)
    
    # 1. Plot log psi*
    logpsi = np.log(psi).reshape(Ntheta, Nv)
    TH, V = np.meshgrid(thetas, vs, indexing="ij")
    
    fig, ax = plt.subplots(figsize=(5, 4))
    # Scale the colour map to the velocity window actually shown: log psi spans
    # ~16 over the full grid but only ~6 over |v|<=8, so scaling to the full grid
    # spends most of the range on tails that are cropped and flattens the view.
    win = np.abs(vs) <= 8
    im = ax.pcolormesh(TH, V, logpsi, cmap="viridis", shading="auto",
                       vmin=logpsi[:, win].min(), vmax=logpsi[:, win].max())
    fig.colorbar(im, ax=ax)
    ax.set_title(r"$\log \psi^\star$ ($\sigma=2.0$)")
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\dot\theta$ (rad/s)")
    ax.set_xlim([thetas.min(), thetas.max()])
    ax.set_ylim([-8, 8])
    fig.tight_layout()
    fig.savefig(FIGDIR / "eigfun_logpsi.pdf")
    plt.close(fig)
    
    # 2. Plot control field
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.pcolormesh(TH, V, uf, cmap="RdBu_r", vmin=-15, vmax=15, shading="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(r"Control field $u^\star$ ($\sigma=2.0$)")
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\dot\theta$ (rad/s)")
    ax.set_xlim([thetas.min(), thetas.max()])
    ax.set_ylim([-8, 8])
    fig.tight_layout()
    fig.savefig(FIGDIR / "eigfun_control.pdf")
    plt.close(fig)

    # 3. Deploy and plot trajectories WITH saturation
    def roll_trajs(n_traj=30):
        rng = np.random.default_rng(42)
        dt = 6.0 / 600
        sq = np.sqrt(dt)
        X = np.array([np.pi - 0.05, 0.0])[None, :] + np.array([0.2, 0.2])[None, :] * rng.standard_normal((n_traj, 2))
        Bv = np.array([0.0, 1.0 / (M * L_LEN**2)])
        
        trajs = [X.copy()]
        from pendulum.eigfun import _bilin
        for _ in range(600):
            u = np.array([_bilin(uf, thetas, vs, X[k]) for k in range(n_traj)])
            u = np.clip(u, -U_MAX, U_MAX)  # STRICT SATURATION
            # RK4 with the control held over the step; the field u is a stored
            # grid interpolant, so it is re-read at each stage.
            def _f(Z):
                th_, v_ = Z[:, 0], Z[:, 1]
                c_ = (G_GRAV / L_LEN) * np.sin(th_) - B_DAMP * v_
                return np.stack([v_, c_ + Bv[1] * u], axis=1)
            k1 = _f(X); k2 = _f(X + dt/2*k1); k3 = _f(X + dt/2*k2); k4 = _f(X + dt*k3)
            X = X + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            X[:, 0] = (X[:, 0] + np.pi) % (2 * np.pi) - np.pi
            X = np.clip(X, [-np.pi, -40], [np.pi, 40])
            trajs.append(X.copy())
            
        return np.stack(trajs, axis=1) # (N, T, 2)
        
    trajs = roll_trajs(n_traj=5)
    fig, ax = plt.subplots(figsize=(5, 4))
    V_star = -sigma**2 * logpsi
    V_star -= V_star.min()
    
    # Plot Lyapunov level sets
    levels = np.linspace(0, np.percentile(V_star, 80), 12)
    cs = ax.contour(TH, V, V_star, levels=levels, colors='0.5', linewidths=0.7, alpha=0.6)
    
    for i in range(trajs.shape[0]):
        theta = trajs[i, :, 0]
        theta_dot = trajs[i, :, 1]
        wrap_idx = np.where(np.abs(np.diff(theta)) > np.pi)[0]
        theta_plot = np.insert(theta, wrap_idx + 1, np.nan)
        theta_dot_plot = np.insert(theta_dot, wrap_idx + 1, np.nan)
        ax.plot(theta_plot, theta_dot_plot, lw=1.5, alpha=0.8, color="C1")
    ax.scatter(trajs[:, 0, 0], trajs[:, 0, 1], s=16, color="C2", label="Start (hanging)", zorder=3)
    ax.scatter(trajs[:, -1, 0], trajs[:, -1, 1], s=16, color="C3", label="End", zorder=3)
    ax.plot(0, 0, "*", ms=16, color="k", label="Target", zorder=3)
    ax.set_title(r"Swing-up phase portrait ($|u| \leq 5$)")
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\dot\theta$ (rad/s)")
    ax.set_xlim([thetas.min(), thetas.max()])
    ax.set_ylim([-8, 8])
    # R12: legend was overlapping the phase-portrait trajectories.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "eigfun_phase.pdf", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    plot_all()
