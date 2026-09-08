"""
Visualize the SOS-certified regions of attraction Omega_gamma = {w2* <= gamma*}
for the pendulum and the cart-pole, as theta-theta_dot slices, with the control
saturation boundary and a few certified closed-loop trajectories flowing in.
"""
from __future__ import annotations
import os
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Uniform on-page text (~7.8pt) across all paper figures: base = 7.8 / scale.
# This 11.5in figure at \linewidth (scale ~0.57) -> base ~14.
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 14,
})

import pendulum_cert as PC
import cartpole_cert as CC
import cartpole_dyn as CD

FIG = Path(__file__).resolve().parents[2] / "notes" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# certified results (from the bisection runs)
GAMMA_P, GAMMA_C = 1.998, 0.3099
UMAX_P, UMAX_C = PC.U_MAX, CC.U_MAX

# ---- lambdified Lyapunov functions / controls ---------------------------------
Wp = sp.lambdify(PC.VARS, PC.W, "numpy")          # (s, c, w)
up = sp.lambdify(PC.VARS, PC.u_ctrl, "numpy")
Wc = sp.lambdify(CC.VARS, CC.W, "numpy")          # (si, co, xc, vth, vx)
p12, p22 = float(PC.P_V[0, 1]), float(PC.P_V[1, 1])
Kc = CC.Klin
a, bb, dd, g = float(CD.a), float(CD.bb), float(CD.dd), float(CD.G_GRAV)
GEAR = float(CD.GEAR)


def rk4(rhs, x0, dt, n):
    xs = [np.array(x0, float)]
    for _ in range(n):
        x = xs[-1]
        k1 = rhs(x); k2 = rhs(x + dt/2*k1); k3 = rhs(x + dt/2*k2); k4 = rhs(x + dt*k3)
        xs.append(x + dt/6*(k1 + 2*k2 + 2*k3 + k4))
    return np.array(xs)


def pend_rhs(x):
    th, w = x
    u = np.clip(-(p12*np.sin(th) + p22*w), -UMAX_P, UMAX_P)
    return np.array([w, g*np.sin(th) - 0.2*w + u])


def cart_rhs(x):
    th, xc, thd, xd = x
    M = np.array([[a, bb*np.cos(th)], [bb*np.cos(th), dd]])
    u = np.clip(-(Kc[0]*np.sin(th) + Kc[1]*xc + Kc[2]*thd + Kc[3]*xd), -UMAX_C, UMAX_C)
    tau = np.array([bb*g*np.sin(th), bb*np.sin(th)*thd**2 + GEAR*u])
    acc = np.linalg.solve(M, tau)
    return np.array([thd, xd, acc[0], acc[1]])


def panel(ax, Wgrid, gamma, Ugrid, umax, title, ylim):
    th = np.linspace(-np.pi, np.pi, Wgrid.shape[1])
    thd = np.linspace(-ylim, ylim, Wgrid.shape[0])
    TH, THD = np.meshgrid(th, thd)
    ax.contourf(TH, THD, (Wgrid <= gamma).astype(float), levels=[0.5, 2],
                colors=["#9ecae1"], alpha=0.65)
    cs = ax.contour(TH, THD, Wgrid, levels=[gamma], colors="C0", linewidths=2.2)
    ax.contour(TH, THD, Ugrid, levels=[umax], colors="0.35", linestyles="--", linewidths=1.0)
    ax.plot(0, 0, "k*", ms=15, zorder=5)
    ax.set_xlabel(r"$\theta$ (rad)"); ax.set_ylabel(r"$\dot\theta$ (rad/s)")
    ax.set_title(title); ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-ylim, ylim)
    return cs


N = 500
th = np.linspace(-np.pi, np.pi, N)

fig, (axp, axc) = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)

# pendulum panel
thd_p = np.linspace(-8, 8, N); TH, THD = np.meshgrid(th, thd_p)
WPg = Wp(np.sin(TH), np.cos(TH), THD)
UPg = np.abs(up(np.sin(TH), np.cos(TH), THD))
panel(axp, WPg, GAMMA_P, UPg, UMAX_P,
      rf"Pendulum: $\Omega_{{\gamma^\star}}$, $\gamma^\star={GAMMA_P:.2f}$", 8)
# trajectories from the boundary of Omega_gamma
for th0 in np.linspace(-0.55, 0.55, 5):
    for w0 in (-1.9, 1.9):
        if Wp(np.sin(th0), np.cos(th0), w0) <= GAMMA_P:
            tr = rk4(pend_rhs, [th0, w0], 0.01, 400)
            axp.plot(tr[:, 0], tr[:, 1], color="C3", lw=0.9, alpha=0.8)

# cart-pole panel: theta-theta_dot slice at xc=0, x_dot=0
thd_c = np.linspace(-8, 8, N); TH, THD = np.meshgrid(th, thd_c)
WCg = Wc(np.sin(TH), np.cos(TH), 0*TH, THD, 0*TH)
UCg = np.abs(Kc[0]*np.sin(TH) + Kc[2]*THD)            # |u| at xc=0, x_dot=0
panel(axc, WCg, GAMMA_C, UCg, UMAX_C,
      rf"Cart-pole ($s=\dot s=0$ slice): $\gamma^\star={GAMMA_C:.2f}$", 8)
for th0 in np.linspace(-0.8, 0.8, 5):
    for w0 in (-2.0, 2.0):
        if Wc(np.sin(th0), np.cos(th0), 0, w0, 0) <= GAMMA_C:
            tr = rk4(cart_rhs, [th0, 0, w0, 0], 0.005, 1200)
            axc.plot(tr[:, 0], tr[:, 2], color="C3", lw=0.9, alpha=0.8)

# legend proxies
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color="C0", lw=2.2),
            Line2D([], [], color="0.35", ls="--"),
            Line2D([], [], color="C3", lw=0.9)],
           [r"$\partial\Omega_{\gamma^\star}$", r"$|u|=U_{\max}$", "closed-loop trajectories"],
           loc="outside lower center", ncol=3, frameon=False)

fig.savefig(FIG / "certified_roa.pdf")
plt.close(fig)
print(f"saved {FIG/'certified_roa.pdf'}")

# write gamma* values for the paper
for name, val in [("roa_gamma_pendulum", GAMMA_P), ("roa_gamma_cartpole", GAMMA_C)]:
    (Path(__file__).parents[2] / "papers" / "eigenfunction" / f"{name}.tex").write_text(f"{val:.2f}")
print("wrote roa_gamma_*.tex")
