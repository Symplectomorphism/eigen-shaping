"""
Visualize Theorem 3: as the noise sigma decreases, the stochastic eigenfunction
control field u*_sigma = sigma^2 d/dtheta_dot log psi*_sigma sharpens about the
deterministic energy-pumping switching manifold (the curve u* = 0). Inverted
pendulum, exact grid eigensolve.

Each panel is normalized to its own field amplitude (divided by the 98th
percentile of |u*| in the window). This strips the sigma^2 magnitude prefactor,
which otherwise dominates the picture and makes the large-sigma panel the most
saturated, and exposes the *shape* of the field: the switching manifold and the
red/blue sign pattern are sigma-robust, and the transition across u* = 0 steepens
as sigma decreases. The raw field magnitude is NOT comparable across panels by
design; the colorbar is in per-panel normalized units.

The grid eigensolve underflows below sigma ~ 0.7 (psi* ~ exp(-H_d/sigma^2) drops
below double precision), so the sweep spans sigma in {2, 1, 0.7}; the field is
masked where psi* has underflowed. No analytic sigma->0 limit is overlaid: the true
limit is the quasi-potential gradient -g^T grad H_d^qp, not the textbook energy
pump.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Uniform on-page text (~7.8pt) across all paper figures: base = 7.8 / scale.
# This 15in figure at \linewidth (scale ~0.43) -> base ~18.
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})
try:
    from eigfun import build_generator, principal_eigfunction, control_field
except ModuleNotFoundError:
    from pendulum.eigfun import build_generator, principal_eigfunction, control_field

FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(parents=True, exist_ok=True)

NTH, NV, VMAX = 161, 201, 16.0
QA, QB = 3.0, 0.3
SIGMAS = [2.0, 1.0, 0.7]
PCTL = 98.0          # robust per-panel amplitude (percentile of |u*|)
VLIM = 10.0          # theta_dot window shown (avoid Dirichlet boundary band)

# sharey: panels 2 and 3 have the same theta_dot axis as panel 1, so their
# y-tick labels are dropped, reclaiming horizontal space for the panels.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True,
                         sharey=True)
im = None
for ax, sigma in zip(axes, SIGMAS):
    A, ths, vs, q = build_generator(NTH, NV, VMAX, sigma, QA, QB, nonlinear=True)
    lam, psi = principal_eigfunction(A)
    uf = control_field(psi, ths, vs, sigma)
    psi2 = psi.reshape(NTH, NV)
    underflow = (psi2 < psi2.max() * 1e-10)
    ufm = np.where(underflow, np.nan, uf)
    TH, V = np.meshgrid(ths, vs, indexing="ij")
    sel = np.abs(vs) <= VLIM
    THs, Vs, Us = TH[:, sel], V[:, sel], ufm[:, sel]
    # Per-panel normalization: divide by a robust amplitude so the field SHAPE
    # (transition steepness about u* = 0) is comparable across sigma, with the
    # sigma^2 magnitude prefactor removed.
    scale = np.nanpercentile(np.abs(Us), PCTL)
    Un = np.clip(Us / scale, -1.0, 1.0)
    im = ax.pcolormesh(THs, Vs, Un, cmap="RdBu_r",
                       vmin=-1.0, vmax=1.0, shading="auto")
    # switching manifold u* = 0 (its location is sigma-robust; the transition
    # across it steepens as sigma -> 0)
    with np.errstate(invalid="ignore"):
        ax.contour(THs, Vs, Us, levels=[0.0], colors="k", linewidths=1.6)
    uf_win = 100.0 * underflow[:, sel].mean()      # underflow within the shown window
    ax.set_title(rf"$\sigma = {sigma}$")
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-VLIM, VLIM)
    ax.plot(0, 0, "k*", ms=12, zorder=5)
axes[0].set_ylabel(r"$\dot\theta$ (rad/s)")
cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.01)
cb.set_label(r"$u^\star_\sigma\,/\,|u^\star_\sigma|_{\max}$")
fig.savefig(FIG / "sigma_sweep.pdf")
plt.close(fig)
print(f"saved {FIG/'sigma_sweep.pdf'}")
