"""
Figure for the matched-noise robustness argument: (left) the optimal ergodic cost
lambda*_Delta depends analytically on the unmatched-noise intensity rho (smooth,
first-order); (right) the matched-synthesized controller degrades gracefully when
deployed on the unmatched plant. No knife-edge at rho = 0.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from pendulum.unmatched_noise import lambda_star, control_field_um, deploy_unmatched

# Uniform on-page text (~7.8pt) across all paper figures: base = 7.8 / scale.
# This 11in figure at \linewidth (scale ~0.59) -> base ~13.
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(parents=True, exist_ok=True)

rhos = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8])
lams = np.array([lambda_star(r) for r in rhos])
uf0, ths, vs, _ = control_field_um(0.0)
succ = np.array([100 * deploy_unmatched(uf0, ths, vs, r) for r in rhos])

# first-order tangent at rho=0 (forward difference on the first interval)
slope0 = (lams[1] - lams[0]) / (rhos[1] - rhos[0])

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

axL.plot(rhos, lams, "o-", color="C0", lw=1.8, ms=6, label=r"$\lambda^\star_\Xi$ (grid)")
axL.plot(rhos, lams[0] + slope0 * rhos, "--", color="0.5", lw=1.2,
         label=rf"first order: $\lambda^\star + {slope0:.2f}\,\rho$")
axL.set_xlabel(r"unmatched-noise intensity $\rho = \|\Xi\|/\sigma^2$")
axL.set_ylabel(r"optimal ergodic cost $\lambda^\star_\Xi$")
axL.set_title(r"Analytic dependence on the diffusion")
axL.legend(frameon=False); axL.grid(alpha=0.3)

axR.plot(rhos, succ, "s-", color="C3", lw=1.8, ms=6)
axR.axhline(succ[0], color="0.6", ls=":", lw=1.0)
axR.set_xlabel(r"unmatched-noise intensity $\rho = \|\Xi\|/\sigma^2$")
axR.set_ylabel(r"swing-up rate ($|\theta| < 0.6$)  [\%]")
axR.set_title(r"Matched controller on the unmatched plant")
axR.set_ylim(0, 100); axR.grid(alpha=0.3)

fig.savefig(FIG / "unmatched_noise.pdf")
plt.close(fig)
print(f"saved {FIG/'unmatched_noise.pdf'}")
print("rho :", np.round(rhos, 3))
print("lam :", np.round(lams, 3))
print("succ:", np.round(succ, 1))
