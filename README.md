# Reproducing the results

Every figure, table and quoted constant in *Certified Energy Shaping of
Port-Hamiltonian Systems via Linearly-Solvable Ergodic Control* is produced by one
of the commands below.  Runtimes are wall-clock on a 2020-era 8-core laptop CPU.

## Requirements

* Python >= 3.12 and [uv](https://docs.astral.sh/uv/).  `uv sync` installs everything
  in `pyproject.toml`; every command below is run through `uv run`.
* **MOSEK** with a valid licence, for the sum-of-squares programs only (Section 8.3).
  Put `mosek.lic` at the repository root or in `~/mosek/`; either is found
  automatically.  Academic licences are free.  The figure and table commands do not
  need it.
* A GPU is optional.  JAX falls back to CPU with a warning; only training benefits.

```bash
uv sync
```

Run every command from the repository root: `plot_unmatched.py` imports
`pendulum.unmatched_noise` as a package, which only resolves with the root on
`sys.path`.

Output goes to `figures/`, both the PDFs and the three small `.tex` files holding
scalars the manuscript `\input`s (the catch rate and the two certified levels).

`checkpoints/eigfun/cartpole_sigma1.0.eqx` is the trained cart-pole value network at
the deployment noise level, included so that Fig. 2 and Table 1 reproduce in under
three minutes on a laptop.  Retraining it from scratch (below) is not necessary to
reproduce anything in the paper.

## Figures and tables

| Artifact | Command | Time |
| --- | --- | --- |
| Fig. 1, pendulum field and phase portrait | `uv run python pendulum/plot_eigfun.py` | 3 s |
| Fig. 2, cart-pole swing-up | `uv run python cartpole/plot_cartpole.py` | 40 s |
| Fig. 3, certified regions of attraction | `uv run python certifier/plot_roa.py` | 2 s |
| Fig. 4, small-noise sweep | `uv run python pendulum/plot_sigma_sweep.py` | 4 s |
| Fig. 5, unmatched noise | `uv run python pendulum/plot_unmatched.py` | 53 s |
| Table 1, swing-up ablation | `uv run python cartpole/ablation.py -n 1024` | 167 s |

`ablation.py` takes `-n` (initial conditions), `--integrator {rk4,euler}`, `--dt` and
`--horizon`; the paper uses the defaults, `rk4` at `dt = 5e-3` over `12 s`.  Halving
the step is `--dt 0.0025`.

## Assumption 1(iv), the drift condition

The constants quoted in Sections 8.1 and 8.2 come from

```bash
uv run python pendulum/drift_condition.py      # c, C, b for the pendulum      (1 s)
uv run python cartpole/drift_condition.py      # the same for the cart-pole   (34 s)
```

## Sum-of-squares certificates (Section 8.3) — MOSEK required

```bash
uv run python certifier/pendulum_cert.py   # certified level gamma*, pendulum
uv run python certifier/cartpole_cert.py   # certified level gamma*, cart-pole
uv run python certifier/khasminskii.py     # residual ball delta*(sigma)
uv run python certifier/om_dual.py         # two-sided occupation-measure bounds on lambda*
```

`plot_roa.py` draws the certified sets from the levels these report and does not
re-solve them, so Fig. 3 can be regenerated without MOSEK.

## Training the cart-pole eigenfunction (Section 7)

The deployed controller is a log-domain network trained with sigma annealed from 2.0
to the deployment value 1.0.  This is the only step that wants a compute node; it
writes `checkpoints/eigfun/cartpole_<tag>.eqx`, which `plot_cartpole.py` and
`ablation.py` then load.

```bash
uv run python cartpole/eigfun_train.py --sigma-start 2.0 --sigma-end 1.0 \
    --steps 200000 --batch-size 4096 --q-cart 0.2
```

`--warm-start <ckpt>` fine-tunes from an existing checkpoint at a smaller sigma, and
`--eval-every` sets the catch-rate evaluation cadence.  Run `--help` for the rest.
