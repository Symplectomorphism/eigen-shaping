"""
Occupation-measure dual: certified two-sided SOS bounds on the optimal ergodic
cost lambda* (the principal eigenvalue of L - q), via the Collatz--Wielandt /
Donsker--Varadhan inequalities. For any psi > 0,
    inf_x [q - L psi / psi]  <=  lambda*  <=  sup_x [q - L psi / psi].
Multiplying by psi > 0 (Hopf--Cole structure) linearizes each side into SOS:
    lower:  (q - lambda) psi - L psi  is SOS  (=> lambda <= lambda*),  maximize lambda
    upper:  L psi - (q - lambda) psi  is SOS  (=> lambda >= lambda*),  minimize lambda
with psi SOS and psi(x*) = 1. The gap (lambda_up - lambda_lo) is the certified
suboptimality bound Delta_r. Pendulum (m l^2 = 1, detM = 1) here; cart-pole adds
the detM clearing.
"""
from __future__ import annotations
import os, sys
import numpy as np
import sympy as sp
import cvxpy as cp
from sos import SOSProblem, CvxPoly, to_dict, monomials, deriv, evaluate

# ---- pendulum prior generator (matches pendulum/eigfun.py, sigma=2) -----------
G_L, B, SIGMA, QA, QB = 9.81, 0.2, 2.0, 3.0, 0.3
s, c, w = sp.symbols("s c w", real=True)
VARS = (s, c, w)
XSTAR = (0.0, 1.0, 0.0)                       # (s, c, w) at upright

Q = to_dict(QA*(1 - c) + 0.5*QB*w**2, VARS)
CONS = to_dict(s**2 + c**2 - 1, VARS)
F0 = [to_dict(c*w, VARS), to_dict(-s*w, VARS), to_dict(G_L*s - B*w, VARS)]  # drift on (s,c,w)


def L_psi(psi):
    """Generator L psi = f0 . grad psi + (sigma^2/2) d^2 psi / dw^2 (CvxPoly)."""
    out = deriv(psi, 0)*F0[0] + deriv(psi, 1)*F0[1] + deriv(psi, 2)*F0[2]
    return out + deriv(deriv(psi, 2), 2) * (SIGMA**2 / 2)


def _q_shift(lam):
    d = dict(Q); d[(0, 0, 0)] = d.get((0, 0, 0), 0.0) - lam      # q - lambda
    return d


def _feasible(lam, dpsi=2):
    """Returns (is_feasible, mosek_status). Tests whether (q-lambda)psi - L psi is
    SOS for some SOS psi with psi(x*)=1, i.e. whether lam is a valid lower bound."""
    p = SOSProblem(VARS)
    psi = p.sos(monomials(VARS, dpsi))
    outer = p.sos(monomials(VARS, dpsi + 1))
    Lm = p.free(monomials(VARS, 2 * dpsi))
    qml = _q_shift(lam)                                          # q - lambda
    ident = psi*qml - L_psi(psi) - Lm*CONS - outer              # (q-lambda)psi - L psi  is SOS
    cons = [Qv >> 0 for Qv in p._psd] + [evaluate(psi, XSTAR) == 1]
    for cc in ident.terms.values():
        if isinstance(cc, (int, float)):
            if abs(cc) > 1e-8:
                return False, "const-mismatch"
        else:
            cons.append(cc == 0)
    try:
        prob = cp.Problem(cp.Minimize(0), cons)
        params = {"MSK_IPAR_NUM_THREADS": MOSEK_THREADS} if MOSEK_THREADS > 0 else {}
        prob.solve(solver=cp.MOSEK, mosek_params=params)
        return prob.status in ("optimal", "optimal_inaccurate"), prob.status
    except cp.error.SolverError:
        return False, "solver_error"


# 0 = MOSEK default (all cores). In --jobs>1 mode the driver sets OMDUAL_THREADS so
# each parallel degree caps its MOSEK threads (avoids oversubscription). Read from
# env so the cap propagates to fork- AND spawn-started workers.
MOSEK_THREADS = int(os.environ.get("OMDUAL_THREADS", "0"))


def lower_bound(dpsi, lo=-2.0, hi=8.0, iters=22):
    """Largest lam with (q-lam)psi - L psi SOS -> certified lam <= lambda*.
    Returns (lam_lo, n_solves, final_status, bracket_width)."""
    ok, st = _feasible(lo, dpsi); n = 1
    if not ok:
        return None, n, st, hi - lo
    last = st
    for _ in range(iters):
        m = 0.5*(lo + hi); ok, st = _feasible(m, dpsi); n += 1
        if ok:
            lo, last = m, st
        else:
            hi = m
    return lo, n, last, hi - lo


def grid_reference():
    """Principal eigenvalue lambda* from the dense grid eigensolve."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from pendulum.eigfun import build_generator, principal_eigfunction
    A, _, _, _ = build_generator(161, 201, 16.0, SIGMA, QA, QB, nonlinear=True)
    lam, _ = principal_eigfunction(A)
    return lam


def _run_one(dpsi):
    """Worker: certified lower bound at psi half-degree dpsi. Returns a dict row."""
    import time
    npsi = len(monomials(VARS, dpsi)); nout = len(monomials(VARS, dpsi + 1))
    nL = len(monomials(VARS, 2 * dpsi))
    t0 = time.time()
    try:
        lb, n, st, _ = lower_bound(dpsi); dt = time.time() - t0
        return dict(d=dpsi, npsi=npsi, nout=nout, nL=nL, lb=lb, n=n, st=st, dt=dt)
    except Exception as e:
        return dict(d=dpsi, npsi=npsi, nout=nout, nL=nL, lb=None, n=0,
                    st="EXC:" + repr(e)[:40], dt=time.time() - t0)


if __name__ == "__main__":
    import time, argparse
    ap = argparse.ArgumentParser(description="Occupation-measure dual lower bounds on lambda*.")
    ap.add_argument("degrees", nargs="*", type=int, help="psi half-degrees (psi_deg = 2x); default 2..6")
    ap.add_argument("--jobs", type=int, default=1, help="parallel degrees (CPU). Caps MOSEK threads per job.")
    ap.add_argument("--threads-per-job", type=int, default=0, help="MOSEK threads per job (0=auto split by cores)")
    args = ap.parse_args()
    dpsis = args.degrees or [2, 3, 4, 5, 6]

    if args.jobs > 1:
        import multiprocessing as mp
        ncpu = mp.cpu_count()
        tpj = args.threads_per_job or max(1, ncpu // args.jobs)
        os.environ["OMDUAL_THREADS"] = str(tpj)         # inherited by workers
        print(f"[parallel: {args.jobs} jobs x {tpj} MOSEK threads on {ncpu} cores]", flush=True)

    print("=" * 78, flush=True)
    print("Occupation-measure (Donsker-Varadhan) DUAL: certified lower bound on the", flush=True)
    print("optimal ergodic cost lambda*  (principal eigenvalue of L - q).", flush=True)
    print(f"system : inverted pendulum, lift z=(sin th, cos th, th_dot), sigma={SIGMA}", flush=True)
    print(f"cost   : q = {QA}(1-cos th) + {0.5*QB} th_dot^2 ;  domain: th_dot unbounded", flush=True)
    print("dual   : max lambda s.t.  (q-lambda) psi - L psi  is SOS,  psi SOS, psi(x*)=1", flush=True)
    print("         -> lambda <= lambda*; degree-r hierarchy increases monotonically.", flush=True)
    print("note   : bound rises slowly because polynomial psi must approximate the", flush=True)
    print("         Gaussian-tailed eigenfunction on the unbounded velocity axis.", flush=True)
    print("-" * 78, flush=True)
    try:
        lam_star = -grid_reference()
        print(f"reference (dense grid eigensolve):  lambda* = {lam_star:.4f}", flush=True)
    except Exception as e:
        lam_star = None
        print("reference grid eigensolve UNAVAILABLE:", repr(e)[:90], flush=True)
    print("-" * 78, flush=True)
    print(f"{'psi_deg':>7} {'psiGram':>7} {'outer':>6} {'L_free':>6} "
          f"{'lambda_lo':>10} {'gap':>8} {'solves':>6} {'MOSEK':>10} {'time_s':>7}", flush=True)

    def _emit(r):
        if r["lb"] is None:
            print(f"{2*r['d']:>7} {r['npsi']:>7} {r['nout']:>6} {r['nL']:>6} "
                  f"{'infeasible':>10} {'-':>8} {r['n']:>6} {r['st']:>10} {r['dt']:>7.0f}", flush=True)
        else:
            gap = f"{lam_star-r['lb']:+.3f}" if lam_star is not None else "n/a"
            print(f"{2*r['d']:>7} {r['npsi']:>7} {r['nout']:>6} {r['nL']:>6} {r['lb']:>10.4f} "
                  f"{gap:>8} {r['n']:>6} {r['st']:>10} {r['dt']:>7.0f}", flush=True)

    if args.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(args.jobs) as pool:
            for r in pool.imap_unordered(_run_one, dpsis):   # may arrive out of order
                _emit(r)
    else:
        for dpsi in dpsis:
            _emit(_run_one(dpsi))
    print("=" * 78, flush=True)
    print("Interpretation: each row certifies lambda* >= lambda_lo (exact, modulo MOSEK", flush=True)
    print("tolerance). A deployed controller of average cost J then has certified", flush=True)
    print("suboptimality at most J - lambda_lo. 'gap' is lambda* - lambda_lo (smaller is", flush=True)
    print("tighter). MOSEK status 'optimal' = trustworthy; 'optimal_inaccurate' = mild", flush=True)
    print("numerical warning; 'solver_error'/'const-mismatch' = ignore that row.", flush=True)
