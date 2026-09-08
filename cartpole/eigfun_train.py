import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from pathlib import Path

from dynamics import ph_drift, G_VEC, hamiltonian, X_STAR, mass_matrix, lqr_gains

# Catch-region LQR value matrix, shared with the evaluator.  Used as a *local*
# (windowed) anchor for V near x*, not a global basin: P couples p_theta into the
# control channel (P[3,2] != 0), so an un-windowed quadratic would damp the pole
# momentum that swing-up needs.  The window confines it to the catch cone.
_P_LQR_np, _ = lqr_gains()
P_LQR = jnp.array(_P_LQR_np, dtype=jnp.float32)

# Weight of the soft anchor pinning V(x*)=0 and grad V(x*)=0 (control -> 0 at target).
ANCHOR_W = 1.0

class LogEigenfunction(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key):
        # Input features: cos(theta), sin(theta), s, theta_dot, s_dot => 5
        self.mlp = eqx.nn.MLP(in_size=5, out_size=1, width_size=128, depth=3,
                              activation=jax.nn.swish, key=key)

    def __call__(self, x):
        theta, s, p_theta, p_s = x
        features = jnp.array([
            jnp.cos(theta),
            jnp.sin(theta),
            s,
            p_theta,
            p_s
        ])
        mlp_val = self.mlp(features)[0]
        # Local LQR value-function anchor, windowed to the catch cone (1-cos theta
        # small).  Gives V the correct curvature and a stabilizing control near the
        # upright target with continuity to the LQR catch controller, while vanishing
        # in the swing-up region so it does not suppress energy pumping.
        xi = jnp.array([jnp.sin(theta), s, p_theta, p_s])
        window = jnp.exp(-(1.0 - jnp.cos(theta)) / 0.2)
        basin = 0.5 * (xi @ (P_LQR @ xi)) * window
        return mlp_val + basin

class TrainState(eqx.Module):
    model: LogEigenfunction
    lambda_param: jnp.ndarray

# q(x): state cost.  Weights are module-level so the trainer can sweep the cart
# penalty Q_CART (read at jit-trace time, i.e. on the first step() call, after
# train() has set them).  Heavier Q_CART shapes V_theta to keep the cart centered
# during swing-up, shrinking the large cart excursion of the pump phase.
Q_THETA = 2.0      # upright (1 - cos theta)
Q_CART  = 0.2      # cart position s^2   <- the swept knob
Q_THDOT = 0.02     # pole angular velocity
Q_SDOT  = 0.02     # cart velocity

def state_cost(x):
    # Scaled to O(1) so that, against the control penalty (1/2 sigma^2)||u||^2 and
    # the GEAR-10 actuator, the unsaturated optimal control lands near the +/-1
    # authority and the V-dependent PDE terms are comparable to q.
    theta, s, p_theta, p_s = x
    M = mass_matrix(theta)
    v = jnp.linalg.solve(M, jnp.array([p_theta, p_s]))
    theta_dot, s_dot = v[0], v[1]
    q = (Q_THETA * (1.0 - jnp.cos(theta)) + Q_CART * s**2
         + Q_THDOT * theta_dot**2 + Q_SDOT * s_dot**2)
    return q

def residual_loss(model, lambda_param, x_batch, sigma):
    """
    Squared residual of the eigenvalue PDE for V_theta approx -sigma^2 log(psi*),
    in the WKB-natural (sigma^2-multiplied) form of eq. (15):

        f0.grad V + (sigma^2/2) tr(gg^T Hess V) - 1/2 (g^T grad V)^2
            + sigma^2 (q - lambda) = 0.

    Unlike dividing through by sigma^2, this form stays O(1) as sigma -> 0 (it
    limits smoothly to the eikonal f0.grad V = 1/2 (g^T grad V)^2), so the drift
    and control terms do not blow up and flatten V at small noise.  Adam absorbs
    the overall sigma-dependent scale.
    """
    sigma_sq = sigma**2

    def single_loss(x):
        f0 = ph_drift(x)
        V, grad_V = jax.value_and_grad(model)(x)

        # Only the (p_s, p_s) Hessian entry is needed: g hits the p_s channel only.
        def V_sdot(s_dot_val):
            x_mod = x.at[3].set(s_dot_val)
            return jax.grad(model)(x_mod)[3]
        d2V_dsdot2 = jax.grad(V_sdot)(x[3])

        g_grad_V = jnp.dot(G_VEC, grad_V)          # g^T grad V
        g_hess_g = (G_VEC[3]**2) * d2V_dsdot2       # g^T (Hess V) g
        q_val = state_cost(x)

        R = (jnp.dot(f0, grad_V)
             + 0.5 * sigma_sq * g_hess_g
             - 0.5 * g_grad_V**2
             + sigma_sq * (q_val - lambda_param))
        return R**2

    pde = jnp.mean(jax.vmap(single_loss)(x_batch))

    # Soft anchor: pin V(x*) = 0 and grad V(x*) = 0 so x* is the value minimum and
    # the eigenfunction control vanishes at the target (no chattering at upright).
    V0, gV0 = jax.value_and_grad(model)(X_STAR)
    anchor = V0**2 + jnp.sum(gV0**2)

    return pde + ANCHOR_W * anchor

def loss_fn(state: TrainState, static_state: TrainState, x_batch, sigma):
    # Combine back to get the full model
    state = eqx.combine(state, static_state)
    return residual_loss(state.model, state.lambda_param, x_batch, sigma)

@eqx.filter_jit
def step(state: TrainState, optimizer, opt_state, x_batch, sigma):
    diff, static = eqx.partition(state, eqx.is_inexact_array)
    
    loss, grads = eqx.filter_value_and_grad(loss_fn)(diff, static, x_batch, sigma)
    
    updates, opt_state = optimizer.update(grads, opt_state, diff)
    diff = eqx.apply_updates(diff, updates)
    
    state = eqx.combine(diff, static)
    return state, opt_state, loss

def generate_batch(key, batch_size):
    """Uniform sampling of state space for collocation, focused on the trajectory domain."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    theta = jax.random.uniform(k1, (batch_size,), minval=-jnp.pi, maxval=jnp.pi)
    s = jax.random.uniform(k2, (batch_size,), minval=-1.0, maxval=1.0)
    theta_dot = jax.random.uniform(k3, (batch_size,), minval=-12.0, maxval=12.0)
    s_dot = jax.random.uniform(k4, (batch_size,), minval=-6.0, maxval=6.0)

    # ── Critical Fix: Convert velocities to canonical momenta ────────────────
    # The port-Hamiltonian state vector is x = (theta, s, p_theta, p_s).
    # Feeding raw velocities as momenta breaks the kinetic energy and drift physics.
    from dynamics import mass_matrix
    def to_p(th, s_pos, th_d, s_d):
        M = mass_matrix(th)
        p = M @ jnp.array([th_d, s_d])
        return jnp.array([th, s_pos, p[0], p[1]])
        
    return jax.vmap(to_p)(theta, s, theta_dot, s_dot)

def train(steps=50000, batch_size=4096, lr=1e-3, alpha=1e-3, init_lambda=2.0,
          sigma_start=2.0, sigma_end=0.1, q_cart=0.2, warm_start=None, eval_every=10000):
    global Q_CART
    Q_CART = q_cart                          # shapes V_theta; read at jit-trace time
    tag = f"sigma{sigma_end}_qc{q_cart}"      # unique checkpoint per sweep value
    print(f"[q_cart = {q_cart}]  cart-position penalty in the state cost", flush=True)
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    model = LogEigenfunction(subkey)
    lambda_param = jnp.array(init_lambda)
    # Warm start: initialize V_theta's weights from a converged checkpoint (e.g. the
    # good sigma=1.0 model) before fine-tuning at a smaller sigma, with a fresh
    # optimizer.  Lands the optimizer in the basin of the correct (non-flat) solution
    # so a small sigma step tracks it instead of collapsing to the trivial eikonal
    # solution.  Use early stopping (keep the best eval, not the last).
    if warm_start is not None:
        templ = TrainState(model=LogEigenfunction(subkey), lambda_param=lambda_param)
        try:
            model = eqx.tree_deserialise_leaves(warm_start, templ).model
        except Exception:                       # checkpoint saved as (state, opt_state)
            model = eqx.tree_deserialise_leaves(warm_start, (templ, None))[0].model
        print(f"[warm start] loaded V_theta weights from {warm_start}", flush=True)
    state = TrainState(model=model, lambda_param=lambda_param)
    
    # Cosine decay schedule for the learning rate
    lr_schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=steps, alpha=alpha)
    optimizer = optax.adam(lr_schedule)
    diff, static = eqx.partition(state, eqx.is_inexact_array)
    opt_state = optimizer.init(diff)
    
    # Resume from checkpoint if it exists (unique per sigma_end and q_cart)
    ckpt_path = Path(__file__).parent.parent / "checkpoints" / "eigfun" / f"cartpole_{tag}.eqx"
    if ckpt_path.exists():
        print(f"Resuming training from existing checkpoint: {ckpt_path} (Warm Start)")
        try:
            # Load the state (model + lambda) from the checkpoint.
            # To ensure the learning rate schedule and noise schedules are perfectly synchronized
            # (both starting at their initial values and decaying together), we initialize a
            # fresh, clean optimizer state while retaining the pre-trained weights and lambda.
            loaded = eqx.tree_deserialise_leaves(ckpt_path, (state, None))
            state = loaded[0]
            print("Successfully loaded model and lambda state.")
        except Exception as e:
            try:
                state = eqx.tree_deserialise_leaves(ckpt_path, state)
                print("Successfully loaded state.")
            except Exception:
                print("Warning: Could not load state cleanly. Starting from scratch...")
        diff, static = eqx.partition(state, eqx.is_inexact_array)
        opt_state = optimizer.init(diff)
            
    print(f"Training Eigenfunction for Cartpole (sigma_start={sigma_start}, sigma_end={sigma_end})")
    
    import time
    start_time = time.time()
    
    for i in range(steps):
        # Generate a fresh, dynamic batch of collocation points at each step to prevent grid overfitting
        key, subkey = jax.random.split(key)
        x_batch = generate_batch(subkey, batch_size)
        
        # Cosine decay of sigma
        if steps > 1:
            frac = i / (steps - 1)
            current_sigma = sigma_end + 0.5 * (sigma_start - sigma_end) * (1.0 + np.cos(np.pi * frac))
        else:
            current_sigma = sigma_end
            
        sigma_jax = jnp.array(current_sigma, dtype=jnp.float32)
        
        state, opt_state, loss = step(
            state, optimizer, opt_state, x_batch, sigma_jax
        )
        
        if i % 1000 == 0 or i == steps - 1:
            actual_lambda = state.lambda_param
            elapsed = time.time() - start_time
            if i > 0:
                steps_per_sec = i / elapsed
                eta_sec = (steps - i) / steps_per_sec
                eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_sec))
            else:
                eta_str = "Calculating..."
                
            print(f"Step {i:06d}/{steps} | Loss: {loss:.4e} | Lambda: {actual_lambda:.4f} | Sigma: {current_sigma:.4f} | ETA: {eta_str}", flush=True)
            ckpt_dir = Path(__file__).parent.parent / "checkpoints" / "eigfun"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            eqx.tree_serialise_leaves(ckpt_dir / f"cartpole_{tag}.eqx", (state, opt_state))
            
        if i % eval_every == 0 or i == steps - 1:
            # Evaluate LQR Catch Statistics
            import sys
            if "plot_cartpole" not in sys.modules:
                from plot_cartpole import evaluate_lqr_catch
            else:
                evaluate_lqr_catch = sys.modules["plot_cartpole"].evaluate_lqr_catch
            evaluate_lqr_catch(state.model, current_sigma)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Eigenfunction for Cartpole")
    parser.add_argument("--sigma-start", type=float, default=2.0, help="Starting diffusion noise scale")
    parser.add_argument("--sigma-end", type=float, default=0.1, help="Ending diffusion noise scale")
    parser.add_argument("--steps", type=int, default=200000, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--alpha", type=float, default=1e-3, help="Fraction of lr for the final cosine decay value")
    parser.add_argument("--init-lambda", type=float, default=2.0, help="Initial guess for lambda (O(few) with the rescaled cost)")
    parser.add_argument("--batch-size", type=int, default=4096, help="Collocation batch size")
    parser.add_argument("--q-cart", type=float, default=0.2, help="Cart-position penalty s^2 weight in q (0.2 is the baseline)")
    parser.add_argument("--warm-start", type=str, default=None, help="Checkpoint path to initialize V_theta from before fine-tuning at a smaller sigma (e.g. the good cartpole_sigma1.0.eqx)")
    parser.add_argument("--eval-every", type=int, default=10000, help="Eval (catch + overshoot) cadence in steps; use a small value (e.g. 2000) to monitor a warm-start fine-tune for early stopping")
    args = parser.parse_args()
    train(steps=args.steps, batch_size=args.batch_size, lr=args.lr, alpha=args.alpha,
          init_lambda=args.init_lambda, sigma_start=args.sigma_start, sigma_end=args.sigma_end,
          q_cart=args.q_cart, warm_start=args.warm_start, eval_every=args.eval_every)
