import math
import logging
from typing import Any, Callable, Optional, Tuple, Dict
import torch
from torch.optim.optimizer import Optimizer

# -----------------------------------------------------------------------------
# Utility Functions (Ported from Megatron-LM spectral_ball_utils.py)
# -----------------------------------------------------------------------------

@torch.no_grad()
def _muon_newton_schulz_step(X: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    """One Newton-Schulz iteration: X ← a·X + X·(b·A + c·A²) where A = X·X^T."""
    A = X @ X.mT
    B = torch.addmm(A, A, A, alpha=c, beta=b)
    X = torch.addmm(X, B, X, alpha=1.0, beta=a)
    return X

@torch.compile
@torch.no_grad()
def _small_msign(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Matrix sign via Newton-Schulz with Polar-Express coefficients."""
    if G.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")
    # Assuming float32 input as per original code
    
    transpose = G.size(-2) > G.size(-1)
    X = G.mT if transpose else G
    X = torch.nn.functional.normalize(X, p=2, dim=(-2, -1), eps=1e-7)
    
    # Cast to bfloat16 for performance if supported/desired, effectively mimicking Megatron's choice
    # Note: Megatron had a warning about NOT using bf16 for msign, but then utilized it for performance in _small_msign?
    # Let's stick to fp32 to be safe as per the warning in the comments of the source.
    # "WARNING: DO NOT run `msign` in bfloat16! ... Always keep `msign` computations in full fp32."
    # But the code also had: # X = X.to(torch.bfloat16)  <-- commented out or present. 
    # The read file output showed `X = X.to(torch.bfloat16)` active in _small_msign. 
    # I will follow exact code I read:
    X = X.to(torch.bfloat16)
    
    coeffs = [
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ]
 
    for i in range(steps):
        if i < 8:
            a, b, c = coeffs[i]
        else:
            a, b, c = coeffs[-1]
        X = _muon_newton_schulz_step(X, a, b, c)
    
    # Cast back to original dtype (likely fp32 for G)
    X = X.to(G.dtype)

    return X.mT if transpose else X

@torch.compile
@torch.no_grad()
def _large_msign(G: torch.Tensor, steps: int) -> torch.Tensor:
    # Requires custom newton_schulz implementation or similar.
    # For now, we fall back to _small_msign or a standard loop if custom CUDA kernel isn't available.
    # Since I don't have the `emerging_optimizers.orthogonalized_optimizers.muon_utils.newton_schulz` source,
    # I will implement a standard loop here similar to _small_msign but strictly fp32 for stability on large matrices.
    
    transpose = G.size(-2) > G.size(-1)
    X = G.mT if transpose else G
    X = torch.nn.functional.normalize(X, p=2, dim=(-2, -1), eps=1e-7)
    
    coeffs = [
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ]
    
    for i in range(steps):
        if i < 8:
            a, b, c = coeffs[i]
        else:
            a, b, c = coeffs[-1]
        X = _muon_newton_schulz_step(X, a, b, c)

    return X.mT if transpose else X

@torch.no_grad()
def msign(G: torch.Tensor, steps: int) -> torch.Tensor:
    # Logic from Megatron: small vs large dispatch.
    # if G.shape[0] <= 512 or G.shape[1] <= 512: ...
    # Original code had `if True: return _small_msign(G, steps)` to force disable Triton branch.
    return _small_msign(G, steps)

@torch.compile
@torch.no_grad()
def power_iteration(w: torch.Tensor, steps: int = 50, eps: float = 1e-20, v_init: Optional[torch.Tensor] = None):
    """Leading singular triplet (σ, u, v) via bilateral power iteration (fp32/bf16)."""
    if w.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")

    # Cast to fp32 for stability during power iteration, although it costs more memory/bandwidth.
    # To be fast but stable: use bf16 for matmuls but float32 for accumulation/normalization loops?
    # PyTorch autocast might handle this, but explicit casting is safer for "nan" issues.
    # Megatron uses bf16 for w, so we stick to it but ensure v is handled robustly.
    w_bf16 = w.to(torch.bfloat16)

    if v_init is not None:
        v = v_init.to(w_bf16.dtype)
        # Add small noise to break symmetry/orthogonality traps if cached
        v = v + 1e-4 * torch.randn_like(v)
    else:
        v = torch.randn_like(w_bf16[..., :1, :].transpose(-2, -1))
        
    for _ in range(steps):
        v = torch.nn.functional.normalize(w_bf16.transpose(-2, -1) @ (w_bf16 @ v), dim=-2)
    
    u = torch.nn.functional.normalize(w_bf16 @ v, dim=-2)
    s = (u.transpose(-2, -1) @ w_bf16 @ v).squeeze(-1).squeeze(-1)

    return s.float(), u.float(), v.float() # Return fp32

@torch.no_grad()
def apply_retract(
    W: torch.Tensor,
    sigma: float,
    target_radius: float,
    mode: str = 'hard',
    alpha: float = 0.05,
    current_lr: Optional[float] = None,
) -> float:
    """Apply retraction to spectral sphere."""
    if mode == 'hard':
        if max(sigma, 0.0) + 1e-8 != target_radius:
            scale_factor = target_radius / (max(sigma, 0.0) + 1e-8)
            W.mul_(scale_factor)
        return 0.0
    elif mode == 'dynamic':
        bias = -1.0 if sigma > target_radius else 1.0
        if current_lr is not None:
            effective_alpha = alpha * current_lr
        else:
            effective_alpha = alpha
        W.mul_(1.0 + effective_alpha * bias)
        return bias
    else:
        raise ValueError(f"Unknown retract mode: {mode}")

@torch.no_grad()
def inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum()

@torch.compile
@torch.no_grad()
def compute_f_tensor(G: torch.Tensor, Theta: torch.Tensor, lambda_value: torch.Tensor, msign_steps: int = 8) -> torch.Tensor:
    """f(λ) = <Θ, msign(G + λΘ)>. Returns 0-d tensor (no GPU sync)."""
    z = G + lambda_value * Theta
    Phi = msign(z, steps=msign_steps)
    return inner_product(Theta, Phi)

@torch.no_grad()
def find_bracket(
    G: torch.Tensor,
    Theta: torch.Tensor,
    initial_guess: float = 0.0,
    initial_step: float = 1e-3,
    max_expansions: int = 10,
    msign_steps: int = 8,
    tolerance_f: float = 1e-8,
) -> Tuple[float, float, float, float]:
    f = compute_f_tensor
    λ0 = initial_guess
    f0 = f(G, Theta, torch.tensor(λ0, device=G.device, dtype=G.dtype), msign_steps).item()

    if abs(f0) < tolerance_f:
        return λ0, λ0, f0, f0

    step = initial_step if f0 < 0 else -initial_step
    λ_prev = λ0
    f_prev = f0

    for _ in range(max_expansions):
        λ_new = λ_prev + step
        f_new = f(G, Theta, torch.tensor(λ_new, device=G.device, dtype=G.dtype), msign_steps).item()

        sign_prev = f_prev <= 0.0
        sign_new  = f_new  <= 0.0

        if sign_prev != sign_new:
            if f_prev <= 0 and f_new >= 0:
                λ_L, f_L = λ_prev, f_prev
                λ_R, f_R = λ_new, f_new
            elif f_new <= 0 and f_prev >= 0:
                λ_L, f_L = λ_new, f_new
                λ_R, f_R = λ_prev, f_prev
            else:
                if abs(f_prev) <= abs(f_new):
                    λ_L, λ_R, f_L, f_R = λ_prev, λ_prev, f_prev, f_prev
                else:
                    λ_L, λ_R, f_L, f_R = λ_new, λ_new, f_new, f_new
            return λ_L, λ_R, f_L, f_R

        step *= 2.0
        λ_prev, f_prev = λ_new, f_new

    return None, None, f0, f0

@torch.no_grad()
def solve_lambda_with_bisection(
    G: torch.Tensor,
    Theta: torch.Tensor,
    initial_guess: float = 0.0,
    initial_step: float = 1e-3,
    tolerance_f: float = 1e-6,
    max_iterations: int = 20,
    max_expansions: int = 10,
    msign_steps: int = 8,
) -> Tuple[float, bool, float, int]:
    
    λ_L, λ_R, f_L, f_R = find_bracket(
        G, Theta,
        initial_guess=initial_guess,
        initial_step=initial_step,
        max_expansions=max_expansions,
        msign_steps=msign_steps,
        tolerance_f=tolerance_f,
    )

    if λ_L is None:
        return 0.0, False, f_L , 0

    if abs(f_L) < abs(f_R):
        best_λ, best_f = λ_L, f_L
    else:
        best_λ, best_f = λ_R, f_R

    if abs(best_f) <= tolerance_f:
        return best_λ, True, abs(best_f), 0

    for it in range(1, max_iterations + 1):
        λ_mid = 0.5 * (λ_L + λ_R)
        f_mid = compute_f_tensor(G, Theta, torch.tensor(λ_mid, device=G.device, dtype=G.dtype), msign_steps).item()

        if abs(f_mid) < abs(best_f):
            best_λ, best_f = λ_mid, f_mid

        if abs(f_mid) <= tolerance_f:
            return λ_mid, True, abs(f_mid), it

        if f_mid < 0:
            λ_L, f_L = λ_mid, f_mid
        else:
            λ_R, f_R = λ_mid, f_mid

    return best_λ, False, abs(best_f), max_iterations

def compute_target_radius(shape: tuple, radius_mode: str, radius_scaler: float = 1.0) -> float:
    if radius_mode == "spectral_mup":
        n_out, n_in = shape
        return radius_scaler * math.sqrt(n_out / n_in)
    elif radius_mode == "identity":
        return radius_scaler * 1.0
    else:
        raise ValueError(f"Invalid radius_mode: {radius_mode}")

def get_spectral_ball_scale_factor(size_out: int, size_in: int, mode: str = "align_adamw_rms") -> float:
    if mode == "shape_scaling":
        return max(1, size_out / size_in) ** 0.5
    elif mode == "align_adamw_rms":
        return 0.2 * max(size_out, size_in) ** 0.5
    elif mode == "spectral_mup":
        return (size_out / size_in) ** 0.5
    else:
        raise ValueError(f"Invalid mode for SpectralBall update scale factor: {mode}")

# -----------------------------------------------------------------------------
# SSO Optimizer Class
# -----------------------------------------------------------------------------

class SSO(Optimizer):
    """
    Spectral Sphere Optimizer (SSO) 

    Implements steepest descent on the spectral sphere.
    Closely follows Megatron-LM implementation logic.
    """
    def __init__(
        self,
        params,
        lr: float = 3e-4,
        momentum_beta: float = 0.95,
        weight_decay: float = 0.0,
        *,
        use_nesterov: bool = False,
        fp32_matmul_prec: str = "medium",
        power_iteration_steps: int = 10,
        msign_steps: int = 5,
        solver: str = "bisection",
        solver_tolerance_f: float = 1e-6,
        solver_max_iterations: int = 20,
        radius_mode: str = "spectral_mup",
        radius_scaler: float = 1.0,
        scale_mode: str = "align_adamw_rms",
        retract_mode: str = "hard",
        retract_alpha: float = 0.05,
    ):
        defaults = dict(
            lr=lr,
            momentum_beta=momentum_beta,
            weight_decay=weight_decay,
            use_nesterov=use_nesterov,
            power_iteration_steps=power_iteration_steps,
            msign_steps=msign_steps,
            solver=solver,
            solver_tolerance_f=solver_tolerance_f,
            solver_max_iterations=solver_max_iterations,
            radius_mode=radius_mode,
            radius_scaler=radius_scaler,
            scale_mode=scale_mode,
            retract_mode=retract_mode,
            retract_alpha=retract_alpha,
        )
        super().__init__(params, defaults)
        
        # Set matmul precision
        if fp32_matmul_prec:
            torch.set_float32_matmul_precision(fp32_matmul_prec)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Extract params from group
            lr = group['lr']
            momentum_beta = group['momentum_beta']
            weight_decay = group['weight_decay']
            use_nesterov = group['use_nesterov']
            
            # SSO specific config
            pi_steps = group['power_iteration_steps']
            msign_steps = group['msign_steps']
            solver = group['solver']
            tol = group['solver_tolerance_f']
            max_iter = group['solver_max_iterations']
            r_mode = group['radius_mode']
            r_scaler = group['radius_scaler']
            s_mode = group['scale_mode']
            ret_mode = group['retract_mode']
            ret_alpha = group['retract_alpha']

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.ndim < 2:
                    # Fallback to AdamW or similar if needed? 
                    # For strictness, SSO shouldn't handle 1D params. 
                    # Assuming user filtered them out (like in train.py).
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SSO does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p)

                state['step'] += 1
                buf = state['momentum']
                
                # Apply weight decay to gradient (decoupled style) if needed?
                # Megatron implementation passes weight_decay to options, 
                # but actually SSO handles "decay" via retraction logic partially.
                # However, the base OrthogonalizedOptimizer in Megatron DOES apply weight decay to params 
                # or gradient based on config.
                # Here we assume decoupled WD on the momentum buffer or similar
                # But actually, in spectral ball, WD is often implicitly handled by the constraint 
                # or applied to the update.
                # Let's align with Megatron:
                # In Megatron `OrthogonalizedOptimizer.step`:
                # if weight_decay_method == "decoupled":
                #    p.data.mul_(1.0 - lr * weight_decay)
                # But `SpectralBall` overrides `orthogonalize` and modifies W in place during retraction!
                # So we should be careful.
                # In `SpectralBall.py`:
                # "weight_decay=weight_decay" is passed to super.
                # But `linear_no_weight_decay_cond` was used in `spectral_ball_optimizer.py` to disable standard WD?
                # "Force all linear params to have wd_mult=0.0 (no weight decay for linear layers)"
                # "This is because SpectralBall already constrains weights to spectral sphere"
                # So we SKIP standard weight decay here!
                
                # Update Momentum
                buf.mul_(momentum_beta).add_(grad)

                # Nesterov
                if use_nesterov:
                    M = buf.mul(momentum_beta).add(grad)
                else:
                    M = buf
                
                # -----------------------------------------------------------
                # SSO Algorithm
                # -----------------------------------------------------------
                
                # 1. Compute target radius
                target_r = compute_target_radius(p.shape, r_mode, radius_scaler=r_scaler)
                
                # 2. Normalize Momentum for numerical stability in power iteration/solver
                M_fp32 = M.to(torch.float32)
                M_norm = torch.linalg.norm(M_fp32, dim=(-2,-1), keepdim=True).clamp_min(1e-8)
                M_normalized = M_fp32 / M_norm

                # 3. Power Iteration (returns fp32)
                # Use cached singular vector v if available
                v_init = state.get('v_cache', None)
                
                # If we have a cache, we can reduce steps significantly (e.g. to 5)
                # But to be safe, let's keep user config or use a smaller default for subsequent steps
                # For now, just pass v_init. 
                # Optimization request: The user asked for SPEED. 
                # If v_init is present, we override steps to a smaller value if strictly following "caching" idea
                # but typically one still does some steps.
                # Let's trust the user config for steps, but assume they will lower it in train.py
                # OR, we can dynamically adjust it. 
                # Megatron doesn't show dynamic adjustment in the snippet I saw, but it's common.
                # Let's just use v_init.
                effective_steps = pi_steps
                if v_init is not None:
                    # If we have a cache, we can reduce steps.
                    # Previous value of 5 was too aggressive causing instability/NaNs.
                    # Increased to 20 to ensure convergence while still being 5x faster than 100.
                    if pi_steps > 20:
                        effective_steps = 20 
                
                sigma, u, v = power_iteration(p, steps=effective_steps, v_init=v_init)
                sigma_val = sigma.item()
                
                # Cache v for next iteration
                state['v_cache'] = v.detach()

                # 4. Retract W
                apply_retract(p, sigma_val, target_r, mode=ret_mode, alpha=ret_alpha, current_lr=lr)

                # 5. Form Theta
                Theta = u @ v.transpose(-2, -1)

                # 6. Solve Lambda
                lambda_val, _, _, _ = solve_lambda_with_bisection(
                    G=M_normalized,
                    Theta=Theta,
                    msign_steps=msign_steps,
                    tolerance_f=tol,
                    max_iterations=max_iter
                )

                # 7. Compute Phi
                Z = M_normalized + lambda_val * Theta
                Phi = msign(Z, steps=msign_steps)

                # 8. Scale and Update
                # "update: W ← W - lr * Φ"
                # Also apply scaling based on shape
                scale_factor = get_spectral_ball_scale_factor(p.shape[0], p.shape[1], mode=s_mode)
                
                # Apply update
                # Note: Phi is fp32/bf16. 
                p.add_(Phi.to(p.dtype), alpha=-lr * scale_factor)

        return loss

