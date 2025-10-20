import torch, numpy as np
from neural_net import SimpleMLP
#testing AGAIN
# f(j) = predicted angle (deg), df/dj via autograd
def f_and_df(model, j_val):
    x = torch.tensor([[j_val]], dtype=torch.float32, requires_grad=True)
    y = model(x)                                # deg
    (dy_dx,) = torch.autograd.grad(y, x, torch.ones_like(y), retain_graph=False)
    return float(y.item()), float(dy_dx.item()) # (deg, deg/rad)

def solve_joint6_for_angle(model, target_deg, j_min=-3.14, j_max=+3.14,
                           x0=None, tol_deg=0.1, max_iters=20):
    """
    Find j in [j_min, j_max] s.t. model(j) ~= target_deg (deg).
    Uses Newton with clamping + bisection fallback.
    Returns (j_solution, succeeded: bool).
    """
    # Quick range check
    f_lo, _ = f_and_df(model, j_min)
    f_hi, _ = f_and_df(model, j_max)
    lo, hi = (j_min, j_max)
    flo, fhi = (f_lo - target_deg), (f_hi - target_deg)

    # If target outside predictions at bounds, we still try Newton but flag it
    out_of_range = not (min(f_lo, f_hi) <= target_deg <= max(f_lo, f_hi))

    # Start at x0 or mid
    x = float(x0 if x0 is not None else 0.5*(lo+hi))
    for it in range(max_iters):
        y, dy = f_and_df(model, x)
        err = y - target_deg                     # deg
        if abs(err) <= tol_deg:
            return float(np.clip(x, lo, hi)), not out_of_range
        # Newton step (deg / (deg/rad) = rad)
        if abs(dy) > 1e-6:
            step = err / dy
            x_new = x - step
        else:
            x_new = x  # derivative too small; will fall back below

        # If Newton jumps out of bounds or is NaN, fall back to bisection
        if not np.isfinite(x_new) or x_new < lo or x_new > hi:
            # Bisection assumes (flo and fhi) have opposite signs.
            # If not, expand toward the closer bound and continue.
            mid = 0.5*(lo+hi)
            fmid, _ = f_and_df(model, mid)
            fmid -= target_deg
            if np.sign(fmid) == np.sign(flo):
                lo, flo = mid, fmid
            else:
                hi, fhi = mid, fmid
            x = mid
        else:
            # Accept Newton but clamp to bounds
            x = float(np.clip(x_new, lo, hi))

        # Optionally tighten bracket around current x
        fx, _ = f_and_df(model, x)
        fx -= target_deg
        if np.sign(fx) == np.sign(flo):
            lo, flo = x, fx
        else:
            hi, fhi = x, fx

    # If we’re here, didn’t converge within iters
    return float(np.clip(x, lo, hi)), False

# ---- Usage ----
# Load your trained model
if __name__ == "__main__":
    model = SimpleMLP()
    model.load_state_dict(torch.load("simple_mlp.pt", map_location="cpu"))
    model.eval()

    targets = np.linspace(-90,90, 10)  # desired beam angles in degrees
    for tgt in targets:
        j6, ok = solve_joint6_for_angle(model, tgt, j_min=-5, j_max=+5, x0=-1.8)
        print(f"target={tgt:6.2f} deg -> joint6 ≈ {j6: .5f} rad  ({'ok' if ok else 'extrapolated'})")
