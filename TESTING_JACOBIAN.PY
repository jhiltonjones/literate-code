import numpy as np
import torch
from neural_net import SimpleMLP
from nn_test import solve_joint6_for_angle
from neural_net import SimpleMLP
import torch
def make_theta_J_from_model(model: torch.nn.Module):
    """Return (theta_fn, J_fn) from a PyTorch model mapping ψ(rad)->θ(deg)."""
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    def theta_fn(psi_rad: float) -> float:
        with torch.no_grad():
            x = torch.tensor([[float(psi_rad)]], dtype=torch.float32, device=device)
            y_deg = model(x).item()           # θ in degrees
        return np.deg2rad(y_deg)              # θ in radians

    def J_fn(psi_rad: float) -> float:
        x = torch.tensor([[float(psi_rad)]], dtype=torch.float32, device=device, requires_grad=True)
        y_deg = model(x)                      # θ in degrees
        (dy_dpsi_deg_per_rad,) = torch.autograd.grad(y_deg, x, torch.ones_like(y_deg), retain_graph=False)
        # Convert deg/rad -> rad/rad:
        return float(dy_dpsi_deg_per_rad.item()) * (np.pi / 180.0)

    return theta_fn, J_fn
def fd_jacobian(theta_fn, psi, h=1e-4):
    return (theta_fn(psi + h) - theta_fn(psi - h)) / (2*h)

def rel_err(a, b, eps=1e-12):
    return abs(a - b) / max(eps, abs(a) + abs(b))

def check_fd(theta_fn, J_fn, psis, h=1e-4, tol=1e-3):
    worst = 0.0
    for psi in psis:
        j_fd = fd_jacobian(theta_fn, psi, h=h)
        j_ad = J_fn(psi)
        err = rel_err(j_fd, j_ad)
        worst = max(worst, err)
        print(f"psi={psi:+.6f} rad | J_fd={j_fd:+.6e}, J_ad={j_ad:+.6e}, rel_err={err:.2e}")
    print(f"worst relative error = {worst:.2e} (tol {tol})")
model = SimpleMLP()
model.load_state_dict(torch.load("simple_mlp.pt", map_location="cpu"))
model.eval()

theta_fn, J_fn = make_theta_J_from_model(model)
psis = np.linspace(-2.5, 2.5, 11)
check_fd(theta_fn, J_fn, psis, h=1e-4, tol=1e-3)
