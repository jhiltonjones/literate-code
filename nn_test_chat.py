import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# ------------ Data prep ------------
def load_excel_dataset(
    xlsx_path: str = "/home/jack/literate-code/grid_results_adv.xlsx",
    dy_target: float = -0.01,
    atol: float = 1e-6,
    batch_size: int = 128,
    output_in_radians: bool = False,
):
    # Read
    df = pd.read_excel(xlsx_path)

    # Filter rows where dy_m == -0.01 (robust to fp error)
    df = df[np.isclose(df["dy_m"].astype(float), dy_target, atol=atol)]

    # Keep necessary columns and drop rows with NaNs
    cols = ["joint_6", "linear advancer step", "beam_angle_deg"]
    df = df[cols].dropna()

    # joint_6 → radians (auto-detect if it's already radians)
    j6 = df["joint_6"].astype(float).to_numpy()
    if np.nanmax(np.abs(j6)) > 2*np.pi:   # likely degrees
        j6_rad = np.deg2rad(j6)
    else:
        j6_rad = j6

    # advancer step → normalize
    adv = df["linear advancer step"].astype(float).to_numpy()
    adv_mean, adv_std = float(np.mean(adv)), float(np.std(adv) if np.std(adv) > 1e-8 else 1.0)
    adv_norm = (adv - adv_mean) / adv_std

    # periodic features
    X = np.column_stack([np.sin(j6_rad), np.cos(j6_rad), adv_norm]).astype(np.float32)

    # target: beam angle
    y_deg = df["beam_angle_deg"].astype(float).to_numpy()
    if output_in_radians:
        y = np.deg2rad(y_deg).astype(np.float32)[:, None]
    else:
        y = y_deg.astype(np.float32)[:, None]

    # tensors
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    # simple train/val split (80/20)
    N = len(X)
    idx = torch.randperm(N)
    n_train = int(0.8 * N)
    tr, va = idx[:n_train], idx[n_train:]

    ds_train = TensorDataset(X[tr], y[tr])
    ds_val   = TensorDataset(X[va], y[va])

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False)

    meta = {
        "adv_mean": adv_mean,
        "adv_std": adv_std,
        "output_unit": "rad" if output_in_radians else "deg",
    }
    return dl_train, dl_val, meta

# ------------ Model ------------
class BendingMLPPeriodic(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)

# ------------ Training example ------------
def train_model(xlsx_path: str, epochs=1000, lr=1e-3, output_in_radians=False):
    dl_train, dl_val, meta = load_excel_dataset(
        xlsx_path, output_in_radians=output_in_radians
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BendingMLPPeriodic(hidden=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        if ep % 100 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for xb, yb in dl_val:
                    xb, yb = xb.to(device), yb.to(device)
                    val_losses.append(loss_fn(model(xb), yb).item())
            print(f"epoch {ep:4d} | train_loss {loss.item():.6f} | val_loss {np.mean(val_losses):.6f} [{meta['output_unit']}]^2")

    # Save checkpoint
    torch.save({"state_dict": model.state_dict(), "meta": meta}, "bending_model.pt")
    return model, meta

# ------------ Inference helper ------------
def predict_beam_angle(model, j6_value, adv_step, meta, j6_in_degrees=False):
    # j6_value can be rad or deg
    j6_rad = math.radians(j6_value) if j6_in_degrees else j6_value
    adv_norm = (adv_step - meta["adv_mean"]) / (meta["adv_std"] if meta["adv_std"] != 0 else 1.0)
    x = torch.tensor([[math.sin(j6_rad), math.cos(j6_rad), adv_norm]], dtype=torch.float32)
    with torch.no_grad():
        y = model(x).item()
    return y  # in meta['output_unit']
# Train from your Excel file
model, meta = train_model("grid_results_adv.xlsx", epochs=800, lr=1e-3, output_in_radians=False)

# Predict (returns degrees if output_in_radians=False)
pred_deg = predict_beam_angle(model, j6_value=2.07, adv_step=6, meta=meta, j6_in_degrees=False)
print("pred beam angle (deg):", pred_deg)
import math
import torch
import numpy as np

# model: trained BendingMLPPeriodic (input = [sinψ, cosψ, s_norm], output θ in deg or rad per meta)
# meta:  dict with {"adv_mean","adv_std","output_unit"} from training
# theta_target_deg: desired beam angle (deg)
def invert_angle(
    model, meta, theta_target_deg,
    adv_step_bounds=(0, 100),          # set to your valid advancer step range
    psi_bounds_deg=(-180.0, 180.0),    # joint_6 range
    prior_psi_deg=None, prior_step=None,
    w_prior=0.0,                       # e.g. 0.1 to bias towards prior
    n_steps=600, lr=0.05, restarts=6
):
    device = next(model.parameters()).device
    model.eval()

    # convert target to the model's unit
    if meta.get("output_unit","deg") == "rad":
        theta_target = math.radians(theta_target_deg)
    else:
        theta_target = theta_target_deg

    # normalize adv step helper
    def norm_step(s):
        std = meta["adv_std"] if abs(meta["adv_std"]) > 1e-9 else 1.0
        return (s - meta["adv_mean"]) / std

    # bounds
    psi_lo = math.radians(psi_bounds_deg[0]); psi_hi = math.radians(psi_bounds_deg[1])
    step_lo, step_hi = adv_step_bounds

    # prior terms (optional)
    use_prior = prior_psi_deg is not None and prior_step is not None and w_prior > 0.0
    if use_prior:
        prior_psi_rad = math.radians(((prior_psi_deg + 180.0) % 360.0) - 180.0)
        prior_step_t = torch.tensor(float(prior_step), dtype=torch.float32, device=device)

    best = {"loss": float("inf"), "psi_deg": None, "step": None}

    # multi-starts over psi; start steps near bounds/prior
    psi_starts_deg = np.linspace(psi_bounds_deg[0], psi_bounds_deg[1], restarts, endpoint=True)
    step_starts = np.linspace(step_lo, step_hi, restarts, endpoint=True)

    for i in range(restarts):
        # initialize variables
        psi0 = math.radians(((psi_starts_deg[i] + 180.0) % 360.0) - 180.0)
        s0   = step_starts[i]

        # unconstrained parameters; map by tanh to respect bounds
        a = torch.tensor(0.0, requires_grad=True, device=device)  # for psi
        b = torch.tensor(0.0, requires_grad=True, device=device)  # for step

        # choose centers so initial maps close to (psi0, s0)
        def inv_tanh_map(x, lo, hi, val):
            # map val in [lo,hi] to z with val = mid + half * tanh(z)
            mid  = 0.5*(lo+hi); half = 0.5*(hi-lo)
            # clamp inside to avoid overflow
            v = max(min(val, hi-1e-6), lo+1e-6)
            return np.arctanh((v - mid)/half)

        a.data = torch.tensor(inv_tanh_map(0, psi_lo, psi_hi, psi0), dtype=torch.float32, device=device)
        b.data = torch.tensor(inv_tanh_map(0, step_lo, step_hi, s0),  dtype=torch.float32, device=device)

        opt = torch.optim.Adam([a, b], lr=lr)

        for _ in range(n_steps):
            # map to bounded variables
            psi = 0.5*(psi_lo+psi_hi) + 0.5*(psi_hi-psi_lo)*torch.tanh(a)
            step = 0.5*(step_lo+step_hi) + 0.5*(step_hi-step_lo)*torch.tanh(b)

            # features for forward model
            x = torch.stack([
                torch.sin(psi), torch.cos(psi),
                torch.tensor(norm_step(step.item()), device=device, dtype=torch.float32)
            ], dim=0).unsqueeze(0)

            pred = model(x).squeeze(0).squeeze(-1)  # scalar

            # primary loss: match target angle
            loss = (pred - torch.tensor(theta_target, device=device, dtype=torch.float32))**2

            # optional prior regularization (keep close to previous solution)
            if use_prior:
                loss = loss + w_prior * ((psi - torch.tensor(prior_psi_rad, device=device))**2 +
                                         (step - prior_step_t)**2)

            opt.zero_grad()
            loss.backward()
            opt.step()

        # evaluate
        with torch.no_grad():
            psi_val = 0.5*(psi_lo+psi_hi) + 0.5*(psi_hi-psi_lo)*torch.tanh(a)
            step_val = 0.5*(step_lo+step_hi) + 0.5*(step_hi-step_lo)*torch.tanh(b)
            # recompute final error
            x = torch.tensor([[math.sin(float(psi_val)), math.cos(float(psi_val)),
                               norm_step(float(step_val))]], dtype=torch.float32, device=device)
            pred = model(x).item()
            err = (pred - theta_target)**2

        if err < best["loss"]:
            best["loss"] = float(err)
            best["psi_deg"] = float(psi_val)
            best["step"] = float(step_val)

    # round step to nearest int for the actuator
    best["step_int"] = int(round(best["step"]))
    return best
# load
ckpt = torch.load("bending_model.pt", map_location="cpu")
meta = ckpt["meta"]
model = BendingMLPPeriodic(hidden=64)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# invert for a desired beam angle (in degrees)
sol = invert_angle(
    model, meta, theta_target_deg=35.0,
    adv_step_bounds=(0, 40),           # <-- set to your hardware range
    psi_bounds_deg=(-120, 120),        # <-- safe joint6 range
    prior_psi_deg=0.0, prior_step=0,   # optional prior
    w_prior=0.0,                       # increase to bias towards prior
    n_steps=600, lr=0.05, restarts=8
)
print(sol)
# {'loss': ..., 'psi_deg': ..., 'step': ..., 'step_int': ...}
