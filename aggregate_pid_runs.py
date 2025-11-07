#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path("runs_pid")  # folder with PID runs

def load_runs():
    runs = []
    for npz_path in sorted(BASE_DIR.glob("run_*/pid_rollout_logs.npz")):
        data = np.load(npz_path, allow_pickle=True)
        # keys mirrored from the MPC NPZ for easy comparison
        t    = data["t"]
        ref  = data["ref_now_deg"]
        meas = data["meas_deg"]
        # prefer aligned error if present; else compute raw
        err  = data["err_aligned_deg"] if "err_aligned_deg" in data else (ref - meas)
        runs.append((t, ref, meas, err))
    return runs

def align_and_stack(runs):
    if not runs:
        raise RuntimeError("No runs found in runs_pid/.")
    # align to shortest for clean stacking
    min_len = min(len(r[0]) for r in runs)
    t_ref   = runs[0][0][:min_len]
    ref_mat  = np.vstack([r[1][:min_len] for r in runs])
    meas_mat = np.vstack([r[2][:min_len] for r in runs])
    err_mat  = np.vstack([r[3][:min_len] for r in runs])  # may have NaNs at the first/last index
    return t_ref, ref_mat, meas_mat, err_mat

def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    t, ref_mat, meas_mat, err_mat = align_and_stack(runs)

    # columns with any finite data (avoid all-NaN columns from alignment)
    valid_cols = np.any(np.isfinite(err_mat), axis=0)
    if not np.any(valid_cols):
        print("[WARN] No finite error samples across runs; skipping plots.")
        return

    t_v        = t[valid_cols]
    ref_mat_v  = ref_mat[:, valid_cols]
    meas_mat_v = meas_mat[:, valid_cols]
    err_mat_v  = err_mat[:, valid_cols]

    # per-run RMSE (NaN-safe)
    finite_mask = np.isfinite(err_mat_v)
    denom = finite_mask.sum(axis=1).clip(min=1)
    per_run_rmse = np.sqrt((np.where(finite_mask, err_mat_v**2, 0.0).sum(axis=1)) / denom)

    mean_rmse = float(np.mean(per_run_rmse))
    std_rmse  = float(np.std(per_run_rmse))
    p50       = float(np.percentile(per_run_rmse, 50))
    p90       = float(np.percentile(per_run_rmse, 90))

    # RMSE of the mean error curve
    err_mean = np.nanmean(err_mat_v, axis=0)
    rmse_of_mean_curve = float(np.sqrt(np.nanmean(err_mean**2)))

    print(f"Found {len(runs)} PID runs.")
    print(f"Per-run RMSE (deg): mean={mean_rmse:.3f}, std={std_rmse:.3f}, median={p50:.3f}, p90={p90:.3f}")
    print(f"RMSE of mean error curve (deg): {rmse_of_mean_curve:.3f}")

    # Save numerics
    out_csv = BASE_DIR / "rmse_summary.csv"
    with open(out_csv, "w") as f:
        f.write("run_idx,per_run_rmse_deg\n")
        for i, rm in enumerate(per_run_rmse):
            f.write(f"{i},{rm:.6f}\n")
        f.write(f"\nmetric,deg\n")
        f.write(f"mean_per_run_rmse,{mean_rmse:.6f}\n")
        f.write(f"std_per_run_rmse,{std_rmse:.6f}\n")
        f.write(f"median_per_run_rmse,{p50:.6f}\n")
        f.write(f"p90_per_run_rmse,{p90:.6f}\n")
        f.write(f"rmse_of_mean_curve,{rmse_of_mean_curve:.6f}\n")
    print(f"[AGG] Wrote {out_csv}")

    # Plots
    ref_mean  = np.nanmean(ref_mat_v,  axis=0)
    meas_mean = np.nanmean(meas_mat_v, axis=0)
    err_std   = np.nanstd(err_mat_v,   axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(t_v, ref_mean,  label="Mean Reference (deg)", linewidth=2)
    plt.plot(t_v, meas_mean, label="Mean Measured (deg)",  linewidth=2)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("Time (s)"); plt.ylabel("Angle (deg)")
    plt.title("PID — Average Tracking Over Runs")
    plt.legend(); plt.tight_layout()
    plt.savefig(BASE_DIR / "avg_ref_meas.png", dpi=200); plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(t_v, err_mean, label=f"Mean Error (deg)")
    plt.fill_between(t_v, err_mean - err_std, err_mean + err_std, alpha=0.2, label="±1 std")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("Time (s)"); plt.ylabel("Error (deg)")
    plt.title(f"PID — Error: Mean ± 1σ (RMSE(mean)={rmse_of_mean_curve:.2f}°)")
    plt.legend(); plt.tight_layout()
    plt.savefig(BASE_DIR / "avg_error_band.png", dpi=200); plt.close()

if __name__ == "__main__":
    main()
