import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path("runs_ltv2")  # folder with run_*/mpc_rollout_logs.npz

def load_runs():
    runs = []
    for npz_path in sorted(BASE_DIR.glob("run_*/mpc_rollout_logs.npz")):
        d = np.load(npz_path, allow_pickle=True)
        t    = d["t"]
        ref  = d["ref_now_deg"]
        meas = d["meas_deg"]                    # synchronous measurement
        err  = d["err_aligned_deg"]             # keep your aligned error for RMSE
        # build lag-1 measurement (first sample = NaN)
        meas_lead1 = np.r_[meas[2:], np.nan]
        runs.append((t, ref, meas_lead1, err))
    return runs

def align_and_stack(runs):
    if not runs:
        raise RuntimeError("No runs found.")
    L = min(len(r[0]) for r in runs)
    t_ref        = runs[0][0][:L]
    ref_mat      = np.vstack([r[1][:L] for r in runs])
    measlag_mat  = np.vstack([r[2][:L] for r in runs])  # lag-1
    err_mat      = np.vstack([r[3][:L] for r in runs])
    return t_ref, ref_mat, measlag_mat, err_mat

def main():
    runs = load_runs()
    t, ref_mat, measlag_mat, err_mat = align_and_stack(runs)

    # For plotting lag-1, drop columns where EVERY run is NaN (the first col)
    valid_cols_meas = np.any(np.isfinite(measlag_mat), axis=0)
    t_m        = t[valid_cols_meas]
    ref_mat_m  = ref_mat[:, valid_cols_meas]
    measlag_m  = measlag_mat[:, valid_cols_meas]

    # For RMSE aggregation, drop columns with no finite error
    valid_cols_err = np.any(np.isfinite(err_mat), axis=0)
    err_mat_v = err_mat[:, valid_cols_err]

    # ---- Per-run RMSEs (NaN-safe) from aligned error ----
    finite = np.isfinite(err_mat_v)
    denom  = finite.sum(axis=1).clip(min=1)
    per_run_rmse = np.sqrt((np.where(finite, err_mat_v**2, 0.0).sum(axis=1)) / denom)

    mean_rmse = float(np.mean(per_run_rmse))
    std_rmse  = float(np.std(per_run_rmse))
    p50       = float(np.percentile(per_run_rmse, 50))
    p90       = float(np.percentile(per_run_rmse, 90))

    err_mean = np.nanmean(err_mat_v, axis=0)
    rmse_of_mean_curve = float(np.sqrt(np.nanmean(err_mean**2)))

    print(f"Found {len(runs)} runs.")
    print(f"Per-run RMSE (deg): mean={mean_rmse:.3f}, std={std_rmse:.3f}, median={p50:.3f}, p90={p90:.3f}")
    print(f"RMSE of mean error curve (deg): {rmse_of_mean_curve:.3f}")

    # ---- Save numerics ----
    out_csv = BASE_DIR / "rmse_summary.csv"
    with open(out_csv, "w") as f:
        f.write("run_idx,per_run_rmse_deg\n")
        for i, rm in enumerate(per_run_rmse):
            f.write(f"{i},{rm:.6f}\n")
        f.write("\nmetric,deg\n")
        f.write(f"mean_per_run_rmse,{mean_rmse:.6f}\n")
        f.write(f"std_per_run_rmse,{std_rmse:.6f}\n")
        f.write(f"median_per_run_rmse,{p50:.6f}\n")
        f.write(f"p90_per_run_rmse,{p90:.6f}\n")
        f.write(f"rmse_of_mean_curve,{rmse_of_mean_curve:.6f}\n")

    # Averages for plotting (lag-1)
    ref_mean      = np.nanmean(ref_mat_m, axis=0)
    meas_lag1_mean= np.nanmean(measlag_m, axis=0)

    avg_csv = BASE_DIR / "avg_ref_vs_meas.csv"
    with open(avg_csv, "w") as f:
        f.write("t,ref_mean_deg,meas_mean_deg\n")
        for ti, rmi, mmi in zip(t_m, ref_mean, meas_lag1_mean):
            f.write(f"{ti:.6f},{rmi:.6f},{mmi:.6f}\n")

    # ---- Plots ----
    plt.figure(figsize=(10, 6))
    plt.plot(t_m, ref_mean,       label="Mean Reference (deg)", linewidth=2)
    plt.plot(t_m, meas_lag1_mean, label="Mean Measured (deg, )", linewidth=2)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("Time (s)"); plt.ylabel("Angle (deg)")
    plt.title("Average Tracking Over Runs")
    plt.legend(); plt.tight_layout()
    plt.savefig(BASE_DIR / "avg_ref_meas.png", dpi=200); plt.close()

    err_std = np.nanstd(err_mat_v, axis=0)
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(err_mean.size), err_mean, label="Mean Aligned Error (deg)")
    plt.fill_between(np.arange(err_mean.size), err_mean - err_std, err_mean + err_std, alpha=0.2, label="±1 std")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("Aligned step index"); plt.ylabel("Error (deg)")
    plt.title(f"Aligned Error: Mean ± 1σ (RMSE(mean)={rmse_of_mean_curve:.2f}°)")
    plt.legend(); plt.tight_layout()
    plt.savefig(BASE_DIR / "avg_error_band.png", dpi=200); plt.close()

    print(f"[AGG] Wrote {out_csv} and {avg_csv}")

if __name__ == "__main__":
    main()
