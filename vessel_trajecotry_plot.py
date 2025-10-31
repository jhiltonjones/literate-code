import numpy as np
import matplotlib.pyplot as plt

def make_ref_tortuous(A_rad, f_slow, f_fast, offset_rad=0.0, phase_rad=0.0):
    A1 = A_rad; A2 = 0.5*A_rad; phi2 = 0.5
    return lambda t: (
        (offset_rad
         + A1*np.sin(2*np.pi*f_slow*t + phase_rad)
         + A2*np.sin(2*np.pi*f_fast*t + phi2)),
        (2*np.pi)*(A1*f_slow*np.cos(2*np.pi*f_slow*t + phase_rad)
                   + A2*f_fast*np.cos(2*np.pi*f_fast*t + phi2))
    )
if __name__ == "__main__":
    # --- configure & build the ref ---
    ref = make_ref_tortuous(A_rad=0.2, f_slow=0.2, f_fast=0.8, offset_rad=0.0, phase_rad=0.0)

    # --- sample over time ---
    T = 10.0            # seconds
    Fs = 50            # samples per second
    t = np.linspace(0, T, int(T*Fs), endpoint=False)

    # Evaluate ref for each t (returns (theta, theta_dot))
    theta, theta_dot = zip(*(ref(tt) for tt in t))
    theta = np.array(theta)
    theta_dot = np.array(theta_dot)

    # (Optional) also show degrees for readability
    theta_deg = np.degrees(theta)

    # --- plot ---
    fig, ax1 = plt.subplots(figsize=(9, 4.5))

    ax1.plot(t, np.rad2deg(theta), label=r"$\theta(t)$ [rad]")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel(r"$\theta$ [rad]")
    ax1.grid(True, alpha=0.3)

    # ax2 = ax1.twinx()
    # ax2.plot(t, theta_dot, linestyle="--", label=r"$\dot\theta(t)$ [rad/s]")
    # ax2.set_ylabel(r"$\dot\theta$ [rad/s]")

    # simple legends
    h1, l1 = ax1.get_legend_handles_labels()

    plt.tight_layout()
    plt.show()
