import numpy as np
import matplotlib.pyplot as plt

# Define a toy function to mimic ε(k0) = θ'(L; k0)
def epsilon(k0):
    return np.sin(k0) - 0.5  # Root when sin(k0) = 0.5

# Setup initial guesses
k0_vals = np.linspace(0, 10, 500)
eps_vals = epsilon(k0_vals)

# Initial bracket values
k0_low = 0.0
k0_high = 1.0
res_low = epsilon(k0_low)
res_high = epsilon(k0_high)

# Store the brackets and evaluated points
bracket_progress = [(k0_low, res_low), (k0_high, res_high)]

# Expand until sign change is found
attempts = 0
while res_low * res_high > 0 and attempts < 10:
    k0_low = k0_high
    res_low = res_high
    k0_high *= 2
    res_high = epsilon(k0_high)
    bracket_progress.append((k0_high, res_high))
    attempts += 1

# Plot the function and the bracketing process
plt.figure(figsize=(10, 5))
plt.plot(k0_vals, eps_vals, label=r"$\varepsilon(k_0) = \sin(k_0) - 0.5$")
plt.axhline(0, color='gray', linestyle='--')
for i, (k, v) in enumerate(bracket_progress):
    plt.plot(k, v, 'o', label=f'Attempt {i}: k0={k:.2f}, ε={v:.2f}')
plt.xlabel(r"$k_0$")
plt.ylabel(r"$\varepsilon(k_0)$")
plt.title("Bracketing ε(k₀) to Find Root")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
