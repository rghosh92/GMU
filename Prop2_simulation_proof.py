import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, gammaln

# Define the function to compute the difference between LHS and RHS

# Function to compute LHS directly without overflow
def compute_lhs(d):
    log_lhs = gammaln((d + 1) / 2) + gammaln((d - 1) / 2) - 2 * gammaln(d / 2)
    return np.exp(log_lhs)

#
def inequality_difference(d):
    # lhs = (gamma((d + 1) / 2) * gamma((d - 1) / 2)) / (gamma(d / 2) ** 2)
    lhs = compute_lhs(d)
    rhs = np.sqrt(d / (d - 1))
    
    return lhs - rhs
plt.rcParams["font.family"] = "Times New Roman"

# Generate values for d from 2 to a wide range (e.g., 2 to 10,000)
d_values = np.floor(np.logspace(0.3, 4, 500))  # Log-spaced values of d from ~2 to 10,000
differences = [inequality_difference(d) for d in d_values]
print(differences)
# Absolute differences for log scale (y-axis cannot handle negative values directly)
log_differences = [abs(diff) for diff in differences]

# Plot setup
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # Two plots side by side

# Left Plot: Normal Scale
axes[0].plot(d_values, differences, color="black", linestyle="-", linewidth=1, label="Trend Line")
for d, diff in zip(d_values, differences):
    color = "green" if diff > 0 else "red"
    axes[0].scatter(d, diff, color=color, s=10)  # Points
axes[0].axhline(0, color="blue", linestyle="--", label="Zero Reference Line")
axes[0].set_title("Normal Scale", fontsize=28)
axes[0].set_xlabel("Dimension (d)", fontsize=24)
axes[0].set_ylabel("Difference (LHS - RHS)", fontsize=24)
axes[0].grid(True)
axes[0].tick_params(axis='both', labelsize=18)
axes[0].legend(fontsize=20)

# Right Plot: Log-Log Scale
axes[1].plot(d_values, log_differences, color="black", linestyle="-", linewidth=1, label="Trend Line")
for d, diff in zip(d_values, log_differences):
    original_diff = inequality_difference(d)
    color = "green" if original_diff > 0 else "red"
    axes[1].scatter(d, diff, color=color, s=10)  # Points
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_title("Log-Log Scale", fontsize=28)
axes[1].set_xlabel("Dimension (d)", fontsize=24)
axes[1].set_ylabel("Difference (LHS - RHS)", fontsize=24)
axes[1].grid(True, which="both", linestyle="--")
axes[1].tick_params(axis='both', labelsize=18)
axes[1].legend(fontsize=20)

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
