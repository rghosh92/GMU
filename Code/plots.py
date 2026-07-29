import numpy as np
import matplotlib.pyplot as plt

# correct_mlp10 = np.array([
#     0.3941,
#     0.3754,
#     0.3671,
#     0.3663,
#     0.3281,
#     0.1799,
#     0.1485
# ])


# import numpy as np

# # Load arrays
# final_errs = np.load("final_errs_mlp.npy")
# final_ests = np.load("final_ests_mlp.npy")

# # Replace ONLY the MLP‑10 empirical errors (row index 1)
# final_errs[1] = correct_mlp10

# # Save corrected file
# np.save("final_errs_mlp_corrected.npy", final_errs)

# Load saved arrays
final_errs = np.load("final_errs_mlp_corrected.npy")   # shape (6, 7)
final_ests = np.load("final_ests_mlp.npy")   # shape (6, 7)

num_slices = [0,1,2,3,4,5,6]
N_list = [1,10,50,100,200,500]

# LaTeX‑friendly styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
})

markers = ['o', 's', 'D', '^', 'v', 'P']

fig, axes = plt.subplots(2, 3, figsize=(9, 5), sharey=True)
axes = axes.flatten()

for idx, N in enumerate(N_list):
    ax = axes[idx]

    empirical = final_errs[idx]
    theory = final_ests[idx]

    # Empirical
    line_emp = ax.plot(
        num_slices, empirical,
        marker=markers[idx % len(markers)],
        color="tab:blue",
        label="Empirical"
    )

    # Theory
    line_theory = ax.plot(
        num_slices, theory,
        linestyle="--",
        marker=markers[(idx+1) % len(markers)],
        color="tab:red",
        label="Theory"
    )

    ax.set_title(f"N = {N}")
    ax.set_xlabel("GMU Order k")

    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    # Only put legend in subplot (row 1, col 2) → index 1
    if idx == 0:
        ax.legend(frameon=False, loc="lower right")
    else:
        ax.legend([], [], frameon=False)

# One shared y‑label
fig.text(0.04, 0.5, "Balanced Test Error", va="center", rotation="vertical")

plt.tight_layout(rect=[0.06, 0, 1, 1])  # leave room for shared y‑label
plt.savefig("MLP_Actual_vs_Theory_latest.pdf", bbox_inches="tight")
