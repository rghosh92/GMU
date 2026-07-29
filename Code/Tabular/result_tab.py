import json
import pandas as pd

# Load results
with open("accuracies_with_probes.json", "r") as f:
    results = json.load(f)

df = pd.DataFrame(results)

# Only keep num_slices 0–3
df = df[df["NumSlices"] <= 3]

# Compute GMU means for each probe
gmu_means_probe1 = df.groupby("NumSlices")["GMU Probe1"].mean()
gmu_means_probe2 = df.groupby("NumSlices")["GMU Probe2"].mean()
gmu_means_probe3 = df.groupby("NumSlices")["GMU Probe3"].mean()

# Compute baseline means
baseline_probe1 = df["Baseline Probe1"].mean()
baseline_probe2 = df["Baseline Probe2"].mean()
baseline_probe3 = df["Baseline Probe3"].mean()

# Build final table
table = pd.DataFrame({
    "k (NumSlices)": ["Baseline", 0, 1, 2, 3],
    "Mean Probe1 Accuracy": [
        baseline_probe1,
        gmu_means_probe1.get(0, float("nan")),
        gmu_means_probe1.get(1, float("nan")),
        gmu_means_probe1.get(2, float("nan")),
        gmu_means_probe1.get(3, float("nan"))
    ],
    "Mean Probe2 Accuracy": [
        baseline_probe2,
        gmu_means_probe2.get(0, float("nan")),
        gmu_means_probe2.get(1, float("nan")),
        gmu_means_probe2.get(2, float("nan")),
        gmu_means_probe2.get(3, float("nan"))
    ],
    "Mean Probe3 Accuracy": [
        baseline_probe3,
        gmu_means_probe3.get(0, float("nan")),
        gmu_means_probe3.get(1, float("nan")),
        gmu_means_probe3.get(2, float("nan")),
        gmu_means_probe3.get(3, float("nan"))
    ]
})

print(table)
# Save to CSV
table.to_csv("probe_results.csv", index=False)

# Or save to Excel
table.to_excel("probe_results.xlsx", index=False)
