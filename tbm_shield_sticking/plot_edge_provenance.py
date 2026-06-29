import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(__file__).resolve().parent
edges = pd.read_csv(base / "edge_provenance_top50.csv")
top = edges.sort_values("weighted_contribution", ascending=False).head(20).copy()
top["rank"] = range(1, len(top) + 1)

plt.figure(figsize=(8, 4))
plt.bar(top["rank"], top["weighted_contribution"])
plt.xlabel("edge rank")
plt.ylabel("weighted contribution")
plt.title("Top rock-shield candidate edge contributions")
plt.tight_layout()
plt.savefig(base / "plot_top_edge_contributions.png", dpi=200)

plt.figure(figsize=(5, 5))
plt.scatter(top["voxel_y_m"], top["voxel_z_m"], s=60)
plt.xlabel("y (m)")
plt.ylabel("z (m)")
plt.title("Cross-section positions of top contributing voxels")
plt.axis("equal")
plt.tight_layout()
plt.savefig(base / "plot_top_edge_voxels_cross_section.png", dpi=200)

print("Saved plot_top_edge_contributions.png and plot_top_edge_voxels_cross_section.png")
