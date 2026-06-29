import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(__file__).resolve().parent
m = pd.read_csv(base / "tbm_monitoring.csv", parse_dates=["timestamp"])
r = pd.read_csv(base / "component_readout.csv", parse_dates=["timestamp"])
df = m.merge(r, on=["timestamp", "face_chainage_m"], how="left")

plt.figure(figsize=(9, 4))
plt.plot(df["timestamp"], df["front_shield"], label="front shield")
plt.plot(df["timestamp"], df["middle_shield"], label="middle shield")
plt.plot(df["timestamp"], df["tail_shield"], label="tail shield")
plt.plot(df["timestamp"], df["shield_group_readout"], label="shield group", linewidth=2)
plt.axvspan(pd.Timestamp("2024-08-03 13:00:00"), pd.Timestamp("2024-08-03 16:30:00"), alpha=0.15)
plt.ylabel("geological exposure readout")
plt.xlabel("time")
plt.title("Shield-indexed geological exposure during the stuck-event window")
plt.legend()
plt.tight_layout()
plt.savefig(base / "plot_component_readout.png", dpi=200)

plt.figure(figsize=(9, 4))
plt.plot(df["timestamp"], df["shield_pressure_bar"], label="shield pressure")
plt.axvspan(pd.Timestamp("2024-08-03 13:00:00"), pd.Timestamp("2024-08-03 16:30:00"), alpha=0.15)
plt.ylabel("bar")
plt.xlabel("time")
plt.title("Shield pressure during the stuck-event window")
plt.legend()
plt.tight_layout()
plt.savefig(base / "plot_shield_pressure.png", dpi=200)

plt.figure(figsize=(9, 4))
plt.plot(df["timestamp"], df["advance_rate_m_per_h"], label="advance rate")
plt.axvspan(pd.Timestamp("2024-08-03 13:00:00"), pd.Timestamp("2024-08-03 16:30:00"), alpha=0.15)
plt.ylabel("m/h")
plt.xlabel("time")
plt.title("Advance rate near the stuck-event window")
plt.legend()
plt.tight_layout()
plt.savefig(base / "plot_advance_rate.png", dpi=200)

print("Saved plot_component_readout.png, plot_shield_pressure.png, plot_advance_rate.png")
