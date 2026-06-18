"""Check actual node counts for paper parameter reporting."""
import sys
sys.path.insert(0, ".")
from src.data.tsp_loader import load_tsp, normalize_coords
from src.data.tbm_geometry import build_tbm_surface
from src.graph.nodes import active_rock_mask
import numpy as np

df = load_tsp("data/raw/tsp.csv")
df = normalize_coords(df)
coords = df[["X_local", "Y", "Z"]].to_numpy(dtype=np.float32)

tbm = build_tbm_surface(
    cutterhead_radius=4.0, shield_radius=3.95,
    front_len=3.0, middle_len=3.5, tail_len=3.5,
    resolution=0.5)
print(f"TBM surface nodes: {len(tbm)}")

# Check active zone at several chainages
for ch in [10, 25, 40, 45]:
    x_face = np.array([float(ch), 0.0, 0.0])
    shield_tail_x = ch - 10.0
    excavated = np.ones(len(coords), dtype=bool)
    mask = active_rock_mask(coords, x_face, excavated,
        cutterhead_look_ahead=5.0, cutterhead_radius=4.0,
        shield_radius=3.95, shield_tail_x=shield_tail_x,
        tau_zone=5.0)
    print(f"Active rock nodes at ch={ch}: {mask.sum()}")
