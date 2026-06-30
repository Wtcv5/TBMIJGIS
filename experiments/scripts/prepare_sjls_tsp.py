"""Prepare the external SJLS Dyk1252+411 TSP data for the case experiment.

The downloaded files contain paired point records:

    X_abs,Y_abs,Z_abs,Vp
    X_abs,Y_abs,Z_abs,Vs

They are a dense 2.5D velocity field along the tunnel alignment. This script
converts them into the project TSP schema and expands the 2.5D cross-section
into a 3D tunnel volume so the existing graph builder can consume it.

This output is the real-data experiment input paired with the real mileage
monitoring labels.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def find_velocity_files(input_dir: Path) -> tuple[Path, Path]:
    files = list(input_dir.glob("*.txt")) + list(input_dir.glob("*.csv"))
    vp_files = [p for p in files if "Vp" in p.name]
    vs_files = [p for p in files if "Vs" in p.name]
    if len(vp_files) != 1 or len(vs_files) != 1:
        raise FileNotFoundError(
            f"Expected one Vp and one Vs file in {input_dir}, "
            f"found Vp={len(vp_files)}, Vs={len(vs_files)}."
        )
    return vp_files[0], vs_files[0]


def scan_alignment(vp_path: Path) -> dict:
    """First pass: coordinate ranges and a Z-on-X trend for detrending."""
    n = 0
    min_x = math.inf
    max_x = -math.inf
    min_y = math.inf
    max_y = -math.inf
    min_z = math.inf
    max_z = -math.inf
    sx = sz = sxx = sxz = 0.0

    with vp_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 4:
                continue
            x, y, z, _ = map(float, parts)
            n += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            min_z = min(min_z, z)
            max_z = max(max_z, z)
            sx += x
            sz += z
            sxx += x * x
            sxz += x * z

    denom = n * sxx - sx * sx
    slope = (n * sxz - sx * sz) / denom if abs(denom) > 1e-12 else 0.0
    intercept = (sz - slope * sx) / n
    return {
        "rows": n,
        "x_abs_min": min_x,
        "x_abs_max": max_x,
        "y_abs_min": min_y,
        "y_abs_max": max_y,
        "z_abs_min": min_z,
        "z_abs_max": max_z,
        "z_trend_slope": slope,
        "z_trend_intercept": intercept,
    }


def round_to_grid(value: float, resolution: float) -> float:
    return round(value / resolution) * resolution


def accumulate_profile(
    vp_path: Path,
    vs_path: Path,
    scan: dict,
    chainage_length: float,
    y_half_width: float,
    x_resolution: float,
    y_resolution: float,
) -> tuple[pd.DataFrame, dict]:
    """Second pass: average the dense external field onto an X-Y profile grid."""
    acc = defaultdict(lambda: [0.0, 0.0, 0])
    mismatches = 0
    malformed = 0
    retained = 0
    min_x_abs = float(scan["x_abs_min"])
    slope = float(scan["z_trend_slope"])
    intercept = float(scan["z_trend_intercept"])

    with vp_path.open("r", encoding="utf-8", errors="replace") as fvp, vs_path.open(
        "r", encoding="utf-8", errors="replace"
    ) as fvs:
        for lv, ls in zip(fvp, fvs):
            av = lv.strip().split(",")
            ass = ls.strip().split(",")
            if len(av) != 4 or len(ass) != 4:
                malformed += 1
                continue
            xv, yv, zv, vp = map(float, av)
            xs, ys, zs, vs = map(float, ass)
            if xv != xs or yv != ys or zv != zs:
                mismatches += 1
                continue

            x_local = xv - min_x_abs
            if x_local < 0.0 or x_local > chainage_length:
                continue
            if abs(yv) > y_half_width:
                continue

            # Detrended Z is retained in the audit, but the external profile is
            # fundamentally a 2.5D line/crossline field. The volume expansion is
            # handled explicitly after this averaging step.
            _ = zv - (slope * xv + intercept)
            x_bin = round_to_grid(x_local, x_resolution)
            y_bin = round_to_grid(yv, y_resolution)
            key = (x_bin, y_bin)
            acc[key][0] += vp
            acc[key][1] += vs
            acc[key][2] += 1
            retained += 1

    rows = []
    for (x_bin, y_bin), (vp_sum, vs_sum, count) in sorted(acc.items()):
        rows.append(
            {
                "X": x_bin,
                "Y": y_bin,
                "Vp": vp_sum / count,
                "Vs": vs_sum / count,
                "n_source_points": count,
            }
        )

    profile = pd.DataFrame(rows)
    audit = {
        "paired_coordinate_mismatches": mismatches,
        "malformed_rows": malformed,
        "retained_source_rows": retained,
        "xy_profile_cells": int(len(profile)),
    }
    return profile, audit


def elastic_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Vp_Vs"] = out["Vp"] / out["Vs"]
    ratio_sq = out["Vp_Vs"] ** 2
    out["Pr"] = (ratio_sq - 2.0) / (2.0 * (ratio_sq - 1.0))
    # Gardner-style density estimate. Vp is in m/s.
    out["rho"] = 1000.0 * 1.74 * (out["Vp"] / 1000.0) ** 0.25
    vp2 = out["Vp"] ** 2
    vs2 = out["Vs"] ** 2
    out["E"] = out["rho"] * vs2 * (3.0 * vp2 - 4.0 * vs2) / (vp2 - vs2)
    return out


def expand_profile_to_volume(
    profile: pd.DataFrame,
    z_half_width: float,
    z_resolution: float,
    vertical_decay: float,
) -> pd.DataFrame:
    z_values = np.arange(-z_half_width, z_half_width + 0.5 * z_resolution, z_resolution)
    volume_rows = []
    for row in profile.itertuples(index=False):
        for z in z_values:
            radial_factor = abs(float(z)) / max(z_half_width, 1e-6)
            # A small deterministic perturbation makes the expanded volume
            # non-degenerate while keeping the real profile as the dominant
            # source of variation.
            vp = float(row.Vp) * (1.0 - vertical_decay * radial_factor)
            vs = float(row.Vs) * (1.0 - 0.8 * vertical_decay * radial_factor)
            volume_rows.append({"X": row.X, "Y": row.Y, "Z": float(z), "Vp": vp, "Vs": vs})
    return elastic_columns(pd.DataFrame(volume_rows))


def write_outputs(volume: pd.DataFrame, audit: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tsp_path = output_dir / "tsp_sjls_dyk1252_411.csv"
    audit_path = output_dir / "tsp_sjls_dyk1252_411_audit.json"
    volume[["X", "Y", "Z", "Vp", "Vs", "rho", "E", "Vp_Vs", "Pr"]].to_csv(
        tsp_path, index=False
    )
    audit["output_rows"] = int(len(volume))
    audit["output_path"] = str(tsp_path)
    audit["columns"] = ["X", "Y", "Z", "Vp", "Vs", "rho", "E", "Vp_Vs", "Pr"]
    audit["intended_use"] = "real-data experiment using external Dyk1252+411 TSP and real mileage monitoring labels"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing one Vp and one Vs raw external file.")
    parser.add_argument("--output-dir", default="data/processed/sjls_dyk1252_411")
    parser.add_argument("--chainage-length", type=float, default=120.0)
    parser.add_argument("--y-half-width", type=float, default=10.0)
    parser.add_argument("--z-half-width", type=float, default=10.0)
    parser.add_argument("--x-resolution", type=float, default=1.0)
    parser.add_argument("--y-resolution", type=float, default=1.0)
    parser.add_argument("--z-resolution", type=float, default=1.0)
    parser.add_argument("--vertical-decay", type=float, default=0.015)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    vp_path, vs_path = find_velocity_files(input_dir)
    scan = scan_alignment(vp_path)
    profile, audit = accumulate_profile(
        vp_path=vp_path,
        vs_path=vs_path,
        scan=scan,
        chainage_length=args.chainage_length,
        y_half_width=args.y_half_width,
        x_resolution=args.x_resolution,
        y_resolution=args.y_resolution,
    )
    volume = expand_profile_to_volume(
        profile=profile,
        z_half_width=args.z_half_width,
        z_resolution=args.z_resolution,
        vertical_decay=args.vertical_decay,
    )
    audit.update(scan)
    audit.update(
        {
            "source_vp_path": str(vp_path),
            "source_vs_path": str(vs_path),
            "chainage_length": args.chainage_length,
            "y_half_width": args.y_half_width,
            "z_half_width": args.z_half_width,
            "x_resolution": args.x_resolution,
            "y_resolution": args.y_resolution,
            "z_resolution": args.z_resolution,
            "vertical_decay": args.vertical_decay,
            "density_model": "rho = 1000 * 1.74 * (Vp/1000)^0.25",
            "volume_expansion": "2.5D external X-Y velocity profile expanded over Z layers",
        }
    )
    write_outputs(volume, audit, output_dir)
    print(f"Wrote {len(volume):,} voxels to {output_dir / 'tsp_sjls_dyk1252_411.csv'}")
    print(f"Wrote audit to {output_dir / 'tsp_sjls_dyk1252_411_audit.json'}")


if __name__ == "__main__":
    main()
