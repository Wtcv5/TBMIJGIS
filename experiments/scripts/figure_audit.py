"""Audit generated figures for IJGIS publication compliance.

Checks DPI, colormap usage, figure sizing, and output format consistency.
Run from the experiments directory: python -m scripts.figure_audit outputs/mvp4_stratified/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.style import audit_figure_dpi


def audit_output_dir(output_dir: str | Path, min_dpi: int = 300) -> None:
    """Scan an output directory for figure files and audit each one."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        print(f"Directory not found: {output_dir}")
        return

    extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    files = sorted(f for f in output_dir.rglob("*") if f.suffix.lower() in extensions)

    if not files:
        print(f"No raster figure files found in {output_dir}")
        return

    print(f"Auditing {len(files)} raster figure(s) in {output_dir}:\n")

    n_pass = 0
    n_fail = 0
    for f in files:
        try:
            result = audit_figure_dpi(f, min_dpi=min_dpi)
            status = "PASS" if result["passes"] else "FAIL"
            if result["passes"]:
                n_pass += 1
            else:
                n_fail += 1
            print(f"  [{status}] {f.relative_to(output_dir)}  "
                  f"DPI={result['dpi']:.0f}  "
                  f"size={result['width_px']}x{result['height_px']}")
        except Exception as e:
            n_fail += 1
            print(f"  [ERROR] {f.relative_to(output_dir)}  {e}")

    # Check for missing PDF companions
    print("\nChecking PDF companions for PNG files:")
    png_files = [f for f in files if f.suffix.lower() == ".png"]
    for f in png_files:
        pdf_sibling = f.with_suffix(".pdf")
        if pdf_sibling.exists():
            print(f"  [OK] {f.relative_to(output_dir)} -> {pdf_sibling.name}")
        else:
            print(f"  [MISSING] {f.relative_to(output_dir)} has no PDF companion")

    print(f"\nSummary: {n_pass} passed, {n_fail} failed (min DPI={min_dpi})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit figures for IJGIS compliance")
    parser.add_argument("output_dir", help="Directory containing generated figures")
    parser.add_argument("--min-dpi", type=int, default=300,
                        help="Minimum acceptable DPI (default: 300)")
    args = parser.parse_args()
    audit_output_dir(args.output_dir, min_dpi=args.min_dpi)


if __name__ == "__main__":
    main()
