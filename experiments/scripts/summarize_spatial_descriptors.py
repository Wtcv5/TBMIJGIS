"""Summarize explicit spatial descriptor outputs across cases."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    exp_dir = Path(__file__).resolve().parent.parent
    descriptor_root = exp_dir / "outputs" / "descriptors"
    case_dirs = sorted(p for p in descriptor_root.iterdir() if p.is_dir())

    summary_rows = []
    association_rows = []
    for case_dir in case_dirs:
        if case_dir.name.endswith("_quick"):
            continue
        summary_path = case_dir / "descriptor_summary.json"
        if not summary_path.exists():
            continue
        data = read_json(summary_path)
        case = data.get("case", {})
        case_id = case.get("id", case_dir.name)
        associations = data.get("association", [])
        if not associations:
            continue

        summary_rows.append(
            {
                "case_id": case_id,
                "tunnel": case.get("tunnel", ""),
                "start_chainage": case.get("start_chainage", ""),
                "test_samples": data.get("split_counts", {}).get("test", ""),
                "association_rows": len(associations),
            }
        )

        for row in associations:
            association_rows.append({"case_id": case_id, **row})

    write_csv(summary_rows, descriptor_root / "descriptor_case_summary.csv")
    write_csv(association_rows, descriptor_root / "descriptor_association_all.csv")
    print(f"Saved {descriptor_root / 'descriptor_case_summary.csv'}")
    print(f"Saved {descriptor_root / 'descriptor_association_all.csv'}")


if __name__ == "__main__":
    main()
