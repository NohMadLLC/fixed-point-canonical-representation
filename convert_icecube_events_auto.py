# convert_icecube_events_auto.py
"""
Auto-convert IceCube 3-year data-release *events* files to CSV without renaming originals.
- Recursively searches under --root for files whose name contains both the year-tag and 'events'
- Ignores AngRes / TabulatedAeff
- Skips '#' comments, preserves all numeric columns
- Writes CSVs to <root>/csv/<basename>.csv
Usage:
    python convert_icecube_events_auto.py --root "E:\\four_forces\\icecube_3yr\\3year-data-release"
"""

from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import csv

YEARS = ["IC79-2010", "IC86-2011", "IC86-2012"]

def find_event_files(root: Path):
    root = root.resolve()
    candidates = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if "events" not in name:
            continue
        if "angres" in name or "tabulatedaeff" in name:
            continue
        # must match one of the year tags (case-insensitive)
        if any(tag.lower() in name for tag in (y.lower() for y in YEARS)):
            candidates.append(p)
    return candidates

def read_numeric_table(path: Path):
    # Load numeric data; skip comment lines beginning with '#'
    # Use genfromtxt to handle arbitrary whitespace and missing headers.
    data = np.genfromtxt(
        path,
        comments="#",
        dtype=float,
        invalid_raise=False,
        autostrip=True
    )
    # Ensure 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data

def write_csv(out_path: Path, data: np.ndarray):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        # Generic column headers col1..colN
        w.writerow([f"col{i+1}" for i in range(data.shape[1])])
        w.writerows(data.tolist())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="Root folder of the IceCube 3-year data release")
    args = ap.parse_args()

    root = args.root
    if not root.exists():
        print(f"✗ Root not found: {root}")
        return

    files = find_event_files(root)
    if not files:
        print(f"✗ No events files found under: {root}")
        return

    out_dir = root / "csv"
    print(f"Found {len(files)} events file(s). Converting to: {out_dir}")
    converted = 0
    for src in sorted(files):
        try:
            data = read_numeric_table(src)
            out = out_dir / (src.stem + ".csv")
            write_csv(out, data)
            print(f"✓ {src.name}  →  {out.relative_to(root)}  [{data.shape[0]}×{data.shape[1]}]")
            converted += 1
        except Exception as e:
            print(f"✗ Failed: {src.name}  ({e})")

    if converted == 0:
        print("✗ Nothing converted.")
    else:
        print(f"Done. Converted {converted} file(s).")

if __name__ == "__main__":
    main()
