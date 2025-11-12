#!/usr/bin/env python3
"""
Convert three HEPData tables into engine-ready single-column CSV trajectories
for a strong-force (QCD) proxy test.

Outputs (in ./hepdata_out):
  - strong_scattering_qcd.csv
  - strong_spectrum_qcd.csv
  - strong_spectrum_qcd_effcorr.csv       (if an efficiency block aligns)
  - *_boot.csv                             (bootstrap variants)

This script is robust to common HEPData CSV layouts:
- It can read whole-table CSVs or per-block sections (e.g., "Data", "SM").
- It auto-picks numeric x/y columns, subtracts SM/background if present,
  normalizes to unit scale, and writes a single-column trajectory.

Run:
  python convert_hepdata_qcd.py
"""

import csv, json, math, re, html
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent.resolve()

# Update these names if your files differ
F_SCAT   = HERE / "HEPData-ins91346-v1-Table_5.csv"      # hadronic scattering (good QCD probe)
F_SPECT  = HERE / "HEPData-ins1644618-v1-Table_1.csv"    # spectrum with possible Data/SM blocks
F_EFF    = HERE / "HEPData-ins1204447-v1-Table_1.csv"    # efficiencies (optional)

OUT_DIR  = HERE / "hepdata_out"
OUT_DIR.mkdir(exist_ok=True)

# ---------- utilities ----------

def canon(col: str) -> str:
    s = col.lower()
    s = re.sub(r'\\[a-zA-Z]+', ' ', s)  # strip LaTeX
    s = s.replace('$', ' ')
    s = html.unescape(s)
    s = re.sub(r'[^0-9a-z_.]+', ' ', s)
    return s.strip()

def to_float_or_nan(x):
    if x is None:
        return float("nan")
    s = str(x).strip()
    if not s:
        return float("nan")
    s = s.replace("±", " ").split()[0]
    try:
        return float(s)
    except:
        s2 = re.sub(r'[^\d\.\-eE]', '', s)
        try:
            return float(s2) if s2 else float("nan")
        except:
            return float("nan")

def read_csv_auto(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return [ {k.strip(): v.strip() for k,v in row.items()} for row in csv.DictReader(f) ]

def pick_numeric_columns(rows):
    if not rows:
        return []
    cols = list(rows[0].keys())
    numeric = []
    for c in cols:
        vals = [to_float_or_nan(r[c]) for r in rows]
        if np.mean([math.isfinite(v) for v in vals]) > 0.7:
            numeric.append(c)
    return numeric

def prefer_data_column(cols):
    # rank likely data columns over bin/SM/eff columns
    pri = []
    for c in cols:
        lc = canon(c)
        score = 0
        if any(k in lc for k in ["data","obs","cross","ds/d","xsec","yield","events","rate","value"]):
            score += 3
        if any(k in lc for k in ["bin","center","low","high","edge"]):
            score -= 1
        if any(k in lc for k in ["sm","bkg","background","mc","pred","model"]):
            score -= 2
        if "eff" in lc or "efficiency" in lc:
            score -= 3
        pri.append((score, c))
    pri.sort(reverse=True)
    return pri[0][1] if pri else (cols[0] if cols else None)

def find_bin_low_high(cols):
    low = high = None
    for c in cols:
        cc = canon(c)
        if any(k in cc for k in ["low","bin low","lowedge","low edge","low_"]):
            low = c
        if any(k in cc for k in ["high","bin high","highedge","high edge","high_"]):
            high = c
    return low, high

def find_center_column(cols):
    for c in cols:
        cc = canon(c)
        if any(k in cc for k in ["center","bin centre","bin_center","mean"]):
            return c
    return None

def compute_bin_centers(rows, low_col, high_col, center_col):
    if center_col:
        centers = np.array([to_float_or_nan(r[center_col]) for r in rows], dtype=float)
    elif low_col and high_col:
        low = np.array([to_float_or_nan(r[low_col]) for r in rows], dtype=float)
        high = np.array([to_float_or_nan(r[high_col]) for r in rows], dtype=float)
        centers = 0.5 * (low + high)
    else:
        numeric = pick_numeric_columns(rows)
        if not numeric:
            return np.array([])
        centers = np.array([to_float_or_nan(r[numeric[0]]) for r in rows], dtype=float)
    return centers

def normalize_and_write_vector(vec, out_path: Path):
    v = np.asarray(vec, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise RuntimeError("Empty vector after filtering")
    maxabs = np.nanmax(np.abs(v))
    v = v / maxabs if maxabs > 0 else v
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        for val in v:
            w.writerow([f"{val:.10g}"])
    return v

def read_hepdata_block(path: Path, block_hint: str):
    """
    Read a 'block' from HEPData CSV with prefixed comment headers (#: ...).
    If block_hint is empty, fall back to reading the whole CSV.
    """
    if not block_hint:
        # whole file
        return read_csv_auto(path)
    lines = []
    in_block = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#:") and block_hint in line:
                in_block = True
                continue
            if in_block:
                if line.startswith("#:") or line.strip() == "":
                    break
                lines.append(line)
    if not lines:
        return []
    # find header start
    start = 0
    for i, L in enumerate(lines[:4]):
        if ',' in L:
            start = i
            break
    return list(csv.DictReader(lines[start:]))

def match_bins_and_subtract(data_rows, sm_rows, key_low, key_high, key_center, ycol):
    data_centers = compute_bin_centers(data_rows, key_low, key_high, key_center)
    data_vals = np.array([to_float_or_nan(r[ycol]) for r in data_rows], dtype=float)

    sm_numeric = pick_numeric_columns(sm_rows)
    sm_center = compute_bin_centers(sm_rows, key_low, key_high, key_center)
    sm_ycol = None
    for c in sm_numeric:
        if canon(c) == canon(ycol):
            sm_ycol = c
            break
    if sm_ycol is None:
        sm_ycol = next((c for c in sm_numeric if not any(k in canon(c) for k in ["error","err","+","-"])), sm_numeric[0])
    sm_vals = np.array([to_float_or_nan(r[sm_ycol]) for r in sm_rows], dtype=float)

    if len(sm_center) >= 2 and len(data_centers) >= 1:
        sm_interp = np.interp(data_centers, sm_center, sm_vals, left=np.nan, right=np.nan)
        mask = np.isfinite(data_centers) & np.isfinite(data_vals) & np.isfinite(sm_interp)
        return data_centers[mask], (data_vals[mask] - sm_interp[mask])
    if len(data_vals) == len(sm_vals):
        mask = np.isfinite(data_vals) & np.isfinite(sm_vals)
        return data_centers[mask], (data_vals[mask] - sm_vals[mask])
    return np.array([]), np.array([])

def interp_efficiency_to_centers(eff_rows, eff_low, eff_high, eff_center_col, eff_val_col, target_centers):
    eff_centers = compute_bin_centers(eff_rows, eff_low, eff_high, eff_center_col)
    eff_vals = np.array([to_float_or_nan(r[eff_val_col]) for r in eff_rows], dtype=float)
    if len(eff_centers) < 2:
        return None
    return np.interp(target_centers, eff_centers, eff_vals, left=np.nan, right=np.nan)

def bootstrap_vals(vals, n_boot=300, frac=0.85, seed=123):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([])
    k = max(1, int(frac * len(vals)))
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(vals), size=k)
        boots.append(np.nanmean(vals[idx]))
    return np.array(boots)

# ---------- converters ----------

def convert_scattering_qcd():
    rows = read_hepdata_block(F_SCAT, "")  # whole table
    if not rows:
        raise RuntimeError("Scattering table is empty")
    num_cols = pick_numeric_columns(rows)
    if len(num_cols) < 2:
        raise RuntimeError("Not enough numeric columns in scattering table")

    xcol = next((c for c in num_cols if "cos" in canon(c) or "angle" in canon(c)), num_cols[0])
    ycol = next((c for c in num_cols if any(k in canon(c) for k in ["dsig","sigma","d(sig)","xsec","d(sig)/d","d(sig)/domega"])), None)
    if ycol is None:
        ycol = [c for c in num_cols if c != xcol][0]

    centers = compute_bin_centers(rows, *find_bin_low_high(num_cols), find_center_column(num_cols))
    yvals = np.array([to_float_or_nan(r[ycol]) for r in rows], dtype=float)
    mask = np.isfinite(centers) & np.isfinite(yvals)
    yvals = yvals[mask]

    out = OUT_DIR / "strong_scattering_qcd.csv"
    norm = normalize_and_write_vector(yvals, out)
    (OUT_DIR / "strong_scattering_qcd.meta.json").write_text(json.dumps({
        "source": F_SCAT.name, "xcol": xcol, "ycol": ycol, "output": out.name, "n_points": int(len(norm))
    }, indent=2))
    print(f"wrote {out}")

    boot = bootstrap_vals(yvals)
    if boot.size:
        bpath = OUT_DIR / "strong_scattering_qcd_boot.csv"
        normalize_and_write_vector(boot, bpath)
        print(f"wrote {bpath}")

def convert_spectrum_qcd(apply_eff=True):
    # Try Data/SM blocks; fall back to whole file if block tags missing.
    data_rows = read_hepdata_block(F_SPECT, "Data") or read_hepdata_block(F_SPECT, "")
    sm_rows   = read_hepdata_block(F_SPECT, "SM")

    num_cols = pick_numeric_columns(data_rows)
    if len(num_cols) < 2:
        raise RuntimeError("Spectrum table lacks numeric columns")

    low_col, high_col = find_bin_low_high(num_cols)
    center_col = find_center_column(num_cols)
    ycol = prefer_data_column([c for c in num_cols if c not in {low_col, high_col, center_col}]) or num_cols[-1]

    if sm_rows:
        centers, residual = match_bins_and_subtract(data_rows, sm_rows, low_col, high_col, center_col, ycol)
    else:
        centers = compute_bin_centers(data_rows, low_col, high_col, center_col)
        residual = np.array([to_float_or_nan(r[ycol]) for r in data_rows], dtype=float)
        mask = np.isfinite(centers) & np.isfinite(residual)
        centers, residual = centers[mask], residual[mask]

    if centers.size == 0:
        raise RuntimeError("No matched spectrum points after subtraction/mask")

    out = OUT_DIR / "strong_spectrum_qcd.csv"
    norm = normalize_and_write_vector(residual, out)
    (OUT_DIR / "strong_spectrum_qcd.meta.json").write_text(json.dumps({
        "source": F_SPECT.name, "ycol": ycol, "background_subtracted": bool(sm_rows),
        "output": out.name, "n_points": int(len(norm))
    }, indent=2))
    print(f"wrote {out}")

    if apply_eff and F_EFF.exists():
        # heuristic: pick an efficiency block; if none, whole file
        eff_rows = (read_hepdata_block(F_EFF, "Prompt") or
                    read_hepdata_block(F_EFF, "TAU")    or
                    read_hepdata_block(F_EFF, "Table")  or
                    read_hepdata_block(F_EFF, ""))
        if eff_rows:
            eff_num = pick_numeric_columns(eff_rows)
            if eff_num:
                eff_val_col = next((c for c in eff_num if "eff" in canon(c) or "fiducial" in canon(c)), eff_num[0])
                eff_low, eff_high = find_bin_low_high(eff_num)
                eff_center = find_center_column(eff_num)
                eff_interp = interp_efficiency_to_centers(eff_rows, eff_low, eff_high, eff_center, eff_val_col, centers)
                if isinstance(eff_interp, np.ndarray) and np.any(np.isfinite(eff_interp)):
                    eff_interp = np.where(eff_interp <= 0, np.nan, eff_interp)
                    corrected = residual / eff_interp
                    corrected = corrected[np.isfinite(corrected)]
                    if corrected.size:
                        out2 = OUT_DIR / "strong_spectrum_qcd_effcorr.csv"
                        normalize_and_write_vector(corrected, out2)
                        (OUT_DIR / "strong_spectrum_qcd_effcorr.meta.json").write_text(json.dumps({
                            "source": F_SPECT.name, "eff_source": F_EFF.name,
                            "eff_val_col": eff_val_col, "output": out2.name,
                            "n_points": int(corrected.size)
                        }, indent=2))
                        print(f"wrote {out2}")
                    else:
                        print("Efficiency interpolation produced no valid corrected points; skipping write.")
                else:
                    print("Efficiency interpolation failed; skipping correction.")
        else:
            print("No efficiency rows found; skipping efficiency correction.")

    boot = bootstrap_vals(residual)
    if boot.size:
        bpath = OUT_DIR / "strong_spectrum_qcd_boot.csv"
        normalize_and_write_vector(boot, bpath)
        print(f"wrote {bpath}")

# ---------- main ----------

if __name__ == "__main__":
    try:
        convert_scattering_qcd()
    except Exception as e:
        print(f"convert_scattering_qcd failed: {e}")
    try:
        convert_spectrum_qcd(apply_eff=True)
    except Exception as e:
        print(f"convert_spectrum_qcd failed: {e}")
    print("Done. Check ./hepdata_out")
