#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_neuro_eeg.py — Real EEG/Neuro validation suite
Style/UX intentionally mirrors: stress_tests.py, test_nine_laws.py, test_real_data.py

- Runs cce_universal.py on EEG CSVs (awake, REM, N3) for SC4001 & SC4002
- Accepts engine’s native JSON output; if absent, synthesizes JSON from stdout
- Emits a ledger JSON and a CSV summary under ./json/
"""

import json, subprocess, sys, csv, re, time, os
from pathlib import Path

# ---------- Suite Setup ----------
BASE   = Path(r"E:\Law_Formal_Proof\New folder\NEW_DATA_NEURO").resolve()
ENGINE = (BASE / "cce_universal.py").resolve()
OUTDIR = (BASE / "json"); OUTDIR.mkdir(exist_ok=True)

FILES = [
    "neuro_eeg_awake.csv",
    "neuro_eeg_rem.csv",
    "neuro_eeg_n3.csv",
    "neuro_eeg_awake_sc4002.csv",
    "neuro_eeg_rem_sc4002.csv",
    "neuro_eeg_n3_sc4002.csv",
]

# Broad physiological expectations (range checks for summary rubric)
EXPECT = {
    "awake": dict(min_=0.80, max_=0.995, label="λ < 1 (mixing/integration)"),
    "rem":   dict(min_=0.98, max_=1.02, label="λ ≈ 1 (near-critical)"),
    "n3":    dict(min_=0.98, max_=1.02, label="λ ≈ 1 (near-critical)"),
}

def expectation_for(stem: str):
    s = stem.lower()
    if "awake" in s: return EXPECT["awake"]
    if "rem"   in s: return EXPECT["rem"]
    if "n3" in s or "sleep_deep" in s or "deep" in s: return EXPECT["n3"]
    return dict(min_=0.90, max_=1.10, label="(unspecified)")

# ---------- Regex Parsers ----------
NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
R_ETA     = re.compile(rf"Intrinsic Contraction Factor \(eta\):\s*({NUM})", re.I)
R_LAMBDA  = re.compile(rf"Estimated Contraction Rate \(lambda\):\s*({NUM}|N/?A)", re.I)
R_GAMMA   = re.compile(rf"Estimated Healing Rate \(gamma\):\s*({NUM}|N/?A)", re.I)
R_STATUS  = re.compile(r"FINAL STATUS:\s*(PASS|FAIL)", re.I)
R_FID     = re.compile(r"Fidelity \(lambda ~ eta\):\s*(True|False)", re.I)

def parse_stdout(txt: str):
    def grab(rgx, cast=float, allow_na=False):
        m = rgx.search(txt)
        if not m: return None
        v = m.group(1)
        if allow_na and isinstance(v, str) and v.strip().upper() in {"N/A","NA"}:
            return None
        return cast(v) if cast and v is not None and v.strip().upper() not in {"N/A","NA"} else v
    return dict(
        eta      = grab(R_ETA, float),
        lambd    = grab(R_LAMBDA, float, allow_na=True),
        gamma    = grab(R_GAMMA, float, allow_na=True),
        status   = (R_STATUS.search(txt).group(1) if R_STATUS.search(txt) else None),
        fidelity = (R_FID.search(txt).group(1) == "True") if R_FID.search(txt) else None,
    )

# ---------- Runner ----------
def run_engine(csv_path: Path):
    cmd = [sys.executable, str(ENGINE),
           "--file", str(csv_path),
           "--domain", "neuro"]
    env = {**os.environ, "CCE_NOISE": "0.0"}
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE), env=env)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    return p.returncode, out

# ---------- Main ----------
def main():
    print("="*70)
    print("NEURO/EEG VALIDATION SUITE (REAL DATA)")
    print("="*70)
    print(f"Engine : {ENGINE}")
    print(f"Data   : {BASE}")
    print("-"*70)

    rows, pass_count, total = [], 0, 0

    for name in FILES:
        total += 1
        csv_path = (BASE / name)
        stem = csv_path.stem
        exp  = expectation_for(stem)

        print(f"\n>>> TEST: {name}")
        if not csv_path.exists():
            print(f"✗ MISSING: {csv_path}")
            rows.append(dict(dataset=name, status="MISSING"))
            continue

        rc, txt = run_engine(csv_path)

        # Compact progress echo
        short = []
        for line in (txt.splitlines()):
            if line.startswith("Estimated Contraction Rate") or line.startswith("FINAL STATUS:"):
                short.append(line)
        if short:
            print("  " + "  ".join(short))

        # Engine-native JSON path (the engine writes <file>.json next to CSV)
        json_path = csv_path.with_suffix(".json")

        # If engine did not write JSON, fallback synthesize from stdout
        if not json_path.exists():
            parsed = parse_stdout(txt)
            if parsed["lambd"] is not None:
                fallback = {
                    "file": str(csv_path),
                    "domain": "neuro",
                    "eta": float(parsed["eta"]) if parsed["eta"] is not None else None,
                    "lambda": float(parsed["lambd"]) if parsed["lambd"] is not None else None,
                    "gamma": float(parsed["gamma"]) if parsed["gamma"] is not None else None,
                    "deviation": (abs(parsed["lambd"] - parsed["eta"])
                                  if parsed["lambd"] is not None and parsed["eta"] is not None else None),
                    "status": parsed["status"] or "UNKNOWN",
                    "fidelity": bool(parsed["fidelity"]) if parsed["fidelity"] is not None else None
                }
                try:
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(fallback, f, ensure_ascii=False, indent=2)
                    print(f"[SAVED] JSON (fallback) → {json_path}")
                except Exception as e:
                    print(f"✗ FAIL: could not write fallback JSON ({e})")

        # Load JSON (engine’s or fallback)
        if not json_path.exists():
            print(f"✗ FAIL: No JSON produced → {json_path}")
            rows.append(dict(dataset=name, status="NO_JSON", rc=rc))
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            rec = json.load(f)

        lam = rec.get("lambda"); eta = rec.get("eta"); gam = rec.get("gamma")
        status = rec.get("status", "UNKNOWN"); fid = rec.get("fidelity")

        lam_f = float(lam) if lam is not None else float("nan")
        eta_f = float(eta) if eta is not None else float("nan")
        gam_f = float(gam) if gam is not None else float("nan")

        in_range = (exp["min_"] <= lam_f <= exp["max_"]) if lam == lam else False
        verdict  = "PASS" if in_range else "PARTIAL" if status == "PASS" else "FAIL"
        if verdict == "PASS":
            pass_count += 1

        print(f"λ={lam_f:.6f}  η={eta_f:.6f}  γ={gam_f:.6e}  → expected {exp['label']}  → {verdict}")

        rows.append(dict(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            dataset=name,
            lambda_val=lam_f,
            eta=eta_f,
            gamma=gam_f,
            expected=exp["label"],
            min_exp=exp["min_"],
            max_exp=exp["max_"],
            engine_status=status,
            fidelity=fid,
            verdict=verdict,
            json=str(json_path),
            rc=0
        ))

    # ---------- Suite Summary ----------
    print("\n" + "#"*70)
    print(f"SUMMARY: {pass_count} PASS / {total} total (range-based)")
    print("#"*70)

    # Write ledger JSON
    ledger = OUTDIR / "eeg_suite_ledger.json"
    with open(ledger, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[WROTE] {ledger}")

    # Write CSV summary
    csv_path = OUTDIR / "eeg_suite_summary.csv"
    cols = ["timestamp","dataset","lambda_val","eta","gamma",
            "expected","min_exp","max_exp","engine_status","fidelity","verdict","json","rc"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            if "dataset" in r:
                w.writerow({k: r.get(k, "") for k in cols})
    print(f"[WROTE] {csv_path}")

if __name__ == "__main__":
    if not ENGINE.exists():
        sys.exit(f"Error: Engine script not found at {ENGINE}")
    main()
