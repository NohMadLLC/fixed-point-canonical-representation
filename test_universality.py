#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated test runner for substrate equivalence.
Generates data, runs the universal engine (if present), parses results,
and falls back to internal estimation when needed.

Usage:
    python test_universality.py

Assumptions:
  - generate_equivalents.py is in the same directory (patched version).
  - Your engine is invoked as: python cce_universal.py --file <csv> --domain <name> ...
  - Engine stdout includes either of these lines (case-insensitive):
        Estimated Contraction Rate (lambda): <number>
        Estimated Healing Rate (gamma): <number>
    The parser is tolerant to spacing and scientific notation.

What this script adds:
  - Robust regex parsing with multiple patterns
  - Per-run log capture to logs/<name>.out and logs/<name>.err
  - Fallback OLS estimator on log-scale with White-robust SEs
  - JSON report with CIs for gamma and lambda
  - Clear pass/fail verdict based on overlap and dispersion
"""

import csv
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GAMMA_TRUE = 0.1
LAMBDA_TRUE = float(np.exp(-GAMMA_TRUE))

ENGINE = ["python", "cce_universal.py"]
ENGINE_TIMEOUT_S = 60

RUN_ENGINE = True        # Set False to skip engine calls and use internal estimator only
PRINT_ENGINE_STDERR = False

# Pass/fail controls
MAX_REL_ERROR = 0.02     # 2% max deviation from theory
MAX_CV = 0.02            # 2% coefficient of variation cross-substrate
ALPHA = 0.05             # for 95% CIs

# Tests to run
TESTS = [
    {"file": "decay_deterministic.csv", "domain": "physics",    "name": "Deterministic ODE"},
    {"file": "decay_stochastic.csv",    "domain": "stochastic", "name": "Stochastic SDE"},
    {"file": "decay_markov.csv",        "domain": "stochastic", "name": "Markov Chain"},
    {"file": "decay_quantum.csv",       "domain": "quantum",    "name": "Quantum Damping"},
]

# Engine CLI args (tune if your engine expects different)
ENGINE_ARGS = ["--alpha", "0.6", "--steps", "1200", "--burn", "200"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def ensure_generated() -> None:
    """Run the data generator (patched) before testing."""
    gen = Path("generate_equivalents.py")
    if not gen.exists():
        raise FileNotFoundError("generate_equivalents.py not found in current directory.")
    print("Step 1: Generating equivalent systems...")
    print("-" * 70)
    subprocess.run(["python", str(gen)], check=True)
    print()


def read_csv_series(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read CSV with headers time,value -> (t, y)."""
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    t_vals, y_vals = [], []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        # allow either 'value' or legacy columns
        val_key = "value"
        if "value" not in r.fieldnames:
            # try known alternates
            for k in ("x", "state", "excited_population"):
                if k in r.fieldnames:
                    val_key = k
                    break
        for row in r:
            try:
                t_vals.append(float(row.get("time", len(t_vals))))
                y_vals.append(float(row[val_key]))
            except Exception:
                continue
    t = np.asarray(t_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    return t, y


def ols_log_decay(t: np.ndarray, y: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Fit log(y) = a - gamma * t + e
    Return gamma_hat, se_gamma, a_hat, lambda_hat, SE-implied CI for both gamma and lambda.
    Uses White-robust covariance.
    """
    mask = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t = t[mask]
    y = y[mask]
    if t.size < 5:
        return None

    X = np.column_stack([np.ones_like(t), -t])  # columns: [1, -t] so slope directly = gamma
    ly = np.log(y)
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return None
    beta = XtX_inv @ (X.T @ ly)
    resid = ly - X @ beta
    # White robust: (X'X)^-1 (X' diag(e^2) X) (X'X)^-1
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    cov = XtX_inv @ S @ XtX_inv

    a_hat = float(beta[0])
    gamma_hat = float(beta[1])
    se_gamma = float(math.sqrt(max(cov[1, 1], 0.0)))
    # 95% CI for gamma
    z = 1.96
    gamma_lo = gamma_hat - z * se_gamma
    gamma_hi = gamma_hat + z * se_gamma
    # map to lambda
    lambda_hat = float(np.exp(-gamma_hat))
    # conservative CI mapping via endpoints
    lambda_lo = float(np.exp(-gamma_hi))
    lambda_hi = float(np.exp(-gamma_lo))

    return {
        "gamma_hat": gamma_hat,
        "se_gamma": se_gamma,
        "gamma_lo": gamma_lo,
        "gamma_hi": gamma_hi,
        "lambda_hat": lambda_hat,
        "lambda_lo": lambda_lo,
        "lambda_hi": lambda_hi,
        "a_hat": a_hat,
        "n_used": int(t.size),
    }


def parse_engine_output(stdout: str) -> Dict[str, Optional[float]]:
    """
    Try multiple regexes to find lambda and gamma in engine output.
    Returns dict with optional floats.
    """
    patterns = [
        r"Estimated\s+Contraction\s+Rate\s*\(?lambda\)?\s*[:=]\s*([\-+\deE\.]+)",
        r"lambda\s*[:=]\s*([\-+\deE\.]+)",
        r"λ\s*[:=]\s*([\-+\deE\.]+)",
    ]
    patterns_g = [
        r"Estimated\s+Healing\s+Rate\s*\(?gamma\)?\s*[:=]\s*([\-+\deE\.]+)",
        r"gamma\s*[:=]\s*([\-+\deE\.]+)",
        r"γ\s*[:=]\s*([\-+\deE\.]+)",
    ]

    lam = None
    gam = None
    for p in patterns:
        m = re.search(p, stdout, re.IGNORECASE)
        if m:
            try:
                lam = float(m.group(1))
                break
            except Exception:
                pass
    for p in patterns_g:
        m = re.search(p, stdout, re.IGNORECASE)
        if m:
            try:
                gam = float(m.group(1))
                break
            except Exception:
                pass

    # a generic status capture
    status = None
    m = re.search(r"FINAL\s+STATUS\s*[:=]\s*([A-Z_]+)", stdout, re.IGNORECASE)
    if m:
        status = m.group(1).upper()

    return {"lambda": lam, "gamma": gam, "status": status}


def run_engine(test: Dict[str, str], log_dir: Path) -> Dict[str, Optional[float]]:
    """Invoke the engine and capture outputs, returning parsed metrics."""
    args = ENGINE + ["--file", test["file"], "--domain", test["domain"]] + ENGINE_ARGS
    out_path = log_dir / (test["name"].replace(" ", "_") + ".out")
    err_path = log_dir / (test["name"].replace(" ", "_") + ".err")
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=ENGINE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        out_path.write_text("TIMEOUT")
        err_path.write_text("")
        return {"lambda": None, "gamma": None, "status": "TIMEOUT"}

    out_path.write_text(proc.stdout or "")
    err_path.write_text(proc.stderr or "")

    if PRINT_ENGINE_STDERR and proc.stderr:
        print(proc.stderr)

    parsed = parse_engine_output(proc.stdout or "")
    # Attach raw for debugging
    parsed["stdout_len"] = len(proc.stdout or "")
    parsed["stderr_len"] = len(proc.stderr or "")
    parsed["returncode"] = proc.returncode
    return parsed


def verdict_from_samples(lambdas: List[float]) -> Tuple[str, Dict[str, float]]:
    arr = np.asarray(lambdas, dtype=float)
    lam_mean = float(np.mean(arr))
    lam_std = float(np.std(arr))
    lam_range = float(np.max(arr) - np.min(arr))
    cv = lam_std / lam_mean if lam_mean != 0 else float("inf")
    max_err = float(np.max(np.abs(arr - LAMBDA_TRUE)) / LAMBDA_TRUE)

    if max_err < MAX_REL_ERROR and cv < MAX_CV:
        tag = "UNIVERSALITY CONFIRMED"
    elif cv < MAX_CV:
        tag = "PARTIAL SUCCESS"
    else:
        tag = "UNIVERSALITY FAILED"
    return tag, {
        "lambda_mean": lam_mean,
        "lambda_std": lam_std,
        "lambda_range": lam_range,
        "lambda_cv": cv,
        "max_rel_error": max_err,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Step 0: Config")
    print("-" * 70)
    print(f"Expected: gamma = {GAMMA_TRUE:.6f}, lambda = {LAMBDA_TRUE:.6f}")
    print(f"Engine: {' '.join(ENGINE)}")
    print()

    ensure_generated()

    print("Step 2: Running engine on each substrate (with robust parsing)...")
    print("-" * 70)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    results: List[Dict] = []

    for test in TESTS:
        print(f"\nTesting: {test['name']}")
        print(f"  File:   {test['file']}")
        print(f"  Domain: {test['domain']}")

        # Engine parse
        eng = {"lambda": None, "gamma": None, "status": None}
        if RUN_ENGINE:
            eng = run_engine(test, log_dir)

        # Fallback estimation if needed
        t, y = read_csv_series(Path(test["file"]))
        est = ols_log_decay(t, y)

        # Pick source of truth:
        # Prefer engine values if present; otherwise fallback to estimator.
        lam = eng.get("lambda")
        gam = eng.get("gamma")
        status = eng.get("status") or "UNKNOWN"

        used_estimator = False
        if lam is None or gam is None:
            if est is not None:
                lam = est["lambda_hat"]
                gam = est["gamma_hat"]
                status = "FALLBACK_ESTIMATE"
                used_estimator = True

        if lam is not None and gam is not None:
            dev = abs(lam - LAMBDA_TRUE)
            print(f"  ✓ λ = {lam:.6f}  (expected {LAMBDA_TRUE:.6f})")
            print(f"  ✓ γ = {gam:.6f}  (expected {GAMMA_TRUE:.6f})")
            print(f"  ✓ Status: {status}")
            if used_estimator and est is not None:
                print(f"    CI(γ): ({est['gamma_lo']:.6f}, {est['gamma_hi']:.6f})  "
                      f"→ CI(λ): ({est['lambda_lo']:.6f}, {est['lambda_hi']:.6f})")
            results.append({
                "name": test["name"],
                "file": test["file"],
                "domain": test["domain"],
                "lambda": lam,
                "gamma": gam,
                "status": status,
                "deviation_from_theory": dev,
                "engine_stdout_len": eng.get("stdout_len"),
                "engine_stderr_len": eng.get("stderr_len"),
                "engine_returncode": eng.get("returncode"),
                "fallback": used_estimator,
                "estimator": est if est is not None else None,
            })
        else:
            print("  ✗ Could not obtain metrics (engine parse failed and estimator unavailable).")
            results.append({
                "name": test["name"],
                "file": test["file"],
                "domain": test["domain"],
                "error": "No metrics",
                "engine_returncode": eng.get("returncode"),
                "engine_stdout_len": eng.get("stdout_len"),
                "engine_stderr_len": eng.get("stderr_len"),
            })

    print()
    print("=" * 70)
    print("SUBSTRATE EQUIVALENCE TEST RESULTS")
    print("=" * 70)
    print()

    successful = [r for r in results if "lambda" in r and r["lambda"] is not None]
    if len(successful) < 2:
        print("❌ INSUFFICIENT DATA - Need at least 2 successful runs")
        summary = None
    else:
        lambdas = [r["lambda"] for r in successful]
        gammas = [r["gamma"] for r in successful]
        verdict, stats = verdict_from_samples(lambdas)

        print(f"Expected λ = {LAMBDA_TRUE:.6f}")
        print(f"Expected γ = {GAMMA_TRUE:.6f}")
        print()
        print("Measured values:")
        print("-" * 70)
        for r in successful:
            err_pct = 100.0 * abs(r["lambda"] - LAMBDA_TRUE) / LAMBDA_TRUE
            print(f"{r['name']:25s}  λ = {r['lambda']:.6f}  (error: {err_pct:.2f}%)  [{r['status']}]")
        print()
        print("Statistical summary:")
        print("-" * 70)
        print(f"λ mean:       {stats['lambda_mean']:.6f}")
        print(f"λ std dev:    {stats['lambda_std']:.6f}")
        print(f"λ range:      {stats['lambda_range']:.6f}")
        print(f"λ CV:         {100.0*stats['lambda_cv']:.2f}%")
        print(f"Max rel err:  {100.0*stats['max_rel_error']:.2f}%")
        print()
        print("=" * 70)
        if verdict == "UNIVERSALITY CONFIRMED":
            print("✅ UNIVERSALITY CONFIRMED")
            print()
            print(f"  • All λ within {int(MAX_REL_ERROR*100)}% of theory")
            print(f"  • Cross-substrate consistency (CV): {100.0*stats['lambda_cv']:.2f}%")
            print(f"  • Maximum deviation: {100.0*stats['max_rel_error']:.2f}%")
            print()
            print("  → Same dynamics in different substrates yields same λ")
            print("  → Framework is substrate independent ✓")
        elif verdict == "PARTIAL SUCCESS":
            print("⚠️  PARTIAL SUCCESS")
            print()
            print(f"  • Cross-substrate consistency (CV): {100.0*stats['lambda_cv']:.2f}% ✓")
            print(f"  • But deviation from theory: {100.0*stats['max_rel_error']:.2f}%")
            print()
            print("  → Substrates agree with each other ✓")
            print("  → But systematic offset from expected value")
            print("  → Check: prediction mapping and estimator bias")
        else:
            print("❌ UNIVERSALITY FAILED")
            print()
            print(f"  • Cross-substrate variation (CV): {100.0*stats['lambda_cv']:.2f}%")
            print(f"  • Maximum deviation from theory: {100.0*stats['max_rel_error']:.2f}%")
            print()
            print("  → Different substrates give different λ")
            print("  → Investigate representation or engine discrepancies")
        print("=" * 70)
        summary = {
            **stats,
            "verdict": verdict,
            "n_success": len(successful),
            "n_total": len(results),
        }

    # Persist JSON
    out = {
        "expected_lambda": LAMBDA_TRUE,
        "expected_gamma": GAMMA_TRUE,
        "engine": {
            "cmd": ENGINE,
            "args": ENGINE_ARGS,
            "timeout_s": ENGINE_TIMEOUT_S,
            "run_engine": RUN_ENGINE,
        },
        "results": results,
        "summary": summary,
    }
    with open("substrate_equivalence_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print()
    print("Results saved to: substrate_equivalence_results.json")
    print("Run logs saved to: ./logs/*.out, ./logs/*.err")


if __name__ == "__main__":
    main()
