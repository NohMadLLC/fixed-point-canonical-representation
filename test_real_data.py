# test_real_data.py (FINAL, Patched Build)
import subprocess, re, pathlib, csv, json
import numpy as np
import unicodedata
import sys
import os

# Windows: force UTF-8 for clean Unicode
if os.name == "nt":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PYTHON = sys.executable

np.random.seed(0)

# --- Paths ---
HERE   = pathlib.Path(__file__).parent.resolve()
ENGINE = str((HERE / "cce_universal.py").resolve())
DATA   = (HERE / "sgce_data")
OUTDIR = (HERE / "real_data_analysis"); OUTDIR.mkdir(exist_ok=True)

# --- Defaults ---
DEFAULT_STEPS = 1200
DEFAULT_BURN  = 200
DEFAULT_TIMEOUT = 300  # seconds

# Extended settings for near-λ≈1 systems (Brownian)
BROWNIAN_FILE = "02_brownian_motion.csv"
EXTENDED_STEPS = 5000
EXTENDED_BURN = 1000
EXTENDED_TIMEOUT = 1200  # seconds

REAL_FILES = {
    'qrng': ['01_qrng_anu.csv', '14_godel_selfreference.csv',
             '15_entropy_forward.csv', 'iid_04639.csv',
             '15_entropy_backward.csv'],
    'physics': ['02_brownian_motion.csv', '03_ising_model.csv',
                '04_heat_diffusion.csv', '05_ca_rule110.csv', '06_ca_rule30.csv'],
    'social': ['07_text_compressible.csv'],
    'climate': ['09_noaa_temperature_synthetic.csv', '10_ndbc_wave_synthetic.csv'],
    'finance': ['11_bitcoin_synthetic.csv', '12_sp500_synthetic.csv', '13_gdp_synthetic.csv']
}

num = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
re_lambda = re.compile(rf"Estimated Contraction Rate.*?:\s*({num}|N/?A)", re.I)
re_gamma  = re.compile(rf"Estimated Healing Rate.*?:\s*({num}|N/?A)", re.I)
re_final  = re.compile(r"FINAL STATUS.*?:\s*(PASS|FAIL)", re.I)

def to_float_or_none(s):
    if s is None: return None
    s = str(s).strip().lower()
    if s in {"n/a","na","nan","none",""}: return None
    try: return float(s)
    except ValueError: return None

def run_engine(file, domain, alpha=0.6, order=4, kappa=25,
               steps=DEFAULT_STEPS, burn=DEFAULT_BURN, timeout=DEFAULT_TIMEOUT):
    if not (DATA / file).exists():
        return {"lambda":"N/A","gamma":"N/A","final":"N/A","rc":-1,
                "error":f"File not found: {DATA / file}"}

    cmd = [PYTHON, ENGINE, "--file", str(DATA/file), "--domain", domain,
           "--alpha", str(alpha), "--order", str(order), "--kappa", str(kappa),
           "--steps", str(steps), "--burn", str(burn)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           cwd=HERE, env={**os.environ, "CCE_NOISE":"0.0"})
    except subprocess.TimeoutExpired:
        p = subprocess.CompletedProcess(cmd, 1, stdout=f"Timeout after {timeout}s", stderr="")

    txt = unicodedata.normalize("NFKD", (p.stdout or "") + "\n" + (p.stderr or ""))

    lam_match = re_lambda.search(txt)
    gam_match = re_gamma.search(txt)
    fin_match = re_final.search(txt)

    lam_raw = lam_match.group(1) if lam_match else None
    gam_raw = gam_match.group(1) if gam_match else None
    fin_raw = fin_match.group(1) if fin_match else None

    # Robust NA check and unique debug log
    if lam_raw is None or (isinstance(lam_raw, str) and lam_raw.upper() in ["N/A","NA"]):
        random_id = int(np.random.randint(1e9))
        debug_path = OUTDIR / f"debug_error_{domain}_{file}_{random_id}.txt"
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(f"--- COMMAND ---\n{' '.join(cmd)}\n\n")
            f.write(f"--- STDOUT ---\n{p.stdout}\n\n")
            f.write(f"--- STDERR ---\n{p.stderr}\n\n")
        if lam_raw is None: lam_raw = "N/A"
        if gam_raw is None: gam_raw = "N/A"
        if fin_raw is None: fin_raw = "FAIL"
        print(f"DEBUG LOG: {debug_path}")

    return {"lambda":lam_raw, "gamma":gam_raw, "final":fin_raw, "rc":p.returncode}

def test_1_parameter_sensitivity():
    print("\nAlpha/Kappa Sensitivity Test:")
    test_cases = [
        ('qrng', '01_qrng_anu.csv'),
        ('physics', BROWNIAN_FILE),
        ('finance', '11_bitcoin_synthetic.csv')
    ]
    alphas = [0.3, 0.5, 0.7, 0.9]
    kappas = [10, 25, 50, 100]
    all_results = []

    print("\nAlpha Sensitivity (Expect STABLE λ):")
    for domain, file in test_cases:
        lambdas, full_results = [], []
        for alpha in alphas:
            if file == BROWNIAN_FILE:
                res = run_engine(file, domain, alpha=alpha,
                                 steps=EXTENDED_STEPS, burn=EXTENDED_BURN, timeout=EXTENDED_TIMEOUT)
            else:
                res = run_engine(file, domain, alpha=alpha)
            lam_val = to_float_or_none(res["lambda"])
            if lam_val is not None: lambdas.append(lam_val)
            full_results.append({'alpha': alpha, **res})
            print(f"  {domain:8s} - α={alpha:.1f}: λ={lam_val:.4f} ({res['final']})"
                  if lam_val is not None else
                  f"  {domain:8s} - α={alpha:.1f}: λ=N/A ({res['final']})")
        valid = [l for l in lambdas if l is not None]
        summary = {'domain':domain, 'file':file, 'param':'alpha', 'values':full_results}
        if len(valid) >= 2:
            std = np.std(valid); rng = max(valid) - min(valid)
            print(f"  → Result: {'❌' if std > 0.01 else '✓'} std={std:.4f}, range={rng:.4f}\n")
            summary.update({'std': float(std), 'range': float(rng)})
        else:
            summary.update({'std': None, 'range': None})
        all_results.append(summary)

    print("\nKappa Sensitivity (Expect STABLE λ):")
    for domain, file in test_cases:
        lambdas, full_results = [], []
        for kappa in kappas:
            if file == BROWNIAN_FILE:
                res = run_engine(file, domain, kappa=kappa,
                                 steps=EXTENDED_STEPS, burn=EXTENDED_BURN, timeout=EXTENDED_TIMEOUT)
            else:
                res = run_engine(file, domain, kappa=kappa)
            lam_val = to_float_or_none(res["lambda"])
            if lam_val is not None: lambdas.append(lam_val)
            full_results.append({'kappa': kappa, **res})
            print(f"  {domain:8s} - κ={kappa:3d}: λ={lam_val:.4f} ({res['final']})"
                  if lam_val is not None else
                  f"  {domain:8s} - κ={kappa:3d}: λ=N/A ({res['final']})")
        valid = [l for l in lambdas if l is not None]
        summary = {'domain':domain, 'file':file, 'param':'kappa', 'values':full_results}
        if len(valid) >= 2:
            std = np.std(valid); rng = max(valid) - min(valid)
            print(f"  → Result: {'❌' if std > 0.01 else '✓'} std={std:.4f}, range={rng:.4f}\n")
            summary.update({'std': float(std), 'range': float(rng)})
        else:
            summary.update({'std': None, 'range': None})
        all_results.append(summary)

    return all_results

def test_2_cross_domain_real():
    print("\n" + "="*70)
    print("TEST 2: Cross-Domain on Real Physical Data (Substrate Independence)")
    print("="*70)

    test_file = BROWNIAN_FILE
    domains = ['qrng', 'physics', 'social', 'climate', 'finance']

    print(f"\nTesting {test_file} across domains:")
    full_results, lambdas_val, csv_rows = [], [], []
    for domain in domains:
        res = run_engine(test_file, domain,
                         steps=EXTENDED_STEPS, burn=EXTENDED_BURN, timeout=EXTENDED_TIMEOUT)
        lam_val = to_float_or_none(res["lambda"])
        full_results.append({'domain': domain, 'file': test_file, **res})
        if lam_val is not None: lambdas_val.append(lam_val)
        csv_rows.append([domain, test_file, res["lambda"], res["gamma"], res["final"], res["rc"]])
        print(f"  {domain:8s}: λ={lam_val:.4f} ({res['final']})"
              if lam_val is not None else
              f"  {domain:8s}: λ=N/A ({res['final']})")

    with open(OUTDIR/"cross_domain_real.csv","w",newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["domain","file","lambda","gamma","final","rc"]); w.writerows(csv_rows)

    summary = {'values': full_results}
    if len(lambdas_val) >= 2:
        std = np.std(lambdas_val); range_val = max(lambdas_val) - min(lambdas_val)
        print(f"\nStd dev: {std:.4f}, Range: {range_val:.4f}")
        print(f"  {'✓' if std < 0.01 else '❌'} {'Substrate-independent' if std < 0.01 else 'Substrate-dependent'}")
        summary.update({'std': float(std), 'range': float(range_val),
                        'result': 'Substrate-independent' if std < 0.01 else 'Substrate-dependent'})
    else:
        print("  Insufficient valid data points to calculate variance.")
        summary.update({'std': None, 'range': None, 'result': 'Insufficient Data'})
    return summary

def test_3_within_domain_consistency():
    print("\n" + "="*70)
    print("TEST 3: Within-Domain Consistency (System Similarity)")
    print("="*70)
    results = {}
    for domain, files in REAL_FILES.items():
        if len(files) < 2: continue
        print(f"\n{domain.upper()} domain:")
        full_results, lambdas_val = [], []
        for file in files:
            if file == BROWNIAN_FILE:
                res = run_engine(file, domain,
                                 steps=EXTENDED_STEPS, burn=EXTENDED_BURN, timeout=EXTENDED_TIMEOUT)
            else:
                res = run_engine(file, domain)
            lam_val = to_float_or_none(res["lambda"])
            full_results.append({'file': file, **res})
            if lam_val is not None: lambdas_val.append(lam_val)
            print(f"  {file:35s}: λ={lam_val:.4f} ({res['final']})"
                  if lam_val is not None else
                  f"  {file:35s}: λ=N/A ({res['final']})")
        summary = {'values': full_results}
        if len(lambdas_val) >= 2:
            std = np.std(lambdas_val); range_val = max(lambdas_val) - min(lambdas_val)
            summary.update({'std': float(std), 'range': float(range_val)})
            print(f"  → std={std:.4f}, range={range_val:.4f}")
            if domain in ['qrng', 'finance'] and std > 0.005:
                print("  ℹ️  High variance observed, check if systems should be truly identical.")
                summary['consistency'] = 'High Variance'
            else:
                print("  ✓ Consistency check passed (or test is non-diagnostic).")
                summary['consistency'] = 'Consistent/Non-Diagnostic'
        else:
            print("  Insufficient valid data points to calculate variance.")
            summary['consistency'] = 'Insufficient Data'
        results[domain] = summary
    return results

def test_4_known_systems():
    print("\n" + "="*70)
    print("TEST 4: Known Systems Analysis (Chaotic/Diffusive)")
    print("="*70)

    # Brownian Motion
    print("\nBrownian Motion (02_brownian_motion.csv):")
    print("  Expected: λ close to 1.0 (✓ if λ ≤ 1.01)")
    res_brown = run_engine(BROWNIAN_FILE, 'physics',
                           steps=EXTENDED_STEPS, burn=EXTENDED_BURN, timeout=EXTENDED_TIMEOUT)
    lam_brown = to_float_or_none(res_brown["lambda"])
    status_brown = "❌" if lam_brown is not None and lam_brown > 1.01 else "✓"
    print(f"  Measured: λ={lam_brown:.4f} {status_brown} ({res_brown['final']})"
          if lam_brown is not None else
          f"  Measured: λ=N/A ⚠️ ({res_brown['final']})")

    # Heat Diffusion
    print("\nHeat Diffusion (04_heat_diffusion.csv):")
    print("  Expected: λ < 1.0 (strong smoothing → contraction)")
    res_heat = run_engine('04_heat_diffusion.csv', 'physics')
    lam_heat = to_float_or_none(res_heat["lambda"])
    status_heat = "✓" if lam_heat is not None and lam_heat < 1.0 else "⚠️"
    print(f"  Measured: λ={lam_heat:.4f} {status_heat} ({res_heat['final']})"
          if lam_heat is not None else
          f"  Measured: λ=N/A ⚠️ ({res_heat['final']})")

    # CA Rule 30
    print("\nCA Rule 30 (06_ca_rule30.csv) - CHAOTIC:")
    print("  Expected: λ ≥ 1.0 (chaotic, should NOT contract)")
    res_rule30 = run_engine('06_ca_rule30.csv', 'physics')
    lam_rule30 = to_float_or_none(res_rule30["lambda"])
    status_rule30 = "❌" if lam_rule30 is not None and lam_rule30 < 1.0 else "✓"
    print(f"  Measured: λ={lam_rule30:.4f} {status_rule30} ({res_rule30['final']})"
          if lam_rule30 is not None else
          f"  Measured: λ=N/A ⚠️ ({res_rule30['final']})")
    if status_rule30 == "❌":
        print("  ⚠️  CRITICAL WARNING: Chaotic system incorrectly reported as contracting.")

    # Entropy forward/backward
    print("\nEntropy Bidirectionality (Time-Reversal Symmetry):")
    res_forward = run_engine('15_entropy_forward.csv', 'qrng')
    res_backward = run_engine('15_entropy_backward.csv', 'qrng')
    forward = to_float_or_none(res_forward["lambda"])
    backward = to_float_or_none(res_backward["lambda"])
    diff = None
    if forward is not None and backward is not None:
        diff = abs(forward - backward)
        status_sym = "✓" if diff < 0.001 else "❌"
        print(f"  Forward: λ={forward:.4f}, Backward: λ={backward:.4f}")
        print(f"  Difference: {diff:.6f} {status_sym} (Expected difference ≈ 0)")
    else:
        print("  N/A: Could not parse both entropy files.")

    return {
        "brownian": {'file':BROWNIAN_FILE, 'status_check':status_brown, **res_brown},
        "heat": {'file':'04_heat_diffusion.csv', 'status_check':status_heat, **res_heat},
        "rule30": {'file':'06_ca_rule30.csv', 'status_check':status_rule30, **res_rule30},
        "forward_backward": {'forward':{**res_forward}, 'backward':{**res_backward}, 'difference':diff}
    }

def generate_real_data_report():
    param_sensitivity = test_1_parameter_sensitivity()
    cross_domain_res = test_2_cross_domain_real()
    within_domain_res = test_3_within_domain_consistency()
    known_systems_res = test_4_known_systems()

    summary = {
        "param_alpha_kappa": param_sensitivity,
        "cross_domain": cross_domain_res,
        "within_domain": within_domain_res,
        "known_systems": known_systems_res
    }
    with open(OUTDIR/"real_data_summary.json","w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "#"*70)
    print("REAL DATA ANALYSIS: SCIENTIFIC VALIDATION REPORT")
    print("#"*70)
    print(f"\nArtifacts saved to {OUTDIR}/real_data_summary.json and {OUTDIR}/cross_domain_real.csv")

    rule30_val = to_float_or_none(known_systems_res['rule30']['lambda'])
    rule30_status = known_systems_res['rule30']['status_check']

    std_val = cross_domain_res.get('std')
    std_txt = (f"{std_val:.4f}" if isinstance(std_val, (int, float)) else "N/A")
    rule30_display = f"{rule30_val:.4f}" if rule30_val is not None else "N/A"

    print("\n" + "="*70)
    print("SUMMARY & INTERPRETATION GUIDE")
    print("="*70)
    print(fr"""
This suite tests the consistency of the Contraction Rate (λ) on real data.

Key Validation Points:
----------------------------------------------------------------------------------
| Test | Claim | Expected Result | Observed |
|:-----|:-----|:----------------|:---------|
| Test 1: Parameter Sensitivity | λ intrinsic to data | STD < 0.01 across sweeps | See printouts |
| Test 2: Substrate Independence | Same data → same λ | STD < 0.01 across domains | STD={std_txt} |
| Test 4: Known Systems (Rule 30) | Chaotic → no contraction | λ ≥ 1.0 | λ={rule30_display} ({rule30_status}) |
----------------------------------------------------------------------------------
""")

if __name__ == "__main__":
    if not pathlib.Path(ENGINE).exists():
        sys.exit(f"Error: Engine script not found at {ENGINE}")
    if not DATA.is_dir():
        sys.exit(f"Error: Data directory not found at {DATA}")
    generate_real_data_report()
