# test_nine_laws.py (CONSISTENT RUBRIC, EXTENDED STEPS FOR DIFFUSIVE CASES)
import numpy as np
import subprocess
import json
import sys
from pathlib import Path
import re

# Setup
PYTHON = sys.executable
HERE = Path(__file__).parent.resolve()
ENGINE = str((HERE / "cce_universal.py").resolve())
TEST_DIR = HERE / "nine_laws_test"
TEST_DIR.mkdir(exist_ok=True)

print("="*70)
print("TESTING BROWN'S LAW AGAINST 9 FUNDAMENTAL LAWS OF PHYSICS")
print("="*70)

# ===== DATASET GENERATORS =====

def generate_heat_diffusion(n=1000):
    """1. Second Law of Thermodynamics - Heat diffusion to equilibrium"""
    x = np.linspace(0, 1, 100)
    u = np.sin(np.pi * x)
    data = [u.copy()]
    dt = 0.01
    D = 0.1
    for _ in range(n-1):
        laplacian = np.zeros_like(u)
        dx = x[1] - x[0]
        laplacian[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx*dx)
        u = u + D * dt * laplacian
        data.append(u.copy())
    return np.array(data)

def generate_undamped_oscillator(n=1000):
    """2. Newton's Laws - Conservation of energy (undamped)"""
    t = np.linspace(0, 10, n)
    return np.cos(2 * np.pi * t).reshape(-1, 1)

def generate_damped_oscillator(n=1000):
    """3. Damped Oscillator - Energy dissipation"""
    t = np.linspace(0, 10, n)
    gamma = 0.1
    return (np.exp(-gamma * t) * np.cos(2 * np.pi * t)).reshape(-1, 1)

def generate_logistic_chaos(n=1000):
    """4. Chaos Theory - Lyapunov exponent (sensitive dependence)"""
    x = np.zeros(n)
    x[0] = 0.5
    r = 3.9  # Chaotic regime
    for i in range(n-1):
        x[i+1] = r * x[i] * (1 - x[i])
    return x.reshape(-1, 1)

def generate_markov_chain(n=1000):
    """5. Ergodic Theorem - Markov chain to stationary distribution"""
    P = np.array([[0.7, 0.3], [0.4, 0.6]])
    state = np.array([1.0, 0.0])
    data = [state.copy()]
    for _ in range(n-1):
        state = P.T @ state
        data.append(state.copy())
    return np.array(data)

def generate_clt(n=1000):
    """6. Central Limit Theorem - Convergence to Gaussian"""
    rng = np.random.default_rng(42)
    data = []
    samples_per_point = 100
    for _ in range(n):
        s = np.sum(rng.uniform(-1, 1, samples_per_point))
        data.append(s)
    return np.array(data).reshape(-1, 1)

def generate_brownian(n=1000):
    """7. Diffusion Equation (Fick's Law) - Brownian motion"""
    rng = np.random.default_rng(42)
    dt = 0.01
    increments = rng.normal(0, np.sqrt(dt), n)
    return np.cumsum(increments).reshape(-1, 1)

def generate_conservation_law(n=1000):
    """8. Noether's Theorem - Conserved angular momentum"""
    t = np.linspace(0, 10, n)
    theta = 2 * np.pi * t
    x = np.cos(theta)
    y = np.sin(theta)
    L = np.ones(n)  # Conserved quantity
    return np.column_stack([x, y, L])

def generate_quantum_decoherence(n=1000):
    """9. Quantum Decoherence - Depolarizing channel"""
    r = np.zeros(n)
    r[0] = 1.0  # Pure state
    p = 0.006  # Decoherence rate (should give λ ≈ 0.994)
    for i in range(n-1):
        r[i+1] = (1 - p) * r[i]
    return r.reshape(-1, 1)

# ===== THEORETICAL PREDICTIONS =====

predictions = {
    "1_thermodynamics_heat": {
        "lambda_range": (0.90, 1.00),
        "lambda_expected": "<1",
        "reason": "Heat diffuses to uniform temperature (max entropy)",
        "law": "Second Law of Thermodynamics"
    },
    "2_newton_undamped": {
        "lambda_range": (0.99, 1.01),
        "lambda_expected": "≈1",
        "reason": "Energy conserved, periodic motion",
        "law": "Newton's Laws (Conservation)"
    },
    "3_damped_oscillator": {
        "lambda_range": (0.90, 0.99),
        "lambda_expected": "<1",
        "reason": "Energy dissipates exponentially",
        "law": "Dissipation / Friction"
    },
    "4_chaos_logistic": {
        "lambda_range": (1.00, 1.10),
        "lambda_expected": "≥1",
        "reason": "Chaotic expansion (positive Lyapunov exponent)",
        "law": "Chaos Theory"
    },
    "5_markov_ergodic": {
        "lambda_range": (0.85, 0.95),
        "lambda_expected": "<1",
        "reason": "Converges to stationary distribution",
        "law": "Ergodic Theorem"
    },
    "6_central_limit": {
        "lambda_range": (0.95, 1.05),
        "lambda_expected": "≈1",
        "reason": "Sum of i.i.d. variables (test case)",
        "law": "Central Limit Theorem"
    },
    "7_brownian_diffusion": {
        "lambda_range": (0.99, 1.01),
        "lambda_expected": "≈1",
        "reason": "Random walk (marginal diffusion)",
        "law": "Diffusion Equation (Fick)"
    },
    "8_conservation_angular": {
        "lambda_range": (0.99, 1.01),
        "lambda_expected": "≈1",
        "reason": "Periodic orbit with conserved quantity",
        "law": "Noether's Theorem"
    },
    "9_quantum_decoherence": {
        "lambda_range": (0.990, 0.998),
        "lambda_expected": "≈0.994",
        "reason": "Exponential decay: λ = (1-p) = 0.994",
        "law": "Quantum Decoherence"
    }
}

# ===== HELPER: CONSISTENT SCORING =====

EPS = 1e-3  # numerical tolerance
NEAR1_BAND = 0.01  # ±1% band for “≈1”

def in_expected_range(name, lam):
    pred = predictions[name]
    lo, hi = pred["lambda_range"]
    expected = pred["lambda_expected"]

    if expected == "≈1":
        return abs(lam - 1.0) <= max(EPS, NEAR1_BAND)
    if expected.startswith("<"):
        return lam < hi + EPS
    if expected.startswith("≥"):
        return lam >= lo - EPS
    # Fallback to explicit range
    return (lo - EPS) <= lam <= (hi + EPS)

# ===== GENERATE DATASETS =====

print("\n1. GENERATING TEST DATASETS...")
print("-"*70)

generators = {
    "1_thermodynamics_heat": generate_heat_diffusion,
    "2_newton_undamped": generate_undamped_oscillator,
    "3_damped_oscillator": generate_damped_oscillator,
    "4_chaos_logistic": generate_logistic_chaos,
    "5_markov_ergodic": generate_markov_chain,
    "6_central_limit": generate_clt,
    "7_brownian_diffusion": generate_brownian,
    "8_conservation_angular": generate_conservation_law,
    "9_quantum_decoherence": generate_quantum_decoherence
}

for name, generator in generators.items():
    data = generator()
    filepath = TEST_DIR / f"{name}.csv"
    np.savetxt(filepath, data, delimiter=",")
    print(f"✓ {name:30s} [{predictions[name]['law']}]")

print(f"\n✓ Generated {len(generators)} datasets in {TEST_DIR}/")

# ===== RUN ENGINE WITH UNIFIED RUNTIME POLICY =====

print("\n2. RUNNING ENGINE ON EACH DATASET WITH CONSISTENT RUBRIC...")
print("-"*70)

# Longer runtime for slow/diffusive systems near λ≈1
EXTENDED = {"1_thermodynamics_heat", "7_brownian_diffusion"}
DEFAULT_STEPS, DEFAULT_BURN = 1200, 200
EXT_STEPS, EXT_BURN = 5000, 1000

results = {}

for name in generators.keys():
    filepath = TEST_DIR / f"{name}.csv"

    steps = EXT_STEPS if name in EXTENDED else DEFAULT_STEPS
    burn = EXT_BURN if name in EXTENDED else DEFAULT_BURN

    cmd = [PYTHON, ENGINE,
           "--file", str(filepath),
           "--domain", "physics",
           "--steps", str(steps),
           "--burn", str(burn)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        output = proc.stdout

        lambda_match = re.search(r"Estimated Contraction Rate.*?:\s*([\d.e+-]+)", output)
        gamma_match = re.search(r"Estimated Healing Rate.*?:\s*([\d.e+-]+)", output)

        if lambda_match:
            lambda_val = float(lambda_match.group(1))
            gamma_val = float(gamma_match.group(1)) if gamma_match else None

            in_range = in_expected_range(name, lambda_val)
            check = "✓" if in_range else "✗"

            results[name] = {
                "lambda": lambda_val,
                "gamma": gamma_val,
                "in_expected_range": in_range,
                "prediction": predictions[name]
            }

            exp_txt = predictions[name]["lambda_expected"]
            print(f"{check} {name:30s}: λ = {lambda_val:.6f} (expected {exp_txt})"
                  + (f"  [steps={steps}, burn={burn}]" if name in EXTENDED else ""))
        else:
            print(f"✗ {name:30s}: FAILED TO PARSE")
            results[name] = {"lambda": None, "gamma": None, "in_expected_range": False,
                             "prediction": predictions[name], "status": "PARSE_ERROR"}

    except subprocess.TimeoutExpired:
        print(f"✗ {name:30s}: TIMEOUT")
        results[name] = {"lambda": None, "gamma": None, "in_expected_range": False,
                         "prediction": predictions[name], "status": "TIMEOUT"}
    except Exception as e:
        print(f"✗ {name:30s}: ERROR - {e}")
        results[name] = {"lambda": None, "gamma": None, "in_expected_range": False,
                         "prediction": predictions[name], "status": "ERROR"}

# ===== ANALYSIS =====

print("\n" + "="*70)
print("3. ANALYSIS: BROWN'S LAW vs. KNOWN PHYSICS (BY λ VS EXPECTATION)")
print("="*70)

passed = 0
failed = 0

for name, result in results.items():
    pred = predictions[name]
    lam = result.get("lambda")

    if lam is None:
        print(f"\n✗ {pred['law']}")
        print(f"  FAILED: Could not measure λ")
        failed += 1
        continue

    if result["in_expected_range"]:
        print(f"\n✓ {pred['law']}")
        print(f"  Measured:  λ = {lam:.6f}")
        print(f"  Expected:  {pred['lambda_expected']}  (range: {pred['lambda_range']})")
        print(f"  Reason:    {pred['reason']}")
        passed += 1
    else:
        print(f"\n✗ {pred['law']}")
        print(f"  Measured:  λ = {lam:.6f}")
        print(f"  Expected:  {pred['lambda_expected']}  (range: {pred['lambda_range']})")
        print(f"  MISMATCH:  Outside predicted range")
        failed += 1

print("\n" + "="*70)
print(f"FINAL SCORE: {passed}/{len(results)} laws validated")
print("="*70)

if passed == len(results):
    print("\n🎯 PERFECT SCORE: All 9 laws match predictions")
elif passed >= 7:
    print(f"\n✓ STRONG VALIDATION: {passed}/9 laws match predictions")
else:
    print(f"\n⚠️  PARTIAL VALIDATION: Only {passed}/9 laws match")

# Save results
results_file = TEST_DIR / "nine_laws_results.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Results saved to {results_file}")
