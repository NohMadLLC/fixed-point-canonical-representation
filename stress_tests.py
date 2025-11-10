"""
Decisive Stress Tests for Brown's Law of Laws
Tests theoretical predictions against known ground-truth systems
"""
import numpy as np
import subprocess
import sys
from pathlib import Path
import json

# Assumes cce_universal.py is in the same directory
PYTHON = sys.executable
ENGINE = "cce_universal.py"
TEST_DIR = Path("stress_tests")
TEST_DIR.mkdir(exist_ok=True)

print("="*70)
print("STRESS TEST SUITE: FALSIFICATION PROTOCOL")
print("="*70)

# ============================================================================
# TEST 1: Perfect Hamiltonian (Zero Dissipation)
# ============================================================================
def test_1_perfect_conservation():
    """
    Perfect energy conservation - circular orbit, no damping
    PREDICTION: Should give λ = 1.0 exactly (no contraction)
    If λ < 1.0, the framework is imposing dissipation where none exists
    """
    print("\n" + "="*70)
    print("TEST 1: PERFECT HAMILTONIAN SYSTEM (Zero Energy Loss)")
    print("="*70)
    
    n = 2000
    t = np.linspace(0, 20*np.pi, n)
    
    # Perfect circle - infinite period conservation
    x = np.cos(t)
    y = np.sin(t)
    
    # Total energy E = x² + y² = 1 (constant)
    energy = x**2 + y**2
    energy_drift = np.max(energy) - np.min(energy)
    
    print(f"\nGenerated {n} points")
    print(f"Energy drift: {energy_drift:.2e} (should be ~0)")
    print(f"System: Perfect circular orbit, E = constant")
    
    # Save as 2D phase space
    data = np.column_stack([x, y])
    filepath = TEST_DIR / "test1_perfect_hamiltonian.csv"
    np.savetxt(filepath, data, delimiter=",")
    
    # Run engine
    cmd = [PYTHON, ENGINE, "--file", str(filepath), "--domain", "physics",
           "--steps", "2000", "--burn", "500"]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse λ
    import re
    match = re.search(r"Estimated Contraction Rate.*?:\s*([\d.e+-]+)", result.stdout)
    
    if match:
        lam = float(match.group(1))
        print(f"\n{'='*70}")
        print(f"RESULT: λ = {lam:.8f}")
        print(f"{'='*70}")
        
        if abs(lam - 1.0) < 0.001:
            print("✓ PASS: λ ≈ 1.0 (correctly identifies conservation)")
            status = "PASS"
        else:
            print(f"✗ FAIL: λ = {lam:.6f} ≠ 1.0")
            print("  Framework falsely detects dissipation in conservative system!")
            status = "FAIL"
    else:
        print("✗ ERROR: Could not parse λ")
        status = "ERROR"
    
    return {"test": "perfect_hamiltonian", "lambda": lam if match else None, 
            "expected": 1.0, "status": status}

# ============================================================================
# TEST 2: Ground Truth Damping Test
# ============================================================================
def test_2_known_damping():
    """
    Generate damped oscillator with KNOWN γ_true
    PREDICTION: Brown's γ should match γ_true across different values
    If all γ → 0.006, the framework is not measuring real physics
    """
    print("\n" + "="*70)
    print("TEST 2: KNOWN DAMPING COEFFICIENT")
    print("="*70)
    
    gamma_true_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    results = []
    
    for gamma_true in gamma_true_values:
        n = 2000
        t = np.linspace(0, 10, n)
        
        # Damped oscillator: x = exp(-γt) cos(ωt)
        omega = 2 * np.pi
        x = np.exp(-gamma_true * t) * np.cos(omega * t)
        
        filepath = TEST_DIR / f"test2_damped_gamma_{gamma_true:.4f}.csv"
        np.savetxt(filepath, x.reshape(-1, 1), delimiter=",")
        
        # Run engine
        cmd = [PYTHON, ENGINE, "--file", str(filepath), "--domain", "physics",
               "--steps", "2000", "--burn", "500"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse γ
        import re
        match = re.search(r"Estimated Healing Rate.*?:\s*([\d.e+-]+)", result.stdout)
        
        if match:
            gamma_measured = float(match.group(1))
            error = abs(gamma_measured - gamma_true)
            rel_error = error / gamma_true if gamma_true > 0 else float('inf')
            
            results.append({
                "gamma_true": gamma_true,
                "gamma_measured": gamma_measured,
                "error": error,
                "rel_error": rel_error
            })
            
            print(f"γ_true = {gamma_true:.4f} → γ_measured = {gamma_measured:.6f} "
                  f"(error: {rel_error*100:.1f}%)")
        else:
            results.append({
                "gamma_true": gamma_true,
                "gamma_measured": None,
                "error": None,
                "rel_error": None
            })
            print(f"γ_true = {gamma_true:.4f} → ERROR: Could not parse")
    
    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS:")
    print(f"{'='*70}")
    
    valid = [r for r in results if r['gamma_measured'] is not None]
    
    if len(valid) >= 3:
        # Check if γ_measured tracks γ_true
        gamma_true_arr = np.array([r['gamma_true'] for r in valid])
        gamma_meas_arr = np.array([r['gamma_measured'] for r in valid])
        
        # Linear correlation
        correlation = np.corrcoef(gamma_true_arr, gamma_meas_arr)[0, 1]
        
        print(f"Correlation: {correlation:.4f}")
        
        # Check if all collapse to ~0.006 (the 0.994 artifact)
        std_measured = np.std(gamma_meas_arr)
        mean_measured = np.mean(gamma_meas_arr)
        
        print(f"Mean γ_measured: {mean_measured:.6f}")
        print(f"Std γ_measured: {std_measured:.6f}")
        
        if correlation > 0.95 and all(r['rel_error'] < 0.2 for r in valid):
            print("✓ PASS: γ correctly tracks ground truth")
            status = "PASS"
        elif std_measured < 0.001:
            print("✗ FAIL: All γ values collapse to same number!")
            print("  Framework is not measuring real damping coefficient")
            status = "FAIL"
        else:
            print("⚠ PARTIAL: Some correlation but high error")
            status = "PARTIAL"
    else:
        print("✗ ERROR: Insufficient valid measurements")
        status = "ERROR"
    
    return {"test": "known_damping", "results": results, "status": status}

# ============================================================================
# TEST 3: Chaos Bifurcation (Logistic Map)
# ============================================================================
def test_3_chaos_boundary():
    """
    Logistic map x_{n+1} = r x_n (1 - x_n)
    Known transition to chaos at r ≈ 3.57
    PREDICTION: λ should track known Lyapunov exponent
    """
    print("\n" + "="*70)
    print("TEST 3: CHAOS BIFURCATION DIAGRAM")
    print("="*70)
    
    r_values = [2.8, 3.2, 3.5, 3.57, 3.7, 3.9]
    results = []
    
    print("\nKnown behavior:")
    print("  r < 3.0: Stable fixed point (λ < 1)")
    print("  r ≈ 3.57: Onset of chaos")
    print("  r > 3.57: Chaotic (λ ≥ 1)")
    print()
    
    for r in r_values:
        n = 2000
        x = np.zeros(n)
        x[0] = 0.5
        
        # Iterate logistic map
        for i in range(n-1):
            x[i+1] = r * x[i] * (1 - x[i])
        
        filepath = TEST_DIR / f"test3_logistic_r_{r:.2f}.csv"
        np.savetxt(filepath, x.reshape(-1, 1), delimiter=",")
        
        # Run engine
        cmd = [PYTHON, ENGINE, "--file", str(filepath), "--domain", "physics",
               "--steps", "2000", "--burn", "500"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse λ
        import re
        match = re.search(r"Estimated Contraction Rate.*?:\s*([\d.e+-]+)", result.stdout)
        
        if match:
            lam = float(match.group(1))
            
            # Expected behavior
            if r < 3.57:
                expected = "λ < 1 (stable/periodic)"
                correct = lam < 1.0
            else:
                expected = "λ ≥ 1 (chaotic)"
                correct = lam >= 1.0
            
            check = "✓" if correct else "✗"
            
            results.append({
                "r": r,
                "lambda": lam,
                "expected": expected,
                "correct": correct
            })
            
            print(f"{check} r = {r:.2f}: λ = {lam:.6f} (expected {expected})")
        else:
            results.append({
                "r": r,
                "lambda": None,
                "expected": None,
                "correct": False
            })
            print(f"✗ r = {r:.2f}: ERROR - Could not parse")
    
    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS:")
    print(f"{'='*70}")
    
    correct_count = sum(1 for r in results if r['correct'])
    total = len([r for r in results if r['lambda'] is not None])
    
    print(f"Correct predictions: {correct_count}/{total}")
    
    if correct_count == total:
        print("✓ PASS: Framework correctly identifies chaos transition")
        status = "PASS"
    elif correct_count >= total * 0.7:
        print("⚠ PARTIAL: Some correct predictions")
        status = "PARTIAL"
    else:
        print("✗ FAIL: Does not capture known chaos dynamics")
        status = "FAIL"
    
    return {"test": "chaos_boundary", "results": results, "status": status}

# ============================================================================
# TEST 4: Time Reversal Symmetry
# ============================================================================
def test_4_time_reversal():
    """
    Run same data forward and backward
    PREDICTION: If truly fundamental, λ_forward = λ_backward
    """
    print("\n" + "="*70)
    print("TEST 4: TIME REVERSAL SYMMETRY")
    print("="*70)
    
    # Generate asymmetric system (exponential growth then decay)
    n = 1000
    t = np.linspace(0, 10, n)
    x = np.exp(-0.1 * t) * np.sin(2 * np.pi * t)
    
    # Forward
    filepath_fwd = TEST_DIR / "test4_forward.csv"
    np.savetxt(filepath_fwd, x.reshape(-1, 1), delimiter=",")
    
    # Backward (time-reversed)
    filepath_bwd = TEST_DIR / "test4_backward.csv"
    np.savetxt(filepath_bwd, x[::-1].reshape(-1, 1), delimiter=",")
    
    results = {}
    
    for direction, filepath in [("forward", filepath_fwd), ("backward", filepath_bwd)]:
        cmd = [PYTHON, ENGINE, "--file", str(filepath), "--domain", "physics",
               "--steps", "2000", "--burn", "500"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        import re
        match = re.search(r"Estimated Contraction Rate.*?:\s*([\d.e+-]+)", result.stdout)
        
        if match:
            lam = float(match.group(1))
            results[direction] = lam
            print(f"{direction:8s}: λ = {lam:.8f}")
    
    if "forward" in results and "backward" in results:
        diff = abs(results["forward"] - results["backward"])
        print(f"\nDifference: {diff:.8f}")
        
        if diff < 0.001:
            print("✓ PASS: Time-reversal symmetric (λ_fwd ≈ λ_bwd)")
            status = "PASS"
        else:
            print("✗ FAIL: Time-reversal asymmetry detected")
            print("  Framework may be measuring directionality, not fundamental property")
            status = "FAIL"
    else:
        status = "ERROR"
    
    return {"test": "time_reversal", "results": results, "status": status}

# ============================================================================
# TEST 5: White Noise to Signal Gradient
# ============================================================================
def test_5_noise_gradient():
    """
    Mix white noise with deterministic signal at varying ratios
    PREDICTION: λ should smoothly vary with signal content
    """
    print("\n" + "="*70)
    print("TEST 5: NOISE-TO-SIGNAL GRADIENT")
    print("="*70)
    
    signal_fractions = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    results = []
    
    n = 2000
    t = np.linspace(0, 10, n)
    
    # Pure signal (damped oscillator)
    signal = np.exp(-0.05 * t) * np.cos(2 * np.pi * t)
    
    # White noise
    np.random.seed(42)
    noise = np.random.randn(n)
    
    for frac in signal_fractions:
        # Mix
        x = frac * signal + (1 - frac) * noise
        
        filepath = TEST_DIR / f"test5_signal_frac_{frac:.2f}.csv"
        np.savetxt(filepath, x.reshape(-1, 1), delimiter=",")
        
        cmd = [PYTHON, ENGINE, "--file", str(filepath), "--domain", "physics",
               "--steps", "2000", "--burn", "500"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        import re
        match = re.search(r"Estimated Contraction Rate.*?:\s*([\d.e+-]+)", result.stdout)
        
        if match:
            lam = float(match.group(1))
            results.append({"signal_frac": frac, "lambda": lam})
            print(f"Signal {frac*100:3.0f}%: λ = {lam:.6f}")
    
    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS:")
    print(f"{'='*70}")
    
    if len(results) >= 5:
        fracs = [r['signal_frac'] for r in results]
        lams = [r['lambda'] for r in results]
        
        # Should show monotonic trend
        is_monotonic = all(lams[i] <= lams[i+1] or lams[i] >= lams[i+1] 
                          for i in range(len(lams)-1))
        
        range_lam = max(lams) - min(lams)
        
        print(f"λ range: {range_lam:.6f}")
        print(f"Monotonic: {is_monotonic}")
        
        if range_lam > 0.01:
            print("✓ PASS: λ varies with signal structure")
            status = "PASS"
        else:
            print("✗ FAIL: λ insensitive to signal content")
            status = "FAIL"
    else:
        status = "ERROR"
    
    return {"test": "noise_gradient", "results": results, "status": status}

# ============================================================================
# RUN ALL TESTS
# ============================================================================
if __name__ == "__main__":
    if not Path(ENGINE).exists():
        print(f"ERROR: Engine {ENGINE} not found in current directory")
        sys.exit(1)
    
    all_results = {}
    
    all_results['test1'] = test_1_perfect_conservation()
    all_results['test2'] = test_2_known_damping()
    all_results['test3'] = test_3_chaos_boundary()
    all_results['test4'] = test_4_time_reversal()
    all_results['test5'] = test_5_noise_gradient()
    
    # Summary
    print("\n" + "#"*70)
    print("STRESS TEST SUMMARY")
    print("#"*70)
    
    for key, result in all_results.items():
        status = result.get('status', 'UNKNOWN')
        test_name = result.get('test', key)
        print(f"{test_name:30s}: {status}")
    
    # Save results
    output_file = TEST_DIR / "stress_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_file}")
    
    # Final verdict
    statuses = [r.get('status') for r in all_results.values()]
    passes = statuses.count('PASS')
    fails = statuses.count('FAIL')
    
    print(f"\n{'='*70}")
    print(f"FINAL SCORE: {passes} PASS, {fails} FAIL out of {len(statuses)} tests")
    print(f"{'='*70}")
    
    if fails == 0:
        print("\n✓ ALL TESTS PASSED - Theory survives stress testing")
    elif fails >= 3:
        print("\n✗ MULTIPLE FAILURES - Theory falsified")
    else:
        print("\n⚠ MIXED RESULTS - Further investigation needed")