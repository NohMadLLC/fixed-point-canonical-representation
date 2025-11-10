# Fixed-Point Canonical Representation of Persistent Systems

This repository contains the artifact accompanying the manuscript:

**Brown, C. L. (2025). _Fixed-Point Representation of Persistent Systems._**

The central result is a necessity theorem: any system exhibiting persistent recursive structure (A1–A4) must admit a canonical factorization

F = R ∘ E

yaml
Copy code

where `R` is an orthogonal projection onto a closed convex set and `E` is a strict contraction under an intrinsic metric. The intrinsic contraction constant `η` is measurable directly from system orbits and is substrate-independent.

This repository includes:

- The canonical engine (`cce_universal.py`)
- Cross-domain test harnesses
- Data used for deterministic, stochastic, quantum, and biological validation
- Reproducibility and audit ledger (SHA-256)

Artifact DOI: **10.5281/zenodo.17565290**  
Artifact commit: `5de157b73409790b48e31972eaf3caf04dfe4c9a`  
License: **Covenant License Agreement** (see `Covenant License Agreement.pdf`)

---

## Repository Structure

root/
│ cce_universal.py Core canonical contraction engine
│ test_nine_laws.py Verification across nine physical law regimes
│ test_neuro_eeg.py EEG state contraction analysis (N3, REM, Wake)
│ test_real_data.py Real-world multivariate systems test panel
│ stress_tests.py Precision, invariance, chaos boundary validation
│ SHA256SUMS.txt Full-file cryptographic audit ledger
│ Covenant License Agreement.pdf License terms (read before any use)
│
├─ nine_laws_test/ Deterministic and stochastic demonstration datasets
├─ NEW_DATA_NEURO/ Public EEG and fMRI used in state discrimination analysis
├─ sgce_data/ Synthetic and generative complexity evaluation datasets
└─ stress_tests/ Structured perturbation stability datasets

yaml
Copy code

---

## Requirements

Python 3.9 or higher.

Install dependencies:

```bash
pip install numpy scipy pandas matplotlib
For EEG and NIfTI processing (optional but recommended):

bash
Copy code
pip install mne nibabel
Windows users: run commands in PowerShell, not Command Prompt.

Running the Core Engine
The canonical contraction engine estimates the intrinsic contraction constant η from system trajectories.

bash
Copy code
python cce_universal.py --input <path_to_csv>
Example:

bash
Copy code
python cce_universal.py --input nine_laws_test/3_damped_oscillator.csv
Output fields:

pgsql
Copy code
η                intrinsic contraction constant
γ = -log(η)      convergence rate
regime           convergent | marginal | expansive
Running the Domain Test Harnesses
1. Nine Physical Law Regimes
bash
Copy code
python test_nine_laws.py
Produces nine_laws_results.json and printed summary.

2. EEG Sleep State and Neural Stability Profiles
bash
Copy code
python test_neuro_eeg.py
Requires the EEG .edf datasets present under NEW_DATA_NEURO/.

3. Real-World Cross-Domain Data
bash
Copy code
python test_real_data.py
4. Stress Tests and Edge Conditions
bash
Copy code
python stress_tests.py
Verifies:

Exponential decay signature

Substrate invariance

Chaos boundary at logistic r≈3.57

γ recovery accuracy

Encoding independence to 10^-15 stability

Reproducibility and Audit Ledger
All results in the manuscript are reproducible from:

Commit: 5de157b73409790b48e31972eaf3caf04dfe4c9a

DOI: 10.5281/zenodo.17565290

Every file in the repository has a corresponding SHA-256 digest recorded in:

Copy code
SHA256SUMS.txt
Verify:

bash
Copy code
certutil -hashfile <file> SHA256
Compare output to ledger entry.

License
All code, figures, mathematical results, and data transformations are licensed under the:

Covenant License Agreement
© Christopher Lamarr Brown / NohMad LLC

No reproduction, modification, derivative work, or redistribution is permitted without explicit written permission.

See Covenant License Agreement.pdf for full legal terms.

Citation
If referencing this work:

mathematica
Copy code
Brown, C. L. (2025). Fixed-Point Representation of Persistent Systems. DOI: 10.5281/zenodo.17565290.
yaml
Copy code
