# convert_cmb_scan.py
from astropy.io import fits
import numpy as np, pandas as pd, sys, pathlib

p = pathlib.Path(sys.argv[1])
with fits.open(p) as hdul:
    # typical Planck full-sky map is in HDU 1 as a 1D array (RING order)
    m = hdul[1].data.field(0).astype(float)  # temperature [K_CMB]
# build a pseudo-time scan by sliding window average to tame noise
w = 1024
x = pd.Series(m).rolling(w, min_periods=w).mean().dropna().values
t = np.arange(len(x))
pd.DataFrame({"t": t, "cmb_scan": x}).to_csv(p.with_suffix(".csv"), index=False)
print("wrote", p.with_suffix(".csv"))
