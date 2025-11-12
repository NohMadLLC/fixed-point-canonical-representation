# convert_ligo.py
import h5py, numpy as np, pandas as pd, sys, pathlib
p = pathlib.Path(sys.argv[1])  # .hdf5
with h5py.File(p, "r") as f:
    s = f["strain/Strain"][()]
    fs = f["strain/Strain"].attrs["Xspacing"]**-1  # sample rate
t = np.arange(len(s))/fs
pd.DataFrame({"t": t, "strain": s}).to_csv(p.with_suffix(".csv"), index=False)
print("wrote", p.with_suffix(".csv"))
