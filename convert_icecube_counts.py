# convert_icecube_counts.py
import zipfile, io, pandas as pd, numpy as np, sys, pathlib
zip_path = pathlib.Path(sys.argv[1])
# guess a text file inside
with zipfile.ZipFile(zip_path) as zf:
    name = [n for n in zf.namelist() if n.lower().endswith((".txt",".dat",".csv"))][0]
    raw = zf.read(name)
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        df = pd.read_csv(io.BytesIO(raw), delim_whitespace=True, comment="#", header=None)
# pick a time-like column if present, else use index
col = df.columns[0]
counts, edges = np.histogram(df[col].values, bins=2048)
t = 0.5*(edges[1:]+edges[:-1])
pd.DataFrame({"t": t, "counts": counts}).to_csv(zip_path.with_suffix(".csv"), index=False)
print("wrote", zip_path.with_suffix(".csv"))
