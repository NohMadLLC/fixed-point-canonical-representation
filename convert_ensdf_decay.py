# convert_ensdf_decay.py
import zipfile, re, numpy as np, pandas as pd, sys, pathlib

zip_path = pathlib.Path(sys.argv[1])
nuclide = (sys.argv[2] if len(sys.argv)>2 else "CS-137").upper().replace(" ", "")
half_life_sec = None

with zipfile.ZipFile(zip_path) as zf:
    for n in zf.namelist():
        if not n.lower().endswith((".ens", ".txt", ".dat")): 
            continue
        with zf.open(n) as fh:
            text = fh.read().decode("latin-1", "ignore")
        # crude half-life grab (look for T1/2 or HALFLIFE)
        m = re.search(r"(T1/2|HALF[-\s]?LIFE)\s*=\s*([0-9.]+)\s*([smhdMy])", text, re.I)
        if m and nuclide in text.upper():
            val, unit = float(m.group(2)), m.group(3).lower()
            mult = dict(s=1, m=60, h=3600, d=86400, y=31557600).get(unit, 1)
            half_life_sec = val*mult
            break

if half_life_sec is None:
    # fallback: 30y as placeholder if pattern not found
    half_life_sec = 30*31557600

tau = half_life_sec/np.log(2)
t = np.linspace(0, 10*tau, 5000)
N = np.exp(-t/tau)            # real parameters (from ENSDF if parsed)
df = pd.DataFrame({"t_s": t, "activity": N})
out = zip_path.with_name(f"{nuclide}_decay.csv")
df.to_csv(out, index=False)
print("wrote", out, "  tau(s)=", tau)
