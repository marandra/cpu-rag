import csv, statistics as st
from collections import defaultdict

def load(path):
    d = defaultdict(lambda: defaultdict(list))   # cell -> proc -> [tok/s]
    walls = defaultdict(list)                     # cell -> [wall_ms]
    meta = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            cid = r["config_id"]; proc = r["procedure"]
            d[cid][proc].append(float(r["decode_tok_s"]))
            walls[cid].append(float(r["wall_ms"]))
            meta[cid] = (int(r["n_threads"]), int(r["n_replicas"]))
    return d, walls, meta

dn, wn, meta = load("reports/sweep/d4_native.csv")
dp, wp, _ = load("reports/sweep/d4_throughput.csv")

def is_sat(cid, N):
    return cid.endswith("K%02d" % N)

print("%-14s %7s %7s %6s | %9s %7s %6s" % (
    "cell(K=N)", "diabP", "diabN", "d%", "allmedP", "allmedN", "d%"))
order = sorted(meta, key=lambda c: (meta[c][0], -meta[c][1], c))
for cid in order:
    nT, N = meta[cid]
    if not is_sat(cid, N):
        continue
    dp_d = st.median(dp[cid]["diabetes"]) if dp[cid].get("diabetes") else 0
    dn_d = st.median(dn[cid]["diabetes"])
    allp = st.median([x for v in dp[cid].values() for x in v]) if cid in dp else 0
    alln = st.median([x for v in dn[cid].values() for x in v])
    dd = (dn_d/dp_d - 1)*100 if dp_d else 0
    da = (alln/allp - 1)*100 if allp else 0
    print("nT%02d N%02d       %7.2f %7.2f %+6.1f | %9.2f %7.2f %+6.1f" % (
        nT, N, dp_d, dn_d, dd, allp, alln, da))
