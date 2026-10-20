import numpy as np, collections
for fn in ("exp4.log", "exp4b.log"):
    rows = [l.strip().split(",") for l in open(fn) if l[:2] in ("10", "20") and l.count(",") == 13]
    g = collections.OrderedDict()
    for r in rows:
        g.setdefault((r[0], r[3], r[4]), []).append(r)
    print(fn)
    print("| N | band (GHz) | Zt (ohm) | inst | infeasible | greedy+prune count min/med/max | max(LB_C, LB_ESL) | LB_Y (MILP) | greedy / simple LB | greedy / LB_Y | local search found k-1 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, rs in g.items():
        ok = [r for r in rs if r[6] != "infeasible"]
        lbs = [max(int(r[7]), int(r[8])) for r in rs]; lby = [int(r[10]) for r in rs]
        band = k[1] if "-" in k[1] else f"0.01-{k[1]}"
        if ok:
            gc = np.array([int(r[6]) for r in ok]); r1 = np.array([float(r[11]) for r in ok]); r2 = np.array([float(r[12]) for r in ok])
            ls = sum(r[13] == "True" for r in ok); lsn = sum(r[13] in ("True", "False") for r in ok)
            print(f"| {k[0]} | {band} | {k[2]} | {len(rs)} | {len(rs)-len(ok)} | {gc.min()}/{int(np.median(gc))}/{gc.max()} | {min(lbs)}-{max(lbs)} | {min(lby)}-{max(lby)} | {r1.min():.2f}-{r1.max():.2f} | {r2.min():.2f}-{r2.max():.2f} | {ls}/{lsn if lsn else '-'} |")
        else:
            print(f"| {k[0]} | {band} | {k[2]} | {len(rs)} | {len(rs)} | - | {min(lbs)}-{max(lbs)} | {min(lby)}-{max(lby)} | - | - | - |")
