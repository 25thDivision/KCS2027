#!/usr/bin/env python3
"""
Does the simulation rank the decoders like the hardware does?
=============================================================
Spearman rank correlation, per (code, distance, backend), between
  x = simulation LER of each ML decoder
  y = LER of the same decoder on that backend
(KCS stim_simulation/analyze_sim_ler.py, analysis (b), on the
results/gathered.csv schema; nothing is hardcoded.)

Aggregation unit: one point per DECODER. Its simulation LER is the mean
of gathered.csv's `ler` over the rows with platform_or_sim == "sim" of
that (code, distance, decoder), i.e. over the (noise_profile, p)
training grid. Its backend LER is the mean over the rows of that backend
(platform "hardware" or "aer") for the same (code, distance, decoder),
i.e. over the checkpoints evaluated on the backend's data and over the
analyzed runs. --match restricts both means to the (noise_profile, p)
conditions present on both sides.
Only the standalone ML decoders are ranked (no MWPM, hybrids, NVIDIA or
raw rows). A group needs at least 3 decoders on both sides.

Output: results/spearman.csv with the columns
  code, distance, platform, backend, n_decoders, rho, p_value, decoders
Usage: python analysis/spearman.py [--gathered results/gathered.csv]
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model import MODEL_NAMES, model_prefix  # noqa: E402

ML_DECODERS = [model_prefix(m) for m in MODEL_NAMES]
COLUMNS = ["code", "distance", "platform", "backend", "n_decoders", "rho",
           "p_value", "decoders"]
MIN_DECODERS = 3


def load(gathered_csv):
    rows = []
    for r in csv.DictReader(open(gathered_csv)):
        try:
            r["ler"] = float(r["ler"])
        except (TypeError, ValueError):
            continue
        rows.append(r)
    return rows


def spearman_table(rows, match=False):
    """List of result dicts (COLUMNS), one per (code, distance, backend)."""
    sim = defaultdict(lambda: defaultdict(list))    # (code,d)->dec->[(cond, ler)]
    hw = defaultdict(lambda: defaultdict(list))     # (code,d,plat,backend)->...
    for r in rows:
        if r["decoder"] not in ML_DECODERS:
            continue
        cond = (r["noise_profile"], str(r["p"]))
        if r["platform_or_sim"] == "sim":
            sim[(r["code"], r["distance"])][r["decoder"]].append(
                (cond, r["ler"]))
        elif r["platform_or_sim"] in ("hardware", "aer"):
            hw[(r["code"], r["distance"], r["platform_or_sim"],
                r["backend"])][r["decoder"]].append((cond, r["ler"]))
    out = []
    for (code, dist, plat, backend), per_dec in sorted(hw.items()):
        xs, ys, names = [], [], []
        for dec in ML_DECODERS:
            s, h = sim[(code, dist)].get(dec, []), per_dec.get(dec, [])
            if match:
                common = {c for c, _ in s} & {c for c, _ in h}
                s = [(c, v) for c, v in s if c in common]
                h = [(c, v) for c, v in h if c in common]
            if not s or not h:
                continue
            xs.append(sum(v for _, v in s) / len(s))
            ys.append(sum(v for _, v in h) / len(h))
            names.append(dec)
        row = {"code": code, "distance": dist, "platform": plat,
               "backend": backend, "n_decoders": len(names), "rho": "",
               "p_value": "", "decoders": " ".join(names)}
        if len(names) >= MIN_DECODERS:
            rho, pval = spearmanr(xs, ys)
            row["rho"], row["p_value"] = f"{rho:.4f}", f"{pval:.4g}"
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--gathered",
                    default=str(_ROOT / "results" / "gathered.csv"))
    ap.add_argument("--out", default=None,
                    help="output CSV (default: next to the input, "
                         "spearman.csv)")
    ap.add_argument("--match", action="store_true",
                    help="use only the (noise_profile, p) conditions "
                         "present in both the simulation and the backend "
                         "rows")
    args = ap.parse_args()
    table = spearman_table(load(args.gathered), args.match)
    out = Path(args.out) if args.out else (Path(args.gathered).parent
                                           / "spearman.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(table)
    print(f"{len(table)} group(s) -> {out}")
    for r in table:
        print(f"  {r['code']:<9} d={r['distance']:<3} {r['platform']:<9}"
              f"{r['backend']:<20} n={r['n_decoders']} rho={r['rho'] or 'N/A'}"
              f" p={r['p_value'] or 'N/A'}")


if __name__ == "__main__":
    main()
