#!/usr/bin/env python3
"""
Figures from results/gathered.csv
=================================
  decoders_<code>_<dtag>.png   LER of MWPM and the 8 ML decoders, one bar
                               per data source, for one (code, distance)
  nvidia_surface.png           MWPM vs NVIDIA(Accurate), per surface
                               distance and data source
Each figure is saved with a CSV of the plotted values next to it (same
name, .csv) in results/figures/.

Data sources (bar colors, fixed per source):
  simulation        gathered platform_or_sim == "sim"
  QPU-cal (Stim)    "sim_qpu_cal": Stim with the QPU calibration profile
  QPU-cal (Aer)     "aer": Aer with an IBM backend's calibration
  hardware          "hardware"
A bar is the MEAN of gathered.csv's `ler` over the rows of that (code,
distance, source, decoder): over the (noise_profile, p) grid for the
simulation sources, over the evaluated checkpoints and analyzed runs for
the others. The CSV lists the mean and the number of rows behind it.

Usage: python analysis/plot.py [--gathered results/gathered.csv]
                               [--outdir results/figures]
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_generation import distance_tag  # noqa: E402
from model import MODEL_NAMES, model_prefix  # noqa: E402

# source -> (label, color); order and colors are fixed per source
SOURCES = [("sim", "simulation", "#2a78d6"),
           ("sim_qpu_cal", "QPU-cal (Stim)", "#eb6834"),
           ("aer", "QPU-cal (Aer)", "#1baf7a"),
           ("hardware", "hardware", "#eda100")]
ML_DECODERS = [model_prefix(m) for m in MODEL_NAMES]
DECODERS = ["MWPM"] + ML_DECODERS
NVIDIA = "NVIDIA(Accurate)"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def load(gathered_csv):
    rows = []
    for r in csv.DictReader(open(gathered_csv)):
        try:
            r["ler"] = float(r["ler"])
        except (TypeError, ValueError):
            continue
        rows.append(r)
    return rows


def mean_table(rows):
    """{(code, distance, source, decoder): (mean ler, n rows)}"""
    acc = defaultdict(list)
    for r in rows:
        acc[(r["code"], str(r["distance"]), r["platform_or_sim"],
             r["decoder"])].append(r["ler"])
    return {k: (sum(v) / len(v), len(v)) for k, v in acc.items()}


def _style(ax, ylabel):
    ax.set_ylabel(ylabel, color=INK)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def _title(ax, text, legend=False):
    """Left-aligned title; the legend sits in a row between the title and
    the plot so it never covers a bar."""
    ax.set_title(text, color=INK, loc="left", pad=30 if legend else 8)
    if legend:
        ax.legend(frameon=False, labelcolor=INK, loc="lower left",
                  bbox_to_anchor=(0.0, 1.0), ncol=4, borderaxespad=0.2,
                  handlelength=1.2, columnspacing=1.6)


def _grouped_bars(ax, categories, series, values):
    """categories on x; series = [(key, label, color)]; values[(cat, key)]
    -> mean or missing. Missing bars are simply absent."""
    n = len(series)
    width = 0.8 / max(n, 1)
    for i, (key, label, color) in enumerate(series):
        xs, hs = [], []
        for j, cat in enumerate(categories):
            if (cat, key) in values:
                xs.append(j - 0.4 + (i + 0.5) * width)
                hs.append(values[(cat, key)])
        if xs:
            ax.bar(xs, hs, width=width * 0.86, color=color, label=label,
                   linewidth=0)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=35, ha="right")


def _save(fig, outdir, name, table_rows, header):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    with open(outdir / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table_rows)
    return outdir / f"{name}.png"


def plot_decoders(table, code, distance, outdir):
    """Decoder LER bars of one (code, distance); None when it has no
    decoder rows."""
    present = [s for s in SOURCES
               if any((code, distance, s[0], d) in table for d in DECODERS)]
    if not present:
        return None
    values, rows = {}, []
    for dec in DECODERS:
        for key, label, _ in present:
            if (code, distance, key, dec) in table:
                mean, n = table[(code, distance, key, dec)]
                values[(dec, key)] = mean
                rows.append([code, distance, label, dec, f"{mean:.6f}", n])
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    _grouped_bars(ax, DECODERS, present, values)
    _style(ax, "logical error rate")
    dtag = distance_tag(code, distance)
    if len(present) >= 2:
        _title(ax, f"{code} {dtag}: decoder LER by data source",
               legend=True)
    else:
        _title(ax, f"{code} {dtag}: decoder LER ({present[0][1]})")
    return _save(fig, outdir, f"decoders_{code}_{dtag}", rows,
                 ["code", "distance", "source", "decoder", "mean_ler",
                  "n_rows"])


def plot_nvidia(table, outdir):
    """MWPM vs NVIDIA(Accurate) on the surface code, per (distance,
    source); None when there is no NVIDIA row."""
    groups = sorted({(d, s) for (c, d, s, dec) in table
                     if c == "surface" and dec == NVIDIA})
    if not groups:
        return None
    labels = {k: lab for k, lab, _ in SOURCES}
    cats = [f"{distance_tag('surface', d)} {labels.get(s, s)}"
            for d, s in groups]
    series = [("MWPM", "MWPM", "#2a78d6"), (NVIDIA, NVIDIA, "#eb6834")]
    values, rows = {}, []
    for cat, (d, s) in zip(cats, groups):
        for key, _, _ in series:
            if ("surface", d, s, key) in table:
                mean, n = table[("surface", d, s, key)]
                values[(cat, key)] = mean
                rows.append(["surface", d, labels.get(s, s), key,
                             f"{mean:.6f}", n])
    fig, ax = plt.subplots(figsize=(max(6.5, 1.6 * len(cats) + 2), 4.0))
    _grouped_bars(ax, cats, series, values)
    _style(ax, "logical error rate")
    _title(ax, "surface: MWPM with and without the NVIDIA pre-decoder",
           legend=True)
    return _save(fig, outdir, "nvidia_surface", rows,
                 ["code", "distance", "source", "decoder", "mean_ler",
                  "n_rows"])


def make_figures(gathered_csv, outdir):
    """Write every figure the data supports; returns the PNG paths."""
    table = mean_table(load(gathered_csv))
    outdir = Path(outdir)
    made = []
    for code, distance in sorted({(c, d) for (c, d, _, _) in table}):
        path = plot_decoders(table, code, distance, outdir)
        if path:
            made.append(path)
    path = plot_nvidia(table, outdir)
    if path:
        made.append(path)
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--gathered",
                    default=str(_ROOT / "results" / "gathered.csv"))
    ap.add_argument("--outdir",
                    default=str(_ROOT / "results" / "figures"))
    args = ap.parse_args()
    made = make_figures(args.gathered, args.outdir)
    print(f"{len(made)} figure(s) -> {args.outdir}")
    for p in made:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
