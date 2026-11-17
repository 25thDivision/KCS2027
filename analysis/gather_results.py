#!/usr/bin/env python3
"""
Collect every result into one long-format table
===============================================
Inputs
  results/train/summary/*.csv   written by train.py, one file per
                                (model, code, distance, cycles, p, noise)
                                config: the model's best head-LER, raw,
                                MWPM and the two hybrids
  results/hardware/*.csv        written by hardware/run_hw.py analyze,
                                one file per analyzed run
Output
  results/gathered.csv with the columns
    code, distance, platform_or_sim, backend, noise_profile, p, decoder,
    ler, n_shots, source_file

platform_or_sim
  sim          training summary on a realistic/* (or any non-qpu) profile
  sim_qpu_cal  training summary on a qpu/* calibration-averaged profile
  aer          hardware report of an aer_* backend (Aer simulation with
               an IBM backend's calibration)
  hardware     hardware report of a QPU backend
decoder
  CNN .. GRAPHMAMBA, MWPM, MWPM+<MODEL>, <MODEL>+MWPM, NVIDIA(Accurate),
  raw. In a hardware report an ML decoder row is one CHECKPOINT; its
  noise_profile and p are the conditions that checkpoint was trained on
  (read from the checkpoint file name). Rows that do not depend on a
  training condition (raw, MWPM, NVIDIA) have noise_profile and p empty.
distance
  as normalized by dataset_generation (heavyhex (3,3) = 33).

Usage: python analysis/gather_results.py [--results results] [--out ...]
"""
import argparse
import csv
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_generation import distance_tag  # noqa: E402

COLUMNS = ["code", "distance", "platform_or_sim", "backend", "noise_profile",
           "p", "decoder", "ler", "n_shots", "source_file"]
# checkpoint file name: <MODEL>_<code>_<dtag>_c<cycles>_p<p>_<noise tag>
# [_<run name>].pt (train.py)
_CKPT = re.compile(r"^(?P<label>[^ ]+) \((?P<file>[^)]+\.pt)\)$")
_CKPT_NAME = re.compile(
    r"^[A-Z]+_(?P<code>[a-z]+)_(?P<dtag>d\d+)_c\d+_p(?P<p>[0-9.eE-]+?)_"
    r"(?P<rest>.+)\.pt$")
_DP_TAG = re.compile(
    r"^(dp[0-9.eE-]+_mf[0-9.eE-]+_rf[0-9.eE-]+_gd[0-9.eE-]+?)(?:_.*)?$")


def profile_of(rest):
    """Noise part of a checkpoint name (noise tag, optionally followed by
    _<run name>) -> noise profile name. Registered profiles are matched
    by their tag; an unregistered 4-parameter tag is taken as
    realistic/<tag>, anything else as qpu/<rest>."""
    from dataset_generation.heavyhex33_stim import NOISE_PROFILES, noise_tag
    tags = {noise_tag(name): name for name in NOISE_PROFILES}
    for tag in sorted(tags, key=len, reverse=True):
        if rest == tag or rest.startswith(tag + "_"):
            return tags[tag]
    m = _DP_TAG.match(rest)
    if m:
        return "realistic/" + m.group(1)
    return "qpu/" + rest


def train_rows(summary_dir):
    rows = []
    for f in sorted(Path(summary_dir).glob("*.csv")):
        for r in csv.DictReader(open(f)):
            noise = r["noise_profile"]
            rows.append({
                "code": r["code"], "distance": r["distance"],
                "platform_or_sim": ("sim_qpu_cal" if noise.startswith("qpu/")
                                    else "sim"),
                "backend": "", "noise_profile": noise, "p": r["p"],
                "decoder": r["decoder"], "ler": r["ler"],
                "n_shots": r["n_shots"], "source_file": str(f)})
    return rows


def parse_hw_decoder(name):
    """Report decoder cell -> (decoder, noise_profile, p).

    'CNN (CNN_surface_d3_c3_p0.005_dp0.001_mf0.01_rf0.01_gd0.008.pt)'
        -> ('CNN', 'realistic/dp0.001_mf0.01_rf0.01_gd0.008', '0.005')
    'MWPM+CNN (...)' / 'CNN+MWPM (...)' keep their label;
    'raw (no decoding)' -> ('raw', '', ''); others unchanged."""
    if name.startswith("raw"):
        return "raw", "", ""
    m = _CKPT.match(name)
    if not m:
        return name, "", ""
    c = _CKPT_NAME.match(m.group("file"))
    if not c:
        return m.group("label"), "", ""
    return m.group("label"), profile_of(c.group("rest")), c.group("p")


def hardware_rows(hw_dir):
    rows = []
    for f in sorted(Path(hw_dir).glob("*.csv")):
        for r in csv.DictReader(open(f)):
            backend = r.get("backend", "")
            if "code" not in r or not r["code"]:
                continue        # reports written before the code column
            decoder, noise, p = parse_hw_decoder(r["decoder"])
            rows.append({
                "code": r["code"], "distance": r["distance"],
                "platform_or_sim": ("aer" if backend.startswith("aer_")
                                    else "hardware"),
                "backend": backend, "noise_profile": noise, "p": p,
                "decoder": decoder, "ler": r["ler"],
                "n_shots": r.get("shots", ""), "source_file": str(f)})
    return rows


def gather(results_dir, out_path=None):
    """Write <results_dir>/gathered.csv and return the rows."""
    results_dir = Path(results_dir)
    rows = (train_rows(results_dir / "train" / "summary")
            + hardware_rows(results_dir / "hardware"))
    out_path = Path(out_path) if out_path else results_dir / "gathered.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--results", default=str(_ROOT / "results"))
    ap.add_argument("--out", default=None,
                    help="output CSV (default: <results>/gathered.csv)")
    args = ap.parse_args()
    rows = gather(args.results, args.out)
    by = {}
    for r in rows:
        key = (r["platform_or_sim"], r["code"],
               distance_tag(r["code"], r["distance"]))
        by[key] = by.get(key, 0) + 1
    out = args.out or str(Path(args.results) / "gathered.csv")
    print(f"{len(rows)} rows -> {out}")
    for key in sorted(by):
        print(f"  {key[0]:<12} {key[1]:<9} {key[2]:<4} {by[key]}")


if __name__ == "__main__":
    main()
