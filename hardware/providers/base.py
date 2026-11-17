#!/usr/bin/env python3
"""
Hardware provider interface
===========================
hardware/run_hw.py talks to a QPU (or a calibrated simulator) only
through a HardwareProvider. A provider is picked from the backend name
(hardware/providers/__init__.py):

    ibm_*   -> ibm.IBMProvider             (heavyhex, surface)
    ionq_*  -> ionq.IonQProvider           (color)
    aer_*   -> aer_calibrated.AerCalibratedProvider   (d=3 evaluation)

Contract of a run folder hardware/runs/<backend>_<YYYYMMDD-HHMMSS>[_dryrun]:
  job.json   written by submit: at least job_id, job_ids, backend, code,
             distance, cycles, shots, pubs, n_pubs, shots_per_pub,
             submitted_at, dry_run
  raw.npz    written by collect: syn (shots, N_CHECKS * cycles) uint8 raw
             syndrome bits in clbit order, dat (shots, num_data) uint8
             final data bits, cycles, pub_shots (shots of each PUB / job,
             concatenated in order)
analyze reads only job.json and raw.npz, so every provider's runs go
through the same report code.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "hardware" / "runs"
FINAL_STATES = ("DONE", "ERROR", "CANCELLED")


def json_dump(obj, path):
    # default=str: calibration snapshots contain datetime objects
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_keys(required=()):
    """keys.json at the repo root; `required` fields must be filled."""
    import sys
    path = ROOT / "keys.json"
    if not path.exists():
        sys.exit(f"keys.json not found at {path}. "
                 f"Copy keys.example.json to keys.json and fill it in.")
    keys = json.load(open(path))
    for k in required:
        if not keys.get(k) or str(keys[k]).startswith("YOUR_"):
            sys.exit(f"keys.json: '{k}' is not set.")
    return keys


def save_raw(run_dir, syn, dat, cycles, pub_shots):
    """Write the provider-independent raw.npz of a run."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_dir / "raw.npz", syn=np.asarray(syn, dtype=np.uint8),
        dat=np.asarray(dat, dtype=np.uint8), cycles=int(cycles),
        pub_shots=np.asarray(pub_shots, dtype=np.int64))
    return run_dir / "raw.npz"


def load_raw(npz_path):
    """raw.npz -> (syn, dat, cycles, pub_shots or None). Files written
    before pub_shots existed count as one PUB."""
    d = np.load(npz_path)
    ps = d["pub_shots"] if "pub_shots" in d.files else None
    return (d["syn"].astype(np.uint8), d["dat"].astype(np.uint8),
            int(d["cycles"]), ps)


class HardwareProvider:
    """One backend of one provider.

    submit(code, distance, cycles, shots, pubs, dry_run=False, **options)
        -> run folder (Path); job ids are recorded in its job.json. With
        dry_run nothing is sent, but everything else (circuit, placement,
        snapshot, job.json) is done.
    status(job_ids) -> list of state names; "DONE", "ERROR" and
        "CANCELLED" are final, anything else means still pending.
    wait(job_ids, poll=30) -> returns when every job is DONE; exits on
        ERROR / CANCELLED.
    collect(run_dir, job_ids=None, cycles=3, poll=30) -> raw.npz path
        (waits for the jobs if needed). job_ids / cycles are only used
        when the run folder has no job.json.
    snapshot(run_dir) -> saves what the backend reports about itself
        (calibration) into the run folder."""

    name = "base"

    def __init__(self, backend):
        self.backend = backend

    def submit(self, code, distance, cycles, shots, pubs, dry_run=False,
               **options):
        raise NotImplementedError

    def status(self, job_ids):
        raise NotImplementedError

    def wait(self, job_ids, poll=30):
        raise NotImplementedError

    def collect(self, run_dir, job_ids=None, cycles=3, poll=30):
        raise NotImplementedError

    def snapshot(self, run_dir):
        raise NotImplementedError
