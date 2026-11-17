#!/usr/bin/env python3
"""
IonQ provider (color code)
==========================
Runs the color code hardware circuit (circuits/color/colorcode.py) on an
IonQ backend through qiskit-ionq (API checked against the installed
1.1.1: IonQProvider(token).get_backend(name), backend.num_qubits,
backend.run(circuit, shots=..., noise_model=...), job.job_id(),
job.status(), job.result().get_counts(), backend.retrieve_job(id),
backend.calibration()).

Backend names of this repository:
    ionq_forte1       -> IonQ "qpu.forte-1"
    ionq_sim_forte1   -> IonQ "simulator" with noise_model "forte-1"

  submit   circuit (ancilla mode: colorcode.DEFAULT_ANCILLA, or the
           ancilla option) -> qubit-count check against the number the
           backend reports (never hardcoded) -> transpile(backend,
           optimization_level=0: the CX order of a check is part of the
           circuit's contract) -> one IonQ job per PUB (IonQ takes one
           such circuit per job; a PUB above the backend's max shots is
           split) -> job.json, characterization.json
  status   job state names
  collect  counts of every job -> raw.npz (syn, dat, cycles, pub_shots)

IonQ is fully connected: there is no placement step. There is no
per-qubit calibration profile either (no [4] profile stage for color).

A circuit that reuses a measured ancilla (the d = 5 default) needs
mid-circuit measurement and reset; qiskit-ionq sends such circuits as
OpenQASM 3 and only one per job. job.json records it as
"mid_circuit": true. Whether a given backend accepts it is decided by
IonQ at submission.

Credentials: keys.json field ionq_api_key. --dry-run needs no key.
"""
import json
import math
import sys
import time

import numpy as np

from hardware.providers.base import (
    HardwareProvider, FINAL_STATES, RUNS_DIR, json_dump, load_keys, save_raw)
from dataset_generation import normalize_distance, require_implemented

# this repository's backend name -> (IonQ backend, simulator noise model)
BACKENDS = {
    "ionq_forte1": ("qpu.forte-1", None),
    "ionq_sim_forte1": ("simulator", "forte-1"),
}
KEY_FIELD = "ionq_api_key"


def plan_jobs(shots, pubs, max_shots):
    """(shots per job, number of jobs): pubs jobs of `shots` shots, each
    split evenly when it exceeds the backend's max shots."""
    k = 1
    if max_shots and shots > max_shots:
        k = math.ceil(shots / max_shots)
    return math.ceil(shots / k), pubs * k


def counts_to_raw(counts, n_syn, n_dat):
    """Qiskit counts {bitstring: n} of a (syn, data) circuit -> (syn,
    dat) uint8 arrays, one row per shot. Clbit i is character -1-i of
    the bitstring once the register separators are removed (syn is the
    first register, data the second)."""
    syn, dat = [], []
    for key, n in counts.items():
        flat = str(key).replace(" ", "")[::-1]
        if len(flat) < n_syn + n_dat:
            flat = flat + "0" * (n_syn + n_dat - len(flat))
        bits = np.frombuffer(flat[:n_syn + n_dat].encode(), dtype=np.uint8) - 48
        n = int(round(n))
        syn.append(np.repeat(bits[None, :n_syn], n, axis=0))
        dat.append(np.repeat(bits[None, n_syn:], n, axis=0))
    if not syn:
        return (np.zeros((0, n_syn), dtype=np.uint8),
                np.zeros((0, n_dat), dtype=np.uint8))
    return (np.concatenate(syn).astype(np.uint8),
            np.concatenate(dat).astype(np.uint8))


def _versions():
    import qiskit
    import qiskit_ionq
    return {"python": sys.version.split()[0],
            "qiskit": qiskit.__version__,
            "qiskit_ionq": getattr(qiskit_ionq, "__version__", "?")}


class IonQProvider(HardwareProvider):
    name = "ionq"

    def __init__(self, backend):
        super().__init__(backend)
        if backend not in BACKENDS:
            raise ValueError(f"unknown IonQ backend '{backend}' "
                             f"(known: {sorted(BACKENDS)})")
        self.ionq_name, self.noise_model = BACKENDS[backend]

    def _backend(self, need_key=True):
        from qiskit_ionq import IonQProvider as _QiskitIonQ
        token = "DRY_RUN_NO_KEY"
        if need_key:
            token = load_keys((KEY_FIELD,))[KEY_FIELD]
        return _QiskitIonQ(token=token).get_backend(self.ionq_name)

    def submit(self, code, distance, cycles, shots, pubs, dry_run=False,
               ancilla=None):
        from qiskit import transpile
        from qiskit_ionq.helpers import circuit_requires_qasm3
        from circuits.color.colorcode import ColorCodeHardware

        require_implemented(code, distance)
        if code != "color":
            sys.exit(f"--code {code} does not run on IonQ backends.")
        backend = self._backend(need_key=not dry_run)
        hw = ColorCodeHardware(cycles, distance, ancilla)
        qc = hw.build_circuit()
        limit = backend.num_qubits
        if qc.num_qubits > limit:
            sys.exit(f"{self.backend}: the circuit needs {qc.num_qubits} "
                     f"qubits (color d={distance}, {cycles} cycles, "
                     f"ancilla '{hw.ancilla}') but the backend reports "
                     f"{limit}.")
        tqc = transpile(qc, backend=backend, optimization_level=0)
        mid_circuit = bool(circuit_requires_qasm3(tqc))
        # the simulator's shot cap (1) is for its ideal state-vector
        # mode; with a noise model it samples any number of shots
        max_shots = (None if self.noise_model
                     else getattr(backend, "_max_shots", None))
        shots_per_job, n_jobs = plan_jobs(shots, pubs, max_shots)
        print(f"{self.backend}: color d={distance}, {cycles} cycles, "
              f"ancilla '{hw.ancilla}', {qc.num_qubits}/{limit} qubits, "
              f"depth={tqc.depth()}, mid-circuit={mid_circuit}")

        ts = time.strftime("%Y%m%d-%H%M%S")
        job_ids = []
        if dry_run:
            run_dir = RUNS_DIR / f"{self.backend}_{ts}_dryrun"
            print(f"--dry-run: not submitting (plan: {n_jobs} job(s) x "
                  f"{shots_per_job} shots).")
        else:
            options = {"shots": shots_per_job}
            if self.noise_model:
                options["noise_model"] = self.noise_model
            for _ in range(n_jobs):
                job_ids.append(backend.run(tqc, **options).job_id())
            run_dir = RUNS_DIR / f"{self.backend}_{ts}"
        while run_dir.exists():                  # same-second collision guard
            run_dir = run_dir.with_name(run_dir.name + "b")
        run_dir.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            self.snapshot(run_dir)

        meta = {"job_id": job_ids[0] if job_ids else None,
                "job_ids": job_ids,
                "n_pubs": n_jobs, "shots_per_pub": shots_per_job,
                "backend": self.backend, "code": code,
                "distance": normalize_distance(code, distance),
                "cycles": cycles, "shots": shots, "pubs": pubs,
                "ancilla": hw.ancilla, "mid_circuit": mid_circuit,
                "num_qubits": qc.num_qubits, "backend_qubits": limit,
                "initial_layout": None,
                "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "dry_run": dry_run,
                "transpiled_depth": tqc.depth(),
                "transpiled_ops": {k: int(v)
                                   for k, v in tqc.count_ops().items()},
                "versions": _versions()}
        json_dump(meta, run_dir / "job.json")
        print(f"run folder: {run_dir}")
        return run_dir

    def status(self, job_ids):
        backend = self._backend()
        names = []
        for jid in job_ids:
            st = backend.retrieve_job(jid).status()
            names.append(getattr(st, "name", str(st)))
        return names

    def wait(self, job_ids, poll=30):
        last = None
        while True:
            names = self.status(job_ids)
            if names != last:
                print(f"[{time.strftime('%H:%M:%S')}] jobs {job_ids}: {names}")
                last = names
            if any(n in ("ERROR", "CANCELLED") for n in names):
                sys.exit(f"jobs {job_ids} ended as {names} - nothing to "
                         f"analyze.")
            if all(n in FINAL_STATES for n in names):
                return
            time.sleep(poll)

    def collect(self, run_dir, job_ids=None, cycles=3, poll=30):
        from circuits.color.colorcode import color_def
        meta_path = run_dir / "job.json"
        if not meta_path.exists():
            sys.exit(f"{run_dir} has no job.json - cannot collect an IonQ "
                     f"run without it.")
        meta = json.load(open(meta_path))
        job_ids = meta.get("job_ids") or job_ids
        cycles = meta.get("cycles", cycles)
        S = color_def(meta["distance"])
        n_syn, n_dat = S.N_CHECKS * cycles, S.NUM_DATA
        self.wait(job_ids, poll=poll)
        backend = self._backend()
        syns, dats, pub_shots = [], [], []
        for jid in job_ids:
            counts = backend.retrieve_job(jid).result().get_counts()
            s, d = counts_to_raw(counts, n_syn, n_dat)
            syns.append(s)
            dats.append(d)
            pub_shots.append(s.shape[0])
        raw = save_raw(run_dir, np.concatenate(syns), np.concatenate(dats),
                       cycles, pub_shots)
        print(f"saved raw results -> {raw} ({len(pub_shots)} job(s))")
        return raw

    def snapshot(self, run_dir):
        """Latest characterization of the QPU (best effort; the simulator
        has none)."""
        try:
            cal = self._backend().calibration()
            if cal is not None:
                data = getattr(cal, "__dict__", None) or cal
                json_dump(data, run_dir / "characterization.json")
                print("   saved characterization.json")
        except Exception as e:
            print(f"   WARNING: could not save characterization: {e}")
