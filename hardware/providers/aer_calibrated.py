#!/usr/bin/env python3
"""
Aer simulation with an IBM backend's calibration (d = 3 evaluation)
===================================================================
Backend name aer_<ibm backend> (e.g. aer_ibm_pittsburgh). The run is a
LOCAL noisy simulation of the circuit a real submission to <ibm backend>
would execute, with qiskit-aer's NoiseModel.from_backend on that
backend's current calibration (KCS ibm_experiment/simulators/
ibm_simulator.py, simulator branch):

  1. the code's hardware circuit and its placement on the device, the
     same as the IBM provider (ibm.build_and_place)
  2. the device target restricted to the placed qubits and renumbered
     0 .. n-1 in circuit order, with every calibration value (gate
     errors and durations, readout errors, T1 / T2) carried over to the
     new index. KCS renumbered the coupling map only, so qubit i got the
     noise of physical qubit i; here the noise follows the placement.
  3. transpile onto that target (optimization_level=1), no dynamical
     decoupling
  4. AerSimulator(noise_model=NoiseModel.from_backend(restricted
     target)), `pubs` runs of `shots` shots
  5. hardware/runs/aer_<ibm backend>_<timestamp>/ with job.json and
     raw.npz in the common format, so analyze reads it like a QPU run

Nothing is sent to the QPU; the IBM service is contacted only to read the
target and the coupling map.

Scope: heavyhex (3,3) and surface d = 3. Larger codes are refused (KCS:
state-vector out of memory, MPS timeouts, stabilizer method rejecting the
non-Clifford relaxation channels).
  * surface d = 3 (17 qubits): full noise model, including thermal
    relaxation
  * heavyhex (3,3) (37 qubits) is too wide for a state vector. It runs
    on Aer's stabilizer method with thermal relaxation switched off
    (depolarizing gate errors and readout errors only) and with the
    circuit kept in a Clifford basis. job.json records which noise was
    used ("noise").
"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from hardware.providers.base import (
    HardwareProvider, RUNS_DIR, json_dump, load_keys, save_raw)
from dataset_generation import normalize_distance, require_implemented

SUPPORTED = {("heavyhex", 33), ("surface", 3)}
STATEVECTOR_MAX_QUBITS = 24
CLIFFORD_BASIS = ["sx", "x", "s", "sdg", "z", "id"]    # no h: H -> s sx s, so the
#                                   calibrated sx error applies to it


def restrict_target(target, qubits):
    """Target of the device restricted to `qubits` (device indices) and
    renumbered 0 .. n-1 in that order, keeping each instruction's
    properties and each qubit's T1 / T2. Instructions that touch a qubit
    outside the set, and non-gate entries (control flow), are dropped."""
    from qiskit.circuit import Instruction
    from qiskit.transpiler import Target
    idx = {q: i for i, q in enumerate(qubits)}
    props = None
    if target.qubit_properties:
        props = [target.qubit_properties[q] for q in qubits]
    sub = Target(num_qubits=len(qubits), dt=target.dt,
                 qubit_properties=props)
    for name in target.operation_names:
        op = target.operation_from_name(name)
        if not isinstance(op, Instruction):
            continue
        entries = {}
        for qargs, ip in target[name].items():
            if qargs is None:
                continue
            if all(q in idx for q in qargs):
                entries[tuple(idx[q] for q in qargs)] = ip
        if entries:
            sub.add_instruction(op, entries)
    return sub


class _TargetBackend:
    """The few attributes NoiseModel.from_backend reads from a BackendV2."""
    version = 2

    def __init__(self, target, name):
        self.target = target
        self.name = name
        self.dt = target.dt
        self.operation_names = list(target.operation_names)


def counts_rows(result, circuit):
    """Aer memory of a (syn, data) circuit -> (syn, dat) uint8 arrays."""
    n_syn = circuit.cregs[0].size
    n_dat = circuit.cregs[1].size
    syn, dat = [], []
    for shot in result.get_memory(circuit):
        d, s = shot.split()
        syn.append([int(b) for b in s[::-1]])
        dat.append([int(b) for b in d[::-1]])
    syn = np.array(syn, dtype=np.uint8).reshape(-1, n_syn)
    dat = np.array(dat, dtype=np.uint8).reshape(-1, n_dat)
    return syn, dat


class AerCalibratedProvider(HardwareProvider):
    name = "aer"

    def __init__(self, backend):
        super().__init__(backend)
        if not backend.startswith("aer_ibm_"):
            raise ValueError(f"aer backend names are aer_<ibm backend>, "
                             f"got '{backend}'")
        self.ibm_backend = backend[len("aer_"):]

    def submit(self, code, distance, cycles, shots, pubs, dry_run=False,
               reselect_layout=False, seed=0):
        from types import SimpleNamespace
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel
        from circuits.heavyhex.fetch_coupling import fetch
        from hardware.providers.ibm import (
            build_and_place, get_service, resolve_instance)

        require_implemented(code, distance)
        key = (code, normalize_distance(code, distance))
        if key not in SUPPORTED:
            sys.exit(f"aer-calibrated runs support {sorted(SUPPORTED)} "
                     f"only (got {key}); larger codes do not fit a "
                     f"noisy Aer simulation.")
        keys = load_keys(("ibm_API", "ibm_instance"))
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        coupling_path = fetch(self.ibm_backend, token=keys["ibm_API"],
                              instance=resolve_instance(keys,
                                                        self.ibm_backend),
                              outdir=str(RUNS_DIR))
        backend = get_service(keys, self.ibm_backend).backend(
            self.ibm_backend)
        args = SimpleNamespace(backend=self.ibm_backend, code=code,
                               distance=distance, cycles=cycles,
                               reselect_layout=reselect_layout)
        try:
            qc, layout, placement_info = build_and_place(args, backend,
                                                         coupling_path)
        except BaseException:
            Path(coupling_path).unlink(missing_ok=True)
            raise

        target = restrict_target(backend.target, layout)
        shim = _TargetBackend(target, self.ibm_backend)
        wide = qc.num_qubits > STATEVECTOR_MAX_QUBITS
        if wide:
            # stabilizer method: Pauli noise only, Clifford basis
            noise = NoiseModel.from_backend(shim, thermal_relaxation=False)
            two_q = [n for n in target.operation_names
                     if target.operation_from_name(n).num_qubits == 2]
            tqc = transpile(qc, basis_gates=CLIFFORD_BASIS + two_q,
                            coupling_map=target.build_coupling_map(),
                            initial_layout=list(range(qc.num_qubits)),
                            optimization_level=1)
            sim = AerSimulator(method="stabilizer", noise_model=noise)
            noise_desc = ("NoiseModel.from_backend(thermal_relaxation="
                          "False), stabilizer method")
        else:
            noise = NoiseModel.from_backend(shim)
            tqc = transpile(qc, target=target,
                            initial_layout=list(range(qc.num_qubits)),
                            optimization_level=1)
            sim = AerSimulator(noise_model=noise)
            noise_desc = "NoiseModel.from_backend, automatic method"
        print(f"{self.backend}: {code} {key[1]}, {qc.num_qubits} qubits "
              f"on device qubits {min(layout)}..{max(layout)}, "
              f"depth={tqc.depth()}, noise: {noise_desc}")

        ts = time.strftime("%Y%m%d-%H%M%S")
        run_dir = RUNS_DIR / (f"{self.backend}_{ts}"
                              + ("_dryrun" if dry_run else ""))
        while run_dir.exists():                  # same-second collision guard
            run_dir = run_dir.with_name(run_dir.name + "b")
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(coupling_path, run_dir / "coupling.json")

        if dry_run:
            print(f"--dry-run: not simulating (plan: {pubs} run(s) x "
                  f"{shots} shots).")
        else:
            syns, dats, pub_shots = [], [], []
            for k in range(pubs):
                res = sim.run(tqc, shots=shots, memory=True,
                              seed_simulator=seed + k).result()
                s, d = counts_rows(res, tqc)
                syns.append(s)
                dats.append(d)
                pub_shots.append(s.shape[0])
            raw = save_raw(run_dir, np.concatenate(syns),
                           np.concatenate(dats), cycles, pub_shots)
            print(f"saved raw results -> {raw} ({pubs} run(s) x {shots} "
                  f"shots)")

        import qiskit
        import qiskit_aer
        meta = {"job_id": None, "job_ids": [],
                "n_pubs": pubs, "shots_per_pub": shots,
                "backend": self.backend, "calibration_of": self.ibm_backend,
                "code": code, "distance": key[1], "cycles": cycles,
                "shots": shots, "pubs": pubs, "dd": None,
                "initial_layout": layout, "noise": noise_desc,
                "seed": seed,
                "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "dry_run": dry_run,
                "transpiled_depth": tqc.depth(),
                "transpiled_ops": {k: int(v)
                                   for k, v in tqc.count_ops().items()},
                "placement": placement_info,
                "versions": {"python": sys.version.split()[0],
                             "qiskit": qiskit.__version__,
                             "qiskit_aer": qiskit_aer.__version__}}
        json_dump(meta, run_dir / "job.json")
        print(f"run folder: {run_dir}")
        return run_dir

    def status(self, job_ids):
        return ["DONE" for _ in job_ids]

    def wait(self, job_ids, poll=30):
        return

    def collect(self, run_dir, job_ids=None, cycles=3, poll=30):
        raw = run_dir / "raw.npz"
        if not raw.exists():
            sys.exit(f"{run_dir} has no raw.npz (aer-calibrated runs are "
                     f"written at submit time).")
        return raw

    def snapshot(self, run_dir):
        return
