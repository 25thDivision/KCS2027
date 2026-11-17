#!/usr/bin/env python3
"""
IBM Quantum provider (heavyhex on Heron, surface on Nighthawk)
==============================================================
The IBM path of hardware/run_hw.py, moved here unchanged behind the
HardwareProvider interface (hardware/providers/base.py):

  submit   fetch_coupling.fetch() -> the code's hardware circuit ->
           calibration-aware placement (circuits/placement.py, pinned in
           hardware/placement_<backend>_<code>.json; static embedding as
           fallback) -> transpile(initial_layout, optimization_level=1)
           -> dd_utils.apply_dd -> QPU snapshot -> SamplerV2
           ONE job per (backend, code) carrying --pubs PUBs of the same
           ISA circuit; backend limits (max_shots / max_experiments)
           split the plan into extra PUBs / jobs
  status   job state names
  collect  results of every PUB -> raw.npz (syn, dat, cycles, pub_shots),
           job_metrics.json, run_started_at in job.json and the
           calibration as of execution time in properties_run.json
  snapshot properties.json / configuration.json / target.pkl

Credentials: keys.json fields ibm_API and ibm_instance.
"""
import json
import pickle
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np

from hardware.providers.base import (
    HardwareProvider, ROOT as _ROOT, RUNS_DIR, json_dump, load_keys,
    save_raw)
from circuits.heavyhex.heavyhex_37q import (
    ALL_PHYS, validate_backend, embedding_for)
from circuits.heavyhex.heavyhex_depth7_opt_for_37q import HeavyHex37QDepthOpt
from circuits.heavyhex.fetch_coupling import fetch
from circuits.heavyhex.dd_utils import apply_dd, dd_pulse_stats
from dataset_generation import normalize_distance, require_implemented


def get_service(keys):
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService(token=keys["ibm_API"],
                                instance=keys["ibm_instance"])



def _local_versions():
    import qiskit
    import qiskit_ibm_runtime
    return {"python": sys.version.split()[0],
            "qiskit": qiskit.__version__,
            "qiskit_ibm_runtime": qiskit_ibm_runtime.__version__}


def snapshot_backend(backend, run_dir):
    """Save everything the backend exposes about the QPU at this moment.

    Best-effort: each item is saved independently so one missing API
    doesn't lose the others."""
    try:
        props = backend.properties()
        if props is not None:
            json_dump(props.to_dict(), run_dir / "properties.json")
            print(f"   saved properties.json (calibration snapshot)")
    except Exception as e:
        print(f"   WARNING: could not save properties: {e}")
    try:
        json_dump(backend.configuration().to_dict(),
                   run_dir / "configuration.json")
        print(f"   saved configuration.json")
    except Exception as e:
        print(f"   WARNING: could not save configuration: {e}")
    try:
        with open(run_dir / "target.pkl", "wb") as f:
            pickle.dump(backend.target, f)
        print(f"   saved target.pkl")
    except Exception as e:
        print(f"   WARNING: could not pickle target: {e}")


def build_and_place(args, backend, coupling_path):
    """The code's hardware circuit and its device placement.

    args needs backend, code, distance, cycles and (optionally)
    reselect_layout. Returns (circuit, initial layout as device qubits in
    the circuit's qubit order, placement info for job.json). Shared with
    the Aer-calibrated provider, which simulates the same placement."""
    placement_info = None
    if args.code == "surface":
        # surface도 캘리브레이션 인지 자동 배치 (heavyhex와 동일 정책:
        # 체인 동안 고정, 임계값 위반 시 재탐색, --reselect-layout).
        # 회로/clbit/logical 정의가 전부 패치-로컬 좌표라 어떤 동형
        # 배치든 의미가 보존된다. 실패 시 정적 45도 임베딩 폴백.
        from circuits.rotatedSurface.rotatedSurface import (
            RotatedSurfaceHardware, surface_def, embedding_for_surface,
            validate_backend_surface, required_edges_surface)
        from circuits.placement import resolve_placement
        s_dist = normalize_distance("surface", args.distance)
        ALL_COORDS = surface_def(s_dist).ALL_COORDS
        qc = RotatedSurfaceHardware(args.cycles, s_dist).build_circuit()
        cm = json.load(open(coupling_path))
        try:
            static_map = embedding_for_surface(args.backend, s_dist)
        except RuntimeError:
            static_map = None
        # the pinned-placement file is per (backend, code); d=3 keeps the
        # plain "surface" key, other distances get their own
        place_key = "surface" if s_dist == 3 else f"surface_d{s_dist}"
        mapping, placement_info = resolve_placement(
            args.backend, place_key, _ROOT / "hardware",
            ALL_COORDS, required_edges_surface(s_dist), qc,
            [tuple(e) for e in cm["coupling_map"]], backend.target,
            static_mapping=static_map,
            reselect=getattr(args, "reselect_layout", False))
        if mapping is None:
            print("WARNING: 유효한 자동 배치 없음 — 정적 45도 임베딩 폴백")
            validate_backend_surface(coupling_path, distance=s_dist)
            mapping = embedding_for_surface(args.backend, s_dist)
            placement_info = {"fallback": "static",
                              **(placement_info or {})}
        else:
            eset = {tuple(sorted(e)) for e in cm["coupling_map"]}
            for u, v in required_edges_surface(s_dist):
                assert tuple(sorted((mapping[u], mapping[v]))) in eset
            print(f"backend '{args.backend}': rotatedSurface d={s_dist} "
                  f"{len(ALL_COORDS)}q patch "
                  f"placed (auto, qubits {min(mapping.values())}"
                  f"–{max(mapping.values())})")
        layout = [mapping[c] for c in ALL_COORDS]
    else:
        # heavyhex: 캘리브레이션 인지 자동 배치 (루프 체인 동안 고정 —
        # hardware/placement_<backend>_heavyhex.json 재사용, 임계값 위반
        # 시에만 재탐색; --reselect-layout으로 강제 재선택)
        from circuits.placement import resolve_placement
        from circuits.heavyhex.heavyhex_37q import required_edges
        qc = HeavyHex37QDepthOpt(args.cycles).build_circuit()
        cm = json.load(open(coupling_path))
        try:
            static_map = embedding_for(args.backend)
        except RuntimeError:
            static_map = None
        mapping, placement_info = resolve_placement(
            args.backend, "heavyhex", _ROOT / "hardware",
            ALL_PHYS, required_edges(), qc,
            [tuple(e) for e in cm["coupling_map"]], backend.target,
            static_mapping=static_map,
            reselect=getattr(args, "reselect_layout", False))
        if mapping is None:
            print("WARNING: 유효한 자동 배치 없음 — 정적 임베딩으로 폴백")
            validate_backend(coupling_path)
            mapping = embedding_for(args.backend)
            placement_info = {"fallback": "static",
                              **(placement_info or {})}
        else:
            # 선택된 배치가 required edges를 실제로 덮는지 이중 확인
            eset = {tuple(sorted(e)) for e in cm["coupling_map"]}
            for u, v in required_edges():
                assert tuple(sorted((mapping[u], mapping[v]))) in eset
            print(f"backend '{args.backend}': 37q patch placed "
                  f"(auto, qubits {min(mapping.values())}"
                  f"–{max(mapping.values())})")
        layout = [mapping[p] for p in ALL_PHYS]

    return qc, layout, placement_info


def _submit(args):
    """Build, place, transpile, apply DD and (unless --dry-run) submit.
    Returns the run folder; job ids are in its job.json."""
    from qiskit import transpile, qpy
    from qiskit_ibm_runtime import SamplerV2

    require_implemented(args.code, args.distance)
    if args.code not in ("heavyhex", "surface"):
        sys.exit(f"--code {args.code} does not run on IBM backends.")
    keys = load_keys(("ibm_API", "ibm_instance"))
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) coupling map + backend handle, then 2) the code's hardware
    #    circuit and its device placement
    coupling_path = fetch(args.backend, token=keys["ibm_API"],
                          instance=keys["ibm_instance"], outdir=str(RUNS_DIR))
    service = get_service(keys)
    backend = service.backend(args.backend)
    qc, layout, placement_info = build_and_place(args, backend,
                                                 coupling_path)

    # 3) transpile with the fixed physical layout (patch labels mapped to
    #    this backend's device qubits), then insert DD
    tqc = transpile(qc, backend=backend, initial_layout=layout,
                    optimization_level=1)
    tqc = apply_dd(tqc, backend.target, sequence=args.dd)
    print(f"transpiled+DD({args.dd}): depth={tqc.depth()}, "
          f"pulses={dd_pulse_stats(tqc)}")

    # 4) submit (unless rehearsing) as ONE job carrying --pubs PUBs of
    #    the same ISA circuit. PUB = Primitive Unified Bloc, the
    #    (circuit, parameter values, shots) execution unit of the V2
    #    primitives — packing N PUBs into one job claims a SINGLE queue
    #    entry for N x shots total. Backend limits (max_shots per PUB,
    #    max_experiments per job) are queried and the plan auto-splits;
    #    if the lookup fails we submit as requested with a warning.
    #    The run folder is named <backend>_<timestamp> (job ids live in
    #    job.json — analyze/collect find the folder by scanning job.json,
    #    so legacy job-id-named folders keep working too).
    ts = time.strftime("%Y%m%d-%H%M%S")
    jobs = []
    if args.dry_run:
        run_dir = RUNS_DIR / f"{args.backend}_{ts}_dryrun"
        shots_per_pub, per_job = args.shots, [args.pubs]
        print(f"--dry-run: not submitting (PUB plan: {args.pubs} PUB x "
              f"{args.shots} shots in 1 job; snapshot is still saved).")
    else:
        max_shots = max_exp = None
        try:
            cfg_b = backend.configuration()
            max_shots = getattr(cfg_b, "max_shots", None)
            max_exp = getattr(cfg_b, "max_experiments", None)
        except Exception as e:
            print(f"WARNING: backend.configuration() lookup failed ({e}) "
                  f"— submitting without limit checks")
        shots_per_pub, per_job = _plan_pubs(args.shots, args.pubs,
                                            max_shots, max_exp)
        if shots_per_pub != args.shots or len(per_job) > 1:
            print(f"backend limits (max_shots={max_shots}, "
                  f"max_experiments={max_exp}) -> {shots_per_pub} "
                  f"shots/PUB, PUBs per job: {per_job}")
        sampler = SamplerV2(mode=backend)
        for n in per_job:
            jobs.append(sampler.run([tqc] * n, shots=shots_per_pub))
        run_dir = RUNS_DIR / f"{args.backend}_{ts}"
    while run_dir.exists():                  # same-second collision guard
        run_dir = run_dir.with_name(run_dir.name + "b")
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.move(coupling_path, run_dir / "coupling.json")
    snapshot_backend(backend, run_dir)
    with open(run_dir / "circuit.qpy", "wb") as f:
        qpy.dump(tqc, f)
    print("   saved circuit.qpy (exact ISA circuit incl. DD delays)")

    job_ids = [j.job_id() for j in jobs]
    meta = {"job_id": job_ids[0] if job_ids else None,
            "job_ids": job_ids,
            "n_pubs": sum(per_job), "shots_per_pub": shots_per_pub,
            "backend": args.backend, "code": args.code,
            "distance": normalize_distance(args.code, args.distance),
            "cycles": args.cycles,
            "shots": args.shots, "pubs": args.pubs, "dd": args.dd,
            "initial_layout": layout,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dry_run": args.dry_run,
            "transpiled_depth": tqc.depth(),
            "transpiled_ops": {k: int(v) for k, v in tqc.count_ops().items()},
            "placement": placement_info,
            "versions": _local_versions()}
    json_dump(meta, run_dir / "job.json")
    print(f"run folder: {run_dir}")
    return run_dir


def wait_for_job(job, poll=30):
    """Poll until the job reaches a final state, printing state changes.
    Returns on DONE; exits on ERROR/CANCELLED."""
    last = None
    while True:
        status = job.status()
        name = getattr(status, "name", status)  # enum (old) or str (new)
        if name != last:
            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"job {job.job_id()}: {name}")
            last = name
        if name == "DONE":
            return
        if name in ("ERROR", "CANCELLED"):
            sys.exit(f"job {job.job_id()} ended as {name} — "
                     f"nothing to analyze.")
        time.sleep(poll)


def _plan_pubs(shots, pubs, max_shots, max_experiments):
    """Backend-limit-aware PUB plan -> (shots per PUB, PUBs per job).

    Keeps the total ~ shots*pubs: a PUB whose shots exceed max_shots is
    subdivided; when the PUB count exceeds max_experiments the job is
    split (list entry = one job's PUB count)."""
    import math
    shots_per_pub, n_pubs = shots, pubs
    if max_shots and shots_per_pub > max_shots:
        k = math.ceil(shots_per_pub / max_shots)
        shots_per_pub = math.ceil(shots / k)
        n_pubs = pubs * k
    per_job, remaining = [], n_pubs
    cap = max_experiments if max_experiments else n_pubs
    while remaining > 0:
        n = min(remaining, cap)
        per_job.append(n)
        remaining -= n
    return shots_per_pub, per_job


def _collect(run_dir, job_ids, cycles, poll):
    """Wait for the run's jobs, write raw.npz (+ job metrics and the
    calibration as of execution time). Returns the raw.npz path."""
    service = get_service(load_keys(("ibm_API", "ibm_instance")))
    meta = {}
    meta_path = run_dir / "job.json"
    if meta_path.exists():
        meta = json.load(open(meta_path))
        cycles = meta.get("cycles", cycles)
        job_ids = (meta.get("job_ids") or [meta.get("job_id")]
                   if (meta.get("job_ids") or meta.get("job_id"))
                   else job_ids)

    syns, dats, pub_shots, metrics_all = [], [], [], []
    first_running = None
    for jid in job_ids:
        job = service.job(jid)
        wait_for_job(job, poll=poll)
        # order='little' -> column i is clbit i, i.e. syn bit cyc*16+j and
        # dat bit i in DATA_PHYS order — exactly the check_values() layout
        for res in job.result():
            s = res.data["syn"].to_bool_array(order="little").astype(np.uint8)
            t = res.data["data"].to_bool_array(order="little").astype(np.uint8)
            syns.append(s)
            dats.append(t)
            pub_shots.append(s.shape[0])
        try:
            mt = job.metrics()
            metrics_all.append(mt)
            r = (mt.get("timestamps") or {}).get("running")
            if r and (first_running is None or r < first_running):
                first_running = r
        except Exception as e:
            print(f"WARNING: could not read metrics of {jid}: {e}")
    syn = np.concatenate(syns)
    dat = np.concatenate(dats)
    pub_arr = np.asarray(pub_shots, dtype=np.int64)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_raw(run_dir, syn, dat, cycles, pub_arr)
    print(f"saved raw results -> {run_dir / 'raw.npz'} "
          f"({len(pub_shots)} PUB x ~{pub_shots[0] if pub_shots else 0} shots)")
    try:
        json_dump({"metrics": metrics_all}, run_dir / "job_metrics.json")
        print(f"saved job_metrics.json")
    except Exception as e:
        print(f"WARNING: could not save job metrics: {e}")

    # execution-time calibration snapshot (submission-time snapshots can
    # be hours older than the actual run when the queue is long)
    try:
        import datetime as _dt
        if first_running:
            run_ts = first_running
            run_dt = _dt.datetime.fromisoformat(run_ts.replace("Z", "+00:00"))
        else:
            run_ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            run_dt = None
            print("WARNING: no 'running' timestamp in job metrics — "
                  "using collection time; properties_run falls back to "
                  "current calibration")
        if meta:
            meta["run_started_at"] = run_ts
            json_dump(meta, meta_path)
        if meta.get("backend"):
            bk = service.backend(meta["backend"])
            props = (bk.properties(datetime=run_dt) if run_dt
                     else bk.properties())
            if props is not None:
                json_dump(props.to_dict(), run_dir / "properties_run.json")
                print("saved properties_run.json (calibration at run time)")
    except Exception as e:
        print(f"WARNING: run-time calibration snapshot failed: {e}")
    return run_dir / "raw.npz"


class IBMProvider(HardwareProvider):
    name = "ibm"

    def submit(self, code, distance, cycles, shots, pubs, dry_run=False,
               dd="XX4", reselect_layout=False):
        return _submit(SimpleNamespace(
            backend=self.backend, code=code, distance=distance,
            cycles=cycles, shots=shots, pubs=pubs, dry_run=dry_run, dd=dd,
            reselect_layout=reselect_layout))

    def status(self, job_ids):
        service = get_service(load_keys(("ibm_API", "ibm_instance")))
        names = []
        for jid in job_ids:
            st = service.job(jid).status()
            names.append(getattr(st, "name", st))
        return names

    def wait(self, job_ids, poll=30):
        service = get_service(load_keys(("ibm_API", "ibm_instance")))
        for jid in job_ids:
            wait_for_job(service.job(jid), poll=poll)

    def collect(self, run_dir, job_ids=None, cycles=3, poll=30):
        return _collect(run_dir, job_ids, cycles, poll)

    def snapshot(self, run_dir):
        service = get_service(load_keys(("ibm_API", "ibm_instance")))
        snapshot_backend(service.backend(self.backend), run_dir)
