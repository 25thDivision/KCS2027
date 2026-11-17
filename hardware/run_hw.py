#!/usr/bin/env python3
"""
Hardware validation on IBM Quantum backends
===========================================
Pipeline (submit):
  keys.json -> fetch_coupling.fetch() -> validate_backend()
    -> HeavyHex37QDepthOpt circuit -> transpile(initial_layout =
    ALL_PHYS mapped through embedding_for(backend), optimization_level=1)
    -> dd_utils.apply_dd(tqc, backend.target)
    -> QPU environment snapshot -> SamplerV2 submit
Submission model: ONE job per (backend, code) carrying --pubs PUBs of
the same ISA circuit (PUB = Primitive Unified Bloc, the V2 primitives'
(circuit, parameter values, shots) execution unit) — a single queue
entry buys shots x pubs. Backend limits (max_shots / max_experiments)
are queried and the plan auto-splits into extra PUBs/jobs (job.json then
carries job_ids). Every submission is registered in
hardware/pending_jobs.json; `collect` (run at the start of each
pipeline loop) analyzes whatever finished and carries the rest over, so
the pipeline never blocks on the queue.

Pipeline (analyze):
  raw results (all PUBs concatenated, pub_shots recorded)
    -> check_values() XOR-chain syndrome recovery
    -> syndrome tensor -> CNN/GNN + MWPM decoding -> LER report with a
    ler_std_over_pubs column (per-PUB LER spread)
    -> results/hardware/<backend>_<code>_<timestamp>.csv
  First collection also stores run_started_at (job metrics) in job.json
  and the calibration AS OF execution time in properties_run.json —
  make_qpu_avg_profile prefers this over the submission-time snapshot.
Pipeline (all): submit -> wait_for_job(s) -> analyze in one go
  (manual use; the sbatch pipeline uses submit + collect).

Every submission gets its own folder so the run can be re-analyzed (and
the QPU environment of that moment stays on record):

  hardware/runs/<backend>_<YYYYMMDD-HHMMSS>/   (dry-run: ..._dryrun/;
    the job id lives in job.json — analyze --job-id finds the folder
    by scanning job.json, so legacy job-id-named folders still work)
    job.json             backend, cycles, shots, dd, timestamps,
                         local package versions, circuit stats
    coupling.json        coupling map / basis gates (fetch_coupling output)
    properties.json      backend.properties() calibration snapshot
                         (T1/T2, gate & readout errors, calibration times)
    configuration.json   backend.configuration() (dt, processor type, ...)
    target.pkl           backend.target pickle (what transpile/DD consumed)
    circuit.qpy          the exact transpiled+DD ISA circuit submitted
    raw.npz              measured raw bits (written by analyze)
    job_metrics.json     job timestamps/usage (written by analyze)

Notes:
  * Default backend is ibm_yonsei; switch with --backend ibm_boston.
  * Credentials come ONLY from keys.json at the repo root (copy
    keys.example.json; keys.json is gitignored). Never hardcode tokens/CRNs.
  * The official metric is the head-LER (logical head). Hardware provides
    no per-qubit ground truth, so `ECR (diagnostic, sim-only)` cannot be
    computed here; `parity_LER (diagnostic)` can (it only needs the
    logical ground truth) and is reported next to the LER, as is the
    `LER/MWPM ratio` (see evaluation/metrics.py).
  * properties.json records the last calibration before the run — it is
    the best available record, not a live snapshot of drift. You can feed
    it to qiskit-aer's NoiseModel.from_backend() to re-simulate the run
    under that day's calibration.

Usage:
  python hardware/run_hw.py         [submit options] [analyze options]
                                    [--poll 30]   # default = all
  python hardware/run_hw.py submit  [--backend ibm_yonsei] [--shots 50000]
                                    [--cycles 3] [--dd XX4] [--dry-run]
  python hardware/run_hw.py analyze --job-id <ID> [--ckpt checkpoint/CNN_....pt]
  python hardware/run_hw.py analyze --npz hardware/runs/<ID>/raw.npz [--ckpt ...]
  (no subcommand runs all: submit -> wait -> analyze in one go;
   analyze without --ckpt evaluates every checkpoint/*.pt)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from circuits.heavyhex.heavyhex_depth7_opt_for_37q import (  # noqa: E402
    check_values, N_CHECKS)
from dataset_generation import (  # noqa: E402
    CODES, distance_tag, normalize_distance, require_implemented)
from hardware.providers import get_provider  # noqa: E402
from hardware.providers.base import (  # noqa: E402
    RUNS_DIR, json_dump as _json_dump, load_raw)
from model import MODEL_NAMES  # noqa: E402

from dataset_generation.heavyhex33_stim import (  # noqa: E402
    check_matrix_from_dict, syndrome_tensor, logical_label, ALL_NOISE)

RESULTS_HW_DIR = _ROOT / "results" / "hardware"


def cmd_submit(args):
    """Submit through the backend's provider; register real jobs in
    pending_jobs.json. Returns (provider, run folder, job ids)."""
    require_implemented(args.code, args.distance)
    provider = get_provider(args.backend)
    options = {}
    if provider.name in ("ibm", "aer"):
        options = dict(reselect_layout=getattr(args, "reselect_layout",
                                               False))
    if provider.name == "ibm":
        options["dd"] = args.dd
    run_dir = provider.submit(args.code, args.distance, args.cycles,
                              args.shots, args.pubs, dry_run=args.dry_run,
                              **options)
    meta = json.load(open(run_dir / "job.json"))
    job_ids = meta.get("job_ids") or []
    if job_ids:
        pend = _load_pending()
        pend.append({"job_id": job_ids[0], "job_ids": job_ids,
                     "backend": args.backend, "code": args.code,
                     "distance": meta["distance"],
                     "run_dir": run_dir.name,
                     "submitted_at": meta["submitted_at"]})
        _save_pending(pend)
        print(f"submitted: {len(job_ids)} job(s) {job_ids} — "
              f"{meta.get('n_pubs')} PUB x {meta.get('shots_per_pub')} "
              f"shots, registered in pending_jobs.json (다음 루프의 수거 "
              f"단계가 자동 분석)")
        print(f"manual: python hardware/run_hw.py analyze "
              f"--job-id {job_ids[0]}")
    return provider, run_dir, job_ids


def find_run_dir(job_id):
    """Locate the run folder of a job id.

    Run folders are named <backend>_<timestamp> (job.json holds the job
    id), so look there first by scanning job.json; a folder literally
    named after the job id (legacy layout) also matches. Returns None
    when no local folder exists (e.g. analyzing on another machine)."""
    direct = RUNS_DIR / job_id
    if direct.exists():
        return direct
    if RUNS_DIR.exists():
        for p in sorted(RUNS_DIR.iterdir()):
            meta_path = p / "job.json"
            if not (p.is_dir() and meta_path.exists()):
                continue
            try:
                if json.load(open(meta_path)).get("job_id") == job_id:
                    return p
            except Exception:
                continue
    return None


PENDING_PATH = _ROOT / "hardware" / "pending_jobs.json"


def _load_pending():
    if not PENDING_PATH.exists():
        return []
    try:
        return json.load(open(PENDING_PATH))
    except Exception as e:
        print(f"WARNING: pending_jobs.json unreadable ({e}) — "
              f"treating as empty")
        return []


def _save_pending(entries):
    PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    _json_dump(entries, PENDING_PATH)


def _remove_pending(job_id):
    entries = _load_pending()
    kept = [e for e in entries
            if e.get("job_id") != job_id
            and job_id not in (e.get("job_ids") or [])]
    if len(kept) != len(entries):
        _save_pending(kept)


def _latest_qpu_profile(backend, code, distance=3):
    """Newest registered qpu/<backend>_<code>_* profile name of the same
    distance (or None) — the collect path uses it for the MWPM DEM
    weights. Profiles without a "distance" field are distance 3."""
    try:
        from dataset_generation import load_config
        profs = load_config().get("noise_profiles", {})
    except Exception:
        return None
    cands = [(v.get("provenance", {}).get("generated_at", ""), k)
             for k, v in profs.items()
             if k.startswith(f"qpu/{backend}_{code}_")
             and (normalize_distance(code, v.get("distance", 3))
                  == normalize_distance(code, distance))]
    return max(cands)[1] if cands else None


def fetch_raw(args):
    """Return (syn, dat, cycles, pub_shots): raw bits in clbit order.

    --npz or a cached raw.npz of the run folder is read directly;
    otherwise the run's provider collects the results (waiting for the
    jobs if needed) and writes raw.npz first. Legacy caches / loose npz
    without pub_shots count as a single PUB (pub_shots=None)."""
    if args.npz:
        return load_raw(args.npz)

    # legacy fallback name if the job has no local run folder yet
    run_dir = find_run_dir(args.job_id) or (RUNS_DIR / args.job_id)
    cached = run_dir / "raw.npz"
    if cached.exists():
        return load_raw(cached)

    backend = None
    meta_path = run_dir / "job.json"
    if meta_path.exists():
        backend = json.load(open(meta_path)).get("backend")
    # runs without a job.json predate the provider split: IBM
    provider = get_provider(backend or "ibm_legacy")
    raw_path = provider.collect(run_dir, job_ids=[args.job_id],
                                cycles=args.cycles,
                                poll=getattr(args, "poll", 30))
    return load_raw(raw_path)


def cmd_analyze(args):
    # run metadata: job.json wins (submit recorded it), else the --code /
    # default heavyhex (offline --npz re-analysis). backend/submitted_at
    # feed the report name and columns.
    code = getattr(args, "code", None) or "heavyhex"
    distance = getattr(args, "distance", None) or 3
    hw_backend, submitted_at, job_id = None, None, args.job_id
    run_dir = None
    if args.job_id:
        run_dir = find_run_dir(args.job_id)
    elif args.npz:
        run_dir = Path(args.npz).resolve().parent
    meta_path = (run_dir / "job.json") if run_dir else None
    if meta_path and meta_path.exists():
        meta = json.load(open(meta_path))
        code = meta.get("code", code)
        distance = meta.get("distance", distance)
        hw_backend = meta.get("backend")
        submitted_at = meta.get("submitted_at")
        job_id = meta.get("job_id", job_id)
    require_implemented(code, distance)
    dtag = distance_tag(code, distance)

    if code == "surface":
        from functools import partial
        from circuits.rotatedSurface import rotatedSurface as rs
        from dataset_generation import rotatedSurface_stim as rss
        s_dist = normalize_distance("surface", distance)
        n_checks = rs.surface_def(s_dist).N_CHECKS
        n_data = rs.surface_def(s_dist).NUM_DATA
        cv_fn = partial(rs.check_values, distance=s_dist)
        mat_fn = partial(rss.check_matrix_from_dict_rotatedSurface,
                         distance=s_dist)
        tensor_fn = partial(rss.syndrome_tensor_rotatedSurface,
                            distance=s_dist)
        logical_fn = partial(rss.logical_label_rotatedSurface,
                             distance=s_dist)
    elif code == "color":
        from functools import partial
        from circuits.color import colorcode as cc
        from dataset_generation import color_stim as cs
        c_dist = normalize_distance("color", distance)
        n_checks = cc.color_def(c_dist).N_CHECKS
        n_data = cc.color_def(c_dist).NUM_DATA
        cv_fn = partial(cc.check_values, distance=c_dist)
        mat_fn = partial(cs.check_matrix_from_dict_color, distance=c_dist)
        tensor_fn = partial(cs.syndrome_tensor_color, distance=c_dist)
        logical_fn = partial(cs.logical_label_color, distance=c_dist)
    else:
        cv_fn, n_checks, n_data = check_values, N_CHECKS, 17
        mat_fn, tensor_fn, logical_fn = (check_matrix_from_dict,
                                         syndrome_tensor, logical_label)

    syn, dat, cycles, pub_shots = fetch_raw(args)
    shots = syn.shape[0]
    assert syn.shape[1] == n_checks * cycles and dat.shape[1] == n_data, \
        f"raw shapes {syn.shape}/{dat.shape} do not match code '{code}'"
    n_pubs = len(pub_shots) if pub_shots is not None else 1
    print(f"shots={shots} ({n_pubs} PUB), cycles={cycles}, code={code}")

    # no-reset raw -> check values via per-ancilla XOR chains
    check_mat = mat_fn(cv_fn(syn, cycles), cycles)
    y_logical = logical_fn(dat)
    raw_ler = float(y_logical.mean())

    # PUB boundaries for the per-PUB LER spread (legacy runs without
    # pub_shots count as one PUB -> std N/A)
    bounds = None
    if pub_shots is not None and len(pub_shots) > 1:
        edges = np.concatenate([[0], np.cumsum(pub_shots)])
        bounds = [(int(edges[i]), int(edges[i + 1]))
                  for i in range(len(pub_shots))]

    def std_over_pubs(pred=None, values=None):
        """Sample std of the per-PUB LERs (pred vs y_logical, or raw
        values)."""
        if bounds is None:
            return None
        lers = []
        for a, b in bounds:
            if values is not None:
                lers.append(float(values[a:b].mean()))
            else:
                lers.append(float((pred[a:b] != y_logical[a:b]).mean()))
        return float(np.std(lers, ddof=1))

    # row = (decoder, LER, ler_std_over_pubs or None,
    #        parity_LER (diagnostic) or None, LER/MWPM ratio or None,
    #        best_epoch or None, total_epochs or None)
    rows = [("raw (no decoding)", raw_ler, std_over_pubs(values=y_logical),
             None, None, None, None)]

    # MWPM baseline (DEM weights from the reference noise profile)
    from baseline.mwpm import build_matching, mwpm_ler_from_hardware
    matching = build_matching(cycles, "X", args.mwpm_p, args.mwpm_profile,
                              code, distance)
    mwpm_ler, mwpm_pred = mwpm_ler_from_hardware(check_mat, dat, cycles,
                                                 matching, code,
                                                 return_pred=True,
                                                 distance=distance)
    rows.append(("MWPM", mwpm_ler, std_over_pubs(pred=mwpm_pred),
                 None, None, None, None))

    # NVIDIA pre-decoder + MWPM on the residual (surface only). The
    # checkpoint path is config.json's nvidia.checkpoint_accurate,
    # relative to the repo root.
    if code == "surface":
        from dataset_generation import load_config
        nv_path = load_config().get("nvidia", {}).get("checkpoint_accurate")
        nv_file = (_ROOT / nv_path) if nv_path else None
        if nv_file is None or not nv_file.exists():
            print(f"(NVIDIA pre-decoder checkpoint not found: "
                  f"{nv_file or 'nvidia.checkpoint_accurate unset'}: "
                  f"skipping)")
        else:
            import torch
            from evaluation.metrics import ler
            from decoders.nvidia_predecoder import (
                load_predecoder, nvidia_mwpm_decode)
            nv_model = load_predecoder(
                nv_file, "cuda" if torch.cuda.is_available() else "cpu")
            nv_pred = nvidia_mwpm_decode(nv_model, check_mat, dat, cycles,
                                         matching, distance)
            nv_ler = ler(nv_pred, y_logical)
            rows.append(("NVIDIA(Accurate)", nv_ler,
                         std_over_pubs(pred=nv_pred), None,
                         nv_ler / mwpm_ler if mwpm_ler else None,
                         None, None))

    # model head: a single --ckpt, or every matching checkpoint if none
    # was given. The architecture is INFERRED from the {MODEL}_ filename
    # prefix, so one analyze evaluates every architecture in the same
    # report; --model narrows it to one. Checkpoints are matched on the
    # "_<code>_<dtag>_" part of their name; names without any code tag
    # are legacy heavyhex (3,3) checkpoints.
    from model import MODEL_REGISTRY, model_from_prefix, model_prefix

    def model_of(path):
        return model_from_prefix(path.name.split("_", 1)[0])

    if args.ckpt:
        p = Path(args.ckpt)
        ckpts = [(model_of(p) or args.model or "cnn", p)]
    else:
        ckpts = []
        for p in sorted((_ROOT / "checkpoint").glob("*.pt")):
            if p.name.endswith(".resume.pt"):
                continue                    # training state, not a model
            mname = model_of(p)
            if mname is None:
                print(f"(skipping {p.name}: unknown model prefix)")
                continue
            if any(f"_{c}_" in p.name for c in CODES):
                if f"_{code}_{dtag}_" not in p.name:
                    continue
            elif (code, dtag) != ("heavyhex", "d3"):
                continue
            if args.model and mname != args.model:
                continue
            ckpts.append((mname, p))
        if not ckpts:
            want = args.model or "/".join(sorted(MODEL_REGISTRY))
            print(f"(no --ckpt and no matching checkpoint/*.pt for "
                  f"code '{code}' {dtag} / model {want}: skipping)")
    if ckpts:
        import torch
        from model import get_model_module, get_model_class, code_spec
        from evaluation.metrics import ler, parity_ler_from_qubit_logits
        from decoders.hybrid import FinalSyndromeMatcher, hybrid_predictions
        hybrid_matcher = None
        base_tensor = tensor_fn(check_mat, cycles)
        prepared = {}       # model name -> model-ready input tensor
        for mname, ckpt_path in ckpts:
            mod = get_model_module(mname)
            if mname not in prepared:
                # model-specific input prep (the final-Z detector
                # channel, computed from the measured final data bits —
                # the hardware counterpart of the simulation labels)
                t = base_tensor
                if hasattr(mod, "prepare_features"):
                    t = np.asarray(mod.prepare_features(base_tensor, dat,
                                                        code, distance))
                prepared[mname] = t
            tensor = prepared[mname]
            model_cls = get_model_class(mname)
            ckpt = torch.load(ckpt_path, map_location="cpu",
                              weights_only=False)
            model = model_cls(in_channels=2 * cycles,
                              num_qubits=code_spec(
                                  code, distance)["num_qubits"],
                              code=code, distance=distance)
            model.load_state_dict(ckpt["model_state_dict"])
            # GPU when there is one (GraphMamba's kernels are CUDA-only)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(dev).eval()
            preds, q_logits = [], []
            with torch.no_grad():
                for i in range(0, shots, 8192):
                    xb = torch.from_numpy(tensor[i:i + 8192]).to(dev)
                    ql, ll = model(xb)
                    preds.append((ll.float().cpu().numpy().ravel() > 0
                                  ).astype(np.uint8))
                    q_logits.append(ql.float().cpu().numpy())
            pred = np.concatenate(preds)
            model_ler = ler(pred, y_logical)
            parity_ler = parity_ler_from_qubit_logits(
                np.concatenate(q_logits), y_logical, code, distance)
            ratio = model_ler / mwpm_ler if mwpm_ler else None
            # "weight from epoch <best> out of <total> trained"
            best_ep = ckpt.get("best_epoch", ckpt.get("epoch"))
            total_ep = ckpt.get("total_epochs")
            rows.append((f"{model_prefix(mname)} ({ckpt_path.name})",
                         model_ler,
                         std_over_pubs(pred=pred), parity_ler, ratio,
                         best_ep, total_ep))
            # hybrids of this checkpoint with the final-syndrome MWPM
            # (decoders/hybrid.py)
            if hybrid_matcher is None:
                hybrid_matcher = FinalSyndromeMatcher(code, distance)
            hyb = hybrid_predictions(model, mod, base_tensor, dat, code,
                                     distance, matcher=hybrid_matcher,
                                     q_logits=np.concatenate(q_logits))
            for label, key in ((f"MWPM+{model_prefix(mname)}", "mwpm_ml"),
                               (f"{model_prefix(mname)}+MWPM", "ml_mwpm")):
                h_ler = ler(hyb[key], y_logical)
                rows.append((f"{label} ({ckpt_path.name})", h_ler,
                             std_over_pubs(pred=hyb[key]), None,
                             h_ler / mwpm_ler if mwpm_ler else None,
                             best_ep, total_ep))

    def _fmt(v, spec=".4f"):
        return format(v, spec) if v is not None else "N/A"

    print(f"\n{'decoder':<55} {'LER':>8} {'ler_std':>8} "
          f"{'parity_LER (diagnostic)':>24} {'LER/MWPM ratio':>15} "
          f"{'best_ep':>8} {'total_ep':>9}")
    for name, v, sd, pl, ratio, be, te in rows:
        print(f"{name:<55} {v:>8.4f} {_fmt(sd):>8} {_fmt(pl):>24} "
              f"{_fmt(ratio):>15} "
              f"{_fmt(be, 'd') if be is not None else 'N/A':>8} "
              f"{_fmt(te, 'd') if te is not None else 'N/A':>9}")

    # persist the report next to the training results
    # report file: <backend>_<code>_<timestamp>.csv, timestamp = the
    # run's submitted_at (falls back to the analysis time when the run
    # metadata is unavailable, e.g. a loose --npz file)
    job_id = job_id or (Path(args.npz).resolve().parent.name if args.npz
                        else None)
    code_label = {"surface": "rotatedSurface", "color": "color"}.get(
        code, "heavyhex")
    ts = (submitted_at.replace("-", "").replace(":", "").replace("T", "-")
          if submitted_at else time.strftime("%Y%m%d-%H%M%S"))
    RESULTS_HW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_HW_DIR / f"{hw_backend or 'unknown'}_{code_label}_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["decoder", "ler", "ler_std_over_pubs",
                    "parity_LER (diagnostic)", "LER/MWPM ratio", "shots",
                    "cycles", "best_epoch", "total_epochs", "backend",
                    "timestamp", "job_id"])
        for name, v, sd, pl, ratio, be, te in rows:
            w.writerow([name, f"{v:.6f}", _fmt(sd, ".6f"),
                        _fmt(pl, ".6f"), _fmt(ratio, ".6f"), shots, cycles,
                        be if be is not None else "",
                        te if te is not None else "",
                        hw_backend or "unknown", submitted_at or ts,
                        job_id or ""])
    print(f"saved -> {csv_path}")
    if args.job_id:
        _remove_pending(args.job_id)


def cmd_collect(args):
    """Check every entry of pending_jobs.json; analyze what finished.

    Called at the start of each pipeline loop: DONE -> analyze and drop,
    QUEUED/RUNNING -> carry over, ERROR/CANCELLED -> log the reason
    (hardware/collect_failed.log) and drop. This path never raises /
    sys.exits, so the sbatch loop chain (set -e) survives any single
    job's failure."""
    pending = _load_pending()
    if not pending:
        print("pending 없음")
        return
    keep = []
    for ent in pending:
        jids = ent.get("job_ids") or [ent.get("job_id")]
        label = f"{ent.get('backend')}/{ent.get('code')} {jids}"
        try:
            names = get_provider(ent.get("backend")).status(jids)
        except (Exception, SystemExit) as e:
            print(f"  {label}: 상태 조회 실패({e}) — 이월")
            keep.append(ent)
            continue
        if any(n in ("ERROR", "CANCELLED") for n in names):
            print(f"  {label}: {names} — 실패로 기록하고 제거")
            try:
                with open(_ROOT / "hardware" / "collect_failed.log",
                          "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} "
                            f"{label} {names}\n")
            except Exception:
                pass
            continue
        if all(n == "DONE" for n in names):
            print(f"  {label}: DONE — 분석 시작")
            mwpm_prof = (_latest_qpu_profile(ent.get("backend"),
                                             ent.get("code"),
                                             ent.get("distance", 3))
                         or args.mwpm_profile)
            ns = argparse.Namespace(
                job_id=jids[0], npz=None, cycles=3,
                code=ent.get("code"), distance=ent.get("distance", 3),
                model=args.model, ckpt=args.ckpt, mwpm_profile=mwpm_prof,
                mwpm_p=args.mwpm_p, poll=30)
            try:
                cmd_analyze(ns)
            except (Exception, SystemExit) as e:
                print(f"  {label}: 분석 실패({e}) — 이월")
                keep.append(ent)
            continue
        print(f"  {label}: {names} — 이월")
        keep.append(ent)
    _save_pending(keep)
    print(f"수거 완료 — pending 잔여 {len(keep)}건")


def cmd_all(args):
    provider, run_dir, job_ids = cmd_submit(args)
    if args.dry_run:
        print("--dry-run: skipping wait/analyze.")
        return
    if job_ids:
        args.job_id, args.npz = job_ids[0], None
        provider.wait(job_ids, poll=args.poll)
    else:              # a local provider wrote raw.npz during submit
        args.job_id, args.npz = None, str(run_dir / "raw.npz")
    cmd_analyze(args)


def _submit_opts(p):
    p.add_argument("--backend", default="ibm_yonsei",
                   help="backend name; its prefix picks the provider "
                        "(ibm_* IBM QPU, ionq_* IonQ, aer_<ibm backend> "
                        "Aer simulation with that backend's calibration)")
    p.add_argument("--code", choices=list(CODES), default="heavyhex",
                   help="code family")
    p.add_argument("--distance", type=int, default=3,
                   help="code distance (heavyhex: 3 or 33 = (3,3), 35, 53; "
                        "surface / color: 3, 5)")
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--shots", type=int, default=50_000,
                   help="shots per PUB")
    p.add_argument("--pubs", type=int, default=1,
                   help="number of PUBs (same ISA circuit repeated) in "
                        "the single submitted job — total shots = "
                        "shots x pubs with ONE queue entry")
    p.add_argument("--dd", default="XX4",
                   choices=["XX2", "XX4", "XY4", "XY8"],
                   help="DD sequence (XX4 default; Heron has no native Y)")
    p.add_argument("--dry-run", action="store_true",
                   help="do everything except the actual submission")
    p.add_argument("--reselect-layout", action="store_true",
                   help="ignore the pinned placement file and search a "
                        "fresh calibration-optimal placement (heavyhex "
                        "auto-placement only; default: reuse the pinned "
                        "placement unless it violates a threshold)")


def _analyze_opts(p, cycles=True):
    if cycles:   # standalone analyze parser ('all' inherits submit's)
        p.add_argument("--cycles", type=int, default=3,
                       help="fallback if job metadata is missing")
        p.add_argument("--code", choices=list(CODES), default=None,
                       help="fallback if job metadata is missing "
                            "(e.g. offline --npz re-analysis)")
        p.add_argument("--distance", type=int, default=None,
                       help="fallback if job metadata is missing")
    p.add_argument("--model", choices=MODEL_NAMES, default=None,
                   help="restrict the evaluation to one architecture; "
                        "default: evaluate every matching checkpoint, "
                        "inferring the model from the {MODEL}_ filename "
                        "prefix")
    p.add_argument("--ckpt", default=None,
                   help="trained model checkpoint (.pt); omit to evaluate "
                        "every checkpoint/{MODEL}_*.pt")
    p.add_argument("--mwpm-profile", default=ALL_NOISE[0],
                   help="noise profile used for the MWPM DEM weights")
    p.add_argument("--mwpm-p", type=float, default=0.005)


def main():
    ap = argparse.ArgumentParser(description="Hardware validation pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="build, transpile, apply DD and submit")
    _submit_opts(s)
    s.set_defaults(func=cmd_submit)

    a = sub.add_parser("analyze", help="fetch results, decode, report LER")
    a.add_argument("--job-id", default=None)
    a.add_argument("--npz", default=None,
                   help="previously saved raw npz (offline re-analysis)")
    _analyze_opts(a)
    a.set_defaults(func=cmd_analyze)

    al = sub.add_parser("all", help="submit, wait for the job, then analyze"
                                    " (manual one-shot; the pipeline uses "
                                    "submit + collect instead)")
    _submit_opts(al)
    _analyze_opts(al, cycles=False)   # submit already owns --cycles
    al.add_argument("--poll", type=int, default=30,
                    help="job status poll interval in seconds")
    al.set_defaults(func=cmd_all)

    co = sub.add_parser("collect",
                        help="analyze finished pending jobs "
                             "(pending_jobs.json), carry the rest over")
    _analyze_opts(co)
    co.set_defaults(func=cmd_collect)

    # 서브커맨드 생략 시 all(제출->대기->분석 원샷)로 동작.
    argv = sys.argv[1:]
    if argv[:1] not in (["submit"], ["analyze"], ["all"], ["collect"],
                        ["-h"], ["--help"]):
        argv.insert(0, "all")
    args = ap.parse_args(argv)
    if args.cmd == "analyze" and not (args.job_id or args.npz):
        ap.error("analyze requires --job-id or --npz")
    args.func(args)


if __name__ == "__main__":
    main()
