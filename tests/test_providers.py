"""Hardware providers (hardware/providers) and the provider-independent
run folder contract: job.json + raw.npz -> analyze."""
import argparse
import csv
import json
import time

import numpy as np
import pytest

from circuits.color.colorcode import color_def
from dataset_generation import color_stim as cs
from hardware import run_hw
from hardware.providers import get_provider, provider_name
from hardware.providers.base import (
    HardwareProvider, json_dump, load_raw, save_raw)
from hardware.providers.ionq import (
    BACKENDS as IONQ_BACKENDS, counts_to_raw, plan_jobs)

NOISE = "realistic/dp0.001_mf0.01_rf0.01_gd0.008"


def test_provider_is_chosen_by_backend_prefix():
    assert provider_name("ibm_pittsburgh") == "ibm"
    assert provider_name("ionq_forte1") == "ionq"
    assert provider_name("aer_ibm_pittsburgh") == "aer"
    with pytest.raises(ValueError):
        provider_name("rigetti_x")
    assert get_provider("ibm_pittsburgh").name == "ibm"
    assert get_provider("ionq_forte1").name == "ionq"
    assert get_provider("aer_ibm_phoenix").ibm_backend == "ibm_phoenix"
    with pytest.raises(ValueError):
        get_provider("ionq_unknown")
    with pytest.raises(ValueError):
        get_provider("aer_forte1")
    assert set(IONQ_BACKENDS) == {"ionq_forte1", "ionq_sim_forte1"}


def test_raw_npz_roundtrip(tmp_path):
    syn = np.random.default_rng(0).integers(0, 2, (10, 18))
    dat = np.random.default_rng(1).integers(0, 2, (10, 7))
    path = save_raw(tmp_path, syn, dat, 3, [6, 4])
    d = np.load(path)
    assert sorted(d.files) == ["cycles", "dat", "pub_shots", "syn"]
    assert d["syn"].dtype == np.uint8 and d["dat"].dtype == np.uint8
    s, t, cycles, pubs = load_raw(path)
    assert np.array_equal(s, syn) and np.array_equal(t, dat)
    assert cycles == 3 and pubs.tolist() == [6, 4]
    # a legacy file without pub_shots counts as one PUB
    np.savez(tmp_path / "old.npz", syn=syn, dat=dat, cycles=3)
    assert load_raw(tmp_path / "old.npz")[3] is None


def test_ibm_pub_plan():
    from hardware.providers.ibm import _plan_pubs
    assert _plan_pubs(50000, 5, None, None) == (50000, [5])
    assert _plan_pubs(50000, 5, 100000, 300) == (50000, [5])
    # a PUB above max_shots is split, the total stays the same
    spp, per_job = _plan_pubs(50000, 2, 20000, 300)
    assert spp <= 20000 and spp * sum(per_job) >= 100000
    # more PUBs than max_experiments -> several jobs
    assert _plan_pubs(1000, 7, None, 3) == (1000, [3, 3, 1])


def test_ionq_job_plan_and_counts():
    assert plan_jobs(5000, 2, 10000) == (5000, 2)
    assert plan_jobs(20000, 2, 10000) == (10000, 4)
    assert plan_jobs(25000, 1, 10000) == (8334, 3)
    assert plan_jobs(20000, 2, None) == (20000, 2)
    # qiskit bitstring: registers right to left, clbit 0 rightmost
    # syn = [1, 0, 0] (clbits 0..2), data = [0, 1] (clbits 3, 4)
    counts = {"10 001": 2, "00 110": 1}
    syn, dat = counts_to_raw(counts, 3, 2)
    assert syn.tolist() == [[1, 0, 0], [1, 0, 0], [0, 1, 1]]
    assert dat.tolist() == [[0, 1], [0, 1], [0, 0]]
    # the same without the register separator
    syn2, dat2 = counts_to_raw({"10001": 2, "00110": 1}, 3, 2)
    assert np.array_equal(syn, syn2) and np.array_equal(dat, dat2)
    assert counts_to_raw({}, 3, 2)[0].shape == (0, 3)


def test_ionq_counts_match_the_circuit_registers():
    """counts_to_raw against a real Aer run of the color circuit with an
    injected error: the rebuilt check values fire on the right faces."""
    from qiskit_aer import AerSimulator
    from circuits.color.colorcode import ColorCodeHardware, check_values
    S = color_def(3)
    qc = ColorCodeHardware(3, 3, "fresh").build_circuit(
        inject=("X", 2, 0))                        # qubit 2: all 3 faces
    counts = AerSimulator(method="stabilizer").run(
        qc, shots=50).result().get_counts()
    syn, dat = counts_to_raw(counts, 3 * S.N_CHECKS, S.NUM_DATA)
    assert syn.shape == (50, 18) and dat.shape == (50, 7)
    vals = check_values(syn, 3, 3)
    for name in S.Z_NAMES:
        assert not vals[name][:, 0].any() and vals[name][:, 1].all()


def test_aer_target_restriction_follows_the_placement():
    """restrict_target renumbers the placed qubits and carries each
    calibration value to the new index."""
    from qiskit.circuit.library import CZGate, Measure, SXGate
    from qiskit.providers import QubitProperties
    from qiskit.transpiler import InstructionProperties, Target
    from qiskit_aer.noise import NoiseModel
    from hardware.providers.aer_calibrated import (
        _TargetBackend, restrict_target)
    t = Target(num_qubits=4, dt=1e-9, qubit_properties=[
        QubitProperties(t1=1e-4 * (q + 1), t2=5e-5 * (q + 1)) for q in range(4)])
    t.add_instruction(SXGate(), {(q,): InstructionProperties(
        duration=3e-8, error=1e-3 * (q + 1)) for q in range(4)})
    t.add_instruction(Measure(), {(q,): InstructionProperties(
        duration=1e-6, error=1e-2 * (q + 1)) for q in range(4)})
    t.add_instruction(CZGate(), {
        (0, 1): InstructionProperties(duration=7e-8, error=0.01),
        (1, 0): InstructionProperties(duration=7e-8, error=0.01),
        (1, 3): InstructionProperties(duration=7e-8, error=0.03),
        (3, 1): InstructionProperties(duration=7e-8, error=0.03),
        (2, 3): InstructionProperties(duration=7e-8, error=0.05),
        (3, 2): InstructionProperties(duration=7e-8, error=0.05)})
    sub = restrict_target(t, [3, 1])               # new 0 = old 3, 1 = old 1
    assert sub.num_qubits == 2
    assert sub["sx"][(0,)].error == pytest.approx(4e-3)
    assert sub["sx"][(1,)].error == pytest.approx(2e-3)
    assert sub["measure"][(0,)].error == pytest.approx(4e-2)
    assert set(sub["cz"]) == {(0, 1), (1, 0)}
    assert sub["cz"][(0, 1)].error == pytest.approx(0.03)
    assert sub.qubit_properties[0].t1 == pytest.approx(4e-4)
    assert sub.qubit_properties[1].t1 == pytest.approx(2e-4)
    noise = NoiseModel.from_backend(_TargetBackend(sub, "test"))
    assert {"sx", "cz"} <= set(noise.noise_instructions)
    assert set(noise.noise_qubits) == {0, 1}


class _FakeProvider(HardwareProvider):
    """Stand-in QPU: samples the abstract color circuit instead of
    running a job (a color raw bit is the check value)."""
    name = "fake"
    states = {}

    def submit(self, code, distance, cycles, shots, pubs, dry_run=False):
        run_dir = run_hw.RUNS_DIR / f"{self.backend}_{len(self.states)}"
        run_dir.mkdir(parents=True)
        jid = f"fake-{len(self.states)}"
        self.states[jid] = "QUEUED"
        json_dump({"job_id": jid, "job_ids": [jid], "backend": self.backend,
                   "code": code, "distance": distance, "cycles": cycles,
                   "shots": shots, "pubs": pubs, "n_pubs": pubs,
                   "shots_per_pub": shots, "dry_run": dry_run,
                   "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  run_dir / "job.json")
        return run_dir

    def status(self, job_ids):
        return [self.states[j] for j in job_ids]

    def wait(self, job_ids, poll=30):
        assert all(self.states[j] == "DONE" for j in job_ids)

    def collect(self, run_dir, job_ids=None, cycles=3, poll=30):
        meta = json.load(open(run_dir / "job.json"))
        self.wait(meta["job_ids"])
        d = meta["distance"]
        c = cs.build_color_stim_circuit(meta["cycles"], "X", 0.005, NOISE,
                                        distance=d)
        n = meta["shots"] * meta["pubs"]
        syn, dat = cs.sample_flips_color(c, n, meta["cycles"], seed=0,
                                         distance=d)
        return save_raw(run_dir, syn, dat, meta["cycles"],
                        [meta["shots"]] * meta["pubs"])


def test_submit_collect_analyze_through_a_provider(tmp_path, monkeypatch):
    """submit -> pending -> collect -> analyze with a fake provider: the
    run folder contract (job.json, raw.npz) is all analyze needs."""
    monkeypatch.setattr(run_hw, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(run_hw, "RESULTS_HW_DIR", tmp_path / "results")
    monkeypatch.setattr(run_hw, "PENDING_PATH", tmp_path / "pending.json")
    monkeypatch.setattr(run_hw, "_ROOT", tmp_path)       # no checkpoints
    _FakeProvider.states = {}
    monkeypatch.setattr(run_hw, "get_provider",
                        lambda backend: _FakeProvider(backend))

    args = argparse.Namespace(backend="fake_qpu", code="color", distance=3,
                              cycles=3, shots=1500, pubs=2, dry_run=False)
    provider, run_dir, job_ids = run_hw.cmd_submit(args)
    assert job_ids == ["fake-0"] and (run_dir / "job.json").exists()
    pending = json.load(open(tmp_path / "pending.json"))
    assert [p["job_id"] for p in pending] == ["fake-0"]
    assert pending[0]["code"] == "color" and pending[0]["distance"] == 3

    collect_args = argparse.Namespace(model=None, ckpt=None,
                                      mwpm_profile=NOISE, mwpm_p=0.005)
    run_hw.cmd_collect(collect_args)                     # still queued
    assert len(json.load(open(tmp_path / "pending.json"))) == 1
    assert not (run_dir / "raw.npz").exists()

    _FakeProvider.states["fake-0"] = "DONE"
    run_hw.cmd_collect(collect_args)
    assert json.load(open(tmp_path / "pending.json")) == []
    syn, dat, cycles, pubs = load_raw(run_dir / "raw.npz")
    S = color_def(3)
    assert syn.shape == (3000, 3 * S.N_CHECKS) and dat.shape == (3000, 7)
    assert cycles == 3 and pubs.tolist() == [1500, 1500]

    reports = list((tmp_path / "results").glob("fake_qpu_color_*.csv"))
    assert len(reports) == 1
    rows = list(csv.DictReader(open(reports[0])))
    assert [r["decoder"] for r in rows] == ["raw (no decoding)", "MWPM"]
    assert float(rows[1]["ler"]) < float(rows[0]["ler"])
    assert rows[0]["shots"] == "3000" and rows[0]["job_id"] == "fake-0"
    assert rows[0]["ler_std_over_pubs"] != "N/A"

    # a failed job is dropped from pending and logged, not analyzed
    args2 = argparse.Namespace(**{**vars(args), "shots": 100, "pubs": 1})
    _, run2, ids2 = run_hw.cmd_submit(args2)
    _FakeProvider.states[ids2[0]] = "ERROR"
    (tmp_path / "hardware").mkdir()
    run_hw.cmd_collect(collect_args)
    assert json.load(open(tmp_path / "pending.json")) == []
    assert not (run2 / "raw.npz").exists()
    assert "ERROR" in (tmp_path / "hardware" / "collect_failed.log").read_text()
