"""analysis/: gather_results, spearman and plot on synthetic result files."""
import csv

import pytest

from analysis import gather_results as gr
from analysis import plot as pl
from analysis import spearman as sp

NOISE = "realistic/dp0.001_mf0.01_rf0.01_gd0.008"
TAG = "dp0.001_mf0.01_rf0.01_gd0.008"
MODELS = ["CNN", "GCN", "GCNII", "GAT", "APPNP", "GNN", "GRAPHTRANSFORMER",
          "GRAPHMAMBA"]
SIM = dict(zip(MODELS, [0.05, 0.20, 0.25, 0.10, 0.30, 0.12, 0.04, 0.03]))


def write_train_summary(d, model, code, distance, p, ler, noise=NOISE):
    d.mkdir(parents=True, exist_ok=True)
    name = f"{model}_{code}_d{distance}_c3_p{p}_{noise.split('/')[-1]}.csv"
    with open(d / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["decoder", "ler", "model", "code", "distance", "cycles",
                    "noise_profile", "p", "error_type", "n_shots",
                    "best_epoch", "total_epochs"])
        for dec, v in ((model, ler), ("raw", 0.4), ("MWPM", 0.08),
                       (f"MWPM+{model}", ler / 2), (f"{model}+MWPM", ler / 3)):
            w.writerow([dec, f"{v:.6f}", model.lower(), code, distance, 3,
                        noise, p, "X", 100000, 5, 9])


def write_hw_report(d, backend, code, distance, lers, extra=()):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{backend}_{code}_20260101-000000.csv", "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["decoder", "ler", "ler_std_over_pubs",
                    "parity_LER (diagnostic)", "LER/MWPM ratio", "shots",
                    "cycles", "best_epoch", "total_epochs", "backend",
                    "timestamp", "job_id", "code", "distance"])

        def row(dec, v):
            w.writerow([dec, f"{v:.6f}", "N/A", "N/A", "N/A", 250000, 3, "",
                        "", backend, "2026-01-01T00:00:00", "job", code,
                        distance])
        row("raw (no decoding)", 0.45)
        row("MWPM", 0.20)
        for dec, v in extra:
            row(dec, v)
        for model, v in lers.items():
            ck = f"{model}_{code}_d{distance}_c3_p0.005_{TAG}.pt"
            row(f"{model} ({ck})", v)
            row(f"MWPM+{model} ({ck})", v / 2)


@pytest.fixture
def results(tmp_path):
    res = tmp_path / "results"
    for model, v in SIM.items():
        write_train_summary(res / "train" / "summary", model, "surface", 3,
                            0.005, v)
        write_train_summary(res / "train" / "summary", model, "surface", 3,
                            0.01, v + 0.02)
    # backend A ranks the decoders like the simulation, backend B the
    # other way round
    write_hw_report(res / "hardware", "ibm_same", "surface", 3,
                    {m: v + 0.1 for m, v in SIM.items()},
                    extra=[("NVIDIA(Accurate)", 0.22)])
    write_hw_report(res / "hardware", "aer_ibm_rev", "surface", 3,
                    {m: 1.0 - v for m, v in SIM.items()})
    return res


def test_parse_hw_decoder():
    ck = f"GRAPHMAMBA_surface_d5_c3_p0.01_{TAG}.pt"
    assert gr.parse_hw_decoder(f"GRAPHMAMBA ({ck})") == (
        "GRAPHMAMBA", NOISE, "0.01")
    assert gr.parse_hw_decoder(f"MWPM+GRAPHMAMBA ({ck})")[0] == \
        "MWPM+GRAPHMAMBA"
    assert gr.parse_hw_decoder("raw (no decoding)") == ("raw", "", "")
    assert gr.parse_hw_decoder("MWPM") == ("MWPM", "", "")
    assert gr.parse_hw_decoder("NVIDIA(Accurate)") == (
        "NVIDIA(Accurate)", "", "")
    named = f"CNN_heavyhex_d3_c3_p0.05_{TAG}_myrun.pt"
    assert gr.parse_hw_decoder(f"CNN ({named})") == ("CNN", NOISE, "0.05")
    # a 4-parameter tag that is not a registered profile, and a QPU
    # calibration profile
    odd = "CNN_surface_d3_c3_p0.005_dp0.002_mf0.03_rf0.01_gd0.004.pt"
    assert gr.parse_hw_decoder(f"CNN ({odd})")[1] == \
        "realistic/dp0.002_mf0.03_rf0.01_gd0.004"
    qpu = "GNN_heavyhex_d3_c3_p0.005_ibm_x_heavyhex_avg5_20260101.pt"
    assert gr.parse_hw_decoder(f"GNN ({qpu})") == (
        "GNN", "qpu/ibm_x_heavyhex_avg5_20260101", "0.005")


def test_gather(results):
    rows = gr.gather(results)
    out = list(csv.DictReader(open(results / "gathered.csv")))
    assert len(out) == len(rows)
    assert list(out[0]) == gr.COLUMNS
    # 8 models x 2 p x 5 decoder lines of training summaries
    sim = [r for r in out if r["platform_or_sim"] == "sim"]
    assert len(sim) == 8 * 2 * 5
    assert {r["noise_profile"] for r in sim} == {NOISE}
    assert {r["p"] for r in sim} == {"0.005", "0.01"}
    hw = [r for r in out if r["platform_or_sim"] == "hardware"]
    aer = [r for r in out if r["platform_or_sim"] == "aer"]
    assert {r["backend"] for r in hw} == {"ibm_same"}
    assert {r["backend"] for r in aer} == {"aer_ibm_rev"}
    cnn = [r for r in hw if r["decoder"] == "CNN"]
    assert len(cnn) == 1 and cnn[0]["noise_profile"] == NOISE
    assert cnn[0]["p"] == "0.005" and cnn[0]["n_shots"] == "250000"
    assert float(cnn[0]["ler"]) == pytest.approx(SIM["CNN"] + 0.1)
    assert any(r["decoder"] == "raw" and r["noise_profile"] == ""
               for r in hw)
    assert all(r["code"] == "surface" and r["distance"] == "3" for r in out)
    assert all(r["source_file"].endswith(".csv") for r in out)


def test_spearman(results):
    gr.gather(results)
    table = sp.spearman_table(sp.load(results / "gathered.csv"))
    by = {r["backend"]: r for r in table}
    assert set(by) == {"ibm_same", "aer_ibm_rev"}
    assert by["ibm_same"]["n_decoders"] == 8
    assert float(by["ibm_same"]["rho"]) == pytest.approx(1.0)
    assert float(by["aer_ibm_rev"]["rho"]) == pytest.approx(-1.0)
    assert by["aer_ibm_rev"]["platform"] == "aer"
    # hybrids, MWPM and raw rows do not enter the ranking
    assert "MWPM" not in by["ibm_same"]["decoders"].split()
    # --match: only conditions present on both sides (here p = 0.005)
    matched = sp.spearman_table(sp.load(results / "gathered.csv"), True)
    assert float({r["backend"]: r for r in matched}["ibm_same"]["rho"]) \
        == pytest.approx(1.0)


def test_spearman_needs_three_decoders(tmp_path):
    res = tmp_path / "results"
    for model in ("CNN", "GNN"):
        write_train_summary(res / "train" / "summary", model, "color", 3,
                            0.005, SIM[model])
    write_hw_report(res / "hardware", "ionq_forte1", "color", 3,
                    {"CNN": 0.2, "GNN": 0.3})
    gr.gather(res)
    table = sp.spearman_table(sp.load(res / "gathered.csv"))
    assert len(table) == 1 and table[0]["n_decoders"] == 2
    assert table[0]["rho"] == ""


def test_plots(results):
    gr.gather(results)
    made = pl.make_figures(results / "gathered.csv", results / "figures")
    names = sorted(p.name for p in made)
    assert names == ["decoders_surface_d3.png", "nvidia_surface.png"]
    for p in made:
        assert p.stat().st_size > 2000
        assert p.with_suffix(".csv").exists()
    vals = list(csv.DictReader(open(results / "figures"
                                    / "decoders_surface_d3.csv")))
    cnn_sim = [r for r in vals if r["decoder"] == "CNN"
               and r["source"] == "simulation"]
    # mean over the two p values: 0.05 and 0.07
    assert float(cnn_sim[0]["mean_ler"]) == pytest.approx(0.06)
    assert cnn_sim[0]["n_rows"] == "2"
    assert {r["source"] for r in vals} == {"simulation", "QPU-cal (Aer)",
                                           "hardware"}
