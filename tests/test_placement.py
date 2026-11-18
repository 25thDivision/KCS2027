"""circuits/placement.py: the search runs on the device graph with bad
qubits and couplers removed (usable_edges)."""
import pytest

from circuits import placement as pl
from circuits.rotatedSurface.rotatedSurface import (
    RotatedSurfaceHardware, required_edges_surface, surface_def)

ROWS, COLS = 8, 9


def grid_edges():
    q = lambda r, c: r * COLS + c                             # noqa: E731
    e = [(q(r, c), q(r, c + 1)) for r in range(ROWS) for c in range(COLS - 1)]
    e += [(q(r, c), q(r + 1, c)) for r in range(ROWS - 1) for c in range(COLS)]
    return e + [(b, a) for a, b in e]                          # directed


def calib(bad_qubits=(), bad_edges=()):
    n = ROWS * COLS
    c = {"readout": {q: 0.01 for q in range(n)},
         "err1q": {q: 1e-4 for q in range(n)},
         "t1": {q: 1e-4 for q in range(n)},
         "t2": {q: 1e-4 for q in range(n)},
         "dur_meas": {}, "dur_1q": {}, "dur_2q": {},
         "err2q": {tuple(sorted(e)): 0.003 for e in grid_edges()}}
    for q, kind in bad_qubits:
        if kind == "T2":
            c["t2"][q] = 5e-6
        else:
            c["readout"][q] = 0.4
    for e in bad_edges:
        c["err2q"][tuple(sorted(e))] = 1.0
    return c


def valid_sets(pattern, labels, layers, edges, cal, cap=10 ** 6):
    out = set()
    for m in pl.enumerate_placements(pattern, edges, cap):
        s, _ = pl.score_placement(m, labels, pattern, layers, cal)
        if s is not None:
            out.add(frozenset(m.values()))
    return out


def test_usable_edges_drops_bad_qubits_and_couplers():
    cal = calib(bad_qubits=[(8, "T2"), (20, "readout")],
                bad_edges=[(1, 2)])
    kept = pl.usable_edges(grid_edges(), cal)
    nodes = {q for e in kept for q in e}
    assert 8 not in nodes and 20 not in nodes
    assert (1, 2) not in kept and (2, 1) not in kept
    assert (0, 1) in kept and (1, 0) in kept


@pytest.mark.parametrize("bad", [
    dict(bad_qubits=[(16, "T2")]),
    dict(bad_qubits=[(15, "readout"), (26, "T2")], bad_edges=[(2, 3)]),
])
def test_same_valid_placements_as_the_full_search(bad):
    """Uncapped: the pruned search finds exactly the valid placements of
    the full search (surface d=3 patch on an 8x9 grid)."""
    S = surface_def(3)
    pattern = required_edges_surface(3)
    layers = pl.circuit_layers(RotatedSurfaceHardware(3, 3).build_circuit(),
                               S.ALL_COORDS)
    cal = calib(**bad)
    full = valid_sets(pattern, S.ALL_COORDS, layers, grid_edges(), cal)
    pruned = valid_sets(pattern, S.ALL_COORDS, layers,
                        pl.usable_edges(grid_edges(), cal), cal)
    assert full and full == pruned


def test_every_pruned_candidate_is_valid():
    """So the enumeration cap is spent on valid placements only: even a
    cap of 1 returns a valid one."""
    S = surface_def(3)
    pattern = required_edges_surface(3)
    cal = calib(bad_qubits=[(16, "T2"), (17, "T2")])
    first = next(iter(pl.enumerate_placements(
        pattern, pl.usable_edges(grid_edges(), cal), cap=1)))
    assert pl.score_placement(first, S.ALL_COORDS, pattern, [], cal)[1] \
        is None


def test_select_placement_avoids_bad_qubits():
    from qiskit.circuit.library import CZGate, Measure, SXGate
    from qiskit.providers import QubitProperties
    from qiskit.transpiler import InstructionProperties, Target
    n = ROWS * COLS
    t2 = {q: 1e-4 for q in range(n)}
    t2[16] = 5e-6
    target = Target(num_qubits=n, dt=1e-9, qubit_properties=[
        QubitProperties(t1=1e-4, t2=t2[q]) for q in range(n)])
    target.add_instruction(SXGate(), {(q,): InstructionProperties(
        duration=3e-8, error=1e-4) for q in range(n)})
    target.add_instruction(Measure(), {(q,): InstructionProperties(
        duration=1e-6, error=0.01) for q in range(n)})
    target.add_instruction(CZGate(), {e: InstructionProperties(
        duration=7e-8, error=0.003) for e in grid_edges()})
    S = surface_def(3)
    qc = RotatedSurfaceHardware(3, 3).build_circuit()
    res = pl.select_placement(S.ALL_COORDS, required_edges_surface(3), qc,
                              grid_edges(), target)
    assert res is not None and 16 not in res["mapping"].values()
    assert res["n_excluded"] == 0
