"""Convention gates: every verification script must print ALL PASS."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GATES = [
    "verification/verify_equivalence.py",
    "verification/verify_rotatedSurface3.py",
    "verification/verify_color.py",
]


@pytest.mark.slow
@pytest.mark.parametrize("script", GATES)
def test_gate_all_pass(script):
    path = ROOT / script
    if not path.exists():
        pytest.skip(f"{script} is not in the repo yet")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, str(path)], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=540)
    tail = (r.stdout + r.stderr)[-2000:]
    assert r.returncode == 0, tail
    assert "ALL PASS" in r.stdout, tail
