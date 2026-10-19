"""The Level 1-3 presets give the same numbers, to the last digit, as before the series
additions of 2026-10-19 (emissions, cloud fraction, ventilation, wall losses, seed salts).

The stored fingerprints (tests/reference/aaar_preset_fingerprints.json) were made at
feat/aaar-tutorial a631e80 with ``python tests/aaar_fingerprint.py``. Every output that
existed then is hashed as float64 bytes; outputs added later are not part of the hash.
The 1 h runs guard every commit; the full presets are ``pytest -m slow``.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from aaar_fingerprint import CASES, FINGERPRINT_FILE, case_key, fingerprint  # noqa: E402

from tomas_jax.box.driver import run  # noqa: E402
from tomas_jax.box.scenario import load_preset  # noqa: E402

STORED = json.loads(FINGERPRINT_FILE.read_text())


@pytest.mark.parametrize("name,hours", [(n, h) for n, h, slow in CASES if not slow])
def test_short_preset_runs_are_bit_identical(name, hours):
    key = case_key(name, hours)
    got = fingerprint(run(load_preset(name).replace(duration_h=hours)))
    assert got["sample"] == STORED[key]["sample"]
    assert got["sha256"] == STORED[key]["sha256"]


@pytest.mark.slow
@pytest.mark.parametrize("name,hours", [(n, h) for n, h, slow in CASES if slow])
def test_full_presets_are_bit_identical(name, hours):
    key = case_key(name, hours)
    got = fingerprint(run(load_preset(name)))
    assert load_preset(name).duration_h == hours
    assert got["sample"] == STORED[key]["sample"]
    assert got["sha256"] == STORED[key]["sha256"]
