"""
Regression gate: re-run the AIBF/dec23 calibration with the regression config
and confirm `results.json` + `dfRtopN.csv` match the snapshot captured from
pristine `main` content.

Skipped if the snapshot or the data file isn't present.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

THIS_DIR = Path(__file__).resolve().parent
SNAPSHOT_DIR = THIS_DIR / "snapshot"
REGRESSION_CONFIG = THIS_DIR / "regression_config.json"
REPO_ROOT = THIS_DIR.parents[1]

_DATA_PATH_ENV = os.environ.get("RCT_REGRESSION_DATA_PATH")
DATA_PATH = Path(_DATA_PATH_ENV) if _DATA_PATH_ENV else None

RTOL = 1e-6
ATOL = 1e-9


@pytest.fixture(scope="module")
def fresh_run(tmp_path_factory):
    if not SNAPSHOT_DIR.exists():
        pytest.skip(f"snapshot missing at {SNAPSHOT_DIR}; run capture_snapshot.py")
    if DATA_PATH is None:
        pytest.skip(
            "regression data path not set; "
            "export RCT_REGRESSION_DATA_PATH=/path/to/data_train.joblib"
        )
    if not DATA_PATH.exists():
        pytest.skip(f"regression data missing at {DATA_PATH}")

    work = tmp_path_factory.mktemp("regression_run")
    cmd = [
        ".venv/bin/rct-train",
        "--building-config", "AIBF",
        "--data-path", str(DATA_PATH),
        "--identification-config", str(REGRESSION_CONFIG),
        "--results-directory-path", str(work),
        "--seed", "23",
        "--stage", "1-3",
        "--save-dfRtopN",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return work


def _flatten(obj, prefix=""):
    """Yield (key_path, value) pairs from a nested dict."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def test_results_json_matches_snapshot(fresh_run):
    fresh = json.loads((fresh_run / "results.json").read_text())
    gold = json.loads((SNAPSHOT_DIR / "results.json").read_text())

    fresh_flat = dict(_flatten(fresh))
    gold_flat = dict(_flatten(gold))

    assert fresh_flat.keys() == gold_flat.keys(), (
        f"results.json key set drift:\n"
        f"  added:   {sorted(fresh_flat.keys() - gold_flat.keys())}\n"
        f"  removed: {sorted(gold_flat.keys() - fresh_flat.keys())}"
    )

    for key in gold_flat:
        a, b = fresh_flat[key], gold_flat[key]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            # Both NaN is considered equal (e.g. validation_error when no val data)
            if np.isnan(a) and np.isnan(b):
                continue
            assert np.isclose(a, b, rtol=RTOL, atol=ATOL), (
                f"{key}: fresh={a} vs gold={b}"
            )
        else:
            assert a == b, f"{key}: fresh={a!r} vs gold={b!r}"


def test_dfrtopn_matches_snapshot(fresh_run):
    fresh = pd.read_csv(fresh_run / "dfRtopN.csv")
    gold = pd.read_csv(SNAPSHOT_DIR / "dfRtopN.csv")

    assert list(fresh.columns) == list(gold.columns), (
        f"dfRtopN columns drift: fresh={list(fresh.columns)} gold={list(gold.columns)}"
    )
    assert len(fresh) == len(gold), f"row count: fresh={len(fresh)} gold={len(gold)}"

    for col in gold.columns:
        if pd.api.types.is_numeric_dtype(gold[col]):
            np.testing.assert_allclose(
                fresh[col].values, gold[col].values,
                rtol=RTOL, atol=ATOL,
                err_msg=f"column {col!r} drift",
            )
        else:
            assert (fresh[col].astype(str).values == gold[col].astype(str).values).all(), (
                f"column {col!r} string drift"
            )
