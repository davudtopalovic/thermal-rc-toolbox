"""
Capture a regression snapshot from the current code.

Run once at the start of the refactor (on pristine `main` content)
to produce the gold reference that test_5r1c_regression.py compares against.

Usage:
    .venv/bin/python tests/regression/capture_snapshot.py
    .venv/bin/python tests/regression/capture_snapshot.py --data-path <path/to/data_train.joblib>

By default, --data-path points to
examples/run_identification/five_r1c_eniso13790/AIBF/data_train.joblib
relative to the repo root.
"""
import argparse
import shutil
import subprocess
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SNAPSHOT_DIR = THIS_DIR / "snapshot"
WORK_DIR = THIS_DIR / "_capture_work"
REGRESSION_CONFIG = THIS_DIR / "regression_config.json"
REPO_ROOT = THIS_DIR.parents[1]
DEFAULT_DATA_PATH = (
    REPO_ROOT
    / "examples/run_identification/five_r1c_eniso13790/AIBF/data_train.joblib"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH),
                        help="Path to data_train.joblib (default: %(default)s)")
    args = parser.parse_args()

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)

    cmd = [
        ".venv/bin/rct-train",
        "--building-config", "AIBF",
        "--data-path", args.data_path,
        "--identification-config", str(REGRESSION_CONFIG),
        "--results-directory-path", str(WORK_DIR),
        "--seed", "23",
        "--stage", "1-3",
        "--save-dfRtopN",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[2])

    SNAPSHOT_DIR.mkdir(exist_ok=True)
    shutil.copy(WORK_DIR / "results.json", SNAPSHOT_DIR / "results.json")
    shutil.copy(WORK_DIR / "dfRtopN.csv", SNAPSHOT_DIR / "dfRtopN.csv")
    print(f"Snapshot written to {SNAPSHOT_DIR}")


if __name__ == "__main__":
    main()
