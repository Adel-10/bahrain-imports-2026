"""Run the whole project, step by step, in order.

    python src/run_all.py                    # everything, from the download to the summary
    python src/run_all.py --skip-download    # use the files already in data/raw
    python src/run_all.py --from shortfall   # start part-way through (earlier steps' results must exist)

The pipeline
------------
Each step reads what the previous ones saved, so they must run in this order:

    download_data    data/raw/*.csv                 the portal's files, untouched
    check_downloads  (checks only)                  stops the run if a file is incomplete
    build_dataset    data/processed/*.parquet       cleaned imports and route figures
    explore          step 3 report                  decisions for the models
    backtest         data/interim/step4_*           tested models, 2026 forecasts, normal ranges
    shortfall        data/interim/step5_gaps.csv    the 2026 gaps
    routes           data/interim/step6_*.csv       route evidence and adapting
    suppliers        data/interim/extra_*.csv       the supplier-concentration extra
    summary          reports/summary.md             reads the saved results above

Each step is run as its own script, exactly as if you had typed it, so a step behaves the same
whether it is run here or on its own. The run stops at the first step that fails and says how to
resume. Re-running rewrites every report and chart from the data.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent

STEPS = [                    # (script name in src/, title printed while it runs)
    ("download_data", "Step 1: download the data"),
    ("check_downloads", "Step 1: check the downloads"),
    ("build_dataset", "Step 2: clean and build the tables"),
    ("explore", "Step 3: explore"),
    ("backtest", "Step 4: train and test the models"),
    ("shortfall", "Step 5: the 2026 shortfall"),
    ("routes", "Step 6: routes and adapting"),
    ("suppliers", "Extra: did relying on few supplier countries matter?"),
    ("summary", "Step 7: the summary"),
]
NAMES = [name for name, _ in STEPS]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-download", action="store_true", help="don't run download_data.py")
    parser.add_argument("--from", dest="start", choices=NAMES, default=NAMES[0], help="first step to run")
    args = parser.parse_args()

    steps = STEPS[NAMES.index(args.start):]
    if args.skip_download:
        steps = [s for s in steps if s[0] != "download_data"]

    t0 = time.time()
    for i, (name, title) in enumerate(steps, 1):
        print(f"\n=== [{i}/{len(steps)}] {title}  (src/{name}.py) ===", flush=True)
        start = time.time()
        result = subprocess.run([sys.executable, str(SRC / f"{name}.py")])
        if result.returncode != 0:
            print(f"\nStopped: src/{name}.py failed (exit code {result.returncode}). Fix it, then run:\n"
                  f"    python src/run_all.py --from {name}")
            return result.returncode
        print(f"=== done in {time.time() - start:.0f}s ===", flush=True)
    print(f"\nAll {len(steps)} steps finished in {(time.time() - t0) / 60:.1f} minutes. Reports are in reports/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
