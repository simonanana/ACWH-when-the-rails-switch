"""
run_upgrade.py — run the upgrade modules u01–u10 in one go

    python src/run_upgrade.py                    # full precision (approx. 40–60 minutes, single core)
    python src/run_upgrade.py --quick            # smoke test (approx. 3–5 minutes, reduced B)
    python src/run_upgrade.py --only u03 u08     # run only the specified steps
    python src/run_upgrade.py --primary transatlantic --legacy idx10

All output is written to --out (default paper_tables_upgrade/); LaTeX tables are written
to paper/tables/. Each step runs as a separate subprocess: a failure in one step does not
stop the subsequent steps, and failed steps are summarised at the end.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsdb", default="data/Global_Sanctions_Data_Base__GSDB__2023.csv")
    ap.add_argument("--out", default="paper_tables_upgrade")
    ap.add_argument("--tex-out", default="paper/tables")
    ap.add_argument("--primary", default="bloc1_k2")
    ap.add_argument("--legacy", default="idx10")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--compare-ref", default=None,
                    help="existing icsid_convention_status.csv, for row-by-row comparison")
    a = ap.parse_args()

    q = a.quick
    defs = [a.primary, a.legacy]
    py = sys.executable
    ref = "data/reference/icsid_convention_status_icsid3.csv"
    steps = {
        "u01": [py, "src/u01_icsid_reference.py", "--out", ref]
               + (["--compare", a.compare_ref] if a.compare_ref else []),
        "u02": [py, "src/u02_gsdb_treatment.py", "--gsdb", a.gsdb,
                "--out-data", "data/derived", "--out", a.out, "--ref", ref],
        "u03": [py, "src/u03_core_table.py", "--defs", *defs, "--out", a.out, "--ref", ref,
                "--B-wcb", "199" if q else "1999", "--B-perm", "199" if q else "999",
                "--B-boot", "99" if q else "999"],
        "u04": [py, "src/u04_interpretive_bounds.py", "--defs", *defs, "--out", a.out,
                "--ref", ref, "--R", "50" if q else "1000"],
        "u05": [py, "src/u05_influence.py", "--defs", *defs, "--out", a.out, "--ref", ref],
        "u06": [py, "src/u06_event_study.py", "--defs", *defs, "--out", a.out, "--ref", ref,
                "--n-draw", "200" if q else "1000"],
        "u07": [py, "src/u07_ovb_sensitivity.py", "--defs", *defs, "--out", a.out,
                "--ref", ref],
        "u08": [py, "src/u08_spec_curve.py", "--B", "49" if q else "499",
                "--primary", a.primary, "--out", a.out, "--ref", ref],
        "u09": [py, "src/u09_incidence_etwfe.py", "--defs", *defs, "transatlantic",
                "--B", "49" if q else "299", "--out", a.out],
        "u10": [py, "src/u10_paper_tables.py", "--in", a.out, "--out", a.tex_out,
                "--primary", a.primary, "--legacy", a.legacy],
    }
    todo = a.only or list(steps)
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    failed = []
    for k in todo:
        t0 = time.time()
        print(f"=== {k} ===", flush=True)
        with open(logs / f"{k}.log", "w") as fh:
            r = subprocess.run(steps[k], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        status = "ok" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
        print(f"    {status}  {dt:.0f}s  log: logs/{k}.log", flush=True)
        if r.returncode:
            failed.append(k)
    if failed:
        print(f"\nFailed steps: {failed}; see the corresponding logs in logs/.")
        sys.exit(1)
    print("\nAll steps completed.")


if __name__ == "__main__":
    main()
