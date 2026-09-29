"""
u03_core_table.py — core table: treatment definition × Convention-status eligibility
coding × outcome × five inference methods

For each combination reports
    β (percentage points), CR1 s.e./p, CV3 jackknife s.e./p, restricted WCB (Webb) p,
    permutation p: full pool / restricted pool (corrected) / timing-only, and the
    minimum attainable p of the timing-only test
Also reports a paired cluster bootstrap confidence interval for the "attenuation"
(no control − three-valued control).

Usage
-----
    python src/u03_core_table.py --defs bloc1_k2 idx10 --out paper_tables \
        --B-wcb 1999 --B-perm 999 --B-boot 999
Quick trial run: --B-wcb 199 --B-perm 199 --B-boot 199
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from u00_common import (AVAIL_CODINGS, OUTCOMES, PP, TIME, UNIT, Projector, attach,
                        append_canonical, cv3, design_frame, estimate, load_base,
                        load_cohorts, outdir, permutation_pvalue, save, wcb_webb)


def attenuation_boot(x: pd.DataFrame, y: str, B: int, seed: int = 20260917) -> dict:
    """Paired cluster bootstrap: estimate the uncontrolled and three-valued-control
    specifications on the same resample and take the difference."""
    rng = np.random.default_rng(seed)
    units = x[UNIT].unique()
    idx = {u: np.where(x[UNIT].to_numpy() == u)[0] for u in units}

    def pair(fr):
        yv, sv = fr[y].to_numpy(float) * PP, fr["S"].to_numpy(float)
        a = Projector(fr, [UNIT, TIME]).fit(yv, sv)["coef"]
        b = Projector(fr, [UNIT, TIME, "avail_three"]).fit(yv, sv)["coef"]
        return a, b

    a0, b0 = pair(x)
    diffs = []
    for _ in range(B):
        pick = rng.choice(units, size=len(units), replace=True)
        parts = []
        for k, u in enumerate(pick):
            p = x.iloc[idx[u]].copy()
            p[UNIT] = f"{u}#{k}"          # a country drawn more than once must be treated as distinct clusters
            parts.append(p)
        bs = pd.concat(parts, ignore_index=True)
        if bs["S"].nunique() < 2:
            continue
        a, b = pair(bs)
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(a - b)
    diffs = np.array(diffs)
    return {"beta_uncond": a0, "beta_three": b0, "attenuation": a0 - b0,
            "ci_lo": float(np.percentile(diffs, 2.5)),
            "ci_hi": float(np.percentile(diffs, 97.5)),
            "p_two_sided": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
            "B_ok": len(diffs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defs", nargs="+", default=["bloc1_k2", "idx10"])
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", default="paper_tables")
    ap.add_argument("--B-wcb", type=int, default=1999)
    ap.add_argument("--B-perm", type=int, default=999)
    ap.add_argument("--B-boot", type=int, default=999)
    ap.add_argument("--outcomes", nargs="+", default=list(OUTCOMES))
    ap.add_argument("--codings", nargs="+", default=list(AVAIL_CODINGS))
    ap.add_argument("--no-boot", action="store_true", help="skip the attenuation bootstrap")
    ap.add_argument("--boot-only", action="store_true", help="run only the attenuation bootstrap")
    ap.add_argument("--append", action="store_true",
                    help="append to existing CSVs (for chunked runs, to avoid a single run timing out)")
    a = ap.parse_args()
    out = outdir(a.out)

    base = load_base(a.ref)
    coh = load_cohorts()
    rows, att, canon = [], [], []
    for dname in a.defs:
        cmap = coh[dname].dropna().to_dict()
        d = attach(base, cmap)
        for y in a.outcomes:
            ylab = OUTCOMES[y]
            for cod in ([] if a.boot_only else a.codings):
                clab = AVAIL_CODINGS[cod]
                r = estimate(d, y, coding=cod)
                j = cv3(d, y, coding=cod)
                rec = {"definition": dname, "outcome": ylab, "coding": clab,
                       "effect": r["coef"], "se_cr1": r["se"], "p_cr1": r["p"],
                       "se_cv3": j["se_cv3"], "p_cv3": j["p_cv3"],
                       "p_wcb_webb": wcb_webb(d, y, coding=cod, B=a.B_wcb)}
                for pool in ("all", "restricted", "timing"):
                    pr = permutation_pvalue(d, y, cmap, pool=pool, coding=cod,
                                            B=a.B_perm)
                    rec[f"p_perm_{pool}"] = pr["p"]
                    if pool == "timing":
                        rec["timing_min_p"] = pr["min_attainable_p"]
                rec.update({k: r[k] for k in ("n", "switchers", "treated_cases",
                                               "identifying_cases")})
                rows.append(rec)
                print(f"  {dname:<14} {y:<17} {cod:<6} β={r['coef']:7.2f}  "
                      f"p_cr1={r['p']:.3f} p_cv3={j['p_cv3']:.3f} "
                      f"p_wcb={rec['p_wcb_webb']:.3f} p_timing={rec['p_perm_timing']:.3f}")
                tag = f"{dname} {y} {cod}"
                canon += [(f"{tag} beta", r["coef"]), (f"{tag} p_wcb", rec["p_wcb_webb"]),
                          (f"{tag} p_cv3", j["p_cv3"]),
                          (f"{tag} p_timing", rec["p_perm_timing"])]
            if a.no_boot:
                continue
            x, _ = design_frame(d, "none")
            b = attenuation_boot(x, y, a.B_boot)
            b.update({"definition": dname, "outcome": ylab})
            att.append(b)
            canon += [(f"{dname} {y} attenuation", b["attenuation"]),
                      (f"{dname} {y} attenuation ci_lo", b["ci_lo"]),
                      (f"{dname} {y} attenuation ci_hi", b["ci_hi"])]

    def _write(new, name):
        f = out / name
        df = pd.DataFrame(new)
        if a.append and f.exists() and len(df):
            old = pd.read_csv(f)
            keys = [k for k in ("definition", "outcome", "coding") if k in df.columns]
            old = old.merge(df[keys], on=keys, how="left", indicator=True)
            old = old[old["_merge"] == "left_only"].drop(columns="_merge")
            df = pd.concat([old, df], ignore_index=True)
        if len(df):
            save(df, out, name)

    print("\n[u03] Core table")
    _write(rows, "tab_core_inference.csv")
    print("\n[u03] Attenuation (paired cluster bootstrap)")
    _write(att, "tab_attenuation_paired.csv")
    append_canonical(canon, out)


if __name__ == "__main__":
    main()
