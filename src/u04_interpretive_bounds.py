"""
u04_interpretive_bounds.py — interpretive range for Article 72 (M1, partial identification)

Problem
-------
The 55 contested cases (filed after the respondent State's denunciation took effect)
are, as a matter of law, either Convention-eligible ("open") or not ("closed"),
depending on the interpretation of Articles 71/72 and on case-specific facts (when the
investor accepted the offer to arbitrate). The true status of each case is unknown.
What is identified is a **set**, not a single number.

Approach
--------
1. Endpoints: strict interpretation (all closed) and permissive interpretation (all open).
2. Country-level combinations of interpretations: VEN / BOL / ECU each open or closed,
   2^3 = 8 combinations (treaty wording, tribunals and denunciation dates differ, so it
   is realistic for different interpretations to apply to different States).
3. Case-by-case random assignment: each contested case is open with probability p,
   p ∈ {0.25, 0.5, 0.75}, R draws each. **The OLS coefficient is not monotone in the
   assignment**, so the two endpoints are not guaranteed to bound all assignments;
   simulation is therefore required.
4. Identified set = [min, max] of β across all of the above assignments.
5. Imbens–Manski (2004) confidence interval, with endpoint standard errors computed
   both by CR1 and by CV3 (jackknife).

Output
------
paper_tables/tab_interpretive_bounds.csv       identified set and IM confidence interval
                                               per definition × outcome
paper_tables/tab_interpretive_assignments.csv  the 8 country-level assignments + quantiles
                                               of the random assignments
paper_tables/fig_interpretive_bounds.pdf       figure

Usage
-----
    python src/u04_interpretive_bounds.py --defs bloc1_k2 idx10 --R 1000 --out paper_tables
"""
from __future__ import annotations

import argparse
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from u00_common import (OUTCOMES, PP, TIME, UNIT, Projector, append_canonical,  # noqa: E402
                        attach, cv3, imbens_manski, load_base, load_cohorts, outdir,
                        save)


def fit_with(x: pd.DataFrame, y: str, open_mask: np.ndarray) -> float:
    """Given an open/closed assignment for the contested cases, estimate β with binary
    Convention-status eligibility as a fixed effect."""
    z = x.copy()
    z["avail_assign"] = np.where(z.convention_status.eq("available") | open_mask,
                                 "open", "closed")
    return Projector(z, [UNIT, TIME, "avail_assign"]).fit(
        z[y].to_numpy(float) * PP, z["S"].to_numpy(float))["coef"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defs", nargs="+", default=["bloc1_k2", "idx10"])
    ap.add_argument("--R", type=int, default=1000)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", default="paper_tables")
    ap.add_argument("--seed", type=int, default=20260917)
    a = ap.parse_args()
    out = outdir(a.out)
    base = load_base(a.ref)
    coh = load_cohorts()
    rng = np.random.default_rng(a.seed)

    contested = base.convention_status.eq("contested").to_numpy()
    states = sorted(base.loc[contested, UNIT].unique())
    print(f"[u04] contested cases: {contested.sum()}, involving {states}")

    rows, assign_rows, canon = [], [], []
    for dname in a.defs:
        d = attach(base, coh[dname].dropna().to_dict()).reset_index(drop=True)
        for y, ylab in OUTCOMES.items():
            betas, masks = {}, {}
            # endpoints
            zero = np.zeros(len(d), bool)
            b_strict = fit_with(d, y, zero)
            b_lax = fit_with(d, y, contested)
            masks["strict"], masks["lax"] = zero, contested
            betas["strict"], betas["lax"] = b_strict, b_lax
            # country-level combinations
            for combo in itertools.product([False, True], repeat=len(states)):
                opened = [s for s, o in zip(states, combo) if o]
                m = contested & d[UNIT].isin(opened).to_numpy()
                lab = "open: " + (",".join(opened) if opened else "none")
                betas[lab], masks[lab] = fit_with(d, y, m), m
                assign_rows.append({"definition": dname, "outcome": ylab,
                                    "assignment": lab, "beta": betas[lab]})
            # case-by-case random assignment
            for p in (0.25, 0.5, 0.75):
                v = []
                for r_ in range(a.R):
                    m = contested & (rng.random(len(d)) < p)
                    lab = f"random p={p} #{r_}"
                    betas[lab], masks[lab] = fit_with(d, y, m), m
                    v.append(betas[lab])
                v = np.array(v)
                assign_rows.append({"definition": dname, "outcome": ylab,
                                    "assignment": f"random p={p}",
                                    "beta": float(np.median(v)),
                                    "q05": float(np.percentile(v, 5)),
                                    "q95": float(np.percentile(v, 95)),
                                    "min": float(v.min()), "max": float(v.max())})
            labs = list(betas)
            vals = np.array([betas[k] for k in labs])
            k_lo, k_hi = labs[int(np.nanargmin(vals))], labs[int(np.nanargmax(vals))]
            lo, hi = betas[k_lo], betas[k_hi]

            def se_pair(mask):
                fr = d.copy()
                fr["avail_three"] = np.where(fr.convention_status.eq("available") | mask,
                                             "open", "closed")   # reuse cv3's "three" slot
                s1 = Projector(fr, [UNIT, TIME, "avail_three"]).fit(
                    fr[y].to_numpy(float) * PP, fr["S"].to_numpy(float))["se"]
                return s1, cv3(fr, y, coding="three")["se_cv3"]

            se_lo_cr1, se_lo_cv3 = se_pair(masks[k_lo])
            se_hi_cr1, se_hi_cv3 = se_pair(masks[k_hi])
            im_cr1 = imbens_manski(lo, se_lo_cr1, hi, se_hi_cr1)
            im_cv3 = imbens_manski(lo, se_lo_cv3, hi, se_hi_cv3)
            rec = {"definition": dname, "outcome": ylab,
                   "beta_strict": b_strict, "beta_lax": b_lax,
                   "set_lo": lo, "set_hi": hi, "argmin": k_lo.split(" #")[0],
                   "argmax": k_hi.split(" #")[0],
                   "endpoints_bound_all": bool(min(b_strict, b_lax) <= lo + 1e-9
                                               and max(b_strict, b_lax) >= hi - 1e-9),
                   "IM95_lo_cr1": im_cr1[0], "IM95_hi_cr1": im_cr1[1],
                   "IM95_lo_cv3": im_cv3[0], "IM95_hi_cv3": im_cv3[1],
                   "set_excludes_zero": bool(lo > 0 or hi < 0),
                   "IM_cv3_excludes_zero": bool(im_cv3[0] > 0 or im_cv3[1] < 0)}
            rows.append(rec)
            canon += [(f"{dname} {y} bounds lo", lo), (f"{dname} {y} bounds hi", hi),
                      (f"{dname} {y} IM cv3 lo", im_cv3[0]),
                      (f"{dname} {y} IM cv3 hi", im_cv3[1])]

    tb = pd.DataFrame(rows)
    ta = pd.DataFrame(assign_rows)
    print("\n[u04] Interpretive range")
    save(tb, out, "tab_interpretive_bounds.csv")
    save(ta, out, "tab_interpretive_assignments.csv", show=False)
    append_canonical(canon, out)

    # Figure: one row per definition, one panel per outcome
    fig, axes = plt.subplots(len(a.defs), 2, figsize=(9, 2.6 * len(a.defs)),
                             squeeze=False)
    for i, dname in enumerate(a.defs):
        for j, ylab in enumerate(OUTCOMES.values()):
            ax = axes[i, j]
            sub = ta[(ta.definition == dname) & (ta.outcome == ylab)]
            det = sub[sub.assignment.str.startswith("open")]
            ax.scatter(det.beta, np.zeros(len(det)) + 1, s=18, color="0.3",
                       label="country-level assignments")
            for k, p in enumerate((0.25, 0.5, 0.75)):
                r = sub[sub.assignment == f"random p={p}"].iloc[0]
                ax.plot([r["q05"], r["q95"]], [0.5 - 0.15 * k] * 2, lw=3,
                        color="tab:blue", alpha=0.4 + 0.2 * k)
            b = tb[(tb.definition == dname) & (tb.outcome == ylab)].iloc[0]
            ax.plot([b.IM95_lo_cv3, b.IM95_hi_cv3], [-0.3, -0.3], color="k", lw=1.2)
            ax.plot([b.set_lo, b.set_hi], [-0.3, -0.3], color="tab:red", lw=4)
            ax.axvline(0, color="0.6", lw=0.8, ls=":")
            ax.set_yticks([1, 0.35, -0.3],
                          ["by country", "random (5–95%)", "set / IM 95% (CV3)"])
            ax.set_title(f"{dname} — {ylab}", fontsize=9)
            ax.set_xlabel("effect (pp)")
    fig.tight_layout()
    fig.savefig(out / "fig_interpretive_bounds.pdf")
    print(f"  -> {out / 'fig_interpretive_bounds.pdf'}")


if __name__ == "__main__":
    main()
