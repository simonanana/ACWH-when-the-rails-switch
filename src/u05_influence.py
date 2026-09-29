"""
u05_influence.py — Which States carry the estimate (M3)

1. Leave-one-State-out (all treated States, not only switching States): β_(-g),
   in two variants: no control and three-valued control;
2. CV3 cluster-jackknife standard errors (consistent with u03);
3. Country effects: S × 1{State j} interactions (each identified from that State's
   own pre- and post-treatment cases).
   **Only point estimates and raw counts are reported, no per-State confidence
   intervals**: cluster-level inference is not possible for a single cluster.
4. Forest plot: one row per treated State, showing β after dropping that State
   (under both controls).

Usage
-----
    python src/u05_influence.py --defs bloc1_k2 idx10 --out paper_tables
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from u00_common import (OUTCOMES, PP, TIME, UNIT, Projector, attach, cv3,  # noqa: E402
                        load_base, load_cohorts, outdir, save)


def country_effects(x: pd.DataFrame, y: str, coding: str) -> pd.DataFrame:
    """One S×1{j} interaction per switching State, estimated in a single regression;
    reports the coefficients and the raw 2×2 counts."""
    sw = sorted(x.groupby(UNIT)["S"].nunique().pipe(lambda s: s[s > 1]).index)
    absorb = [UNIT, TIME] + ([] if coding == "none" else ["avail_three"])
    P = Projector(x.reset_index(drop=True), absorb)
    Xs = np.column_stack([(x["S"] * (x[UNIT] == u)).to_numpy(float) for u in sw])
    Xt = P.resid(Xs)
    yt = P.resid(x[y].to_numpy(float) * PP)
    b = np.linalg.lstsq(Xt, yt, rcond=None)[0]
    rows = []
    for k, u in enumerate(sw):
        g = x[x[UNIT] == u]
        pre, post = g[g.S == 0], g[g.S == 1]
        rows.append({"iso3": u, "coding": coding, "beta_country": b[k],
                     "n_pre": len(pre), "n_post": len(post),
                     "rate_pre": pre[y].mean() * PP, "rate_post": post[y].mean() * PP,
                     "conv_available_pre": pre.convention_status.eq("available").mean(),
                     "conv_available_post": post.convention_status.eq("available").mean()})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defs", nargs="+", default=["bloc1_k2", "idx10"])
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", default="paper_tables")
    a = ap.parse_args()
    out = outdir(a.out)
    base = load_base(a.ref)
    coh = load_cohorts()

    loo_rows, ce_rows = [], []
    for dname in a.defs:
        d = attach(base, coh[dname].dropna().to_dict()).reset_index(drop=True)
        treated = sorted(d.loc[d.S == 1, UNIT].unique())
        for y, ylab in OUTCOMES.items():
            res = {}
            for cod in ("none", "three"):
                res[cod] = cv3(d, y, coding=cod)
            for u in ["(none)"] + treated:
                rec = {"definition": dname, "outcome": ylab, "dropped": u,
                       "switcher": bool(u != "(none)" and d[d[UNIT] == u].S.nunique() > 1)}
                for cod in ("none", "three"):
                    b = res[cod]["coef"] if u == "(none)" else res[cod]["loo"].get(u, res[cod]["coef"])
                    rec[f"beta_{cod}"] = b
                    rec[f"dfbeta_{cod}"] = b - res[cod]["coef"]
                    rec[f"se_cv3_{cod}"] = res[cod]["se_cv3"]
                loo_rows.append(rec)
            for cod in ("none", "three"):
                ce = country_effects(d, y, cod)
                ce.insert(0, "outcome", ylab)
                ce.insert(0, "definition", dname)
                ce_rows.append(ce)

    loo = pd.DataFrame(loo_rows)
    ce = pd.concat(ce_rows, ignore_index=True)
    print("[u05] Leave-one-State-out (all treated States)")
    save(loo, out, "tab_influence_loo.csv")
    print("\n[u05] Country effects (point estimates + raw counts; no per-State confidence intervals)")
    save(ce, out, "tab_country_effects.csv")

    for dname in a.defs:
        sub = loo[loo.definition == dname]
        fig, axes = plt.subplots(1, 2, figsize=(9, 0.32 * sub.dropped.nunique() + 1.2),
                                 sharey=True)
        for ax, ylab in zip(axes, OUTCOMES.values()):
            s = sub[sub.outcome == ylab].reset_index(drop=True)
            ypos = np.arange(len(s))[::-1]
            ax.scatter(s.beta_none, ypos + 0.12, marker="o", color="0.25",
                       label="no availability control")
            ax.scatter(s.beta_three, ypos - 0.12, marker="s", color="tab:red",
                       label="three-valued availability")
            full = s[s.dropped == "(none)"].iloc[0]
            ax.axvline(full.beta_none, color="0.25", lw=0.7, ls="--")
            ax.axvline(full.beta_three, color="tab:red", lw=0.7, ls="--")
            ax.axvline(0, color="0.6", lw=0.7, ls=":")
            labels = [f"{u}{'*' if sw else ''}" for u, sw in zip(s.dropped, s.switcher)]
            ax.set_yticks(ypos, labels, fontsize=8)
            ax.set_title(ylab, fontsize=9)
            ax.set_xlabel("β after dropping the state (pp)")
        axes[0].legend(fontsize=7, loc="lower left")
        fig.suptitle(f"Influence of each treated respondent ({dname}); * = switching state",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"fig_influence_{dname}.pdf")
        print(f"  -> {out / f'fig_influence_{dname}.pdf'}")


if __name__ == "__main__":
    main()
