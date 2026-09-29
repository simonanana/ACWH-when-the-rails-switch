"""
u06_event_study.py — Event study of forum outcomes, imputation estimator and
sensitivity to parallel trends (M4)

A. Case-level repeated cross-section event study
     y_c = Σ_k β_k·1{et_c ∈ k} + α_resp + γ_year (+ eligibility FE) + ε
     k ∈ {≤-4, -3, -2, 0, 1, 2, 3, ≥4}, reference period -1; event dummies are all 0
     for never-treated States.
     CR1 variance matrix (clustered by respondent State).
B. Borusyak–Jaravel–Spiess (2024) imputation estimator
     Estimate α_resp + γ_year (+ eligibility) on untreated cases only, impute Y(0)
     for treated cases, τ_c = y_c − Ŷ_c(0); average by event time and overall.
     Respondent-level cluster-jackknife standard errors.
     Treated States with no untreated case cannot be imputed; the number of
     dropped cases is reported separately.
C. Rambachan–Roth (2023) relative-magnitude restriction (via the project's honest_did.py)
     Robust confidence intervals for the average post-treatment effect for
     M̄ ∈ {0, 0.5, 1, 1.5, 2}.
     Note: the binned end periods (≤-4, ≥4) are not single years, so the
     first-difference restriction is only an approximation for them.

Usage
-----
    python src/u06_event_study.py --defs bloc1_k2 idx10 --out paper_tables
"""
from __future__ import annotations

import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from honest_did import honest_ci  # noqa: E402
from u00_common import (OUTCOMES, PP, TIME, UNIT, Projector, _dummies, attach,  # noqa: E402
                        load_base, load_cohorts, outdir, save)

BINS = [-4, -3, -2, 0, 1, 2, 3, 4]


def event_dummies(x: pd.DataFrame) -> np.ndarray:
    et = x["et"].to_numpy(float)
    cols = []
    for k in BINS:
        if k == -4:
            m = et <= -4
        elif k == 4:
            m = et >= 4
        else:
            m = et == k
        cols.append(np.where(np.isnan(et), 0.0, m.astype(float)))
    return np.column_stack(cols)


def event_study(x: pd.DataFrame, y: str, coding: str) -> tuple[np.ndarray, np.ndarray]:
    absorb = [UNIT, TIME] + ([] if coding == "none" else ["avail_three"])
    P = Projector(x, absorb)
    X = P.resid(event_dummies(x))
    keep = X.std(axis=0) > 1e-10
    Xk = X[:, keep]
    yt = P.resid(x[y].to_numpy(float) * PP)
    XtXi = np.linalg.pinv(Xk.T @ Xk)
    b = XtXi @ (Xk.T @ yt)
    e = yt - Xk @ b
    U = np.zeros((P.G, Xk.shape[1]))
    np.add.at(U, P.cl, Xk * e[:, None])
    k = P.k_fe + Xk.shape[1]
    adj = (P.G / (P.G - 1)) * ((P.n - 1) / max(P.n - k, 1))
    V = adj * XtXi @ (U.T @ U) @ XtXi
    beta = np.full(len(BINS), np.nan)
    beta[keep] = b
    Vf = np.full((len(BINS), len(BINS)), np.nan)
    ii = np.where(keep)[0]
    Vf[np.ix_(ii, ii)] = V
    return beta, Vf


def imputation(x: pd.DataFrame, y: str, coding: str) -> dict:
    absorb = [UNIT, TIME] + ([] if coding == "none" else ["avail_three"])
    D = _dummies(x, absorb)
    yv = x[y].to_numpy(float) * PP
    untreated = x["S"].to_numpy() == 0
    ok_units = set(x.loc[untreated, UNIT])
    imputable = (~untreated) & x[UNIT].isin(ok_units).to_numpy()
    coef = np.linalg.lstsq(D[untreated], yv[untreated], rcond=None)[0]
    tau = yv[imputable] - D[imputable] @ coef
    et = x.loc[imputable, "et"].clip(upper=4).to_numpy()
    out = {"att": float(tau.mean()), "n_imputed": int(imputable.sum()),
           "n_not_imputable": int(((~untreated) & ~imputable).sum())}
    for k in range(0, 5):
        m = et == k
        out[f"att_et{k}{'+' if k == 4 else ''}"] = float(tau[m].mean()) if m.any() else np.nan
    return out


def imputation_jackknife(x: pd.DataFrame, y: str, coding: str) -> dict:
    full = imputation(x, y, coding)
    units = sorted(x[UNIT].unique())
    reps = []
    for u in units:
        xu = x[x[UNIT] != u].reset_index(drop=True)
        if xu.S.sum() == 0:
            continue
        reps.append(imputation(xu, y, coding)["att"])
    reps = np.array(reps)
    G = len(reps)
    se = math.sqrt((G - 1) / G * np.sum((reps - full["att"]) ** 2))
    full["se_jack"] = se
    full["p_jack"] = float(2 * stats.t.sf(abs(full["att"] / se), df=G - 1)) if se > 0 else np.nan
    return full


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defs", nargs="+", default=["bloc1_k2", "idx10"])
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", default="paper_tables")
    ap.add_argument("--n-draw", type=int, default=1000)
    a = ap.parse_args()
    out = outdir(a.out)
    base = load_base(a.ref)
    coh = load_cohorts()

    es_rows, imp_rows, rr_rows = [], [], []
    for dname in a.defs:
        d = attach(base, coh[dname].dropna().to_dict()).reset_index(drop=True)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        for ax, (y, ylab) in zip(axes, OUTCOMES.items()):
            for cod, col, off in (("none", "0.25", -0.1), ("three", "tab:red", 0.1)):
                b, V = event_study(d, y, cod)
                se = np.sqrt(np.diag(V))
                for k, bk, sk in zip(BINS, b, se):
                    es_rows.append({"definition": dname, "outcome": ylab, "coding": cod,
                                    "event_time": k, "beta": bk, "se": sk})
                xs = np.array(BINS) + off
                ax.errorbar(xs, b, yerr=1.96 * se, fmt="o", ms=3, color=col, capsize=2,
                            label="no availability control" if cod == "none"
                            else "three-valued availability")
                # Rambachan–Roth
                ok = np.isfinite(b)
                per = [k for k, o in zip(BINS, ok) if o]
                bb, VV = b[ok], V[np.ix_(ok, ok)]
                if any(p < -1 for p in per) and any(p >= 0 for p in per):
                    for mbar in (0.0, 0.5, 1.0, 1.5, 2.0):
                        ci = honest_ci(per, bb, VV, kind="RM", param=mbar,
                                       n_draw=a.n_draw)
                        rr_rows.append({"definition": dname, "outcome": ylab,
                                        "coding": cod, "Mbar": mbar,
                                        "ci_lo": ci[0] if ci else np.nan,
                                        "ci_hi": ci[1] if ci else np.nan})
                imp = imputation_jackknife(d, y, cod)
                imp.update({"definition": dname, "outcome": ylab, "coding": cod})
                imp_rows.append(imp)
            ax.axhline(0, color="0.6", lw=0.7)
            ax.axvline(-0.5, color="0.6", lw=0.7, ls=":")
            ax.set_xticks(BINS + [-1], ["≤-4", "-3", "-2", "0", "1", "2", "3", "≥4", "-1"])
            ax.set_title(ylab, fontsize=9)
            ax.set_xlabel("years relative to onset")
        axes[0].set_ylabel("pp (95% CI, CR1)")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"Event study ({dname})", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"fig_event_study_{dname}.pdf")
        print(f"  -> {out / f'fig_event_study_{dname}.pdf'}")

    print("\n[u06] Event-study coefficients")
    save(pd.DataFrame(es_rows), out, "tab_event_study.csv")
    print("\n[u06] Imputation estimator (BJS), respondent-level jackknife standard errors")
    save(pd.DataFrame(imp_rows), out, "tab_imputation.csv")
    print("\n[u06] Rambachan–Roth relative-magnitude robust intervals (post-treatment average)")
    save(pd.DataFrame(rr_rows), out, "tab_honest_did_forum.csv")


if __name__ == "__main__":
    main()
