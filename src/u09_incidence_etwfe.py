"""
u09_incidence_etwfe.py — Incidence: extended two-way fixed effects Poisson
(M6, Wooldridge 2023)

Model
-----
    E[N_it | ·] = exp(α_i + γ_t + Σ_{g} Σ_{s ≥ g} τ_{gs} · 1{G_i = g, t = s})
N_it is the number of new cases against State i as respondent in year t (rebuilt from
the case file and checked against the panel's isds_n_resp). τ_gs is estimated
separately for each (cohort, post-treatment year), so never-treated and not-yet-treated
States serve only as controls and the "forbidden comparisons" of staggered DiD do
not arise.

Aggregation (on the ratio scale, weighted by treated observations)
    μ1 = exp(η̂),  μ0 = exp(η̂ − τ̂_gs)
    RR  = Σ μ1 / Σ μ0             (incidence rate ratio averaged over the treated)
    ATT = mean(μ1 − μ0)           (level effect in cases per State-year)
Inference: country-level cluster bootstrap (B draws; States drawn more than once are
relabelled).
Also reported: TWFE-Poisson with a single post-treatment dummy (reference), a TOST on
      log RR (bounds [2/3, 3/2]), and the MDE at 80% power (converted to a rate ratio).

Separation and cell granularity (important)
    A fully saturated τ_gs (each cohort × each post-treatment year) often yields cells
    whose counts are all zero under sparse counts. The MLE of τ_gs then does not exist
    (→ −∞) and the cell must be dropped; but the dropped cells are exactly those with
    the most negative effects, which biases the aggregate RR **upwards**. The default
    (--modes etwfe) therefore uses **cohort-level** τ_g (heterogeneous across cohorts,
    constant across years within a cohort), which still avoids the forbidden
    comparisons of staggered DiD; the fully saturated version is kept as etwfe_cell
    for comparison, and the number of dropped observations is reported.
    States with zero counts in every year have an unidentified country FE and are
    always dropped (the number of States is reported).

Usage
-----
    python src/u09_incidence_etwfe.py --defs bloc1_k2 idx10 transatlantic --B 299 --out paper_tables
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
from scipy import stats

from u00_common import UNIT, append_canonical, load_cohorts, outdir, save


LABELS = {"etwfe": "ETWFE-Poisson (cohort x post)",
          "etwfe_cell": "ETWFE-Poisson (cohort x year; separation-prone)",
          "twfe": "TWFE-Poisson (single post dummy)"}


def ppml_dense(y: np.ndarray, X: np.ndarray, maxiter: int = 100, tol: float = 1e-9):
    beta = np.zeros(X.shape[1])
    beta[0] = math.log(max(y.mean(), 1e-8))
    for _ in range(maxiter):
        eta = np.clip(X @ beta, -30, 30)
        mu = np.exp(eta)
        z = eta + (y - mu) / np.maximum(mu, 1e-10)
        W = mu
        A = X.T @ (X * W[:, None])
        new = np.linalg.lstsq(A, X.T @ (W * z), rcond=None)[0]
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    return beta


def build_design(p: pd.DataFrame, cmap: dict, mode: str, require_pre: bool = True):
    d = p.copy()
    d["g"] = d.iso3.map(cmap)
    d = d[d.groupby("iso3").N.transform("sum") > 0].reset_index(drop=True)
    n_no_pre = 0
    if require_pre:
        # States with zero counts in every pre-treatment year: their country fixed
        # effect can only be identified from post-treatment observations, and τ absorbs
        # the entire level difference (quasi-separation, τ → +∞). Such States provide
        # no within-unit identification and must be dropped; otherwise the aggregate
        # RR is dominated by degenerate cohorts (see the separation note in the docstring).
        pre_ok = d.assign(pre=np.where(d.g.isna() | (d.year < d.g), d.N, 0.0)) \
            .groupby("iso3").pre.sum()
        treated_units = d.loc[d.g.notna(), "iso3"].unique()
        bad = [u for u in treated_units if pre_ok.get(u, 0) == 0]
        n_no_pre = len(bad)
        d = d[~d.iso3.isin(bad)].reset_index(drop=True)
    cells = []
    if mode in ("etwfe", "etwfe_cell"):
        post = d.g.notna() & (d.year >= d.g)
        lab = (d.g.astype("Int64").astype(str) if mode == "etwfe"
               else d.g.astype("Int64").astype(str) + "_" + d.year.astype(str))
        d["cell"] = np.where(post, lab, "")
        zero = d[post].groupby("cell").N.sum().pipe(lambda s: s[s == 0]).index
        n_drop = int(d.cell.isin(zero).sum())
        d = d[~d.cell.isin(zero)].reset_index(drop=True)
        cells = sorted(c for c in d.cell.unique() if c)
    else:
        n_drop = 0
        d["D"] = (d.g.notna() & (d.year >= d.g)).astype(float)
    units = sorted(d.iso3.unique())
    years = sorted(d.year.unique())
    n = len(d)
    blocks = [np.ones((n, 1))]
    blocks.append((d.iso3.to_numpy()[:, None] == np.array(units[1:])[None, :]).astype(float))
    blocks.append((d.year.to_numpy()[:, None] == np.array(years[1:])[None, :]).astype(float))
    if mode in ("etwfe", "etwfe_cell"):
        C = (d.cell.to_numpy()[:, None] == np.array(cells)[None, :]).astype(float)
        blocks.append(C)
    else:
        blocks.append(d[["D"]].to_numpy())
    X = np.hstack(blocks)
    k_tau = len(cells) if mode in ("etwfe", "etwfe_cell") else 1
    return d, X, k_tau, n_drop, n_no_pre


def aggregate(d, X, beta, k_tau):
    eta = X @ beta
    tau = X[:, -k_tau:] @ beta[-k_tau:]
    treated = X[:, -k_tau:].sum(axis=1) > 0
    mu1 = np.exp(eta[treated])
    mu0 = np.exp(eta[treated] - tau[treated])
    return {"RR": float(mu1.sum() / mu0.sum()), "ATT_level": float((mu1 - mu0).mean()),
            "n_treated_obs": int(treated.sum())}


def fit_once(p, cmap, mode, require_pre=True):
    d, X, k_tau, n_drop, n_no_pre = build_design(p, cmap, mode, require_pre)
    beta = ppml_dense(d.N.to_numpy(float), X)
    if mode in ("etwfe", "etwfe_cell"):
        agg = aggregate(d, X, beta, k_tau)
    else:
        agg = {"RR": float(math.exp(beta[-1])), "ATT_level": np.nan,
               "n_treated_obs": int(d.D.sum())}
    agg.update({"n_obs": len(d), "n_cells": k_tau, "n_dropped_zero_cells": n_drop,
                "n_treated_dropped_no_pre": n_no_pre, "n_countries": d.iso3.nunique()})
    return agg


def bootstrap(p, cmap, mode, B, seed, require_pre=True):
    rng = np.random.default_rng(seed)
    units = p.iso3.unique()
    idx = {u: np.where(p.iso3.to_numpy() == u)[0] for u in units}
    out = []
    for _ in range(B):
        pick = rng.choice(units, size=len(units), replace=True)
        parts, cm = [], {}
        for k, u in enumerate(pick):
            q = p.iloc[idx[u]].copy()
            q["iso3"] = f"{u}#{k}"
            parts.append(q)
            if u in cmap:
                cm[f"{u}#{k}"] = cmap[u]
        bs = pd.concat(parts, ignore_index=True)
        try:
            r = fit_once(bs, cm, mode, require_pre)
            if np.isfinite(r["RR"]) and r["n_treated_obs"] > 0:
                out.append(r)
        except np.linalg.LinAlgError:
            continue
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defs", nargs="+", default=["bloc1_k2", "idx10", "transatlantic"])
    ap.add_argument("--B", type=int, default=299)
    ap.add_argument("--modes", nargs="+", default=["etwfe", "twfe"],
                    choices=["etwfe", "etwfe_cell", "twfe"])
    ap.add_argument("--out", default="paper_tables")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--allow-no-pre", action="store_true",
                    help="keep States with zero pre-treatment counts "
                         "(produces quasi-separation; for diagnostics only)")
    a = ap.parse_args()
    out = outdir(a.out)

    from io_load import load_cases, load_panel
    panel = load_panel()[["iso3", "year", "isds_n_resp"]]
    cases = load_cases()
    cnt = cases.groupby([UNIT, "year"]).size().rename("N_cases").reset_index() \
        .rename(columns={UNIT: "iso3"})
    p = panel.merge(cnt, on=["iso3", "year"], how="left").fillna({"N_cases": 0})
    lost = cases[~cases[UNIT].isin(panel.iso3.unique())]
    print(f"[u09] panel isds_n_resp total {int(p.isds_n_resp.sum())}, rebuilt from case file "
          f"{int(p.N_cases.sum())}; cases outside the 170-economy panel: {len(lost)} "
          f"(respondent States: {sorted(lost[UNIT].unique())[:15]})")
    mism = (p.isds_n_resp != p.N_cases).sum()
    print(f"      State-years where the two disagree: {int(mism)}")
    p["N"] = p.N_cases.astype(float)

    coh = load_cohorts()
    rows, canon = [], []
    for dname in a.defs:
        cmap = coh[dname].dropna().to_dict()
        for mode in a.modes:
            est = fit_once(p, cmap, mode, not a.allow_no_pre)
            bs = bootstrap(p, cmap, mode, a.B, a.seed, not a.allow_no_pre)
            lrr = np.log(bs.RR.to_numpy())
            se_l = float(lrr.std(ddof=1))
            l0 = math.log(est["RR"])
            p_tost = max(stats.norm.sf((l0 - math.log(2 / 3)) / se_l),
                         stats.norm.cdf((l0 - math.log(1.5)) / se_l))
            mde = 2.8 * se_l
            rec = {"definition": dname, "estimator": LABELS[mode], **est,
                   "RR_ci_lo": float(np.percentile(bs.RR, 2.5)),
                   "RR_ci_hi": float(np.percentile(bs.RR, 97.5)),
                   "se_logRR_boot": se_l,
                   "p_RR_ne_1": float(2 * stats.norm.sf(abs(l0) / se_l)),
                   "p_TOST_[2/3,3/2]": float(p_tost),
                   "MDE80_RR_up": math.exp(mde), "MDE80_RR_down": math.exp(-mde),
                   "B_ok": len(bs)}
            if mode != "twfe":
                rec["ATT_level_ci_lo"] = float(np.percentile(bs.ATT_level, 2.5))
                rec["ATT_level_ci_hi"] = float(np.percentile(bs.ATT_level, 97.5))
            rows.append(rec)
            print(f"  {dname:<14} {mode:<6} RR={est['RR']:.3f} "
                  f"[{rec['RR_ci_lo']:.2f}, {rec['RR_ci_hi']:.2f}]  "
                  f"p_TOST={p_tost:.3f}  MDE80 RR ∈ [{rec['MDE80_RR_down']:.2f}, "
                  f"{rec['MDE80_RR_up']:.2f}]")
            canon += [(f"{dname} {mode} RR", est["RR"]),
                      (f"{dname} {mode} RR ci_lo", rec["RR_ci_lo"]),
                      (f"{dname} {mode} RR ci_hi", rec["RR_ci_hi"])]
    print("\n[u09] Incidence (ratio scale)")
    save(pd.DataFrame(rows), out, "tab_incidence_etwfe.csv")
    append_canonical(canon, out)


if __name__ == "__main__":
    main()
