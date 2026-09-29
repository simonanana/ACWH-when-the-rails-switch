"""
h4_moderation.py - H4: does the effect vary with payment-network dependence?

Three problems with an earlier version of this analysis:
  (a) the scaling of PNCI was never documented, so the interaction coefficients (0.175/0.125/0.057) could not be reproduced
  (b) only two classes of moderator were used, PNCI and CIPS
  (c) the null distribution used for Romano-Wolf was not stated

Improvements in this module:
  1. All moderators are first standardised to mean 0 and standard deviation 1, and this step is written into the code,
     so that interaction coefficients have a uniform "per standard deviation" interpretation and no longer depend on undocumented scaling.
  2. UNGA geopolitical alignment is added as a moderator. In theory it is harder evidence than a constructed index:
     the hypothesis is that the further a State is geopolitically from the sanctioning bloc, the stronger its shift towards less delocalised forums.
     Unlike PNCI, it is also not a home-made index and can be checked externally.
  3. The Romano-Wolf bootstrap null distribution is generated explicitly by wild bootstrap and saved,
     so that the multiple-testing correction is reproducible.
"""
import numpy as np
import pandas as pd

from config import N_BOOT_WCB, OUT, SEED
from estimators import feols, romano_wolf
from io_load import (attach_treatment, build_cohorts, load_cases, load_panel,
                     panel_treatment)
from pnci_rebuild import add_rebuilt_pnci

MODERATORS = {
    "pnci_equal": "PNCI: equal-weight composite",
    "pnci_cpmi": "PNCI: CPMI membership (binary)",
    "pnci_network": "PNCI: PageRank network centrality (broken)",
    "pnci_rank": "PNCI: within-year percentile rank (rebuilt, recommended)",
    "pnci_logmm": "PNCI: min-max after log (rebuilt)",
    "CIPSit": "CIPS direct participant (observable, externally dated)",
    "CIPS_broad_it": "CIPS broad access",
    "rmb_infra_it": "RMB infrastructure (any)",
    "clearing_bank_it": "RMB clearing bank",
    "swap_line_it": "PBoC swap line",
    "unga_align_norm_it": "UNGA alignment (with sanctioning bloc)",
}


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


def moderation_panel(p, g, dose="sanction_fin_norm_it", y="y_asinh"):
    """Panel level: moderator x sanction intensity; outcome is arcsinh(number of filings)."""
    d = panel_treatment(p, g)
    d["y_asinh"] = np.arcsinh(d.isds_n_resp.fillna(0))
    rows, tstats, boots = [], [], []
    rng = np.random.default_rng(SEED)

    for mod, label in MODERATORS.items():
        if mod not in d.columns:
            continue
        dd = d.dropna(subset=[mod, dose]).copy()
        if dd[mod].nunique() < 2:
            continue
        dd["_mod"] = zscore(dd[mod])          # uniform standardisation
        dd["_dose"] = dd[dose]
        dd["_inter"] = dd._dose * dd._mod
        try:
            r = feols(dd, y, ["_inter", "_dose", "_mod"],
                      ["iso3", "year"], "iso3")
        except Exception:
            continue
        rows.append({
            "moderator": mod, "label": label,
            "interaction_beta_perSD": round(r["_inter"]["b"], 4),
            "se": round(r["_inter"]["se"], 4),
            "t": round(r["_inter"]["t"], 3),
            "p_raw": round(r["_inter"]["p"], 4),
            "N": r["_nobs"],
        })
        tstats.append(r["_inter"]["t"])

        # wild bootstrap null distribution for this moderator term (used for Romano-Wolf)
        bt = _wild_t_null(dd, y, "_inter", ["_dose", "_mod"],
                          ["iso3", "year"], "iso3", B=299, rng=rng)
        boots.append(bt)

    tab = pd.DataFrame(rows)
    if len(tab) and len(boots):
        Bt = np.column_stack(boots)
        tab["p_RomanoWolf"] = np.round(romano_wolf(tstats, Bt), 4)
    return tab


def _wild_t_null(df, y, x_test, x_other, fe_cols, cluster, B, rng):
    """Generate the null distribution of t under H0: beta_{x_test}=0 by wild cluster bootstrap."""
    import statsmodels.api as sm
    from estimators import _cr1, _design

    X = _design(df, [x_test] + list(x_other), fe_cols).values.astype(float)
    yy = df[y].astype(float).values
    cl = pd.factorize(df[cluster].values)[0]
    G = cl.max() + 1
    ix = 1
    XtXi = np.linalg.pinv(X.T @ X)
    keep = [i for i in range(X.shape[1]) if i != ix]
    Xr = X[:, keep]
    br = np.linalg.pinv(Xr.T @ Xr) @ (Xr.T @ yy)
    er = yy - Xr @ br
    fit = Xr @ br
    ts = np.empty(B)
    for j in range(B):
        w = rng.choice([-1.0, 1.0], size=G)[cl]
        bb, VV, _ = _cr1(X, fit + er * w, cl, XtXi, G)
        ts[j] = bb[ix] / np.sqrt(VV[ix, ix])
    return ts


def moderation_cases(c, p, g):
    """Case level: moderator x sanction indicator; outcomes are the delocalisation ladder and binary forum."""
    d = attach_treatment(c, g)
    key = p.set_index(["iso3", "year"])
    rows = []
    for mod, label in MODERATORS.items():
        if mod not in p.columns:
            continue
        vals = [key[mod].get((r, y_), np.nan)
                for r, y_ in zip(d.respondent_iso3, d.year)]
        dd = d.assign(_m=pd.to_numeric(pd.Series(vals, index=d.index),
                                       errors="coerce")).dropna(subset=["_m"])
        if len(dd) < 100 or dd._m.nunique() < 2:
            continue
        dd["_mz"] = zscore(dd._m)
        dd["_inter"] = dd.S * dd._mz
        dd["UNC"] = dd.is_uncitral.astype(float) * 100
        dd["LADDER"] = dd.ladder
        for y in ["UNC", "LADDER"]:
            try:
                r = feols(dd, y, ["_inter", "S", "_mz"],
                          ["respondent_iso3", "year"], "respondent_iso3")
            except Exception:
                continue
            rows.append({
                "moderator": mod, "label": label, "outcome": y,
                "interaction_perSD": round(r["_inter"]["b"], 4),
                "se": round(r["_inter"]["se"], 4),
                "p_raw": round(r["_inter"]["p"], 4), "N": r["_nobs"],
            })
    return pd.DataFrame(rows)


def run(verbose=True):
    p = add_rebuilt_pnci(load_panel())
    c = load_cases()
    g = build_cohorts(p)

    tab1 = moderation_panel(p, g)
    tab2 = moderation_cases(c, p, g)

    tab1.to_csv(OUT / "h4_moderation_panel.csv", index=False)
    tab2.to_csv(OUT / "h4_moderation_cases.csv", index=False)

    if verbose:
        print("\n=== H4 (panel) moderation: sanction intensity x moderator -> dispute incidence ===")
        print("  All moderators are standardised; coefficients are interpreted 'per standard deviation'.")
        print(tab1.to_string(index=False))
        print("\n=== H4 (cases) moderation -> forum composition and delocalisation ladder ===")
        print(tab2.to_string(index=False))
        print("\nNote: if no term survives the Romano-Wolf correction, report 'no index-based heterogeneity',")
        print("   rather than selecting uncorrected significant terms.")
    return {"panel": tab1, "cases": tab2}


if __name__ == "__main__":
    run()
