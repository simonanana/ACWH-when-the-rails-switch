"""
r12_pnci_rebuild.py - repair pnci_network and re-estimate H4 / H3
=====================================================================
Diagnosis (traced in the source of notebooks 00/03)
------------------------------------
`pnci_network` is constructed in notebook 00, cell[19]:
    net['pnci_network_new'] = net.groupby('year')['pr_new'].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()))     # within-year min-max
and then written into the panel via `fill_only()` - whose semantics are
**"fill NaN only, never overwrite existing values"**.

Two compounding consequences make the column unusable:

[A] min-max meets the heavy-tailed PageRank distribution -> 90% of the sample is squashed to 0
      Distribution of pr in 2018: min=0.00353, median=0.00354, max=0.08385
      => min-max value of the median country = (0.00354-0.00353)/(0.08385-0.00353) = **0.0001**
      This is not "missing proxy values"; the normalisation itself has squashed the variable into a point mass at 0.

[B] fill_only does not overwrite -> two normalisations mixed in the same column
      src=''    448 obs (32 countries): values left over from the old panel, mean 0.288
      src='wdi' 1900 obs (138 countries): min-max values from this run, mean 0.0002
      (Note: 'wdi' is an audit label hard-coded in fill_only and unrelated to the actual source;
        these 1900 cells in fact come from BIS PageRank, so the label is misleading.)

=> `pnci_network` = 448 cells on scale A + 1900 cells on scale B. **The two scales are not comparable.**
   The abstract describes it as "PageRank centrality for 170 countries", but for 138 countries it is in fact approximately the constant 0.

Fix
----
The raw `pr` column is clean (2348 obs, genuine PageRank). Three normalisations are rebuilt from it:
    rank  : within-year percentile rank      <- (key) recommended; robust to heavy tails and interpretable
    logmm : within-year min-max of log(pr)   <- retains cardinal information
    z     : within-year z-score
H4 is re-estimated with these side by side with the legacy column, together with a Romano-Wolf multiple-testing correction.

Outputs: p1_pnci_variants.csv, p2_h4_rebuilt.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

# --- Path resolution: defaults ./data/ and ./out_fix/, both overridable from the command line ---
import argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ap = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else "")
_ap.add_argument("--panel", default=None, help="panel parquet path (default ./data/analysis_panel_v4.parquet)")
_ap.add_argument("--cases", default=None, help="case parquet path (default ./data/cases_caselevel_final.parquet)")
_ap.add_argument("--out",   default=None, help="output directory (default ./out_fix)")
_args, _ = _ap.parse_known_args()

PANEL = Path(_args.panel) if _args.panel else _HERE / "data" / "analysis_panel_v4.parquet"
CASES = Path(_args.cases) if _args.cases else _HERE / "data" / "cases_caselevel_final.parquet"
OUT   = Path(_args.out)   if _args.out   else _HERE / "out_fix"
OUT.mkdir(parents=True, exist_ok=True)


def _need(p, what):
    if not Path(p).exists():
        raise SystemExit(
            f"\n[error] cannot find {what}: {Path(p).resolve()}\n"
            f"       place the file under {_HERE / 'data'}, or specify the path with --panel / --cases.\n")
# ------------------------------------------------------------------------


# ---------------- minimal HDFE OLS (cluster-robust) ----------------
def _gm(v, c, k):
    n = np.bincount(c, weights=v, minlength=k)
    d = np.bincount(c, minlength=k)
    return (n / np.where(d == 0, 1, d))[c]


def _demean(M, fes):
    M = np.asarray(M, float); one = M.ndim == 1
    if one: M = M[:, None]
    ng = [int(c.max()) + 1 for c in fes]; out = M.copy()
    for j in range(out.shape[1]):
        v = out[:, j]
        for _ in range(3000):
            v0 = v
            for c, k in zip(fes, ng): v = v - _gm(v, c, k)
            if np.max(np.abs(v - v0)) < 1e-11 * max(1, np.max(np.abs(v0))): break
        out[:, j] = v
    return out[:, 0] if one else out


def ols(df, y, X, fes, cl):
    d = df[list(dict.fromkeys([y] + X + fes + [cl]))].dropna()
    fc = [pd.factorize(d[f])[0] for f in fes]
    yd = _demean(d[y].to_numpy(float), fc); Xd = _demean(d[X].to_numpy(float), fc)
    XtXi = np.linalg.pinv(Xd.T @ Xd)
    b = XtXi @ (Xd.T @ yd); r = yd - Xd @ b
    g = pd.factorize(d[cl])[0]; G = int(g.max()) + 1
    su = Xd * r[:, None]; S = np.zeros((G, Xd.shape[1]))
    for j in range(Xd.shape[1]):
        S[:, j] = np.bincount(g, weights=su[:, j], minlength=G)
    V = XtXi @ (S.T @ S) @ XtXi * (G / (G - 1))
    se = np.sqrt(np.maximum(np.diag(V), 0)); t = b / se
    return b, se, 2 * stats.t.sf(np.abs(t), G - 1), len(d), G, r, Xd, g


def wcr(df, y, X, fes, cl, test, B=999, seed=7):
    """restricted wild cluster bootstrap-t"""
    rng = np.random.default_rng(seed)
    b, se, p, N, G, _, _, _ = ols(df, y, X, fes, cl)
    j = X.index(test); t_obs = b[j] / se[j]
    Xr = [v for v in X if v != test]
    d = df[list(dict.fromkeys([y] + X + fes + [cl]))].dropna().copy()
    _, _, _, _, _, r0, _, g = ols(d, y, Xr, fes, cl)
    ts = []
    for _ in range(B):
        d["_ys"] = r0 * rng.choice([-1.0, 1.0], size=int(g.max()) + 1)[g]
        try:
            bb, ss, _, _, _, _, _, _ = ols(d, "_ys", X, fes, cl)
            ts.append(bb[j] / ss[j])
        except Exception:
            pass
    ts = np.array(ts)
    return float(b[j]), float(se[j]), float(p[j]), \
        float((1 + np.sum(np.abs(ts) >= abs(t_obs))) / (1 + len(ts))), ts, t_obs


def romano_wolf(t_obs, t_boot):
    t_obs = np.abs(t_obs); t_boot = np.abs(t_boot)
    k = len(t_obs); order = np.argsort(-t_obs); out = np.empty(k); prev = 0.0
    rem = list(order)
    for idx in order:
        mx = np.nanmax(t_boot[:, rem], axis=1)
        p = max((1 + np.nansum(mx >= t_obs[idx])) / (1 + len(mx)), prev)
        out[idx] = p; prev = p; rem = rem[1:]
    return out


def main():
    _need(PANEL, "panel file")
    d = pd.read_parquet(PANEL)
    print(f"[load] {d.shape}")

    # ---------------- rebuild three normalisations ----------------
    pr = pd.to_numeric(d["pr"], errors="coerce")
    d["pnci_rank"] = pr.groupby(d.year).transform(lambda s: s.rank(pct=True))
    lp = np.log(pr.where(pr > 0))
    d["pnci_logmm"] = lp.groupby(d.year).transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else np.nan)
    d["pnci_z"] = pr.groupby(d.year).transform(
        lambda s: (s - s.mean()) / s.std() if s.std() else np.nan)

    VAR = ["pnci_network_filled", "pnci_rank", "pnci_logmm", "pnci_z",
           "pnci_equal", "pnci_cpmi"]
    rows = []
    for c in VAR:
        s = pd.to_numeric(d[c], errors="coerce")
        rows.append({"variable": c, "n": int(s.notna().sum()), "mean": s.mean(),
                     "sd": s.std(), "median": s.median(),
                     "share<0.01": float((s < 0.01).mean()),
                     "n_time_varying_countries": int((d.groupby("iso3")[c].std() > 1e-9).sum())})
    v = pd.DataFrame(rows)
    v.to_csv(OUT / "p1_pnci_variants.csv", index=False)
    print("\n[PNCI variants compared]\n", v.round(4).to_string(index=False))
    print("\n  (key) legacy pnci_network_filled has >80% of observations < 0.01 (point mass at 0);")
    print("    rank / logmm have well-behaved distributions after the rebuild.")

    print("\n  Top countries (2018):")
    for c in ["pnci_network_filled", "pnci_rank", "pnci_logmm"]:
        top = d[d.year == 2018].nlargest(6, c)[["iso3", c]]
        print(f"    {c:22s}", {r.iso3: round(getattr(r, c), 3) for r in top.itertuples()})

    # ---------------- H4 re-estimation ----------------
    S = "sanction_fin_norm_it"
    d["_S"] = pd.to_numeric(d[S], errors="coerce")
    tests = ["pnci_equal", "pnci_cpmi", "pnci_network_filled",
             "pnci_rank", "pnci_logmm"]
    res, tb, tobs = [], [], []
    for c in tests:
        z = pd.to_numeric(d[c], errors="coerce")
        d["_P"] = (z - z.mean()) / z.std()          # uniform z-standardisation, so coefficients are comparable
        d["_X"] = d["_S"] * d["_P"]
        b, se, p, pw, ts, t_o = wcr(d, "isds_ihs_resp", ["_S", "_P", "_X"],
                                    ["iso3", "year"], "iso3", "_X", B=999)
        res.append({"PNCI variant": c, "interaction coef (z-standardised)": b, "se": se,
                    "p_CR1": p, "p_wild_cluster": pw})
        tb.append(ts); tobs.append(t_o)
    L = min(len(x) for x in tb)
    p_rw = romano_wolf(np.array(tobs), np.column_stack([x[:L] for x in tb]))
    h4 = pd.DataFrame(res); h4["p_RomanoWolf"] = p_rw
    h4.to_csv(OUT / "p2_h4_rebuilt.csv", index=False)
    print("\n[H4 re-estimated: uniform z-standardisation + three inference methods + Romano-Wolf]\n",
          h4.round(4).to_string(index=False))
    print("\n  Reading: only variants with p<0.05 after the Romano-Wolf correction may be reported as a finding.")
    print(f"\n[output] {OUT}")


if __name__ == "__main__":
    main()
