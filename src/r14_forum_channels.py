"""
r14_forum_channels.py - (key) linking H2 (forum) to the demand/supply framework
=====================================================================
Motivation
----------
The paper currently contains two disconnected empirical parts:
  Section 3  forum choice: sanctioned respondent State -> UNCITRAL +25.5pp (the only robustly significant finding)
  Section 5  bilateral gravity: supply differential -0.49 to -0.65, none significant (DV = number of filings)
The low power in Section 5 stems from the DV: only 837 of the 142,562 cells in the bilateral panel are non-zero.

But the data contain **forum choice** for 958 cases, and the forum is precisely the strongest signal.
Moving the test of the supply channel from "counts" to "forum" shifts the source of power from 837 non-zero cells
to 958 cases, on the same causal chain as the main finding.

This module estimates
----------
    y_c = b1*S_j(c)t(c) + b2*B_i(c) + b3*(B_i x S_j) + FE + e

  y_c   whether case c proceeds under UNCITRAL / non-ICSID rules
  S_jt  respondent State j is under financial sanctions in year t (after first reaching intensity >= 0.10)
  B_i   investor's home State i belongs to the sanctioning bloc (default B1: EU27+GBR+CHE+NOR+USA)

  b1  demand-side forum effect (forum shift for sanctioned States as a whole)
  b3 (key) supply differential: for **the same sanctioned respondent State**, whether investors from the sanctioning bloc
      are more likely than investors from outside it to use non-ICSID forums.
      This is exactly the parameter Section 5 set out to test but lacked the power for, moved to the forum margin.

Fixed effects are tightened step by step:
    (1) year FE                                   - coarsest
    (2) year FE + respondent State FE             - within-group identification of b1 (= Section 3 reference specification)
    (3) year FE + respondent FE + home State FE   - also absorbs home-State litigation culture/preferences
    (4) respondent State x year FE                - b1 is absorbed, **only b3 is identified**,
                                                    i.e. comparing home States within the same respondent-year
                                                    => the cleanest estimate of the supply differential

Additionally: re-estimation of H4 on the forum margin (PNCI triple interaction).
H4 is conclusively dead on filing counts (all p>0.30 after Romano-Wolf), but has never been tried on the forum.

Outputs: f1_forum_channels.csv, f2_h4_on_forum.csv, f3_cells.csv
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# --- Path resolution: defaults ./data/ and ./out_fix2/, both overridable from the command line ---
import argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ap = argparse.ArgumentParser(description="Demand/supply channel decomposition on the forum margin")
_ap.add_argument("--panel", default=None, help="panel parquet (default ./data/analysis_panel_v4.parquet)")
_ap.add_argument("--cases", default=None, help="case parquet (default ./data/cases_caselevel_final.parquet)")
_ap.add_argument("--out", default=None, help="output directory (default ./out_fix2)")
_ap.add_argument("--boot", type=int, default=999, help="number of wild cluster bootstrap replications (default 999)")
_args, _ = _ap.parse_known_args()

PANEL = Path(_args.panel) if _args.panel else _HERE / "data" / "analysis_panel_v4.parquet"
CASES = Path(_args.cases) if _args.cases else _HERE / "data" / "cases_caselevel_final.parquet"
OUT = Path(_args.out) if _args.out else _HERE / "out_fix2"
OUT.mkdir(parents=True, exist_ok=True)


def _need(p, what):
    if not Path(p).exists():
        raise SystemExit(f"\n[error] cannot find {what}: {Path(p).resolve()}\n"
                         f"       place it under {_HERE/'data'}, or specify with --panel / --cases.\n")
# ------------------------------------------------------------------------

EU27 = {"AUT","BEL","BGR","HRV","CYP","CZE","DNK","EST","FIN","FRA","DEU","GRC",
        "HUN","IRL","ITA","LVA","LTU","LUX","MLT","NLD","POL","PRT","ROU","SVK",
        "SVN","ESP","SWE"}
B1 = EU27 | {"GBR", "CHE", "NOR", "USA"}          # reference specification (legal-services ban core)
WEST_RULES = {"ICSID", "ICSID_AF", "SCC", "ICC"}


# ---------------- minimal HDFE OLS + cluster-robust standard errors ----------------
def _gm(v, c, k):
    n = np.bincount(c, weights=v, minlength=k); d = np.bincount(c, minlength=k)
    return (n / np.where(d == 0, 1, d))[c]


def _dm(M, fes):
    M = np.asarray(M, float); one = M.ndim == 1
    if one: M = M[:, None]
    ng = [int(c.max()) + 1 for c in fes]; o = M.copy()
    for j in range(o.shape[1]):
        v = o[:, j]
        for _ in range(3000):
            v0 = v
            for c, k in zip(fes, ng): v = v - _gm(v, c, k)
            if np.max(np.abs(v - v0)) < 1e-11 * max(1, np.max(np.abs(v0))): break
        o[:, j] = v
    return o[:, 0] if one else o


def ols(df, y, X, fes, cl):
    d = df[list(dict.fromkeys([y] + X + fes + [cl]))].dropna()
    fc = [pd.factorize(d[f])[0] for f in fes]
    yd = _dm(d[y].to_numpy(float), fc); Xd = _dm(d[X].to_numpy(float), fc)
    keep = Xd.std(0) > 1e-11
    Xd = Xd[:, keep]; names = [n for n, k in zip(X, keep) if k]
    if Xd.shape[1] == 0:
        return None
    XtXi = np.linalg.pinv(Xd.T @ Xd); b = XtXi @ (Xd.T @ yd); r = yd - Xd @ b
    g = pd.factorize(d[cl])[0]; G = int(g.max()) + 1
    su = Xd * r[:, None]; S = np.zeros((G, Xd.shape[1]))
    for j in range(Xd.shape[1]):
        S[:, j] = np.bincount(g, weights=su[:, j], minlength=G)
    V = XtXi @ (S.T @ S) @ XtXi * (G / (G - 1))
    se = np.sqrt(np.maximum(np.diag(V), 0)); t = b / se
    return dict(names=names, b=b, se=se, p=2 * stats.t.sf(np.abs(t), G - 1),
                t=t, N=len(d), G=G, resid=r, d=d, fc=fc, dropped=[n for n, k in zip(X, keep) if not k])


def wcr(df, y, X, fes, cl, test, B=999, seed=11):
    rng = np.random.default_rng(seed)
    full = ols(df, y, X, fes, cl)
    if full is None or test not in full["names"]:
        return np.nan
    j = full["names"].index(test); t_obs = full["t"][j]
    Xr = [v for v in X if v != test]
    d = df[list(dict.fromkeys([y] + X + fes + [cl]))].dropna().copy()
    r0 = ols(d, y, Xr, fes, cl)
    if r0 is None:
        return np.nan
    g = pd.factorize(d[cl])[0]; ts = []
    for _ in range(B):
        d["_ys"] = r0["resid"] * rng.choice([-1.0, 1.0], size=int(g.max()) + 1)[g]
        rb = ols(d, "_ys", X, fes, cl)
        if rb is not None and test in rb["names"]:
            ts.append(rb["t"][rb["names"].index(test)])
    ts = np.array(ts)
    return float((1 + np.sum(np.abs(ts) >= abs(t_obs))) / (1 + len(ts))) if len(ts) else np.nan


def bucket_rules(s):
    u = s.astype(str).str.upper().str.strip()
    o = pd.Series("UNKNOWN", index=s.index, dtype=object)
    o = o.mask(u.str.contains("UNCITRAL", na=False), "UNCITRAL")
    o = o.mask(u.str.contains("SCC", na=False), "SCC")
    o = o.mask(u.str.startswith("ICC", na=False), "ICC")
    o = o.mask(u.str.contains("MCCI", na=False), "MCCI")
    o = o.mask(u.str.contains("NONE", na=False), "AD_HOC")
    o = o.mask(u.str.contains("ICSID", na=False), "ICSID")
    o = o.mask(u.str.contains("ICSID AF", na=False), "ICSID_AF")
    o = o.mask(u.str.contains("DATA NOT AVAILABLE", na=False), "UNKNOWN")
    return o


def main():
    _need(PANEL, "panel file"); _need(CASES, "case file")
    p = pd.read_parquet(PANEL); c = pd.read_parquet(CASES)
    c["year"] = pd.to_numeric(c["year"], errors="coerce")
    c = c.dropna(subset=["year"]); c["year"] = c.year.astype(int)
    c = c[(c.year >= 2010) & (c.year <= 2023)].reset_index(drop=True)
    for k in ("respondent_iso3", "claimant_iso3"):
        c[k] = c[k].astype(str).str.upper().str.strip()

    c["forum"] = bucket_rules(c["rules"])
    c["is_uncitral"] = (c.forum == "UNCITRAL").astype(float)
    c["non_icsid"] = (~c.forum.isin(["ICSID", "ICSID_AF"])).astype(float)
    c["forum_west"] = c.forum.isin(WEST_RULES).astype(float)

    S = pd.to_numeric(p["sanction_fin_norm_it"], errors="coerce")
    g = p[S >= 0.10].groupby("iso3")["year"].min(); g = g[(g >= 2011) & (g <= 2022)]
    c["G"] = c.respondent_iso3.map(g)
    c["Sj"] = ((c.year >= c.G) & c.G.notna()).astype(float)
    c["Bi"] = c.claimant_iso3.isin(B1).astype(float)
    c["BiSj"] = c.Bi * c.Sj
    c["resp_year"] = c.respondent_iso3 + "_" + c.year.astype(str)

    # ---------- cells ----------
    cells = (c.groupby(["Bi", "Sj"]).agg(n=("year", "size"),
             uncitral=("is_uncitral", "mean"), non_icsid=("non_icsid", "mean")).reset_index())
    cells.to_csv(OUT / "f3_cells.csv", index=False)
    print("[cells: B_i x S_jt]\n", cells.round(3).to_string(index=False))
    print(f"\n  total cases {len(c)}; cases against sanctioned respondent States {int(c.Sj.sum())}; "
          f"of which by sanctioning-bloc investors {int(c.BiSj.sum())}")

    # ---------- main table ----------
    SPECS = [("(1) year FE", ["year"], ["Sj", "Bi", "BiSj"]),
             ("(2) year + respondent FE", ["year", "respondent_iso3"], ["Sj", "Bi", "BiSj"]),
             ("(3) year + respondent + home State FE", ["year", "respondent_iso3", "claimant_iso3"],
              ["Sj", "Bi", "BiSj"]),
             ("(4) respondent x year FE (identifies b3 only)", ["resp_year"], ["Bi", "BiSj"])]
    rows = []
    for dv, dvl in [("is_uncitral", "UNCITRAL rules"), ("non_icsid", "Non-ICSID rules")]:
        for lab, fes, X in SPECS:
            r = ols(c, dv, X, fes, "respondent_iso3")
            if r is None:
                continue
            for nm in r["names"]:
                j = r["names"].index(nm)
                pw = np.nan
                if nm == "BiSj":
                    pw = wcr(c, dv, X, fes, "respondent_iso3", "BiSj", B=_args.boot)
                rows.append({"DV": dvl, "specification": lab, "parameter": nm,
                             "coef (pp)": r["b"][j] * 100, "se(pp)": r["se"][j] * 100,
                             "p_CR1": r["p"][j], "p_WCR": pw, "N": r["N"], "clusters": r["G"]})
    f1 = pd.DataFrame(rows)
    f1.to_csv(OUT / "f1_forum_channels.csv", index=False)
    print("\n[(key) channel decomposition on the forum margin]")
    print(f1.round(3).to_string(index=False))
    print("\n  b1 = Sj      demand side: forum shift for sanctioned respondent States as a whole")
    print("  b3 = BiSj  (key) supply differential: same sanctioned respondent State, sanctioning-bloc vs non-bloc investors")
    print("  Specification (4) uses respondent x year FE; Sj is absorbed, giving the cleanest estimate of b3.")

    # ---------- H4 on the forum margin ----------
    pr = pd.to_numeric(p["pr"], errors="coerce")
    p["pnci_rank"] = pr.groupby(p.year).transform(lambda s: s.rank(pct=True))
    for src, nm in [("pnci_rank", "pnci_rank"), ("pnci_equal", "pnci_equal")]:
        m = p.set_index(["iso3", "year"])[src]
        c[nm] = pd.MultiIndex.from_arrays([c.respondent_iso3, c.year]).map(m)
    rows = []
    for dv, dvl in [("is_uncitral", "UNCITRAL rules"), ("non_icsid", "Non-ICSID rules")]:
        for nm in ["pnci_rank", "pnci_equal"]:
            z = c[nm]; c["_P"] = (z - z.mean()) / z.std()
            c["_SxP"] = c.Sj * c["_P"]
            r = ols(c, dv, ["Sj", "_P", "_SxP"], ["year", "respondent_iso3"],
                    "respondent_iso3")
            if r is None:
                continue
            j = r["names"].index("_SxP")
            pw = wcr(c, dv, ["Sj", "_P", "_SxP"], ["year", "respondent_iso3"],
                     "respondent_iso3", "_SxP", B=_args.boot)
            rows.append({"DV": dvl, "PNCI": nm, "sanction x PNCI (pp)": r["b"][j] * 100,
                         "se": r["se"][j] * 100, "p_CR1": r["p"][j], "p_WCR": pw,
                         "N": r["N"]})
    f2 = pd.DataFrame(rows)
    f2.to_csv(OUT / "f2_h4_on_forum.csv", index=False)
    print("\n[H4 re-estimated on the forum margin (previously tried only on filing counts)]")
    print(f2.round(3).to_string(index=False))
    print(f"\n[output] {OUT}")


if __name__ == "__main__":
    main()
