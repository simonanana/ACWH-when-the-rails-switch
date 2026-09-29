"""
r13_bloc_choice.py - choice of the reference definition of B_i (sanctioning bloc) and multiple-testing correction
=====================================================================
Problem: the H3 supply differential yields different conclusions under different definitions of B_i. Which one should be the reference specification?

Principle
---------
**A definition must not be chosen because it is significant.** Testing the same hypothesis once under each of three definitions
and taking the smallest p gives an actual significance level of about 1-0.95^3 ~ 14%, not 5%. The paper already
rejects H4 on multiple-testing grounds; applying a double standard to H3 would be an easy target for a referee.

Correct approach: fix the reference specification **a priori from the legal texts**, report all three definitions,
and apply a Romano-Wolf stepdown correction.

Three candidate B_i (all with a priori justification, none data-driven)
--------------------------------------------
  B1 "legal-services ban core"  EU27 + GBR + CHE + NOR + USA
     Basis: EU Reg 833/2014 Art 5n (legal advisory services ban, 2022-10-07)
           + follow-on measures by the UK/Switzerland/Norway + the US OFAC general licence regime
     (key) This is the set that **maps directly** onto the supply mechanism; recommended as the reference specification
  B2 "broad Western camp"      B1 + remaining Five Eyes + Japan and Korea
     Basis: members of the sanctions-coordination mechanism, but with weaker legal constraints on lawyers' practice
  B3 "original notebook definition"  the hard-coded 17-country set in notebook 00, cell[6]/[22]
     Basis: consistent with the existing pipeline, for comparison

Outputs: b1_bloc_definitions.csv, b2_supply_by_bloc.csv
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

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

EU27 = {"AUT","BEL","BGR","HRV","CYP","CZE","DNK","EST","FIN","FRA","DEU","GRC",
        "HUN","IRL","ITA","LVA","LTU","LUX","MLT","NLD","POL","PRT","ROU","SVK",
        "SVN","ESP","SWE"}
B1 = EU27 | {"GBR","CHE","NOR","USA"}
B2 = B1 | {"CAN","AUS","NZL","JPN","KOR"}
B3 = {"GBR","FRA","USA","DEU","NLD","SWE","AUT","CAN","CHE","BEL","LUX",
      "ITA","ESP","IRL","DNK","FIN","POL"}
BLOCS = [("B1 legal-services ban core (EU27+GBR+CHE+NOR+USA)", B1),
         ("B2 broad Western camp (+Five Eyes+JPN/KOR)", B2),
         ("B3 original notebook, 17 countries", B3)]


def _gm(v, c, w, k):
    n = np.bincount(c, weights=v * w, minlength=k)
    dd = np.bincount(c, weights=w, minlength=k)
    return (n / np.where(dd == 0, 1, dd))[c]


def _dm(M, fes, w):
    M = np.asarray(M, float); one = M.ndim == 1
    if one: M = M[:, None]
    ng = [int(c.max()) + 1 for c in fes]; o = M.copy()
    for j in range(o.shape[1]):
        v = o[:, j]
        for _ in range(2000):
            v0 = v
            for c, k in zip(fes, ng): v = v - _gm(v, c, w, k)
            if np.max(np.abs(v - v0)) < 1e-11 * max(1, np.max(np.abs(v0))): break
        o[:, j] = v
    return o[:, 0] if one else o


def ppml(df, y, X, fes, cl, maxiter=150):
    d = df[list(dict.fromkeys([y] + X + fes + [cl]))].dropna().copy()
    for _ in range(20):
        n0 = len(d)
        for f in fes:
            d = d[d.groupby(f, observed=True)[y].transform("sum") > 0]
        if len(d) == n0: break
    yv = d[y].to_numpy(float); Xv = d[X].to_numpy(float)
    fc = [pd.factorize(d[f])[0] for f in fes]
    mu = np.maximum((yv + yv.mean()) / 2, 1e-6); eta = np.log(mu); b = np.zeros(len(X))
    for _ in range(maxiter):
        z = eta + (yv - mu) / mu; w = mu
        zt = _dm(z, fc, w); Xt = _dm(Xv, fc, w)
        bn = np.linalg.pinv(Xt.T @ (Xt * w[:, None])) @ (Xt.T @ (zt * w))
        eta = np.clip(z - (zt - Xt @ bn), -50, 50); mu = np.maximum(np.exp(eta), 1e-10)
        if np.max(np.abs(bn - b)) < 1e-9: b = bn; break
        b = bn
    w = mu; Xt = _dm(Xv, fc, w)
    br = np.linalg.pinv(Xt.T @ (Xt * w[:, None])); u = yv - mu
    g = pd.factorize(d[cl])[0]; G = int(g.max()) + 1
    su = Xt * u[:, None]; S = np.zeros((G, len(X)))
    for j in range(len(X)): S[:, j] = np.bincount(g, weights=su[:, j], minlength=G)
    V = br @ (S.T @ S) @ br * (G / (G - 1))
    se = np.sqrt(np.maximum(np.diag(V), 0)); t = b / se
    return b, se, 2 * stats.norm.sf(np.abs(t)), len(d), G, t


def main():
    _need(PANEL, "panel file"); _need(CASES, "case file")
    p = pd.read_parquet(PANEL); c = pd.read_parquet(CASES)
    c["year"] = pd.to_numeric(c["year"], errors="coerce")
    c = c.dropna(subset=["year"]); c["year"] = c.year.astype(int)
    c = c[(c.year >= 2010) & (c.year <= 2023)]
    for k in ("respondent_iso3", "claimant_iso3"):
        c[k] = c[k].astype(str).str.upper().str.strip()

    S = pd.to_numeric(p["sanction_fin_norm_it"], errors="coerce")
    g = p[S >= 0.10].groupby("iso3")["year"].min()
    g = g[(g >= 2011) & (g <= 2022)]

    cc = c[["claimant_iso3", "respondent_iso3", "year"]].copy()
    cc.columns = ["i", "j", "year"]
    cc = cc[(cc.i != cc.j) & ~cc.i.isin(["", "NAN"]) & ~cc.j.isin(["", "NAN"])]
    cnt = cc.groupby(["i", "j", "year"]).size().rename("n").reset_index()
    Is, Js = sorted(cnt.i.unique()), sorted(cnt.j.unique())
    yr = list(range(2010, 2024))
    sk = pd.MultiIndex.from_product([Is, Js, yr], names=["i", "j", "year"]).to_frame(index=False)
    sk = sk[sk.i != sk.j]
    dy = sk.merge(cnt, on=["i", "j", "year"], how="left")
    dy["n"] = dy.n.fillna(0).astype(int)
    dy["it"] = [f"{a}_{t}" for a, t in zip(dy.i, dy.year)]
    dy["jt"] = [f"{a}_{t}" for a, t in zip(dy.j, dy.year)]
    dy["dyad"] = [f"{a}_{b}" for a, b in zip(dy.i, dy.j)]
    dy["Gj"] = dy.j.map(g)
    dy["Sj"] = ((dy.year >= dy.Gj) & dy.Gj.notna()).astype(float)
    print(f"[dyad] {len(dy):,} rows, non-zero {int((dy.n>0).sum())}, "
          f"i={len(Is)} j={len(Js)}, cases against sanctioned respondent States {int(dy.loc[dy.Sj==1,'n'].sum())}")

    rows, tv = [], []
    for lab, B in BLOCS:
        dy["Bi"] = dy.i.isin(B).astype(float)
        dy["X"] = dy.Bi * dy.Sj
        n_bloc = int(dy.loc[(dy.Bi == 1) & (dy.Sj == 1), "n"].sum())
        n_non = int(dy.loc[(dy.Bi == 0) & (dy.Sj == 1), "n"].sum())
        b, se, pv, N, G, t = ppml(dy, "n", ["Sj", "X"], ["it", "jt"], "dyad")
        rows.append({"B_i definition": lab, "n_i in bloc": len(set(Is) & B),
                     "cases bloc x sanc": n_bloc, "cases nonbloc x sanc": n_non,
                     "supply differential beta": b[1], "se": se[1], "p": pv[1],
                     "semi-elasticity %": (np.exp(b[1]) - 1) * 100, "N": N})
        tv.append(abs(t[1]))
    r = pd.DataFrame(rows)
    # Bonferroni (conservative; Romano-Wolf is recommended for the final draft, which requires a bootstrap)
    r["p_Bonferroni"] = np.minimum(r["p"] * len(r), 1.0)
    r.to_csv(OUT / "b2_supply_by_bloc.csv", index=False)
    pd.DataFrame([{"definition": l, "n_countries": len(B), "members": " ".join(sorted(B))}
                  for l, B in BLOCS]).to_csv(OUT / "b1_bloc_definitions.csv", index=False)
    print("\n[H3 supply differential: three B_i definitions side by side]\n", r.round(4).to_string(index=False))
    print("\n  Specification: n_ijt ~ PPML(Sj + Bi x Sj | i-year FE, j-year FE), clusters = dyad")
    print("  Note: with j-year FE, Sj is absorbed, so Bi x Sj is the only identifiable supply differential.")
    print(f"\n[output] {OUT}")


if __name__ == "__main__":
    main()
