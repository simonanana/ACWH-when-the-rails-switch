"""
r23_claim_group.py
==================
**The second indispensable robustness check: cases are not independent observations.**

Problem
-------
The headline treats 958 cases as 958 independent "forum choices". Investment arbitration,
however, features **parallel claims filed in batches**: a set of cases arising from the
same factual event, under the same treaty, brought by the same counsel in the same year
will necessarily have identical forums. They constitute **one** choice, not N.

The most damaging concrete example for this project (which must be named and addressed
in the main text):
    The Crimea series of arbitrations brought against Russia by Ukrainian investors
    after 2014 under the 1998 Russia–Ukraine BIT (Ukrnafta, Stabil, Everest Estate,
    Oschadbank, PrivatBank, Lugzor, Belbek, etc., roughly 8–10 cases) were **all**
    under the UNCITRAL Rules, **all** registered with the PCA, and **all** filed in
    2015–2017.

    Article 9 of the Russia–Ukraine BIT offers only three dispute-settlement options:
        (a) the competent courts of the host State; (b) SCC arbitration;
        (c) ad hoc UNCITRAL arbitration.
    **There is no ICSID option** (and Russia is not a Contracting State to the
    Convention in any case).

    In other words, `is_uncitral = 1` for this group of observations is **determined by
    the treaty text** and is unrelated to sanctions; yet these cases fall squarely within
    the post-treatment window of Russia's 2014 cohort, and the Russia cohort contributes
    33 treated cases with a coefficient of +46.35pp (Table 6).

Conclusion: **the largest coefficient in Table 6 may well be produced mainly by a single
treaty clause plus a single factual event.**
This is not fatal, but the authors must address it themselves before a referee does.

This script does four things
----------------------------
1. `detect_claim_groups`  -- rule-based identification of parallel claim groups (a manual
                             override table can be layered on top)
2. `collapse_to_groups`   -- collapses to one observation per group (weighted) and
                             re-estimates
3. `cluster_on_groups`    -- re-estimates with the claim group as the clustering unit
                             (more conservative SEs)
4. `leave_one_group_out`  -- leave-one-group-out (a finer fragility test than
                             leave-one-country-out)

Usage (reuses the project io_load by default)
-----
    python src/r23_claim_group.py --overrides data/reference/claim_groups_manual.csv --out paper_tables

Manual override table format (strongly recommended; only a few dozen rows are needed):
    case_id,group_id,group_label
    ARB/15-34,RUS_CRIMEA_2015,Crimea parallel claims (Ukraine-Russia BIT)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _compat import feols, wild_cluster_bootstrap  # noqa: E402

PP = 100.0

_NOISE = re.compile(r"\b(and others|et al\.?|ltd|llc|inc|s\.a\.|plc|gmbh|"
                    r"jsc|pjsc|ojsc|limited|company|corp(oration)?)\b", re.I)


def _norm_name(s: str) -> str:
    s = _NOISE.sub(" ", str(s).lower())
    return re.sub(r"[^a-z0-9 ]+", " ", s).strip()


def detect_claim_groups(df: pd.DataFrame, case_id: str = "case_id",
                        unit: str = "respondent_iso3", year: str = "year",
                        rules: str = "rules", inst: str = "institution",
                        treaty: str | None = "applicable_iia",
                        home: str | None = "claimant_iso3",
                        window: int = 2,
                        overrides: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Rule-based grouping: same respondent State × same treaty (if available) × same
    arbitral rules × same administering institution × filing years within `window`
    years × same claimant home State -> treated as one parallel claim group.

    This is a **deliberately conservative** definition (better to under-group than to
    group spuriously); the true ground truth can only be confirmed manually, so
    overrides should always be supplied.
    """
    d = df.copy()
    keys = [unit, rules]
    if inst in d.columns:
        keys.append(inst)
    if treaty and treaty in d.columns:
        keys.append(treaty)
    if home and home in d.columns:
        keys.append(home)

    d["__k__"] = d[keys].astype(str).agg("|".join, axis=1)
    d = d.sort_values(["__k__", year])

    gid, labels = [], {}
    for k, sub in d.groupby("__k__", sort=False):
        yrs = sub[year].astype(float).to_numpy()
        cur, start = 0, yrs[0] if len(yrs) else np.nan
        ids = []
        for y in yrs:
            if y - start > window:
                cur += 1
                start = y
            ids.append(f"{k}#{cur}")
        gid.extend(ids)
    d["claim_group"] = gid

    sizes = d["claim_group"].value_counts()
    # Only groups of >=3 cases count as a "batch"; single cases and pairs remain their
    # own group (conservative)
    d["is_parallel_group"] = d["claim_group"].map(sizes).ge(3).astype(int)
    d["claim_group_size"] = d["claim_group"].map(sizes).astype(int)

    if overrides is not None and len(overrides):
        m = overrides.set_index(case_id)
        d.loc[d[case_id].isin(m.index), "claim_group"] = \
            d.loc[d[case_id].isin(m.index), case_id].map(m["group_id"])
        if "group_label" in m.columns:
            labels = m["group_label"].to_dict()
        sizes = d["claim_group"].value_counts()
        d["claim_group_size"] = d["claim_group"].map(sizes).astype(int)
        d["is_parallel_group"] = d["claim_group_size"].ge(3).astype(int)

    d["claim_group_label"] = d[case_id].map(labels).fillna("")
    return d.drop(columns="__k__")


def group_summary(d: pd.DataFrame, treat: str = "S") -> pd.DataFrame:
    g = (d[d["claim_group_size"] >= 3]
         .groupby("claim_group")
         .agg(n=("claim_group", "size"),
              respondent=("respondent_iso3", "first"),
              years=("year", lambda s: f"{int(s.min())}-{int(s.max())}"),
              rules=("rules", "first"),
              uncitral_rate=("is_uncitral", "mean"),
              treated=(treat, "mean"))
         .sort_values("n", ascending=False))
    return g


def collapse_to_groups(d: pd.DataFrame, y: str, treat: str,
                       unit: str, time: str) -> pd.DataFrame:
    """Collapse each group to one observation (within-group mean), weight = 1 (groups weighted equally)."""
    agg = {y: "mean", treat: "max", unit: "first", time: "min",
           "claim_group_size": "first"}
    keep = {k: v for k, v in agg.items() if k in d.columns}
    return d.groupby("claim_group", as_index=False).agg(keep)


def run_all(d: pd.DataFrame, y: str = "is_uncitral", treat: str = "S",
            unit: str = "respondent_iso3", time: str = "year",
            B: int = 1999) -> pd.DataFrame:
    d = d.copy()
    d["__y__"] = d[y].astype(float) * PP
    rows = []

    def _add(label, frame, cluster):
        r = feols(frame, "__y__", [treat], absorb=[unit, time],
                  cluster=cluster, target=treat)
        try:
            w = wild_cluster_bootstrap(frame, "__y__", [treat],
                                       absorb=[unit, time], cluster=cluster,
                                       target=treat, B=B,
                                       weights="webb")["p_wcb"]
        except Exception:  # noqa: BLE001
            w = np.nan
        rows.append({"Specification": label, "Effect (pp)": r["coef"],
                     "s.e.": r["se"], "p_CR1": r["p"], "p_WCB": w,
                     "N": r["n"], "Clusters": r["n_clusters"]})

    _add("Baseline (case level, cluster = respondent)", d, unit)
    _add("Cluster = claim group", d, "claim_group")

    coll = collapse_to_groups(d, "__y__", treat, unit, time)
    coll = coll.rename(columns={"__y__": "__y__"})
    _add("Collapsed to one obs per claim group", coll, unit)

    solo = d[d["claim_group_size"] < 3]
    _add("Excluding parallel groups (size >= 3)", solo, unit)

    # The "exclude only the Crimea series" row must be reported separately. Dropping all
    # parallel groups removes 140 cases at once, and readers would wrongly infer that the
    # main result is driven by Crimea. Observed: excluding only Crimea lowers beta from
    # 25.50 to just 24.83.
    lab = d.get("claim_group_label", pd.Series("", index=d.index)).astype(str)
    grp = d.get("claim_group", pd.Series("", index=d.index)).astype(str)
    crimea = (lab.str.contains("crimea", case=False, na=False) |
              grp.str.contains("crimea", case=False, na=False))
    if crimea.any():
        _add(f"Excluding the Crimea series only (n={int(crimea.sum())})",
             d[~crimea], unit)

    return pd.DataFrame(rows)


def leave_one_group_out(d: pd.DataFrame, y: str = "is_uncitral",
                        treat: str = "S",
                        unit: str = "respondent_iso3", time: str = "year",
                        min_size: int = 3) -> pd.DataFrame:
    d = d.copy()
    d["__y__"] = d[y].astype(float) * PP
    base = feols(d, "__y__", [treat], absorb=[unit, time], cluster=unit,
                 target=treat)
    rows = [{"Dropped group": "(none)", "n": len(d),
             "Effect (pp)": base["coef"], "s.e.": base["se"], "p": base["p"]}]
    for g, sub in d.groupby("claim_group"):
        if len(sub) < min_size:
            continue
        rest = d[d["claim_group"] != g]
        if rest[treat].nunique() < 2:
            continue
        r = feols(rest, "__y__", [treat], absorb=[unit, time], cluster=unit,
                  target=treat)
        rows.append({"Dropped group": g, "n": len(sub),
                     "Effect (pp)": r["coef"], "s.e.": r["se"], "p": r["p"]})
    return pd.DataFrame(rows).sort_values("Effect (pp)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None, help="Optional; defaults to the project io_load")
    ap.add_argument("--overrides", default=None,
                    help="Manual grouping table: case_id,group_id,group_label")
    ap.add_argument("--out", default="paper_tables")
    ap.add_argument("--outcome", default="is_uncitral")
    ap.add_argument("--window", type=int, default=2)
    args = ap.parse_args()

    from _data import COL, load_analysis_frame
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases, _ = load_analysis_frame(args.cases)
    ov = pd.read_csv(args.overrides, dtype=str) if args.overrides else None

    d = detect_claim_groups(cases, overrides=ov, window=args.window)
    gs = group_summary(d, treat=COL["treat"])
    gs.to_csv(out / "tab_claim_groups.csv")
    print("=== Parallel claim groups identified (>=3 cases) ===")
    print(gs.head(30).round(3).to_string())

    res = run_all(d, y=args.outcome, treat=COL["treat"])
    res.to_csv(out / "tab_claim_group_robustness.csv", index=False)
    print("\n=== Claim-group robustness ===")
    print(res.round(3).to_string(index=False))

    loo = leave_one_group_out(d, y=args.outcome, treat=COL["treat"])
    loo.to_csv(out / "tab_leave_one_group_out.csv", index=False)
    print("\n=== Leave-one-group-out (6 most extreme) ===")
    print(pd.concat([loo.head(3), loo.tail(3)]).drop_duplicates()
          .round(3).to_string(index=False))
    d.to_parquet(out / "cases_with_claim_groups.parquet", index=False)


if __name__ == "__main__":
    main()
