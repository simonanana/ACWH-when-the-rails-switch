"""
r26_eligibility_audit.py -- an audit of the pivotal variable itself: Convention-status eligibility

This addresses referee concerns about the coding of the eligibility variable. It audits
the very variable on which the paper's central attenuation result depends.

Four tasks
----------
A. **Consistency check**: how many cases were actually conducted under ICSID Convention
   proceedings yet are coded as "Convention unavailable"? This is the same check that
   §2.5 applies to the treaty forum menu but that had not been applied to the core
   variable. Result already obtained on the project data: **9 cases** (6 Venezuela,
   1 Bolivia, 1 Spain, 1 Turkmenistan). The authors must report this proactively.

B. **The Article 71/72 contested window**: whether consent perfected by an investor
   before the notice of denunciation survives once the denunciation takes effect is a
   disputed legal question (*Valores Mundiales* upheld jurisdiction vs *Fábrica de
   Vidrios (Favianca)* declined it). "Denunciation ⇒ unavailable" is therefore not a
   mechanical rule. This script recodes eligibility as three-valued:
   available / contested / unavailable, and reports estimates under four codings,
   giving **lower and upper bounds reflecting coding uncertainty**.

C. **The 2022 Additional Facility Rules**: from 2022-07-01 the AF is **also** available
   where "neither party is a Contracting State" and where "a regional economic
   integration organisation is a party". The earlier three-cell table is wrong for cases
   filed after 2022-07-01.

D. **Exposure to date precision**: changes in Contracting-State status within the window
   go well beyond two instances. The script lists all changes and counts how many cases
   have a filing year coinciding with a status-change year of either party (the
   mid-year rule would misclassify these cases).

Usage
-----
    python src/r26_eligibility_audit.py \\
        --cases paper_tables/cases_with_treaty_choiceset.parquet \\
        --ref   data/reference/icsid_convention_status.csv \\
        --out   paper_tables
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _compat import feols, permutation_test, wild_cluster_bootstrap  # noqa: E402

PP = 100.0
AF2022 = pd.Timestamp("2022-07-01")   # entry into force of the 2022 Additional Facility Rules
REIO = {"EUU", "European Union", "EU"}


# ==========================================================================
def load_ref(path: str) -> pd.DataFrame:
    r = pd.read_csv(path, dtype=str).fillna("")
    for c in ("in_force_from", "out_of_force_from"):
        r[c] = pd.to_datetime(r[c].replace("", np.nan), errors="coerce")
    return r.drop_duplicates("iso3").set_index("iso3")


def filing_date(row, date_col=None) -> pd.Timestamp:
    if date_col and pd.notna(row.get(date_col)):
        return pd.Timestamp(row[date_col])
    return pd.Timestamp(f"{int(row['year'])}-07-01")


def classify(cases: pd.DataFrame, ref: pd.DataFrame,
             host="respondent_iso3", home="claimant_iso3",
             date_col=None) -> pd.DataFrame:
    """
    Three-valued eligibility + 2022 AF rules + date-precision flag.

    convention_status:
        available    both parties are Contracting States on the filing date
        contested    the respondent State was a Contracting State but has denounced the
                     Convention, and the filing date falls after the denunciation
                     (Art. 71/72: whether consent perfected before denunciation survives
                     is disputed)
        unavailable  all other cases
    af_status_2006 / af_status_2022: whether the Additional Facility is available under
                                     each version of the rules
    date_precision_risk: the filing year coincides with a status-change year of either
                         party
    """
    d = cases.copy()
    out = {k: [] for k in ("convention_status", "af_2006", "af_2022",
                           "date_precision_risk", "host_ever_denounced")}
    for _, r in d.iterrows():
        when = filing_date(r, date_col)
        h, m = str(r.get(host, "")), str(r.get(home, ""))

        def party(iso):
            if iso in REIO:
                return False                      # a REIO is not a State and cannot be a Contracting State
            if iso not in ref.index:
                return False
            inf, o = ref.loc[iso, "in_force_from"], ref.loc[iso, "out_of_force_from"]
            if pd.isna(inf):
                return False
            if pd.isna(o):
                return when >= inf
            if inf < o:
                return inf <= when < o
            return (when < o) or (when >= inf)

        hp, mp = party(h), party(m)
        denounced = (h in ref.index and pd.notna(ref.loc[h, "out_of_force_from"])
                     and when >= ref.loc[h, "out_of_force_from"]
                     and not hp)
        if hp and mp:
            st = "available"
        elif denounced:
            st = "contested"
        else:
            st = "unavailable"
        out["convention_status"].append(st)
        out["host_ever_denounced"].append(bool(denounced))

        # Additional Facility
        out["af_2006"].append(bool(hp ^ mp))
        if when >= AF2022:
            out["af_2022"].append(not (hp and mp))       # available unless both parties are Contracting States
        else:
            out["af_2022"].append(bool(hp ^ mp))

        # date precision
        risk = False
        for iso in (h, m):
            if iso in ref.index:
                for c in ("in_force_from", "out_of_force_from"):
                    v = ref.loc[iso, c]
                    if pd.notna(v) and v.year == int(r["year"]):
                        risk = True
        out["date_precision_risk"].append(risk)

    for k, v in out.items():
        d[k] = v
    d["conv_available_strict"] = (d.convention_status == "available")
    d["conv_available_permissive"] = d.convention_status.isin(
        ["available", "contested"])
    return d


# ==========================================================================
def consistency_check(d: pd.DataFrame) -> pd.DataFrame:
    """Cases actually conducted under Convention proceedings but coded as unavailable."""
    conv = d["forum_class"].eq("ICSID Convention")
    bad = d[conv & (d.convention_status == "unavailable")]
    cols = [c for c in ("case_name", "year", "respondent_iso3", "claimant_iso3",
                        "convention_status", "S") if c in d.columns]
    return bad[cols].sort_values("year", ascending=False)


def status_change_inventory(ref: pd.DataFrame, cases: pd.DataFrame,
                            y0=2010, y1=2023) -> pd.DataFrame:
    rows = []
    iso_in_sample = set(cases["respondent_iso3"]) | set(cases["claimant_iso3"])
    for iso, r in ref.iterrows():
        for c, lab in (("in_force_from", "entry into force"),
                       ("out_of_force_from", "denunciation effective")):
            v = r[c]
            if pd.notna(v) and y0 <= v.year <= y1:
                rows.append({"iso3": iso, "state": r.get("state", ""),
                             "event": lab, "date": v.date(),
                             "in_sample": iso in iso_in_sample,
                             "as_respondent": iso in set(cases["respondent_iso3"]),
                             "as_claimant_home": iso in set(cases["claimant_iso3"])})
    return pd.DataFrame(rows).sort_values("date")


# ==========================================================================
def variants(d: pd.DataFrame, outcomes=("is_uncitral", "natcourt_exposed"),
             unit="respondent_iso3", time="year", treat="S",
             B_wcb: int = 1999, cohort_map: dict | None = None,
             B_perm: int = 999) -> pd.DataFrame:
    """Estimates under four eligibility codings; the core table for bounding coding uncertainty."""
    specs = {
        "(0) no availability control": None,
        "(1) strict: denunciation ends availability": "strict",
        "(2) three-valued: contested as own category": "three",
        "(3) permissive: Art. 72 consent survives": "permissive",
        "(4) drop contested cases": "drop",
    }
    rows = []
    for y in outcomes:
        if y not in d.columns:
            continue
        for lab, mode in specs.items():
            x = d.copy()
            x["__y__"] = x[y].astype(float) * PP
            if mode == "drop":
                x = x[x.convention_status != "contested"]
            if mode is None:
                absorb = [unit, time]
            else:
                if mode == "strict":
                    x["__c__"] = x.conv_available_strict.astype(str)
                elif mode == "three":
                    x["__c__"] = x.convention_status
                elif mode in ("permissive", "drop"):
                    x["__c__"] = (x.conv_available_permissive.astype(str)
                                  if mode == "permissive"
                                  else x.conv_available_strict.astype(str))
                absorb = [unit, time, "__c__"]
            r = feols(x, "__y__", [treat], absorb=absorb, cluster=unit,
                      target=treat)
            rec = {"outcome": y, "specification": lab, "effect": r["coef"],
                   "se": r["se"], "p_cr1": r["p"], "n": r["n"]}
            try:
                rec["p_wcb_webb"] = wild_cluster_bootstrap(
                    x, "__y__", [treat], absorb=absorb, cluster=unit,
                    target=treat, B=B_wcb, weights="webb")["p_wcb"]
            except Exception:  # noqa: BLE001
                rec["p_wcb_webb"] = np.nan
            if cohort_map:
                try:
                    cm = {k: v for k, v in cohort_map.items()
                          if k in set(x[unit])}
                    rec["p_perm"] = permutation_test(
                        x, "__y__", absorb=absorb, cluster=unit, unit=unit,
                        time=time, cohort_map=cm, B=B_perm)["p_perm"]
                except Exception:  # noqa: BLE001
                    rec["p_perm"] = np.nan
            rows.append(rec)
    t = pd.DataFrame(rows)
    base = t.groupby("outcome")["effect"].transform("first")
    t["share_explained"] = 1 - t["effect"] / base
    return t


# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--ref", default="data/reference/icsid_convention_status.csv")
    ap.add_argument("--out", default="paper_tables")
    ap.add_argument("--date-col", default=None,
                    help="Column with the exact filing date (e.g. if ICSID registration dates have been "
                         "collected)")
    ap.add_argument("--B-wcb", type=int, default=1999)
    ap.add_argument("--B-perm", type=int, default=999)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cases = (pd.read_parquet(args.cases) if args.cases.endswith("parquet")
             else pd.read_csv(args.cases))
    ref = load_ref(args.ref)
    d = classify(cases, ref, date_col=args.date_col)

    print("=== A. Eligibility consistency check ===")
    bad = consistency_check(d)
    bad.to_csv(out / "tab_elig_consistency_violations.csv", index=False)
    n_conv = int(d.forum_class.eq("ICSID Convention").sum())
    print(f"  Convention-proceeding cases: {n_conv}, of which coded 'unavailable': {len(bad)} "
          f"({len(bad)/max(n_conv,1):.1%})")
    if len(bad):
        print(bad.to_string(index=False))
        print("\n  By respondent State:", bad.respondent_iso3.value_counts().to_dict())
        print("  [This figure must be reported proactively in the paper]")

    print("\n=== B. Distribution of three-valued eligibility ===")
    print(pd.crosstab(d.convention_status, d.S, margins=True).to_string())
    print("\n  Contested cases by respondent State:",
          d[d.convention_status == "contested"].respondent_iso3
          .value_counts().to_dict())

    print("\n=== C. Effect of the 2022 Additional Facility Rules ===")
    diff = d[(d.af_2006 != d.af_2022)]
    print(f"  Cases whose AF availability differs between the two rule versions: {len(diff)}")
    if len(diff):
        print(diff.groupby(["year"]).size().to_string())

    print("\n=== D. Inventory of Contracting-State status changes within the window ===")
    inv = status_change_inventory(ref, d)
    inv.to_csv(out / "tab_icsid_status_changes.csv", index=False)
    print(inv[inv.in_sample].to_string(index=False))
    print(f"\n  Cases whose filing year coincides with a party's status-change year "
          f"(misclassified by the mid-year rule): "
          f"{int(d.date_precision_risk.sum())}")
    d[d.date_precision_risk][["case_name", "year", "respondent_iso3",
                              "claimant_iso3"]].to_csv(
        out / "tab_date_precision_risk.csv", index=False)

    print("\n=== E. Estimates under coding variants (core table) ===")
    cm = None
    if "g" in d.columns:
        cm = d.dropna(subset=["g"]).groupby("respondent_iso3")["g"].first().to_dict()
        cm = {u: cm.get(u, np.nan) for u in sorted(d.respondent_iso3.unique())}
    t = variants(d, cohort_map=cm, B_wcb=args.B_wcb, B_perm=args.B_perm)
    t.to_csv(out / "tab_elig_coding_variants.csv", index=False)
    print(t.round(3).to_string(index=False))

    u = t[t.outcome == "is_uncitral"]
    lo, hi = u.effect.min(), u.effect.max()
    print(f"\n  Coding-dependent range of the conditional UNCITRAL estimate: [{lo:.2f}, {hi:.2f}]")
    print("  If the range spans 'significant/not significant', the paper must report the "
          "range rather than a single number.")

    d.to_parquet(out / "cases_with_elig_variants.parquet", index=False)
    print(f"\n[done] -> {out.resolve()}")


if __name__ == "__main__":
    main()
