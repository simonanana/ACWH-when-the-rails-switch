"""
r25_iia_choiceset.py  (v2 -- adapted to the actual layout of the official UNCTAD Excel file)
==========================================================================================
Upgrades Convention-status eligibility from the **Convention layer** (r22: whether both
parties are ICSID Contracting States) to the **treaty layer** (this script: which forums
the dispute-settlement clause of the applicable IIA actually offers the investor).

Changes relative to an earlier version of this script (addressing the "required column
not found" error raised by `--stage worklist`)
------------------------------------------------------------------------------------------
The first 11 rows of the official file are explanatory text (title, citation format,
links); column names only appear in row 12, and they are **all upper case**
(`APPLICABLE IIA` / `YEAR OF INITIATION` / `RESPONDENT STATE` ...).
The earlier version called `read_excel` directly and treated row 1 as the header, which
produced `Unnamed: 1..27`. This version does four things:

 1. **Automatic header-row detection**: reads the first 60 rows without a header, scores
    each row, and takes the row matching the most known fields.
 2. **Column names are matched against the official upper-case spelling**; the older
    spellings are kept as fallback regular expressions.
 3. **Multi-valued cells**: the official file separates values with ";" plus a line break
    (e.g. `NAFTA (1992);\\nUSMCA (2018)`), so the splitting rule has been relaxed
    accordingly.
 4. **Post-award proceedings**: the official Excel file lacks the two columns
    "Judicial review by national courts" / "ICSID annulment proceedings" shown in the web
    filter panel; the information is held in the two columns
    `FOLLOW-ON PROCEEDING TYPE` + `FOLLOW-ON PROCEEDING STATUS` (paired, multi-valued).
    This version parses those two columns and derives `fo_annulment` / `fo_setaside` /
    `fo_upheld` etc., for use in the §8 wording revision and in descriptive analysis.

A new `--stage inspect` has also been added: it shows the parsing result before the
full pipeline is run, to avoid silent mismatches.

Run order
---------
    # 0) First confirm that the file is parsed correctly (running this step first is
    #    strongly recommended)
    python src/r25_iia_choiceset.py --stage inspect \\
        --navigator data/unctad_isds_navigator.xlsx

    # 1) Match + generate the manual-coding worklist
    python src/r25_iia_choiceset.py --stage worklist \\
        --navigator data/unctad_isds_navigator.xlsx --out paper_tables

    # 2) Fill in paper_tables/iia_forum_menu_TEMPLATE.csv by hand
    #    -> save as data/reference/iia_forum_menu.csv

    # 3) Merge + two-layer decomposition (requires r22's output cases_with_eligibility.parquet)
    python src/r25_iia_choiceset.py --stage merge \\
        --navigator data/unctad_isds_navigator.xlsx \\
        --coding data/reference/iia_forum_menu.csv --out paper_tables

If automatic detection fails, specify the row manually with --header-row (0-based; 11 for
the official 31Dec2023 file).

Data sources
    https://investmentpolicy.unctad.org/pages/1057/isds-navigator-about-and-methodology
    Full treaty texts: https://investmentpolicy.unctad.org/international-investment-agreements
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _compat import feols  # noqa: E402

PP = 100.0

FORUM_COLS = [
    "allows_icsid_convention",
    "allows_icsid_af",
    "allows_uncitral",
    "allows_scc",
    "allows_icc",
    "allows_other_institution",
    "allows_domestic_courts",
    "fork_in_the_road",
]

COLUMN_PATTERNS = {
    "no":        [r"^\s*no\.?\s*$"],
    "year":      [r"year\s+of\s+initiation", r"^\s*year\s*$"],
    "short":     [r"short\s+case\s+name"],
    "full":      [r"full\s+case\s+name", r"case\s+name"],
    "iia":       [r"applicable\s+iia", r"^\s*iia\s*$", r"applicable\s+treaty"],
    "rules":     [r"arbitral\s+rules"],
    "inst":      [r"administering\s+institution"],
    "outcome":   [r"status/?\s*outcome\s+of\s+original", r"outcome\s+of\s+original"],
    "resp":      [r"respondent\s+state", r"^\s*respondent\s*$"],
    "home":      [r"home\s+state\s+of\s+investor", r"home\s+state"],
    "claimed":   [r"amount\s+claimed"],
    "awarded":   [r"amount\s+awarded"],
    "fo_type":   [r"follow-?on\s+proceeding\s+type"],
    "fo_status": [r"follow-?on\s+proceeding\s+status"],
    "italaw":    [r"italaw"],
}
REQUIRED = ["year", "iia", "resp"]

MULTI_SPLIT = r"[;\n\r]+"
NA_TOKENS = {"", "nan", "none", "data not available", "not available",
             "no follow-up proceedings initiated"}


# ==========================================================================
# 1. Loading: automatic header-row detection + column-name mapping
# ==========================================================================
def _clean(x) -> str:
    return re.sub(r"\s+", " ", str(x)).strip()


def detect_header_row(path: str, max_scan: int = 60) -> int:
    raw = (pd.read_excel(path, header=None, nrows=max_scan)
           if str(path).lower().endswith((".xlsx", ".xls"))
           else pd.read_csv(path, header=None, nrows=max_scan, dtype=str))
    best, best_score = None, 0
    for i in range(len(raw)):
        cells = [_clean(v).lower() for v in raw.iloc[i].tolist() if pd.notna(v)]
        if not cells:
            continue
        score = sum(
            any(any(re.search(p, c) for p in pats) for c in cells)
            for pats in COLUMN_PATTERNS.values())
        if score > best_score:
            best, best_score = i, score
    if best is None or best_score < 3:
        raise SystemExit(
            f"Could not locate the header within the first {max_scan} rows (best match: "
            f"{best_score} fields). Specify it manually with --header-row (0-based; "
            f"usually 11 for the official 31Dec2023 file).")
    return best


def map_columns(cols: list[str]) -> dict:
    clean = {c: _clean(c).lower() for c in cols}
    out: dict[str, str] = {}
    for key, pats in COLUMN_PATTERNS.items():
        for p in pats:
            hit = [c for c, cc in clean.items()
                   if re.search(p, cc) and c not in out.values()]
            if hit:
                out[key] = hit[0]
                break
    return out


def load_navigator(path: str, header_row: int | None = None,
                   verbose: bool = True) -> pd.DataFrame:
    hr = header_row if header_row is not None else detect_header_row(path)
    nav = (pd.read_excel(path, header=hr)
           if str(path).lower().endswith((".xlsx", ".xls"))
           else pd.read_csv(path, header=hr))
    nav.columns = [_clean(c) for c in nav.columns]
    nav = nav.dropna(axis=1, how="all")
    cmap = map_columns(list(nav.columns))
    missing = [k for k in REQUIRED if k not in cmap]
    if missing:
        raise SystemExit(
            f"With header row={hr}, required columns {missing} are still missing after parsing.\n"
            f"Actual column names: {list(nav.columns)}\n"
            f"Specify the correct row with --header-row.")

    nav = nav.rename(columns={v: k for k, v in cmap.items()})
    nav["year"] = pd.to_numeric(nav["year"], errors="coerce")
    nav = nav[nav["year"].notna()].reset_index(drop=True)   # drop residual note rows
    if verbose:
        print(f"[load] header row = {hr} (0-based), valid data rows = {len(nav)}, "
              f"years {int(nav.year.min())}–{int(nav.year.max())}")
        print("[load] column mapping:")
        for k, v in cmap.items():
            print(f"        {k:<10} <- {v}")
        unmapped = [c for c in nav.columns if c not in cmap]
        if unmapped:
            print(f"[load] unmapped (does not affect the run): {unmapped[:12]}")
    nav.attrs["header_row"] = hr
    nav.attrs["colmap"] = cmap
    return nav


# ==========================================================================
# 2. Multi-valued fields and post-award proceedings
# ==========================================================================
def split_multi(s) -> list[str]:
    if pd.isna(s):
        return []
    parts = [p.strip() for p in re.split(MULTI_SPLIT, str(s))]
    return [p for p in parts if p and p.lower() not in NA_TOKENS]


def parse_followon(nav: pd.DataFrame) -> pd.DataFrame:
    """
    The official Excel file merges the web interface's "Judicial review by national
    courts" and "ICSID annulment proceedings" into the two columns FOLLOW-ON PROCEEDING
    TYPE / STATUS (paired, multi-valued, matched by position). This function splits them
    back into usable indicator variables.

    Note: these fields are **conditional on an award having been rendered**; they form a
    selected sample and may be used descriptively only, not as DiD outcomes.
    """
    d = nav.copy()
    if "fo_type" not in d.columns:
        for c in ("fo_any", "fo_annulment", "fo_setaside"):
            d[c] = np.nan
        return d

    types = d["fo_type"].map(split_multi)
    stats = (d["fo_status"].map(split_multi) if "fo_status" in d.columns
             else types.map(lambda x: []))

    def _has(lst, *pats):
        return int(any(any(re.search(p, x, re.I) for p in pats) for x in lst))

    d["fo_any"] = types.map(lambda x: int(len(x) > 0))
    d["fo_annulment"] = types.map(lambda x: _has(x, r"annul"))
    d["fo_setaside"] = types.map(lambda x: _has(
        x, r"set\s*aside", r"setting\s*aside", r"judicial\s*review",
        r"national\s*court"))
    d["fo_enforcement"] = types.map(lambda x: _has(x, r"enforc", r"recognit"))
    d["fo_resubmission"] = types.map(lambda x: _has(x, r"resubmis"))
    d["fo_upheld"] = stats.map(lambda x: _has(x, r"upheld"))
    d["fo_annulled_full"] = stats.map(
        lambda x: _has(x, r"(annulled|set aside).*entirety"))
    d["fo_annulled_part"] = stats.map(lambda x: _has(x, r"partially"))
    d["fo_pending"] = stats.map(lambda x: _has(x, r"pending"))

    pairs = []
    for i, (ts, ss) in enumerate(zip(types, stats)):
        for j, t in enumerate(ts):
            pairs.append({"row": i, "fo_type_item": t,
                          "fo_status_item": ss[j] if j < len(ss) else "(none)"})
    d.attrs["followon_pairs"] = pd.DataFrame(pairs)
    return d


# ==========================================================================
# 3. Matching
# ==========================================================================
def _norm_name(s) -> str:
    """
    Normalise a case name. **The trailing Roman sequence numeral is preserved**
    ((I)/(II)/(III)): UNCTAD uses it to distinguish multiple cases brought by the same
    investor against the same State (e.g. Gargour Family v. Libya (I) and (II)).
    It is extracted as a separate token before parentheses are stripped; otherwise the
    two cases would normalise to the same string and both would fail the
    "unique match" requirement.
    """
    s = str(s)
    seq = ""
    m = re.search(r"\(\s*(i{1,3}|iv|v|vi{0,3})\s*\)\s*$", s, re.I)
    if m:
        seq = " seq" + m.group(1).lower()
        s = s[:m.start()]
    s = s.lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"\b(v\.?|vs\.?|versus)\b", " ", s)
    s = re.sub(r"\b(and others|et al\.?|and other|ltd|inc|llc)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return (s + seq).strip()


CARRY = ["iia", "rules", "inst", "outcome", "home", "resp", "claimed", "awarded",
         "italaw", "fo_any", "fo_annulment", "fo_setaside", "fo_enforcement",
         "fo_resubmission", "fo_upheld", "fo_annulled_full", "fo_annulled_part",
         "fo_pending"]


def match_navigator(cases: pd.DataFrame, nav: pd.DataFrame,
                    case_name: str = "case_name", year: str = "year",
                    verbose: bool = True) -> pd.DataFrame:
    """
    Three matching rounds, each requiring a **unique match**; no fuzzy matching.
      R1  normalised short case name + year
      R2  normalised short case name (across years)
      R3  normalised full case name
    Unmatched cases are written to iia_unmatched_priority.csv for manual completion.
    """
    c = cases.copy().reset_index(drop=True)
    n = nav.copy().reset_index(drop=True)
    n["__short__"] = n["short"].map(_norm_name) if "short" in n.columns else np.nan
    n["__full__"] = n["full"].map(_norm_name) if "full" in n.columns else np.nan
    c["__k__"] = c[case_name].map(_norm_name)
    c["__yr__"] = pd.to_numeric(c[year], errors="coerce")

    carry = [x for x in CARRY if x in n.columns]
    for col in carry:
        c[col] = pd.Series([np.nan] * len(c), dtype=object)
    c["__match__"] = ""

    def _round(key_right: str, use_year: bool, tag: str):
        todo = c.index[c["iia"].isna()]
        if len(todo) == 0 or key_right not in n.columns:
            return
        sub = n[n[key_right].notna() & (n[key_right] != "")]
        keys = [key_right, "year"] if use_year else [key_right]
        cnt = sub.groupby(keys).size()
        uniq = sub.set_index(keys)
        uniq = uniq[cnt.reindex(uniq.index).to_numpy() == 1]
        if use_year:
            lookup = list(zip(c.loc[todo, "__k__"], c.loc[todo, "__yr__"]))
        else:
            lookup = list(c.loc[todo, "__k__"])
        idx = pd.Index(lookup)
        hit = idx.isin(uniq.index)
        if not hit.any():
            return
        sel = uniq.reindex(idx[hit])
        rows = todo[hit]
        for col in carry:
            c.loc[rows, col] = sel[col].to_numpy()
        c.loc[rows, "__match__"] = tag

    _round("__short__", True, "short+year")
    _round("__short__", False, "short")
    _round("__full__", False, "full")

    c = c.drop(columns=["__k__", "__yr__"])
    if verbose:
        got, tot = int(c["iia"].notna().sum()), len(c)
        print(f"\n[match] {got}/{tot} = {got/tot:.1%} matched")
        print(c["__match__"].replace("", "UNMATCHED").value_counts().to_string())
    return c


# ==========================================================================
# 4. worklist
# ==========================================================================
def explode_iia(df: pd.DataFrame, col: str = "iia") -> pd.DataFrame:
    d = df.copy()
    d["__items__"] = d[col].map(split_multi)
    d = d.explode("__items__")
    d = d[d["__items__"].notna()]
    d[col] = d["__items__"].astype(str).str.strip()
    return d.drop(columns="__items__")


def build_worklist(m: pd.DataFrame, switchers: list[str], out: Path,
                   unit: str = "respondent_iso3", treat: str = "S",
                   case_id: str = "case_id") -> None:
    m = m.copy()
    m["is_identifying"] = m[unit].isin(switchers).astype(int)
    m["is_treated"] = (m[treat] > 0).astype(int)
    m["priority_case"] = ((m.is_identifying == 1) | (m.is_treated == 1)).astype(int)

    cov = pd.DataFrame([{"Sample": nm, "N": len(s),
                         "IIA matched": round(float(s["iia"].notna().mean()), 3)}
                        for nm, s in [
                            ("All cases", m),
                            ("Identifying (switcher respondents)",
                             m[m.is_identifying == 1]),
                            ("Treated", m[m.is_treated == 1]),
                            ("Priority (identifying or treated)",
                             m[m.priority_case == 1])]])
    cov.to_csv(out / "tab_iia_coverage.csv", index=False)
    print("\n=== A. IIA match rate (if below 0.95 in the identifying sample, complete the "
          "matching before proceeding) ===")
    print(cov.to_string(index=False))

    un = m[m["iia"].isna() & (m.priority_case == 1)]
    if len(un):
        cols = [c for c in (case_id, "case_name", "year", unit) if c in un.columns]
        un[cols].to_csv(out / "iia_unmatched_priority.csv", index=False)
        print(f"\n[!] {len(un)} priority cases unmatched -> iia_unmatched_priority.csv; "
              f"fill in the Applicable IIA manually from the Navigator website.")

    e = explode_iia(m[m["iia"].notna()])
    wl = (e.groupby("iia")
            .agg(n_cases=("iia", "size"),
                 n_identifying=("is_identifying", "sum"),
                 n_treated=("is_treated", "sum"),
                 respondents=(unit, lambda s: ",".join(sorted(set(s)))),
                 years=("year", lambda s: f"{int(s.min())}-{int(s.max())}"))
            .sort_values(["n_identifying", "n_treated", "n_cases"], ascending=False))
    q = wl.index.to_series().str.replace(r"\s+", "+", regex=True)
    wl["unctad_search"] = (
        "https://investmentpolicy.unctad.org/international-investment-agreements?q=" + q)
    wl.to_csv(out / "iia_coding_worklist.csv")

    prio = wl[(wl.n_identifying > 0) | (wl.n_treated > 0)]
    tmpl = pd.DataFrame({"iia": prio.index})
    tmpl["ds_article"] = ""
    for col in FORUM_COLS:
        tmpl[col] = ""
    for col in ("source_url", "coder", "coded_on", "note"):
        tmpl[col] = ""
    tmpl.to_csv(out / "iia_forum_menu_TEMPLATE.csv", index=False)

    tot_id = int(prio["n_identifying"].sum())
    n80 = int((prio["n_identifying"].cumsum() < 0.8 * tot_id).sum() + 1) if tot_id else 0
    print("\n=== B. Coding workload ===")
    print(f"  Treaties in total (full sample)              : {len(wl)}")
    print(f"  To be coded (identifying or treated cases)   : {len(prio)}")
    print(f"  Cases covered                                : {int(m.priority_case.sum())}")
    print(f"  Treaties needed to cover 80% of ident. cases : {n80}   <- code these first")
    print(f"\n  Template: {out/'iia_forum_menu_TEMPLATE.csv'}")
    print("\n  Top 20 by priority:")
    print(prio.head(20)[["n_cases", "n_identifying", "n_treated",
                         "respondents", "years"]].to_string())

    fo = [c for c in ("fo_any", "fo_annulment", "fo_setaside", "fo_enforcement",
                      "fo_upheld", "fo_annulled_full", "fo_annulled_part")
          if c in m.columns and m[c].notna().any()]
    if fo:
        mm = m.copy()
        for c_ in fo:
            mm[c_] = pd.to_numeric(mm[c_], errors="coerce")
        t = mm.groupby(treat)[fo].agg(["sum", "mean"])
        t.to_csv(out / "tab_followon_by_treatment.csv")
        print("\n=== C. Post-award (FOLLOW-ON) proceedings, by treatment status ===")
        print(t.round(3).to_string())
        print("  Note: conditional on an award having been rendered; a selected sample, "
              "descriptive only, not usable as a DiD outcome.")


# ==========================================================================
# 5. merge
# ==========================================================================
def attach_treaty_choice_set(m: pd.DataFrame, coding: pd.DataFrame,
                             strict_af: bool = True) -> pd.DataFrame:
    """Monotone (partial-identification) rule; implemented in src/menu_logic.py."""
    from menu_logic import attach_menu, prepare_coding
    out = attach_menu(m, coding, iia_col="iia", strict_af=strict_af)
    cd = prepare_coding(coding, strict_af)
    out["treaty_offers_icsid"] = np.nan
    out["treaty_offers_uncitral"] = np.nan
    for c in FORUM_COLS:
        out[c] = np.nan
    return out


def two_layer_decomposition(d: pd.DataFrame, y: str, treat: str,
                            unit: str, time: str) -> pd.DataFrame:
    d = d.copy()
    d["__y__"] = d[y].astype(float) * PP
    conv = d["icsid_convention_available"] if "icsid_convention_available" in d \
        else pd.Series(False, index=d.index)
    d["__conv__"] = conv.fillna(False).astype(bool).astype(str)
    # Uncoded treaties are kept as a separate "unknown" category rather than dropped:
    # dropping them would change the year fixed effects, so (0) Baseline would no longer
    # equal the headline 25.50 and the decomposition could not be interpreted.
    d["__menu__"] = np.where(d["treaty_menu_known"] == 1,
                             d["menu_icsid_and_uncitral"].fillna(-1)
                             .astype(int).astype(str), "unknown")
    d["__both__"] = d["__conv__"] + "|" + d["__menu__"]
    specs = [("(0) Baseline", [unit, time]),
             ("(1) + Convention availability FE", [unit, time, "__conv__"]),
             ("(2) + Treaty forum-menu FE", [unit, time, "__menu__"]),
             ("(3) + both", [unit, time, "__both__"])]
    rows = []
    for lab, ab in specs:
        r = feols(d, "__y__", [treat], absorb=ab, cluster=unit, target=treat)
        rows.append({"Specification": lab, "Effect (pp)": r["coef"],
                     "s.e.": r["se"], "p": r["p"], "N": r["n"]})
    t = pd.DataFrame(rows)
    t["share of baseline explained"] = 1 - t["Effect (pp)"] / t.loc[0, "Effect (pp)"]
    return t


# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["inspect", "worklist", "merge"], required=True)
    ap.add_argument("--navigator", required=True)
    ap.add_argument("--header-row", type=int, default=None,
                    help="0-based header row number; auto-detected by default (usually 11 for the "
                         "official file)")
    ap.add_argument("--coding", default=None)
    ap.add_argument("--elig-cases",
                    default="paper_tables/cases_with_eligibility.parquet")
    ap.add_argument("--out", default="paper_tables")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    nav = parse_followon(load_navigator(args.navigator, args.header_row))

    if args.stage == "inspect":
        keep = [c for c in ("year", "short", "iia", "rules", "inst", "resp",
                            "home", "fo_type", "fo_status") if c in nav.columns]
        print("\n=== First 5 rows (after mapping) ===")
        print(nav[keep].head(5).to_string())
        yr = nav.year.value_counts().sort_index()
        win = yr.reindex(range(2010, 2024)).fillna(0).astype(int)
        print("\n=== Cases per year 2010–2023 (should match the analysis sample) ===")
        print(win.to_string())
        print(f"  Total = {int(win.sum())}   (analysis sample = 958)")
        print("\n=== APPLICABLE IIA multi-value distribution (treaties per case) ===")
        print(nav["iia"].map(lambda s: len(split_multi(s)))
              .value_counts().sort_index().to_string())
        pairs = nav.attrs.get("followon_pairs")
        if pairs is not None and len(pairs):
            print("\n=== FOLLOW-ON TYPE × STATUS cross-tabulation ===")
            print(pd.crosstab(pairs.fo_type_item, pairs.fo_status_item).to_string())
        return

    from _data import COL, load_analysis_frame
    U, T, Y = COL["unit"], COL["time"], COL["treat"]

    cases = (load_analysis_frame()[0] if args.stage == "worklist"
             else pd.read_parquet(args.elig_cases))
    sw = sorted(cases.groupby(U)[Y].nunique().pipe(lambda s: s[s > 1]).index)
    m = match_navigator(cases, nav, year=T)

    if args.stage == "worklist":
        build_worklist(m, sw, out, unit=U, treat=Y)
        m.to_parquet(out / "cases_with_navigator.parquet", index=False)
        print(f"\n[done] -> {out.resolve()}")
        return

    if not args.coding:
        raise SystemExit("--stage merge requires --coding data/reference/iia_forum_menu.csv")
    d = attach_treaty_choice_set(m, pd.read_csv(args.coding))
    print("\n=== Treaty forum menu coding coverage (by treatment status) ===")
    print(d.groupby(Y)["treaty_menu_known"].agg(["size", "mean"]).round(3).to_string())
    print("\n=== Share of menus offering both ICSID and UNCITRAL (cases with known menu only) ===")
    kn = d[d.treaty_menu_known == 1]
    print(kn.groupby(Y)["menu_icsid_and_uncitral"]
          .agg(["size", "mean"]).round(3).to_string())
    print("  (unknown is coded -1 and excluded from the table above; it is a separate "
          "category in the decomposition)")
    cov = d.groupby(Y)["treaty_menu_known"].mean()
    if len(cov) > 1 and abs(cov.iloc[1] - cov.iloc[0]) > 0.15:
        print(f"\n  [!] Coding coverage is severely unbalanced between the treated and "
              f"control groups ({cov.iloc[0]:.1%} vs {cov.iloc[1]:.1%}).\n"
              f"      The 'unknown' category is therefore correlated with treatment status, "
              f"so the layer (2) coefficient is not clean.\n"
              f"      Remedy: code a random batch of control-group treaties until coverage "
              f"in the two groups is similar.")

    # Reference specification: full sample, uncoded treaties as a separate category
    # -> (0) Baseline must equal the headline
    dec = two_layer_decomposition(d, "is_uncitral", Y, U, T)
    dec.to_csv(out / "tab_two_layer_decomposition.csv", index=False)
    print("\n=== Two-layer decomposition (full sample; uncoded treaties as a separate category) ===")
    print(dec.round(3).to_string(index=False))
    if abs(dec.loc[0, "Effect (pp)"] - 25.50) > 0.05:
        print(f"  [!] (0) Baseline = {dec.loc[0,'Effect (pp)']:.2f} does not equal the "
              f"headline 25.50, indicating that the sample or treatment variable differs "
              f"from r22; resolve this before interpreting the remaining rows.")

    dd = d[d.treaty_menu_known == 1]
    if dd[Y].nunique() > 1 and len(dd) > 100:
        dec2 = two_layer_decomposition(dd, "is_uncitral", Y, U, T)
        dec2.to_csv(out / "tab_two_layer_decomposition_codedonly.csv", index=False)
        print("\n=== Two-layer decomposition (coded subsample only; year FE change, so Baseline "
              "is not comparable with 25.50) ===")
        print(dec2.round(3).to_string(index=False))

    conv = (d["icsid_convention_available"] if "icsid_convention_available" in d
            else pd.Series(True, index=d.index))
    sub = d[(d.treaty_menu_known == 1) & (d.menu_icsid_and_uncitral == 1) &
            conv.fillna(False).astype(bool)]
    swz = sub.groupby(U)[Y].nunique().pipe(lambda s: s[s > 1]).index.tolist() \
        if len(sub) else []
    nsw = len(swz)
    print(f"\n=== Genuine-choice subsample (Convention available AND treaty offers both "
          f"ICSID and UNCITRAL) ===")
    print(f"  N = {len(sub)}, switching clusters = {nsw} {swz}, "
          f"treated = {int(sub[Y].sum()) if len(sub) else 0}")
    try:
        from config import MIN_SWITCHING_CLUSTERS as GATE
    except Exception:  # noqa: BLE001
        GATE = 5
    if nsw < GATE:
        print(f"  [BLOCKED] switching clusters {nsw} < gate {GATE}: under this "
              f"project's feasibility gate, **no point estimate should be reported** "
              f"here.\n"
              f"  Correct wording: 'this subsample cannot identify the coefficient' "
              f"(report only N and the number of switching clusters).")
        if len(sub) and sub[Y].nunique() > 1:
            sub = sub.copy()
            sub["__y__"] = sub["is_uncitral"].astype(float) * PP
            r = feols(sub, "__y__", [Y], absorb=[U, T], cluster=U, target=Y)
            print(f"  (internal reference only; not for the main text) beta = {r['coef']:.2f} "
                  f"(s.e. {r['se']:.2f}), p = {r['p']:.4f}")
    elif len(sub) and sub[Y].nunique() > 1:
        sub = sub.copy()
        sub["__y__"] = sub["is_uncitral"].astype(float) * PP
        r = feols(sub, "__y__", [Y], absorb=[U, T], cluster=U, target=Y)
        print(f"  beta = {r['coef']:.2f} (s.e. {r['se']:.2f}), p = {r['p']:.4f}")
    d.to_parquet(out / "cases_with_treaty_choiceset.parquet", index=False)
    print(f"\n[done] -> {out.resolve()}")


if __name__ == "__main__":
    main()
