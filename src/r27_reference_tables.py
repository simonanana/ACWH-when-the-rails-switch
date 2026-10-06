"""
r27_reference_tables.py — Treaty-layer decomposition and parallel-claims robustness
under the reference treatment definition (paper Table 11, Table 13, and the
genuine-choice subsample of Section 5.10).

Why this module exists
----------------------
The paper reports these results under the reference treatment definition
(`bloc1_k2`) for both outcomes. The command-line entry points of `r23` and `r25`
construct treatment from the legacy intensity index (`config.py`) and report the
UNCITRAL outcome only, so running them does not reproduce the published tables.

This module adds **no new estimator**. It composes functions that already exist
in the repository:

    u00_common.load_base / load_cohorts / attach / Projector   (data, FE engine, CR1,
                                                                analytic weights)
    r23_claim_group.detect_claim_groups / collapse_to_groups   (parallel-claim groups)
    r25_iia_choiceset.attach_treaty_choice_set                 (treaty forum menus)
    legal_eligibility.add_forum_availability                   (four-way Convention-
                                                                status configuration)
    _compat.feols                                              (claim-group clustering)

and checks each number against the value printed in the paper.

Specification choices (they are what the published numbers correspond to)
----------------------------------------------------------------------------
Table 13 (parallel claims)
  * Groups: respondent × rules × administering institution × claimant home State,
    filing years within a two-year window (`detect_claim_groups` defaults). No
    applicable-treaty key is used because the case file carries no treaty column,
    and no manual override is applied to rows 3–5.
  * "Excluding the Crimea claims only" removes the ten cases flagged by the rule in
    `u00_common.load_base` (respondent RUS, claimant home UKR, 2015–2019, UNCITRAL).
  * "Excluding all parallel groups" keeps only cases whose group has one member
    (i.e. it drops every group of size >= 2).
  * "Inverse group-size weights" uses analytic weights 1/size in the `u00` engine.
Table 11 (two layers)
  * Row (1) absorbs the four-way configuration `choice_set_type`
    (both_parties / af_only / none / unknown) from `add_forum_availability`
    with the ICSID/3 reference table; rows (2) and (3) add the treaty-menu category,
    with uncoded treaties kept as their own category.

Inputs: only files committed to this repository (no Navigator Excel file, no panel).

Usage
-----
    python src/r27_reference_tables.py --out outputs
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from u00_common import (PP, TIME, UNIT, ROOT, Projector, attach, load_base,
                        load_cohorts, outdir, save)

warnings.filterwarnings("ignore")

OUTCOMES = ["natcourt_exposed", "is_uncitral"]
LABEL = {"natcourt_exposed": "Non-Convention", "is_uncitral": "UNCITRAL"}
NAV = ROOT / "data" / "cases_with_navigator.parquet"
MENU = ROOT / "data" / "reference" / "iia_forum_menu_v4.csv"
ICSID3 = ROOT / "data" / "reference" / "icsid_convention_status_icsid3.csv"

# Values printed in the paper (effect, s.e.), used only for the check column.
PAPER_T13 = {
    ("Baseline", "natcourt_exposed"): (33.28, 13.53),
    ("Baseline", "is_uncitral"): (15.53, 5.25),
    ("Excluding the Crimea claims only", "natcourt_exposed"): (33.99, 13.37),
    ("Excluding the Crimea claims only", "is_uncitral"): (14.84, 5.03),
    ("Inverse group-size weights", "natcourt_exposed"): (25.60, 11.31),
    ("Inverse group-size weights", "is_uncitral"): (9.77, 6.23),
    ("Collapsed to one case per group", "natcourt_exposed"): (22.96, 12.43),
    ("Collapsed to one case per group", "is_uncitral"): (8.36, 6.94),
    ("Excluding all parallel groups", "natcourt_exposed"): (20.73, 9.70),
    ("Excluding all parallel groups", "is_uncitral"): (7.71, 9.18),
}
PAPER_T11 = {
    ("(0) Year + respondent FE", "natcourt_exposed"): (33.28, 13.53),
    ("(1) + Convention-status configuration FE", "natcourt_exposed"): (8.86, 7.44),
    ("(2) + treaty forum-menu FE", "natcourt_exposed"): (34.36, 14.74),
    ("(3) + both", "natcourt_exposed"): (10.79, 9.12),
    ("(0) Year + respondent FE", "is_uncitral"): (15.53, 5.25),
    ("(1) + Convention-status configuration FE", "is_uncitral"): (0.74, 6.86),
    ("(2) + treaty forum-menu FE", "is_uncitral"): (17.12, 5.77),
    ("(3) + both", "is_uncitral"): (2.80, 6.30),
}


def _fit(x: pd.DataFrame, y: str, absorb: list[str], w: str | None = None) -> dict:
    x = x.reset_index(drop=True)
    P = Projector(x, absorb, None if w is None else x[w].to_numpy(float))
    r = P.fit(x[y].to_numpy(float) * PP, x["S"].to_numpy(float))
    r["n"] = len(x)
    return r


def _check(paper: tuple[float, float] | None, coef: float, se: float) -> str:
    if paper is None:
        return ""
    ok = abs(round(coef, 2) - paper[0]) < 0.006 and abs(round(se, 2) - paper[1]) < 0.006
    return "match" if ok else f"differs (paper {paper[0]:+.2f} ({paper[1]:.2f}))"


def table13(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from r23_claim_group import collapse_to_groups, detect_claim_groups
    from _compat import feols

    crimea = x["claim_group"].eq("CRIMEA")          # rule defined in u00_common.load_base
    g = detect_claim_groups(x.drop(columns=["claim_group", "claim_group_size"]))
    g["w_inv"] = 1.0 / g["claim_group_size"]
    rows, notes = [], []
    for y in OUTCOMES:
        cg = g.copy()
        cg["__y__"] = cg[y]
        coll = collapse_to_groups(cg, "__y__", "S", UNIT, TIME).rename(columns={"__y__": y})
        specs = [("Baseline", _fit(x, y, [UNIT, TIME])),
                 ("Excluding the Crimea claims only", _fit(x[~crimea], y, [UNIT, TIME])),
                 ("Inverse group-size weights", _fit(g, y, [UNIT, TIME], "w_inv")),
                 ("Collapsed to one case per group", _fit(coll, y, [UNIT, TIME])),
                 ("Excluding all parallel groups",
                  _fit(g[g["claim_group_size"] < 2], y, [UNIT, TIME]))]
        for lab, r in specs:
            rows.append({"Specification": lab, "Outcome": LABEL[y],
                         "Effect (pp)": r["coef"], "s.e.": r["se"], "p_CR1": r["p"],
                         "N": r["n"],
                         "check vs paper": _check(PAPER_T13.get((lab, y)), r["coef"], r["se"])})
        # Footnote: standard errors clustered on claim groups instead of respondent States
        g["__y__"] = g[y] * PP
        rc = feols(g, "__y__", ["S"], absorb=[UNIT, TIME], cluster="claim_group",
                   target="S", use_project=False)
        notes.append({"Outcome": LABEL[y], "Effect (pp)": rc["coef"],
                      "s.e. (cluster = claim group)": rc["se"], "p": rc["p"],
                      "clusters": rc["n_clusters"]})
    return pd.DataFrame(rows), pd.DataFrame(notes)


def table11_and_choice(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    import legal_eligibility as le
    from r25_iia_choiceset import attach_treaty_choice_set

    nav = pd.read_parquet(NAV)[["case_id", "iia"]]
    d = x.merge(nav, on="case_id", how="left", validate="one_to_one")
    d = le.add_forum_availability(d, le.EligibilityTable.load(ICSID3))
    d = attach_treaty_choice_set(d, pd.read_csv(MENU))
    d["__conv__"] = d["choice_set_type"].astype(str)
    d["__menu__"] = np.where(d["treaty_menu_known"] == 1,
                             d["menu_icsid_and_uncitral"].fillna(-1).astype(int).astype(str),
                             "unknown")
    d["__both__"] = d["__conv__"] + "|" + d["__menu__"]
    specs = [("(0) Year + respondent FE", []),
             ("(1) + Convention-status configuration FE", ["__conv__"]),
             ("(2) + treaty forum-menu FE", ["__menu__"]),
             ("(3) + both", ["__both__"])]
    rows = []
    for y in OUTCOMES:
        base = None
        for lab, extra in specs:
            r = _fit(d, y, [UNIT, TIME] + extra)
            base = r["coef"] if base is None else base
            rows.append({"Outcome": LABEL[y], "Specification": lab,
                         "Effect (pp)": r["coef"], "s.e.": r["se"], "p_CR1": r["p"],
                         "Change from (0)": r["coef"] - base, "N": r["n"],
                         "check vs paper": _check(PAPER_T11.get((lab, y)), r["coef"], r["se"])})
    cov = d.groupby("S")["treaty_menu_known"].mean()
    rows.append({"Outcome": "", "Specification":
                 f"menu coverage: treated {cov.get(1.0, np.nan):.3f}, "
                 f"others {cov.get(0.0, np.nan):.3f}"})

    # Genuine-choice subsample: Convention open on status grounds AND a coded treaty
    # whose clause offers both ICSID and UNCITRAL arbitration.
    conv = d["icsid_convention_available"].fillna(False).astype(bool)
    sub = d[(d["treaty_menu_known"] == 1) & (d["menu_icsid_and_uncitral"] == 1) & conv]
    sw = sorted(sub.groupby(UNIT)["S"].nunique().pipe(lambda s: s[s > 1]).index)
    share = sub.groupby("S")["is_uncitral"].mean()
    choice = pd.DataFrame([{
        "cases": len(sub), "respondent States": sub[UNIT].nunique(),
        "treated cases": int(sub["S"].sum()),
        "UNCITRAL share, treated": share.get(1.0, np.nan),
        "UNCITRAL share, others": share.get(0.0, np.nan),
        "switching States": " ".join(sw)}])
    return pd.DataFrame(rows), choice


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--definition", default="bloc1_k2")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    out = outdir(a.out)

    x = attach(load_base(), load_cohorts()[a.definition].dropna())

    print("\n=== Table 13: parallel claims ===")
    t13, fn = table13(x)
    save(t13, out, "tab_parallel_claims_reference.csv")
    print("\n=== Table 13 footnote: clustering on claim groups ===")
    save(fn, out, "tab_parallel_claims_groupcluster.csv")

    print("\n=== Table 11: two layers of legal predetermination ===")
    t11, choice = table11_and_choice(x)
    save(t11, out, "tab_two_layer_reference.csv")
    print("\n=== Section 5.10: genuine-choice subsample ===")
    save(choice, out, "tab_genuine_choice.csv")

    bad = [r for r in pd.concat([t13, t11])["check vs paper"].dropna()
           if r and r != "match"]
    print(f"\nChecks against the paper: {'all match' if not bad else bad}")


if __name__ == "__main__":
    main()
