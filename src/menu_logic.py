"""
menu_logic.py — **monotone (partial-identification)** construction of `menu_icsid_and_uncitral`

Why a separate module is needed
-------------------------------
r25 and coding_sensitivity each once had their own implementation, with inconsistent
counts (239 vs 267), and **both were wrong**. The problem lies in how to handle a case
that invokes several treaties of which only some have been coded.

The correct logic is monotone:
    The treaty forum menu is a union (the investor may sue under any one treaty)
    => an uncoded treaty can **only add** forums, never remove them.

Therefore:
    menu_both = 1   when **the coded part alone** already provides both ICSID and UNCITRAL
                    — the conclusion is then **determined**, whatever the remaining
                    treaties turn out to be;
    menu_both = 0   only when **all** invoked treaties are coded and the union lacks one
                    of the two;
    otherwise       unknown.

Errors in the earlier implementations:
    r25: required every treaty to be coded before a case counted as known -> needlessly
         discarded 22 cases that were already determined;
    sensitivity: declared a case known from the coded subset alone -> over-asserted in
         cases where the subset is not sufficient to decide.

This rule has a by-product: **revealed bounds can be used directly**.
A lower bound supplies only a few 1s and leaves the rest blank; under the monotone rule,
as long as both the ICSID and the UNCITRAL 1 are present, that treaty is sufficient to set
the case to menu_both=1, without filling in the other six columns.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

FORUM_COLS = ["allows_icsid_convention", "allows_icsid_af", "allows_uncitral",
              "allows_scc", "allows_icc", "allows_other_institution",
              "allows_domestic_courts", "fork_in_the_road"]


def split_multi(s):
    if pd.isna(s):
        return []
    return [p.strip() for p in re.split(r"[;\n\r]+", str(s))
            if p.strip() and p.strip().lower() not in
            ("", "nan", "none", "data not available")]


def prepare_coding(coding: pd.DataFrame, strict_af: bool = True) -> pd.DataFrame:
    """Reshape the coding table into a form indexed by iia with three decision columns."""
    cd = coding.copy()
    cd["iia"] = cd["iia"].astype(str).str.strip()
    for c in FORUM_COLS + ["icsid_af_conditional"]:
        if c not in cd.columns:
            cd[c] = np.nan
        cd[c] = pd.to_numeric(cd[c], errors="coerce")
    cd = cd.drop_duplicates("iia").set_index("iia")

    # Conditional AF merely restates Article 2 of the AF Rules and is collinear with
    # the Convention-eligibility layer in r22, so it is not counted
    af_eff = cd["allows_icsid_af"].where(
        cd["icsid_af_conditional"].fillna(0) != 1, 0)

    cd["__icsid_yes__"] = ((cd["allows_icsid_convention"].fillna(0) > 0) |
                           (af_eff.fillna(0) > 0))
    cd["__unc_yes__"] = cd["allows_uncitral"].fillna(0) > 0
    # "Fully coded" = both the ICSID and the UNCITRAL channel have an explicit value
    # (0 or 1); the other six columns need not be filled — they do not enter
    # menu_icsid_and_uncitral
    cd["__complete__"] = (cd["allows_icsid_convention"].notna() &
                          cd["allows_uncitral"].notna())
    return cd


def attach_menu(cases: pd.DataFrame, coding: pd.DataFrame,
                iia_col: str = "iia", strict_af: bool = True) -> pd.DataFrame:
    """
    Return a copy with the following columns added:
      treaty_menu_known         1 = menu_icsid_and_uncitral has been determined
      menu_icsid_and_uncitral   1 / 0 / -1 (unknown)
      menu_basis                'resolved_by_subset' / 'all_coded' / 'unknown'
    """
    cd = prepare_coding(coding, strict_af)
    d = cases.copy()
    items = d[iia_col].map(split_multi)

    known, both, basis = [], [], []
    for lst in items:
        coded = [x for x in lst if x in cd.index and cd.loc[x, "__complete__"]]
        if coded and cd.loc[coded, "__icsid_yes__"].any() and \
                cd.loc[coded, "__unc_yes__"].any():
            known.append(1); both.append(1); basis.append("resolved_by_subset")
            continue
        all_coded = bool(lst) and len(coded) == len(lst)
        if all_coded:
            known.append(1); both.append(0); basis.append("all_coded")
        else:
            known.append(0); both.append(-1); basis.append("unknown")

    d["treaty_menu_known"] = known
    d["menu_icsid_and_uncitral"] = both
    d["menu_basis"] = basis
    return d
