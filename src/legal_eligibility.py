"""
legal_eligibility.py — turn "Convention-status eligibility of the forum" and "enforcement
regime" into estimable variables.

Why this layer is necessary
---------------------------
The current design treats `is_uncitral` / `non_icsid` as a **choice**. In investment
arbitration, however, the availability of ICSID Convention arbitration is not a choice but
a **jurisdictional precondition**:

    ICSID Convention Art. 25(1) — the dispute must arise between a Contracting State and
    **a national of another Contracting State**.
    => Convention eligible  <=>  the respondent State and the claimant's home State are
       **both Contracting States** on the filing date.

    Additional Facility Rules, Art. 2 — the Centre may administer proceedings where
    either the State party to the dispute **or** the investor's home State is **not** a
    Contracting State.
    => AF eligible          <=>  **exactly one** of the respondent State and the
       claimant's home State is a Contracting State (XOR).

    Neither is a Contracting State => the ICSID system as a whole is unavailable;
    `non_icsid = 1` is **mechanical** and unrelated to sanctions.

Direct consequences of this layer (three things this project must check)
------------------------------------------------------------------------
1. Among the 8 switching States, **Russia never ratified the Convention** (signed 1992,
   not ratified); **for Kyrgyzstan it entered into force only on 2022-05-21**;
   **Venezuela's denunciation took effect on 2012-07-25**; Iraq's ratification status is
   still to be checked. In other words, for the States contributing most to the
   coefficient, `non_icsid`/`is_uncitral` is determined to a considerable extent by
   Contracting-State status rather than by sanctions.
2. Respondent-State fixed effects absorb **time-invariant** Contracting-State status
   (Russia) but **do not absorb** **within-sample switches** such as Venezuela 2012 and
   Kyrgyzstan 2022; the latter are correlated with treatment timing and may be exactly
   the mechanical source of the long-horizon lead coefficient of -34.6 in the event
   study, and hence of the Rambachan–Roth breakdown value M̄=0.04.
3. The interpretation "leaving the ICSID Convention's self-contained annulment and
   recognition regime" is legally **wrong** for Russia — it was never inside that
   regime. If a referee with a background in arbitration law picked up on this
   sentence, the cost would be very high.

Usage
-----
    from legal_eligibility import EligibilityTable, add_forum_availability, \
        add_enforcement_regime

    elig = EligibilityTable.load("data/reference/icsid_convention_status.csv")
    cases = add_forum_availability(cases, elig,
                                   host="respondent_iso3",
                                   home="claimant_home_iso3",
                                   date="filing_date")   # or the year column
    cases = add_enforcement_regime(cases, rules="rules",
                                   institution="institution")

Required columns (override via arguments if named differently)
---------------------------------------------------------------
    respondent_iso3, claimant_home_iso3, filing_date (or year), rules, institution
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_REF = Path(__file__).resolve().parents[1] / "data" / "reference" / \
    "icsid_convention_status.csv"

# Non-ISO3 entities in the case file -> reference-table keys
ISO_ALIASES = {"European Union": "EUU", "EU": "EUU"}
MISSING_TOKENS = {"", "nan", "None", "Data not available", "DATA NOT AVAILABLE"}


@dataclass
class EligibilityTable:
    """ICSID Convention Contracting-State status (by date).

    Reference-table schema: iso3, state, in_force_from, out_of_force_from, status_note, verified
      - in_force_from empty          -> never a Contracting State (signed but not
                                        ratified / never signed)
      - out_of_force_from non-empty and > in_force_from -> denunciation (VEN/BOL)
      - out_of_force_from non-empty and < in_force_from -> re-accession after
                                        denunciation (ECU)
    States not listed in the reference table are **no longer assumed to be Contracting
    States**: by default is_party=False and the State is recorded as unknown, to rule out
    silent failure. If Contracting-State status really should be the default, pass
    default_is_party=True explicitly.
    """

    ref: pd.DataFrame
    default_is_party: bool = False
    _unknown: set = None  # noqa: RUF012

    @classmethod
    def load(cls, path: str | Path = DEFAULT_REF,
             default_is_party: bool = False) -> "EligibilityTable":
        ref = pd.read_csv(path, dtype=str).fillna("")
        for c in ("in_force_from", "out_of_force_from"):
            ref[c] = pd.to_datetime(ref[c].replace("", np.nan), errors="coerce")
        unver = ref[~ref["verified"].str.upper().eq("TRUE")]
        if len(unver):
            warnings.warn(
                "The reference table has %d rows with verified=FALSE (%s). Check against the "
                "official ICSID/3 list before submission: "
                "https://icsid.worldbank.org/resources/lists/icsid-3"
                % (len(unver), ", ".join(unver["iso3"].tolist()[:12])),
                stacklevel=2)
        obj = cls(ref=ref.drop_duplicates("iso3").set_index("iso3"),
                  default_is_party=default_is_party)
        obj._unknown = set()
        return obj

    def is_party(self, iso3, when: pd.Timestamp) -> bool:
        """Whether the State is a Contracting State to the Convention on the given date."""
        if not isinstance(iso3, str) or not iso3 or pd.isna(when):
            return False
        iso3 = ISO_ALIASES.get(iso3, iso3)
        if iso3 not in self.ref.index:
            self._unknown.add(iso3)
            return self.default_is_party
        row = self.ref.loc[iso3]
        inf, out = row["in_force_from"], row["out_of_force_from"]
        if pd.isna(inf):
            return False
        if pd.isna(out):
            return when >= inf
        if inf < out:                      # accession followed by denunciation (VEN, BOL)
            return inf <= when < out
        return (when < out) or (when >= inf)   # re-accession after denunciation (ECU)

    def unknown_states(self) -> list[str]:
        return sorted(self._unknown or [])


# --------------------------------------------------------------------------
def add_forum_availability(df: pd.DataFrame, elig: EligibilityTable,
                           host: str = "respondent_iso3",
                           home: str = "claimant_iso3",
                           date: str | None = "filing_date",
                           year: str | None = "year") -> pd.DataFrame:
    """
    Add Convention-status eligibility flags:

      icsid_convention_available : host and home are both Contracting States
      icsid_af_available         : exactly one is a Contracting State (AF Rules Art. 2)
      icsid_any_available        : either of the above
      icsid_unavailable_mech     : neither is a Contracting State -> non_icsid is
                                   mechanically 1
      home_missing               : claimant's home State missing (cannot be determined;
                                   kept as a separate category)
      choice_set_type            : {both_parties, af_only, none, unknown}

    Date convention: `date` (day precision) is preferred; if missing, fall back to
    1 July of `year` (mid-year approximation). **Venezuela 2012-07-25 and Kyrgyzstan
    2022-05-21 both fall close to mid-year, so the annual approximation would
    misclassify them; this is the second reason for the planned supplementary
    collection of monthly data.**
    """
    d = df.copy()
    if date and date in d.columns:
        when = pd.to_datetime(d[date], errors="coerce")
    else:
        when = pd.Series(pd.NaT, index=d.index)
    if year and year in d.columns:
        fallback = pd.to_datetime(d[year].astype("Int64").astype(str) + "-07-01",
                                  errors="coerce")
        when = when.fillna(fallback)
    d["__when__"] = when

    host_party = [elig.is_party(h, w) if pd.notna(w) else np.nan
                  for h, w in zip(d[host], d["__when__"])]
    home_raw = d[home] if home in d.columns else pd.Series(np.nan, index=d.index)
    home_missing = home_raw.isna() | home_raw.astype(str).str.strip().isin(MISSING_TOKENS)
    home_party = [np.nan if m else (elig.is_party(h, w) if pd.notna(w) else np.nan)
                  for h, m, w in zip(home_raw, home_missing, d["__when__"])]

    d["host_icsid_party"] = pd.Series(host_party, index=d.index).astype("boolean")
    d["home_icsid_party"] = pd.Series(home_party, index=d.index).astype("boolean")
    d["home_missing"] = home_missing

    hp, mp = d["host_icsid_party"], d["home_icsid_party"]
    d["icsid_convention_available"] = (hp & mp).astype("boolean")
    d["icsid_af_available"] = (hp ^ mp).astype("boolean")
    d["icsid_any_available"] = (hp | mp).astype("boolean")
    d["icsid_unavailable_mech"] = (~hp & ~mp).astype("boolean")

    def _ctype(r):
        if bool(r["home_missing"]) or pd.isna(r["host_icsid_party"]) or \
                pd.isna(r["home_icsid_party"]):
            return "unknown"
        if r["icsid_convention_available"]:
            return "both_parties"
        if r["icsid_af_available"]:
            return "af_only"
        return "none"

    d["choice_set_type"] = d.apply(_ctype, axis=1)
    d = d.drop(columns="__when__")

    if elig.unknown_states():
        warnings.warn(
            f"The following ISO3 codes do not appear in the reference table and were treated "
            f"with the default is_party={elig.default_is_party} (default False = treated "
            f"as a non-Contracting State). Please add them to the reference table: "
            f"{elig.unknown_states()[:40]}", stacklevel=2)
    return d


# --------------------------------------------------------------------------
_CONVENTION_PAT = r"^\s*ICSID(?!\s*AF)(?!.*ADDITIONAL)"
_AF_PAT = r"ADDITIONAL\s*FACILITY|ICSID\s*\(AF\)|ICSID\s*AF"


def add_enforcement_regime(df: pd.DataFrame, rules: str = "rules",
                           institution: str = "institution") -> pd.DataFrame:
    """
    Fully separate the ICSID Convention and the Additional Facility in legal terms, and
    construct the **enforcement-regime** outcome variable.

      forum_class ∈ {ICSID Convention, ICSID Additional Facility, UNCITRAL,
                     SCC, ICC, Other institutional, Expressly ad hoc/undisclosed}

      convention_regime      = 1  the award is subject only to ICSID's internal annulment
                                  procedure (Convention Arts 53–55) and does not pass
                                  through national-court recognition proceedings
      natcourt_exposed       = 1  the award requires recognition by national courts / can
                                  be set aside (New York Convention system: AF, UNCITRAL,
                                  SCC, ICC, ad hoc)

    This is the key to upgrading "away from ICSID" into a defensible legal proposition:
        away from a self-contained, national-court-insulated enforcement
        architecture, towards awards exposed to national-court review and
        to the New York Convention public-policy exception.
    It also **directly recovers** the "enforcement" dimension promised in the original
    abstract — not through compliance data, which do not exist, but through the
    **institutional track** on which the award sits, which has 100% coverage.
    """
    d = df.copy()
    r = d[rules].astype(str)

    is_af = r.str.contains(_AF_PAT, case=False, regex=True, na=False)
    is_conv = r.str.contains(_CONVENTION_PAT, case=False, regex=True, na=False) & ~is_af
    is_unc = r.str.contains("UNCITRAL", case=False, na=False)
    is_scc = r.str.contains(r"\bSCC\b|Stockholm", case=False, regex=True, na=False)
    is_icc = r.str.contains(r"\bICC\b", case=False, regex=True, na=False)
    is_adhoc = r.str.contains("AD HOC|NOT AVAILABLE|UNDISCLOSED|UNKNOWN|^NONE",
                              case=False, regex=True, na=False)

    cls = np.select(
        [is_conv, is_af, is_unc, is_scc, is_icc, is_adhoc],
        ["ICSID Convention", "ICSID Additional Facility", "UNCITRAL",
         "SCC", "ICC", "Expressly ad hoc / undisclosed"],
        default="Other institutional")
    d["forum_class"] = cls

    d["convention_regime"] = (d["forum_class"] == "ICSID Convention").astype(int)
    d["natcourt_exposed"] = (1 - d["convention_regime"]).astype(int)

    if institution in d.columns:
        inst = d[institution].astype(str)
        d["is_pca_admin"] = inst.str.contains("PCA|Permanent Court",
                                              case=False, regex=True,
                                              na=False).astype(int)
        d["inst_undisclosed"] = inst.str.contains(
            "NOT AVAILABLE|UNDISCLOSED|^NONE|UNKNOWN", case=False,
            regex=True, na=False).astype(int)
    return d


# --------------------------------------------------------------------------
def eligibility_audit(df: pd.DataFrame, treat: str = "S",
                      unit: str = "respondent_iso3") -> pd.DataFrame:
    """
    Audit table in the style of the feasibility gate: examine **mechanical eligibility**
    before estimating anything.
    If the share of `icsid_unavailable_mech` in the treated group is markedly higher than
    in the control group, the current headline result is at least partly a product of
    legal eligibility rather than of sanctions.
    """
    rows = []
    for name, sub in [("All", df),
                      ("Treated", df[df[treat] == 1]),
                      ("Control", df[df[treat] == 0])]:
        n = len(sub)
        if n == 0:
            continue
        rows.append({
            "Sample": name, "N": n,
            "Convention available": float(sub["icsid_convention_available"]
                                          .fillna(False).mean()),
            "AF only": float(sub["icsid_af_available"].fillna(False).mean()),
            "ICSID unavailable (mechanical)":
                float(sub["icsid_unavailable_mech"].fillna(False).mean()),
            "Home state unknown": float(sub["home_missing"].mean()),
            "UNCITRAL rate": float(sub.get("is_uncitral", pd.Series(dtype=float))
                                   .mean()) if "is_uncitral" in sub else np.nan,
        })
    out = pd.DataFrame(rows)
    return out
