"""
u02_gsdb_treatment.py — build a **transparent** family of treatment variables from the
GSDB case-level file (D1)

Motivation
----------
1. The current reference specification, `config.SANCTION_INTENSITY_THRESHOLD = 0.10`,
   applies to the panel's `sanction_fin_norm_it` — the index that the paper's appendix
   acknowledges **cannot be reproduced from the public GSDB**. The "sanctioning-bloc
   member count ≥3" described in §3.2 of the main text is not the variable that
   produces 25.50.
2. The GSDB dyadic file expands the EU into 27 sender States, which renders the "≥3"
   threshold meaningless. The **case-level file** (this script's input) records the EU
   as a single entity, `EU`, so the "EU counted as 1" convention can be constructed
   directly.

Input
-----
Global_Sanctions_Data_Base__GSDB__2023.csv (Release 4, case level, 1547 cases, 1949–2023)
Columns: case_id, sanctioned_state, sanctioning_state, begin, end, trade, …, financial,
         travel, other, target_mult, sender_mult, objective, success
Multiple countries are comma-separated; World Bank-style names ("Korea, South",
"Gambia, The") contain commas and are re-joined during parsing using a suffix list.

Treatment definitions (all absorbing; the cohort is the first year in 2010–2023 in
which the condition holds; only cohorts 2011–2022 are kept, consistent with
config.COHORT_MIN/MAX; States already meeting the condition in 2010 are treated before
the window = left-censored, enter the control group and are flagged separately)
----------------------------------------------------------------------
idx10          panel sanction_fin_norm_it ≥ 0.10 (current reference specification, for comparison)
bloc1_k1       ≥1 entity in the sanctioning bloc (EU counted as 1, USA, GBR, CHE, NOR)
               imposes financial sanctions
bloc1_k2       as above, ≥2
bloc1_k3       as above, ≥3
transatlantic  EU and USA both impose financial sanctions
us_fin         USA imposes financial sanctions
eu_fin         EU (including unilateral measures by member States) imposes financial sanctions
raw27_k3       ≥3 with the EU expanded into its 27 member States (reproduces the problem
               with the dyadic convention, for comparison)

**Unilateral** financial measures by EU member States are assigned to the EU entity
(convention: EU counted as 1). When the G7 is the sender it is expanded into USA, GBR,
EU (DEU/FRA/ITA assigned to the EU), CAN and JPN. The UN is recorded separately (un_fin)
and does not count towards the bloc.

Output (--out-data and --out)
-----------------------------
data/derived/gsdb_fin_country_year.csv     country-year measures (2005–2023)
data/derived/gsdb_treatment_cohorts.csv    cohort year for each definition (input to u03–u09)
data/derived/gsdb_name_iso3_crosswalk.csv  name-to-ISO3 crosswalk + list of unmatched names
paper_tables/tab_treatment_family.csv      identifying sample and headline estimate per definition
paper_tables/tab_switchers_gsdb.csv        appendix table of switching-State onsets
                                           (GSDB case ID, senders, objectives)
paper_tables/tab_treatment_overlap.csv     overlap of switching-State sets across definitions

Usage
-----
    python src/u02_gsdb_treatment.py \
        --gsdb data/Global_Sanctions_Data_Base__GSDB__2023.csv \
        --out-data data/derived --out paper_tables
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from u00_common import (OUTCOMES, PP, TIME, UNIT, Projector, attach, identification,
                        load_base, outdir, save)

EU27 = {"Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
        "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary",
        "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta",
        "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia",
        "Spain", "Sweden"}
BLOC_ENTITIES = ["EU", "USA", "GBR", "CHE", "NOR"]

SUFFIXES = {"The", "South", "North", "Arab Rep.", "Islamic Rep.", "Dem. Rep.", "Rep.",
            "RB", "Fed. Sts.", "PDR", "Democratic Republic of the", "Republic of",
            "Rep. of"}

NAME2ISO = {
    "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA", "Angola": "AGO",
    "Argentina": "ARG", "Armenia": "ARM", "Australia": "AUS", "Austria": "AUT",
    "Azerbaijan": "AZE", "Bahrain": "BHR", "Bangladesh": "BGD", "Belarus": "BLR",
    "Belgium": "BEL", "Belize": "BLZ", "Benin": "BEN", "Bolivia": "BOL",
    "Bosnia and Herzegovina": "BIH", "Brazil": "BRA", "Bulgaria": "BGR",
    "Burkina Faso": "BFA", "Burundi": "BDI", "Cambodia": "KHM", "Cameroon": "CMR",
    "Canada": "CAN", "Central African Republic": "CAF", "Ceylon": "LKA",
    "Sri Lanka": "LKA", "Chad": "TCD", "Chile": "CHL", "China": "CHN",
    "Colombia": "COL", "Comoros": "COM", "Congo": "COG",
    "Congo, Democratic Republic of the": "COD", "Congo, Dem. Rep.": "COD",
    "Congo, Rep.": "COG", "Costa Rica": "CRI", "Cote d'Ivoire": "CIV",
    "Croatia": "HRV", "Cuba": "CUB", "Cyprus": "CYP", "Czech Republic": "CZE",
    "Denmark": "DNK", "Djibouti": "DJI", "Dominican Republic": "DOM",
    "Ecuador": "ECU", "Egypt, Arab Rep.": "EGY", "El Salvador": "SLV",
    "Equatorial Guinea": "GNQ", "Eritrea": "ERI", "Estonia": "EST",
    "Ethiopia (excludes Eritrea)": "ETH", "Ethiopia": "ETH", "Fiji": "FJI",
    "Finland": "FIN", "France": "FRA", "Gabon": "GAB", "Gambia, The": "GMB",
    "Georgia": "GEO", "Germany": "DEU", "Ghana": "GHA", "Greece": "GRC",
    "Guatemala": "GTM", "Guinea": "GIN", "Guinea-Bissau": "GNB", "Guyana": "GUY",
    "Haiti": "HTI", "Honduras": "HND", "Hong Kong": "HKG", "Hungary": "HUN",
    "Iceland": "ISL", "India": "IND", "Indonesia": "IDN", "Iran": "IRN",
    "Iran, Islamic Rep.": "IRN", "Iraq": "IRQ", "Ireland": "IRL", "Israel": "ISR",
    "Italy": "ITA", "Jamaica": "JAM", "Japan": "JPN", "Jordan": "JOR",
    "Transjordan": "JOR", "Kazakhstan": "KAZ", "Kenya": "KEN",
    "Korea, North": "PRK", "Korea, South": "KOR", "Kosovo": "XKX", "Kuwait": "KWT",
    "Kyrgyzstan": "KGZ", "Laos": "LAO", "Latvia": "LVA", "Lebanon": "LBN",
    "Lesotho": "LSO", "Liberia": "LBR", "Libya": "LBY", "Liechtenstein": "LIE",
    "Lithuania": "LTU", "Luxembourg": "LUX", "Macedonia": "MKD",
    "North Macedonia": "MKD", "Malagasy Republic": "MDG", "Madagascar": "MDG",
    "Malawi": "MWI", "Malaya": "MYS", "Malaysia": "MYS", "Maldives": "MDV",
    "Mali": "MLI", "Malta": "MLT", "Marshall Islands": "MHL", "Mauritania": "MRT",
    "Mexico": "MEX", "Moldova": "MDA", "Monaco": "MCO", "Montenegro": "MNE",
    "Morocco": "MAR", "Mozambique": "MOZ", "Myanmar": "MMR", "Burma": "MMR",
    "Nepal": "NPL", "Netherlands": "NLD", "New Zealand": "NZL", "Nicaragua": "NIC",
    "Niger": "NER", "Nigeria": "NGA", "Norway": "NOR", "Pakistan": "PAK",
    "Palestine": "PSE", "Panama": "PAN", "Paraguay": "PRY", "Peru": "PER",
    "Philippines": "PHL", "Poland": "POL", "Portugal": "PRT", "Qatar": "QAT",
    "Romania": "ROU", "Russia": "RUS", "Rwanda": "RWA", "Saudi Arabia": "SAU",
    "Senegal": "SEN", "Serbia": "SRB", "Sierra Leone": "SLE", "Singapore": "SGP",
    "Slovakia": "SVK", "Slovenia": "SVN", "Somalia": "SOM", "South Africa": "ZAF",
    "South Sudan": "SSD", "Spain": "ESP", "Sudan": "SDN", "Sweden": "SWE",
    "Switzerland": "CHE", "Syria": "SYR", "Taiwan": "TWN", "Tajikistan": "TJK",
    "Tanzania": "TZA", "Thailand": "THA", "Togo": "TGO", "Tunisia": "TUN",
    "Turkey": "TUR", "Turkmenistan": "TKM", "Uganda": "UGA", "Ukraine": "UKR",
    "United Arab Emirates": "ARE", "United Kingdom": "GBR", "United States": "USA",
    "Uruguay": "URY", "Uzbekistan": "UZB", "Venezuela": "VEN", "Vietnam": "VNM",
    "Yemen": "YEM", "Yemen, North": "YEM", "Yemen, Rep.": "YEM", "Zambia": "ZMB",
    "Zimbabwe": "ZWE",
}
ORGS = {"EU": "EU", "UN": "UN", "G7": "G7"}  # other organisations do not count towards the bloc; recorded as OTHER_ORG


def split_names(s) -> list[str]:
    if pd.isna(s):
        return []
    toks = [t.strip() for t in re.split(r",", str(s))]
    out: list[str] = []
    for t in toks:
        if not t:
            continue
        if out and t in SUFFIXES:
            out[-1] = f"{out[-1]}, {t}"
        else:
            out.append(t)
    return out


def sender_entities(names: list[str]) -> set[str]:
    ent = set()
    for n in names:
        if n == "EU" or n in EU27:
            ent.add("EU")
        elif n == "G7":
            ent |= {"USA", "GBR", "EU", "CAN", "JPN"}
        elif n == "UN":
            ent.add("UN")
        elif n in NAME2ISO:
            ent.add(NAME2ISO[n])
        else:
            ent.add("OTHER_ORG")
    return ent


def build_country_year(gsdb: pd.DataFrame, y0: int = 2005, y1: int = 2023):
    g = gsdb[(gsdb.financial == 1) & (gsdb.end >= y0) & (gsdb.begin <= y1)].copy()
    rows, unmatched = [], {}
    for _, r in g.iterrows():
        tnames = split_names(r.sanctioned_state)
        snames = split_names(r.sanctioning_state)
        ents = sender_entities(snames)
        eu_members = {n for n in snames if n in EU27}
        for n in snames:
            if n not in NAME2ISO and n not in EU27 and n not in ORGS:
                unmatched[n] = unmatched.get(n, 0) + 1
        for tn in tnames:
            iso = NAME2ISO.get(tn)
            if iso is None:
                unmatched[tn] = unmatched.get(tn, 0) + 1
                continue
            for yr in range(max(int(r.begin), y0), min(int(r.end), y1) + 1):
                rows.append({"iso3": iso, "year": yr, "case_id": int(r.case_id),
                             "entities": ents, "eu_entity": "EU" in snames,
                             "eu_members": eu_members, "senders": r.sanctioning_state,
                             "objective": r.objective})
    long = pd.DataFrame(rows)

    def agg(grp):
        ents = set().union(*grp.entities)
        bloc = ents & set(BLOC_ENTITIES)
        eu_mem = set().union(*grp.eu_members)
        eu_raw = 27 if grp.eu_entity.any() else len(eu_mem)
        return pd.Series({
            "fin_any": 1,
            "n_bloc_eu1": len(bloc),
            "n_bloc_raw27": eu_raw + len(bloc - {"EU"}),
            "eu_fin": int("EU" in ents), "us_fin": int("USA" in ents),
            "uk_fin": int("GBR" in ents), "ch_fin": int("CHE" in ents),
            "no_fin": int("NOR" in ents), "un_fin": int("UN" in ents),
            "n_cases": grp.case_id.nunique(),
            "case_ids": " ".join(map(str, sorted(grp.case_id.unique()))),
            "objectives": ";".join(sorted(set(grp.objective.astype(str)))),
        })

    cy = long.groupby(["iso3", "year"]).apply(agg, include_groups=False).reset_index()
    cw = pd.DataFrame(sorted(unmatched.items(), key=lambda kv: -kv[1]),
                      columns=["name", "n_occurrences"])
    return cy, long, cw


DEFS = {
    "bloc1_k1": lambda t: t.n_bloc_eu1 >= 1,
    "bloc1_k2": lambda t: t.n_bloc_eu1 >= 2,
    "bloc1_k3": lambda t: t.n_bloc_eu1 >= 3,
    "transatlantic": lambda t: (t.eu_fin == 1) & (t.us_fin == 1),
    "us_fin": lambda t: t.us_fin == 1,
    "eu_fin": lambda t: t.eu_fin == 1,
    "raw27_k3": lambda t: t.n_bloc_raw27 >= 3,
}


def cohorts_from(cy: pd.DataFrame, panel_iso: list[str], cmin=2011, cmax=2022,
                 ymin=2010, ymax=2023) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = pd.MultiIndex.from_product([panel_iso, range(ymin, ymax + 1)],
                                      names=["iso3", "year"]).to_frame(index=False)
    t = grid.merge(cy, on=["iso3", "year"], how="left").fillna(
        {c: 0 for c in cy.columns if c not in ("iso3", "year", "case_ids", "objectives")})
    out = pd.DataFrame(index=sorted(panel_iso))
    flags = pd.DataFrame(index=sorted(panel_iso))
    for name, f in DEFS.items():
        on = t[f(t)]
        first = on.groupby("iso3").year.min()
        flags[f"{name}_left_censored"] = first.reindex(out.index).eq(ymin).astype(int)
        out[name] = first.where((first >= cmin) & (first <= cmax))
    return out, flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsdb", required=True)
    ap.add_argument("--out-data", default="data/derived")
    ap.add_argument("--out", default="paper_tables")
    ap.add_argument("--ref", default=None, help="ICSID reference table (default: u01 output)")
    a = ap.parse_args()
    od, ot = outdir(a.out_data), outdir(a.out)

    gsdb = pd.read_csv(a.gsdb)
    cy, long, cw = build_country_year(gsdb)
    cy.to_csv(od / "gsdb_fin_country_year.csv", index=False)
    cw.to_csv(od / "gsdb_name_iso3_crosswalk.csv", index=False)
    print(f"[u02] financial-sanction country-years: {len(cy)} rows; unmatched names: {len(cw)}"
          f" (mostly organisations or former States; see crosswalk)")

    from config import COHORT_MAX, COHORT_MIN, SANCTION_INTENSITY_THRESHOLD
    from io_load import build_cohorts, load_panel
    panel = load_panel()
    iso = sorted(panel.iso3.unique())
    coh, flags = cohorts_from(cy, iso, COHORT_MIN, COHORT_MAX)
    coh.insert(0, "idx10", build_cohorts(panel, SANCTION_INTENSITY_THRESHOLD)
               .reindex(coh.index))
    coh.index.name = "iso3"
    both = pd.concat([coh, flags], axis=1)
    both.index.name = "iso3"
    both.reset_index().to_csv(
        od / "gsdb_treatment_cohorts.csv", index=False)
    print(f"  -> {od / 'gsdb_treatment_cohorts.csv'}")

    # Consistency on the panel: transparent counts vs the panel's sanction_fin_count_it
    chk = panel[["iso3", "year", "sanction_fin_count_it", "sanction_fin_norm_it"]].merge(
        cy[["iso3", "year", "n_bloc_eu1", "n_bloc_raw27"]], on=["iso3", "year"],
        how="left").fillna(0)
    print("\n[u02] Correlation with the panel's legacy measures:")
    print(chk[["sanction_fin_norm_it", "sanction_fin_count_it", "n_bloc_eu1",
               "n_bloc_raw27"]].corr().round(3).to_string())

    # Identifying sample and headline estimate per definition (no Convention-status
    # eligibility control; for the three-valued control see u03)
    d = load_base(a.ref)
    rows, sets = [], {}
    for name in coh.columns:
        x = attach(d, coh[name].dropna().to_dict())
        info = identification(x)
        sets[name] = set(info["switcher_list"].split()) if info["switcher_list"] else set()
        rec = {"definition": name, "n_cohort_states": int(coh[name].notna().sum()), **info}
        for y in OUTCOMES:
            for cod in ("none", "three"):
                xx = x.reset_index(drop=True)
                absorb = [UNIT, TIME] + ([] if cod == "none" else ["avail_three"])
                r = Projector(xx, absorb).fit(xx[y].to_numpy(float) * PP,
                                              xx["S"].to_numpy(float))
                rec[f"{y}_{cod}_b"] = r["coef"]
                rec[f"{y}_{cod}_se"] = r["se"]
                rec[f"{y}_{cod}_p"] = r["p"]
        rows.append(rec)
    fam = pd.DataFrame(rows)
    print("\n[u02] Family of treatment definitions:")
    save(fam, ot, "tab_treatment_family.csv")

    names = list(sets)
    ov = pd.DataFrame([[len(sets[i] & sets[j]) / max(len(sets[i] | sets[j]), 1)
                        for j in names] for i in names], index=names, columns=names)
    ov.index.name = "jaccard_switchers"
    ov.reset_index().to_csv(ot / "tab_treatment_overlap.csv", index=False)

    # Appendix table of switching States: for each definition, the switching States, their
    # onset year, and the senders and GSDB case IDs in the onset year
    app = []
    cyi = cy.set_index(["iso3", "year"])
    for name in coh.columns:
        for u in sorted(sets[name]):
            gy = int(coh.loc[u, name])
            rec = {"definition": name, "iso3": u, "onset": gy}
            if (u, gy) in cyi.index:
                r = cyi.loc[(u, gy)]
                rec.update({"n_bloc_eu1": int(r.n_bloc_eu1), "eu": int(r.eu_fin),
                            "us": int(r.us_fin), "uk": int(r.uk_fin),
                            "ch": int(r.ch_fin), "no": int(r.no_fin),
                            "un": int(r.un_fin), "gsdb_case_ids": r.case_ids,
                            "objectives": r.objectives})
            app.append(rec)
    save(pd.DataFrame(app), ot, "tab_switchers_gsdb.csv", show=False)


if __name__ == "__main__":
    main()
