"""
io_load.py — load the two parquet files and construct the treatment variable and all
derived outcome variables.

All downstream scripts obtain their data only through this module, so that no definition
can diverge between two places.
"""
import numpy as np
import pandas as pd

from config import (BLOCS, CASES_FP, COHORT_MAX, COHORT_MIN, PANEL_FP,
                    SANCTION_INTENSITY_THRESHOLD)

ICSID_CONV = "ICSID (INTERNATIONAL CENTRE FOR SETTLEMENT OF INVESTMENT DISPUTES)"
ICSID_AF = "ICSID AF (ICSID ADDITIONAL FACILITY)"
UNADMIN = ["NONE (NO ADMINISTERING INSTITUTION)", "DATA NOT AVAILABLE"]


def load_panel():
    p = pd.read_parquet(PANEL_FP)
    return p


def load_cases():
    c = pd.read_parquet(CASES_FP).reset_index(drop=True)

    c["is_icsid_conv"] = (c.rules == ICSID_CONV)
    c["is_icsid_af"] = (c.rules == ICSID_AF)
    c["is_icsid_any"] = c.rules.str.startswith("ICSID")
    c["is_uncitral"] = (c.rules == "UNCITRAL")
    c["non_icsid"] = ~c.is_icsid_any

    c["is_pca"] = c.institution.str.startswith("PCA")
    c["is_unadmin"] = c.institution.isin(UNADMIN)

    # --- Delocalisation ladder (the new ordinal outcome variable) --------------
    # See config.LADDER_LABELS. This is a feasible replacement for H2 "choice of seat"
    # in the original design: the seat has only 6.8% coverage and is unusable, but the
    # theoretical core, "enforcement exposure", can be measured with 100% coverage from
    # the combination of arbitral rules and administering institution.
    lad = np.full(len(c), np.nan)
    lad[c.is_icsid_conv.values] = 0
    lad[c.is_icsid_af.values] = 1
    lad[(~c.is_icsid_any & ~c.is_unadmin).values] = 2
    lad[(~c.is_icsid_any & c.is_unadmin).values] = 3
    c["ladder"] = lad

    # --- Full forum categories (for the multinomial logit) ---------------------
    def forum_cat(r):
        if r.is_icsid_conv:
            return "ICSID"
        if r.is_icsid_af:
            return "ICSID_AF"
        if r.is_uncitral and r.is_pca:
            return "UNCITRAL_PCA"
        if r.is_uncitral:
            return "UNCITRAL_other"
        return "Other_institutional"

    c["forum"] = c.apply(forum_cat, axis=1)
    return c


def build_cohorts(panel, threshold=SANCTION_INTENSITY_THRESHOLD,
                  cohort_min=COHORT_MIN, cohort_max=COHORT_MAX,
                  var="sanction_fin_norm_it"):
    """Return iso3 -> the first year in which the threshold is reached (onset year g).

    When cohort_min/max is None no window is imposed. Whether a window is applied must
    be reported in the output.
    """
    d = panel[panel[var] >= threshold]
    g = d.groupby("iso3").year.min()
    if cohort_min is not None:
        g = g[g >= cohort_min]
    if cohort_max is not None:
        g = g[g <= cohort_max]
    return g


def attach_treatment(cases, g):
    """Attach the treatment indicator S and event time et to the case table."""
    c = cases.copy()
    c["g"] = c.respondent_iso3.map(g)
    c["S"] = ((c.g.notna()) & (c.year >= c.g)).astype(float)
    c["et"] = c.year - c.g
    return c


def panel_treatment(panel, g):
    d = panel.copy()
    d["g"] = d.iso3.map(g)
    d["D"] = ((d.g.notna()) & (d.year >= d.g)).astype(float)
    d["et"] = d.year - d.g
    return d


def add_bloc(cases, bloc_name):
    """Add to the case table whether the claimant's home State belongs to the sanctioning bloc B_i."""
    c = cases.copy()
    c["bloc"] = c.claimant_iso3.isin(BLOCS[bloc_name]).astype(float)
    return c


def switching_clusters(cases_with_S, unit="respondent_iso3"):
    """Return the units whose treatment status switches within the sample — those that actually identify beta."""
    v = cases_with_S.groupby(unit).S.nunique()
    return sorted(v[v > 1].index.tolist())


def identification_summary(cases_with_S, unit="respondent_iso3"):
    sw = switching_clusters(cases_with_S, unit)
    return {
        "n_obs": int(len(cases_with_S)),
        "n_clusters": int(cases_with_S[unit].nunique()),
        "n_clusters_any_treated": int(cases_with_S.groupby(unit).S.max().sum()),
        "n_switching_clusters": len(sw),
        "switching_clusters": sw,
        "n_treated_obs": int(cases_with_S.S.sum()),
        "n_treated_obs_in_switchers": int(
            cases_with_S[cases_with_S[unit].isin(sw)].S.sum()),
        "n_identifying_obs": int(cases_with_S[unit].isin(sw).sum()),
    }
