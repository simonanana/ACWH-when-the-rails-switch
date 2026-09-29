"""
config.py — the single source of parameters for the whole project.

Design principle: every "researcher degree of freedom" (thresholds, bloc definitions,
windows) must be declared explicitly here and must not be scattered across individual
scripts. This addresses the concern that, in an earlier version of the paper, B_i was
never defined and the cohort window was not disclosed; the root cause was that these
choices had no single source.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

CASES_FP = DATA / "cases_caselevel_final.parquet"
PANEL_FP = DATA / "analysis_panel_v4.parquet"

YEAR_MIN, YEAR_MAX = 2010, 2023

# --- Treatment definition --------------------------------------------------
# Reference specification. COHORT_MIN/MAX is the previously undisclosed window,
# now made explicit. Setting it to None imposes no restriction; both variants must
# be reported in the robustness checks.
SANCTION_INTENSITY_THRESHOLD = 0.10
COHORT_MIN = 2011
COHORT_MAX = 2022

# --- Sanctioning bloc B_i --------------------------------------------------
# An earlier version never defined B_i, which made §5 entirely irreproducible.
# Three nested definitions are given here; every result that uses B_i must be
# reported across all three.
# Note: the panel's g7_flag omits the United States, so it is never used to
# construct a bloc.
BLOC_G7 = {"USA", "GBR", "DEU", "FRA", "ITA", "CAN", "JPN"}
BLOC_EU27 = {
    "DEU", "FRA", "ITA", "ESP", "NLD", "BEL", "LUX", "AUT", "IRL", "SWE",
    "DNK", "FIN", "POL", "CZE", "HUN", "PRT", "GRC", "SVK", "SVN", "EST",
    "LVA", "LTU", "HRV", "ROU", "BGR", "CYP", "MLT",
}
BLOC_EXTRA = {"CHE", "AUS", "NZL", "NOR", "KOR", "ISL", "LIE"}

BLOCS = {
    "G7": BLOC_G7,
    "G7_EU27": BLOC_G7 | BLOC_EU27,
    "G7_EU27_EXTRA": BLOC_G7 | BLOC_EU27 | BLOC_EXTRA,
}
BLOC_PRIMARY = "G7_EU27"

# --- Forum "delocalisation ladder" (the new H2 outcome variable) -----------
# Theoretical core: what the respondent avoids is not an "institution" but the
# self-contained annulment and recognition mechanism of the ICSID Convention.
# The ladder is ordered from lowest enforcement exposure (fully delocalised) to
# highest (dependent on national courts).
# Level 0 = ICSID Convention; 1 = ICSID Additional Facility; 2 = administered
# non-ICSID; 3 = no administering institution / expressly ad hoc.
LADDER_LABELS = {
    0: "ICSID Convention",
    1: "ICSID Additional Facility",
    2: "Administered non-ICSID",
    3: "Unadministered / ad hoc",
}

# --- Inference -------------------------------------------------------------
N_BOOT_WCB = 1999      # number of wild cluster bootstrap draws
N_PERM_RI = 999        # number of permutations for randomisation inference
SEED = 20260831

# --- Feasibility gate ------------------------------------------------------
# Below these thresholds, scripts refuse to report estimates (see feasibility.py).
MIN_COVERAGE = 0.50        # minimum outcome-variable coverage
MIN_POSITIVES = 30         # minimum number of positives for a binary outcome
MIN_SWITCHING_CLUSTERS = 5 # minimum number of switching clusters

PANEL_CONTROLS = ["ln_gdp", "gdp_growth_it", "trade_pct_gdp", "fdi_inflow_gdp_it_w"]
