"""
_data.py — the single data-access entry point for r22, r23 and r25, layered on top of the
project's existing `io_load.py`.

Why it is needed
----------------
An earlier version of r22 failed with `KeyError: 'sanctioned'`: the case-level parquet
**never contained** a treatment variable. In this project the treatment variable is not
data; it is generated at run time by `io_load.build_cohorts()` + `attach_treatment()`
from the thresholds/windows in config (columns S / g / et). This is the correct design
("every number has a single source"), so r22, r23 and r25 should not construct their own
`sanctioned` but should **reuse the same generation chain**. This module is that
point of reuse.

It also performs schema validation: missing columns raise an error immediately instead
of failing silently.

Usage
-----
    from _data import load_analysis_frame
    d, g = load_analysis_frame()          # uses the project's config/io_load
    # d contains: S, g, et, claimant_iso3, respondent_iso3, year, rules, institution,
    #       is_uncitral, non_icsid, is_pca, is_unadmin, ladder, forum
    # g : iso3 -> cohort year
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Let both "python src/r22_*.py" and "python -m src.r22_*" find io_load / config
_HERE = Path(__file__).resolve().parent
for cand in (_HERE, _HERE.parent / "src"):
    if str(cand) not in sys.path:
        sys.path.insert(0, str(cand))

# Canonical column names (consistent with the output of the project's io_load)
COL = dict(unit="respondent_iso3", home="claimant_iso3", time="year",
           treat="S", cohort="g", event_time="et", rules="rules",
           institution="institution", case_id="case_id")

REQUIRED = ["case_id", "respondent_iso3", "claimant_iso3", "year", "rules",
            "institution", "is_uncitral", "non_icsid", "S", "g", "et"]


class SchemaError(RuntimeError):
    pass


def validate(d: pd.DataFrame, required=REQUIRED) -> None:
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise SchemaError(f"Analysis frame is missing columns: {missing}. "
                          f"Available columns: {sorted(d.columns)[:40]}…")
    if d["S"].nunique() < 2:
        raise SchemaError("Treatment variable S has no variation: check the config "
                          "threshold / cohort window.")
    if not np.issubdtype(d["year"].dtype, np.number):
        raise SchemaError("year is not numeric.")
    if d["respondent_iso3"].isna().any():
        raise SchemaError("respondent_iso3 has missing values.")


def load_analysis_frame(cases_path: str | None = None,
                        panel_path: str | None = None,
                        threshold: float | None = None,
                        cohort_min: int | None = None,
                        cohort_max: int | None = None,
                        treat_var: str | None = None,
                        cohort_def: str | None = None,
                        cohort_file: str | None = None):
    """
    Uses the project chain by default: config -> io_load.load_panel/load_cases/
    build_cohorts/attach_treatment. The threshold/window/treatment variable can be
    overridden via arguments (for robustness checks); by default everything follows config.

    If io_load cannot be imported, falls back to reading the parquet directly; the file
    must then already contain S/g/et.

    cohort_def (added 2026-09-17): use one column of the transparent cohort table
    produced by u02_gsdb_treatment.py (data/derived/gsdb_treatment_cohorts.csv) as the
    treatment definition, e.g. "bloc1_k2". When given, threshold / treat_var are ignored.
    """
    try:
        import config  # noqa: F401
        from io_load import (attach_treatment, build_cohorts, load_cases,
                             load_panel)
        if cases_path or panel_path:
            import config as _cfg
            if cases_path:
                _cfg.CASES_FP = Path(cases_path)
            if panel_path:
                _cfg.PANEL_FP = Path(panel_path)
        p = load_panel()
        c = load_cases()
        kw = {}
        if threshold is not None:
            kw["threshold"] = threshold
        if cohort_min is not None or cohort_max is not None:
            kw["cohort_min"], kw["cohort_max"] = cohort_min, cohort_max
        if treat_var is not None:
            kw["var"] = treat_var
        if cohort_def is not None:
            cf = Path(cohort_file) if cohort_file else (
                Path(_cfg_root()) / "data" / "derived" / "gsdb_treatment_cohorts.csv")
            tab = pd.read_csv(cf).set_index("iso3")
            if cohort_def not in tab.columns:
                raise SchemaError(f"{cf} has no treatment definition {cohort_def}; available: "
                                  f"{[c for c in tab.columns if not c.endswith('_left_censored')]}")
            g = tab[cohort_def].dropna()
        else:
            g = build_cohorts(p, **kw)
        d = attach_treatment(c, g)
    except ImportError:
        if not cases_path:
            raise SchemaError("Cannot import the project's io_load, and --cases was not given.")
        d = pd.read_parquet(cases_path)
        if "S" not in d.columns:
            raise SchemaError("Fallback mode requires the case file to contain S/g/et columns.")
        g = d.dropna(subset=["g"]).groupby("respondent_iso3")["g"].first()

    d = d.reset_index(drop=True).copy()
    d["year"] = d["year"].astype(int)
    for b in ("is_uncitral", "non_icsid", "is_pca", "is_unadmin",
              "is_icsid_conv", "is_icsid_af", "is_icsid_any"):
        if b in d.columns:
            d[b] = d[b].astype(int)
    validate(d)
    return d, g


def _cfg_root():
    import config
    return config.ROOT


def cohort_map_from(g, d: pd.DataFrame | None = None, unit: str = "respondent_iso3",
                    timing_only: bool = False) -> dict:
    """
    Cohort-assignment vector for the permutation test.
      timing_only=False (same convention as the project's randomisation_test):
          all respondent States enter the permutation pool, untreated States as NaN
          -> tests "which States are sanctioned + when"
      timing_only=True:
          onset years are permuted among ever-treated States only
          -> the stricter "timing-only" test
    """
    m = {k: float(v) for k, v in dict(g).items()}
    if timing_only or d is None:
        return m
    return {u: m.get(u, np.nan) for u in sorted(d[unit].unique())}


def treated_pre_post_table(d: pd.DataFrame, y: str = "is_uncitral") -> pd.DataFrame:
    """Raw 2x2: outcome rates and case counts for ever-treated / never × pre / post."""
    d = d.copy()
    d["ever"] = np.where(d["g"].notna(), "Ever-sanctioned respondent",
                         "Never-sanctioned respondent")
    d["period"] = np.where(d["S"] == 1, "Post-onset", "Pre-onset / never")
    t = d.groupby(["ever", "period"])[y].agg(rate="mean", n="size")
    t["rate"] = (t["rate"] * 100).round(1)
    return t
