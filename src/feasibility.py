"""
feasibility.py - feasibility gate.

This is the most important module in the project, and the largest departure from
an earlier version of this analysis.

The three errata to the earlier version of the paper (Section 7) were all the same
kind of failure: an estimate ran, the number looked usable, and so it was written
into the paper, without anyone first asking "does this dataset actually support
this question?"

The gate here runs before estimation. When verdict != "estimable", downstream
scripts must print the reason and skip, rather than emit a number that could be
written into the paper.

Hypotheses from the original architecture that the gate blocks:
  H2 "choice of seat"                       -> seat coverage 6.8%, 5 values        -> BLOCKED
  H3 "enforcement friction"                 -> no compliance-outcome field in data -> BLOCKED
  "reallocation to Singapore/HK/Dubai"      -> 5 non-Western institution cases     -> BLOCKED
These are not regrets but findings: they are themselves empirical conclusions about
"what this literature can ask".
"""
import numpy as np
import pandas as pd

from config import MIN_COVERAGE, MIN_POSITIVES, MIN_SWITCHING_CLUSTERS


def gate_outcome(series, name, min_cov=MIN_COVERAGE, min_pos=MIN_POSITIVES):
    """Check coverage and number of positives for a candidate outcome variable."""
    n = len(series)
    cov = series.notna().mean()
    if series.dropna().isin([0, 1, True, False]).all():
        pos = int(series.fillna(0).astype(float).sum())
        binary = True
    else:
        pos = int(series.notna().sum())
        binary = False

    reasons = []
    if cov < min_cov:
        reasons.append(f"coverage {cov:.1%} < threshold {min_cov:.0%}")
    if binary and pos < min_pos:
        reasons.append(f"positives {pos} < threshold {min_pos}")
    if binary and pos > 0 and (n - pos) < min_pos:
        reasons.append(f"negatives {n - pos} < threshold {min_pos}")

    return {
        "outcome": name, "n": n, "coverage": round(float(cov), 4),
        "positives": pos, "binary": binary,
        "verdict": "estimable" if not reasons else "NOT estimable",
        "reasons": "; ".join(reasons) if reasons else "-",
    }


def gate_design(ident, min_switch=MIN_SWITCHING_CLUSTERS):
    """Check the number of switching clusters for the identification design."""
    reasons = []
    if ident["n_switching_clusters"] < min_switch:
        reasons.append(
            f"switching clusters {ident['n_switching_clusters']} < threshold {min_switch}")
    if ident["n_treated_obs_in_switchers"] < 20:
        reasons.append(
            f"treated observations within switching units {ident['n_treated_obs_in_switchers']} < 20")
    return {
        "verdict": "estimable" if not reasons else "UNDERPOWERED",
        "reasons": "; ".join(reasons) if reasons else "-",
        **{k: v for k, v in ident.items() if k != "switching_clusters"},
    }


def mde(se, power=0.80, alpha=0.05):
    """Minimum detectable effect. Two-sided alpha, at the given power."""
    from scipy.stats import norm
    return (norm.ppf(1 - alpha / 2) + norm.ppf(power)) * se


def posthoc_power(beta, se, alpha=0.05):
    """Post-hoc power at the observed point estimate. A low value means the design cannot distinguish a moderate effect from zero."""
    from scipy.stats import norm
    crit = norm.ppf(1 - alpha / 2)
    return float(norm.cdf(abs(beta) / se - crit) + norm.cdf(-abs(beta) / se - crit))


def tost(beta, se, delta):
    """Equivalence p-value from two one-sided tests. H0: |beta| >= delta."""
    from scipy.stats import norm
    p1 = norm.cdf((delta - beta) / se)
    p2 = norm.cdf((delta + beta) / se)
    return float(1 - min(p1, p2))


def audit_all_outcomes(cases):
    """Replicate and extend the feasibility audit in Table 1 of the paper."""
    rows = []
    cand = {
        "UNCITRAL rules": cases.is_uncitral.astype(float),
        "Non-ICSID rules": cases.non_icsid.astype(float),
        "UNCITRAL x PCA administered": (cases.is_uncitral & cases.is_pca).astype(float),
        "Unadministered / undisclosed": cases.is_unadmin.astype(float),
        "Delocalisation ladder (ordered)": cases.ladder,
        "admin_class = ad hoc": cases.admin_adhoc.astype(float),
        "admin_class = non-West": cases.admin_nonwest.astype(float),
        "Payment barrier (broad)": cases.payment_barrier_broad.astype(float),
        "Payment barrier (strict)": cases.payment_barrier_strict.astype(float),
        "Alleged expropriation (placebo)": cases.alleged_expropriation.astype(float),
        "Amount claimed (USD)": cases.amount_claimed_usd,
    }
    for k, v in cand.items():
        rows.append(gate_outcome(v, k))

    # Seat: must be handled separately, because notna coverage is itself the problem
    seat = cases.seat_iso3
    rows.append({
        "outcome": "Arbitral seat (original architecture H2)", "n": len(cases),
        "coverage": round(float(seat.notna().mean()), 4),
        "positives": int(seat.nunique()), "binary": False,
        "verdict": "NOT estimable",
        "reasons": f"coverage {seat.notna().mean():.1%}; only {seat.nunique()} distinct values",
    })
    rows.append({
        "outcome": "Award enforcement (original architecture H3)", "n": len(cases),
        "coverage": 0.0, "positives": 0, "binary": False,
        "verdict": "NOT estimable",
        "reasons": "UNCTAD Navigator does not record compliance outcomes; no such field in the data",
    })
    return pd.DataFrame(rows)
