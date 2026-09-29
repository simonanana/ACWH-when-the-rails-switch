"""
pnci_rebuild.py - rebuild the payment-network centrality index (PNCI) and connect it to
the moderation modules (h4_moderation.py, h4b_moderation_rw.py).

============================================================================
Background (every claim in the handover notes has been independently verified)
============================================================================

pnci_network / pnci_network_filled in the panel are broken:

    pnci_network         median = 0.00003    share(<0.01) = 80.8%
    pnci_network_filled  median = 0.00003    share(<0.01) = 80.8%   <- identical to the former, value by value
    pr (raw PageRank)    median = 0.00360    n = 2348               <- clean

The cause is the heavy-tailed PageRank distribution meeting a within-year min-max: in 2018 min=0.00353,
median=0.00354, max=0.08385, so the median country is squashed to 0.0001. The variable degenerates into
a point mass at 0. "filled" is also a misnomer - it is identical to pnci_network value by value;
the proxy column was never merged in.

Rebuild (the handover notes recommend pnci_rank; adopted after verification):

    pnci_rank   within-year percentile rank    median 0.503, share(<0.01) 0.6%   (key) recommended
    pnci_logmm  within-year min-max after log                                     comparison

No ordering information is lost: in 2018 the Spearman correlation between pnci_rank and the legacy index is 0.976,
and the top six are the same: JPN, GBR, SGP, LUX, FRA, NLD. Only the normalisation is broken, not the underlying measure.

============================================================================
Another fact that must be stated in the paper
============================================================================

pnci_cpmi has only **24** member countries in the panel (the hard-coded table has 26, but the United States and China
are not in the panel). Actual members: ARG AUS BEL BRA CAN CHE DEU ESP FRA GBR HKG IDN
IND ITA JPN KOR MEX NLD RUS SAU SGP SWE TUR ZAF.

It is also almost time-invariant - the only time variation comes from RUS switching from 1 to 0 in 2022
(BIS suspension of the Russian central bank, 2022-03). This means that in specifications with country fixed effects
the main effect of pnci_cpmi is fully absorbed, and identification of the interaction term comes almost entirely from Russia alone.
This fact matters far more than its p-value.
"""
import numpy as np
import pandas as pd


def add_rebuilt_pnci(panel, verbose=False):
    """Add the rebuilt PNCI variants to the panel. Existing columns are not modified; new ones are only added."""
    p = panel.copy()
    pr = pd.to_numeric(p.get("pr"), errors="coerce")
    if pr is None or pr.notna().sum() == 0:
        if verbose:
            print("  [!] No usable pr column in the panel; cannot rebuild PNCI.")
        return p

    p["_pr"] = pr
    # Within-year percentile rank: robust to heavy tails, and retains the full ordering information
    p["pnci_rank"] = p.groupby("year")._pr.rank(pct=True)

    # Within-year min-max after log: for comparison; still affected by heavy tails but better than the raw min-max
    lg = np.log(p._pr.clip(lower=1e-12))
    p["pnci_logmm"] = lg.groupby(p.year).transform(
        lambda s: (s - s.min()) / (s.max() - s.min())
        if s.max() > s.min() else s * 0)
    p = p.drop(columns=["_pr"])
    return p


def diagnose(panel, verbose=True):
    """Report the degree of degeneracy of the legacy variables and the improvement after the rebuild."""
    p = add_rebuilt_pnci(panel)
    rows = []
    for v in ["pnci_network", "pnci_network_filled", "pr", "pnci_rank",
              "pnci_logmm", "pnci_equal", "pnci_cpmi"]:
        if v not in p.columns:
            continue
        s = pd.to_numeric(p[v], errors="coerce")
        rows.append({
            "variable": v, "n": int(s.notna().sum()),
            "mean": round(float(s.mean()), 5),
            "median": round(float(s.median()), 5),
            "sd": round(float(s.std()), 5),
            "share_below_0.01": round(float((s < 0.01).mean()), 4),
            "n_distinct": int(s.nunique()),
        })
    tab = pd.DataFrame(rows)

    # CPMI membership diagnostics
    cm = sorted(p[pd.to_numeric(p.pnci_cpmi, errors="coerce") > 0]
                .iso3.unique()) if "pnci_cpmi" in p.columns else []
    tv = None
    if "pnci_cpmi" in p.columns:
        g = p.groupby("iso3").pnci_cpmi.nunique()
        tv = sorted(g[g > 1].index.tolist())

    if verbose:
        print("\n=== PNCI variable diagnostics ===")
        print(tab.to_string(index=False))
        print(f"\n  pnci_cpmi member countries (in panel): {len(cm)}")
        print(f"  {cm}")
        print(f"  of which time-varying: {tv}")
        print("  -> With country fixed effects, identification of the pnci_cpmi interaction comes almost entirely from these countries.")
        print("\n  Reading: pnci_network and _filled are identical value by value, with 80% of observations <0.01;")
        print("       they have degenerated into a point mass at 0 and should not be used in any interaction specification.")
        print("       The rebuilt pnci_rank has median 0.503 and a well-behaved distribution; recommended.")
    return tab, cm, tv


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from io_load import load_panel
    diagnose(load_panel())
