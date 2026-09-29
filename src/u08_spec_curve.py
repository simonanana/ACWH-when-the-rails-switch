"""
u08_spec_curve.py — Specification curve + joint permutation inference over the whole
curve (M2)

Specification space (default 7 × 4 × 2 × 2 × 2 = 224)
    Treatment    idx10, bloc1_k1, bloc1_k2, bloc1_k3, transatlantic, eu_fin, us_fin
    Eligibility  none, strict, three, lax
    Outcome      natcourt_exposed, is_uncitral
    Sample       all, excl_VEN
    Weights      none, inverse claim-group size (one forum decision = one vote)
(raw27_k3 is identical to eu_fin at the case level and is not repeated.)

Joint inference (Simonsohn, Simmons & Nelson 2020; the joint-randomisation idea of
Young 2019)
    Test statistics, within each outcome: (i) the median β across specifications;
    (ii) the share of specifications with β>0 and p_CR1<0.05.
    Null distribution: within the set U of States treated under any definition,
            **permute each State's entire onset profile**
            (the onset vector moves jointly across definitions, preserving the
            dependence structure between definitions), B times, re-estimating
            every specification on each draw.
    This is the sharp null that which States were sanctioned, and when, is unrelated
    to forum composition.
    A "timing-only" version is also reported: each definition independently permutes
    onset years within its own set of treated States, holding fixed which States are
    sanctioned. This corresponds to the paper's claim that the effect is not a
    response to timing.

Specifications with fewer than 5 switching clusters are estimated as usual but
flagged low_switchers=True.

Output
------
paper_tables/tab_spec_curve.csv          β, CR1 s.e./p and number of switching States per specification
paper_tables/tab_spec_curve_joint.csv    joint test statistics and p-values
paper_tables/fig_spec_curve_<outcome>.pdf

Usage
-----
    python src/u08_spec_curve.py --B 499 --primary bloc1_k2 --out paper_tables
    # If a single run is time-limited, split it: run chunks with different seeds, then combine
    python src/u08_spec_curve.py --B 200 --seed 1 --out paper_tables
    python src/u08_spec_curve.py --B 200 --seed 2 --out paper_tables
    python src/u08_spec_curve.py --combine-only --out paper_tables
"""
from __future__ import annotations

import argparse
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from u00_common import (OUTCOMES, PP, TIME, UNIT, Projector, load_base,  # noqa: E402
                        load_cohorts, outdir, save)

DEFS = ["idx10", "bloc1_k1", "bloc1_k2", "bloc1_k3", "transatlantic", "eu_fin", "us_fin"]
CODINGS = ["none", "strict", "three", "lax"]
SAMPLES = ["all", "excl_VEN"]
WEIGHTS = ["none", "group"]


class SpecEngine:
    def __init__(self, base: pd.DataFrame):
        self.frames, self.proj, self.yres = {}, {}, {}
        for smp, cod, w in itertools.product(SAMPLES, CODINGS, WEIGHTS):
            x = base if smp == "all" else base[base[UNIT] != "VEN"]
            x = x.reset_index(drop=True)
            absorb = [UNIT, TIME] + ([] if cod == "none" else [f"avail_{cod}"])
            P = Projector(x, absorb, None if w == "none" else x["w_group"].to_numpy(float))
            key = (smp, cod, w)
            self.frames[key], self.proj[key] = x, P
            for y in OUTCOMES:
                self.yres[key + (y,)] = P.resid(x[y].to_numpy(float) * PP)

    def run(self, cohorts: dict[str, dict]) -> pd.DataFrame:
        rows = []
        S_cache = {}
        for key, x in self.frames.items():
            yrs = x[TIME].to_numpy(float)
            for dn in DEFS:
                g = x[UNIT].map(cohorts[dn]).to_numpy(float)
                S = ((~np.isnan(g)) & (yrs >= g)).astype(float)
                S_cache[(key, dn)] = S
        for (key, dn), S in S_cache.items():
            P, x = self.proj[key], self.frames[key]
            sw = x.assign(S=S).groupby(UNIT).S.nunique().gt(1).sum()
            for y in OUTCOMES:
                r = P.fit(None, S, yr=self.yres[key + (y,)])
                rows.append({"definition": dn, "sample": key[0], "coding": key[1],
                             "weights": key[2], "outcome": y, "beta": r["coef"],
                             "se": r["se"], "p": r["p"], "switchers": int(sw)})
        return pd.DataFrame(rows)


SUBSETS = {"all": lambda s: s,
           "availability-controlled": lambda s: s[s.coding != "none"]}


def joint_stats(t: pd.DataFrame) -> dict:
    """Joint test statistics, computed separately for all specifications and for the
    subset with an eligibility control. The paper's claim is made **after controlling
    for Convention-status eligibility**, so the subset is the test that matches the
    claim; the full set also includes the uncontrolled specifications, for comparison."""
    out = {}
    for y in OUTCOMES:
        for sub, f in SUBSETS.items():
            s = f(t[t.outcome == y])
            out[(y, sub, "median_beta")] = float(s.beta.median())
            out[(y, sub, "share_pos_sig")] = float(((s.beta > 0) & (s.p < 0.05)).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=499)
    ap.add_argument("--primary", default="bloc1_k2")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", default="paper_tables")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--combine-only", action="store_true",
                    help="no new draws; combine all spec_null_draws_*.csv under --out "
                         "into the joint-test table and figures")
    a = ap.parse_args()
    out = outdir(a.out)
    base = load_base(a.ref)
    coh = load_cohorts()
    cohorts = {dn: coh[dn].dropna().to_dict() for dn in DEFS}
    eng = SpecEngine(base)

    if a.combine_only:
        obs = pd.read_csv(out / "tab_spec_curve.csv")
    else:
        obs = eng.run(cohorts)
        obs["low_switchers"] = obs.switchers < 5
        print(f"[u08] {len(obs)} specifications")
        save(obs, out, "tab_spec_curve.csv", show=False)

        rng = np.random.default_rng(a.seed)
        U = sorted(set().union(*[set(c) for c in cohorts.values()]) & set(base[UNIT]))
        prof = {u: {dn: cohorts[dn].get(u, np.nan) for dn in DEFS} for u in U}
        recs = []
        for b in range(a.B):
            perm = rng.permutation(U)
            pc = {dn: {} for dn in DEFS}
            for u, v in zip(U, perm):
                for dn in DEFS:
                    if np.isfinite(prof[v][dn]):
                        pc[dn][u] = prof[v][dn]
            for (yy, sb, st), v in joint_stats(eng.run(pc)).items():
                recs.append({"design": "profile", "outcome": yy, "subset": sb,
                             "statistic": st, "value": v})
            # Timing only: each definition permutes onset years within its own treated
            # States (which States are treated is unchanged)
            tc = {}
            for dn in DEFS:
                tr = sorted(set(cohorts[dn]) & set(base[UNIT]))
                vals = [cohorts[dn][u] for u in tr]
                tc[dn] = dict(zip(tr, rng.permutation(vals)))
            for (yy, sb, st), v in joint_stats(eng.run(tc)).items():
                recs.append({"design": "timing", "outcome": yy, "subset": sb,
                             "statistic": st, "value": v})
            if (b + 1) % 50 == 0:
                print(f"  permutation {b + 1}/{a.B}")
        pd.DataFrame(recs).to_csv(out / f"spec_null_draws_{a.seed}.csv", index=False)

    s_obs = joint_stats(obs)
    draws = pd.concat([pd.read_csv(f) for f in sorted(out.glob("spec_null_draws_*.csv"))],
                      ignore_index=True)
    rows = []
    labels = {"profile": "profile permutation (all specs)",
              "timing": "timing only (within each definition)"}
    for key, v in s_obs.items():
        for des, lab in labels.items():
            null = draws[(draws.design == des) & (draws.outcome == key[0])
                         & (draws.subset == key[1])
                         & (draws.statistic == key[2])].value.to_numpy()
            if not len(null):
                continue
            if key[2] == "median_beta":
                p = (np.sum(np.abs(null) >= abs(v)) + 1) / (len(null) + 1)
            else:
                p = (np.sum(null >= v) + 1) / (len(null) + 1)
            rows.append({"outcome": key[0], "spec_subset": key[1],
                         "statistic": key[2], "observed": v,
                         "null_design": lab, "null_median": float(np.median(null)),
                         "null_q95": float(np.percentile(null, 95)), "p_joint": float(p),
                         "B": len(null)})
    print("\n[u08] Joint inference")
    save(pd.DataFrame(rows), out, "tab_spec_curve_joint.csv")

    for y, ylab in OUTCOMES.items():
        s = obs[obs.outcome == y].sort_values("beta").reset_index(drop=True)
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                      gridspec_kw={"height_ratios": [2.2, 2]})
        xs = np.arange(len(s))
        sig = (s.p < 0.05) & (s.beta > 0)
        ax.vlines(xs, s.beta - 1.96 * s.se, s.beta + 1.96 * s.se,
                  color=np.where(sig, "tab:red", "0.7"), lw=0.6)
        ax.scatter(xs, s.beta, s=6, color=np.where(sig, "tab:red", "0.3"), zorder=3)
        ax.axhline(0, color="k", lw=0.6)
        ax.axhline(s.beta.median(), color="tab:blue", lw=0.8, ls="--")
        ax.set_ylabel("effect (pp), 95% CI (CR1)")
        ax.set_title(f"Specification curve — {ylab}; red = positive and p < 0.05", fontsize=9)
        rows_ind = ([("def: " + dn, s.definition == dn) for dn in DEFS]
                    + [("avail: " + c, s.coding == c) for c in CODINGS]
                    + [("excl. VEN", s["sample"] == "excl_VEN"),
                       ("group weights", s.weights == "group")])
        for i, (lab, m) in enumerate(rows_ind):
            ax2.scatter(xs[m.to_numpy()], np.full(m.sum(), i), s=3, color="0.2", marker="|")
        ax2.set_yticks(range(len(rows_ind)), [r[0] for r in rows_ind], fontsize=7)
        ax2.invert_yaxis()
        ax2.set_xlabel("specifications, ordered by estimate")
        fig.tight_layout()
        fig.savefig(out / f"fig_spec_curve_{y}.pdf")
        print(f"  -> {out / f'fig_spec_curve_{y}.pdf'}")


if __name__ == "__main__":
    main()
