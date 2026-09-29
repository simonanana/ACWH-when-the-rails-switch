"""
u10_paper_tables.py — Upgrade-module CSV outputs → booktabs tables that main.tex can
\\input directly

If a CSV is missing, the corresponding table is skipped without aborting.

Usage
-----
    python src/u10_paper_tables.py --in paper_tables_upgrade --out paper/tables \
        --primary bloc1_k2 --legacy idx10
In main.tex:
    \\input{tables/tab_u_core_bloc1_k2}
    \\input{tables/tab_u_bounds}
    \\input{tables/tab_u_family}
    \\input{tables/tab_u_specjoint}
    \\input{tables/tab_u_incidence}
    \\input{tables/tab_u_eventstudy}
    \\input{tables/tab_u_ovb}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEF_LABEL = {
    "idx10": "Legacy index $\\geq 0.10$",
    "bloc1_k1": "$\\geq$1 Western sanctioning jurisdiction",
    "bloc1_k2": "$\\geq$2 Western sanctioning jurisdictions",
    "bloc1_k3": "$\\geq$3 Western sanctioning jurisdictions",
    "transatlantic": "EU and United States jointly",
    "eu_fin": "European Union",
    "us_fin": "United States",
    "raw27_k3": "EU expanded to 27 senders, $\\geq 3$",
}
CODING_SHORT = {
    "(0) no availability control": "None",
    "(1) strict: denunciation ends availability": "Strict",
    "(2) three-valued: contested as own category": "Three-valued",
    "(3) permissive: Art. 72 consent survives": "Permissive",
}


def f(v, d=2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "---"
    return "---" if not np.isfinite(v) else f"{v:.{d}f}"


def table(body, spec, caption, label, note):
    return "\n".join([
        r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}",
        f"\\label{{{label}}}", r"\small", f"\\begin{{tabular}}{{{spec}}}", r"\toprule",
        *body, r"\bottomrule", r"\end{tabular}", r"\par\medskip",
        r"\begin{minipage}{0.95\linewidth}\footnotesize", note, r"\end{minipage}",
        r"\end{table}", ""])


def t_core(src: Path, dname: str) -> str | None:
    d = pd.read_csv(src)
    d = d[d.definition == dname]
    if d.empty:
        return None
    body = [r"Outcome & Availability & Effect & (CR1) & $p_{\mathrm{CR1}}$ & "
            r"$p_{\mathrm{CV3}}$ & $p_{\mathrm{WCB}}$ & $p_{\mathrm{restr}}$ & "
            r"$p_{\mathrm{timing}}$ \\", r"\midrule"]
    for oc, g in d.groupby("outcome", sort=False):
        for i, (_, r) in enumerate(g.iterrows()):
            body.append(" & ".join([
                oc.split(" (")[0] if i == 0 else "", CODING_SHORT.get(r.coding, r.coding),
                f(r.effect), f"({f(r.se_cr1)})", f(r.p_cr1, 3), f(r.p_cv3, 3),
                f(r.p_wcb_webb, 3), f(r.p_perm_restricted, 3), f(r.p_perm_timing, 3)])
                + r" \\")
        body.append(r"\addlinespace")
    r0 = d.iloc[0]
    note = (f"Treatment: {DEF_LABEL.get(dname, dname)}. {int(r0.n)} cases; "
            f"{int(r0.switchers)} respondent States change status inside the sample; "
            f"{int(r0.identifying_cases)} cases identify the coefficient. Case-level linear "
            r"probability models with respondent and year fixed effects, outcomes in "
            r"percentage points. $p_{\mathrm{CR1}}$: cluster-robust by respondent; "
            r"$p_{\mathrm{CV3}}$: cluster jackknife; $p_{\mathrm{WCB}}$: restricted wild "
            r"cluster bootstrap-$t$ with Webb six-point weights ($B=1999$); "
            r"$p_{\mathrm{restr}}$: permutation of cohort assignment among all treated "
            r"States and untreated States with cases in at least two years ($B=999$); "
            r"$p_{\mathrm{timing}}$: permutation of onset years among treated States only.")
    return table(body, "llrrrrrrr",
                 "Sanctions onset and forum: five inference procedures.",
                 f"tab:ucore{dname.replace('_', '')}", note)


def t_bounds(src: Path) -> str:
    d = pd.read_csv(src)
    body = [r"Treatment & Outcome & Strict & Perm. & Range examined & "
            r"95\% CI (CR1) & 95\% CI (CV3) \\", r"\midrule"]
    for _, r in d.iterrows():
        body.append(" & ".join([
            DEF_LABEL.get(r.definition, r.definition), r.outcome.split(" (")[0],
            f(r.beta_strict), f(r.beta_lax), f"[{f(r.set_lo)}, {f(r.set_hi)}]",
            f"[{f(r.IM95_lo_cr1)}, {f(r.IM95_hi_cr1)}]",
            f"[{f(r.IM95_lo_cv3)}, {f(r.IM95_hi_cv3)}]"]) + r" \\")
    note = (r"The legal status of the 55 cases filed after the respondent's denunciation "
            r"took effect is not identified. The interpretive range reported is the range of estimates "
            r"over the strict and permissive readings of Article~72, all eight "
            r"country-level combinations, and 1{,}000 case-level random assignments at "
            r"each of $p\in\{0.25, 0.5, 0.75\}$; because the estimator is not monotone "
            r"in the assignment, the two readings alone need not bound the range, and the "
            r"search is not claimed to recover global extrema. The bands in the last two "
            r"columns apply the construction of \citet{imbens2004} to the endpoints of the "
            r"range examined; because global extrema are not proven, they are "
            r"sampling-uncertainty bands conditional on the assignments explored, not "
            r"confidence intervals for a sharply identified parameter.")
    return table(body, "llrrccc",
                 "Interpretive ranges under alternative Article~72 codings.", "tab:ubounds", note)


def t_family(src: Path) -> str:
    d = pd.read_csv(src)
    body = [r"Treatment definition & States & Switchers & \multicolumn{2}{c}{Non-Convention} "
            r"& \multicolumn{2}{c}{UNCITRAL} \\",
            r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
            r" & & & None & Three-valued & None & Three-valued \\", r"\midrule"]
    for _, r in d.iterrows():
        def cell(y, c):
            p = r[f"{y}_{c}_p"]
            s = "$^{***}$" if p < .01 else "$^{**}$" if p < .05 else "$^{*}$" if p < .10 else ""
            return f"{f(r[f'{y}_{c}_b'])}{s}"
        body.append(" & ".join([DEF_LABEL.get(r.definition, r.definition),
                                str(int(r.n_cohort_states)), str(int(r.switchers)),
                                cell("natcourt_exposed", "none"),
                                cell("natcourt_exposed", "three"),
                                cell("is_uncitral", "none"), cell("is_uncitral", "three")])
                    + r" \\")
    note = (r"Treatment definitions constructed from the case-level Global Sanctions Data "
            r"Base (Release~4), counting the European Union as one sanctioning "
            r"jurisdiction and assigning national measures of EU member States to it; "
            r"Western jurisdictions are the European Union, the United States, the United "
            r"Kingdom, Switzerland and Norway. Onset is the first year in 2011--2022 in which "
            r"the condition holds; States meeting it in 2010 are left-censored and "
            r"assigned to the comparison group. Stars: CR1 $p$-values, for orientation only.")
    return table(body, "lrrrrrr", "Transparent treatment definitions.",
                 "tab:ufamily", note)


def t_specjoint(src: Path) -> str:
    d = pd.read_csv(src)
    body = [r"Outcome & Statistic & Observed & Null design & Null 95th pct. & $p$ \\",
            r"\midrule"]
    lab = {"median_beta": "Median effect", "share_pos_sig": "Share positive, $p<0.05$"}
    for _, r in d.iterrows():
        body.append(" & ".join([
            "Non-Convention" if r.outcome == "natcourt_exposed" else "UNCITRAL",
            lab[r.statistic], f(r.observed), r.null_design.split(" (")[0],
            f(r.null_q95), f(r.p_joint, 3)]) + r" \\")
    note = (r"The specification space crosses seven treatment definitions, four codings of "
            r"Convention-status eligibility, two outcomes, two samples (all; excluding Venezuela) and "
            r"two weighting schemes (unweighted; inverse claim-group size), 224 "
            r"specifications in all. Joint null distributions re-estimate every "
            r"specification under each draw \citep{simonsohn2020,young2019}. ``Profile'' "
            r"permutes the vector of onset years across States treated under any "
            r"definition; ``timing'' permutes onset years within each definition's treated "
            r"States, holding fixed which States are sanctioned.")
    return table(body, "llrlrr",
                 "Specification curve: joint inference across 224 specifications.",
                 "tab:uspecjoint", note)


def t_incidence(src: Path) -> str:
    d = pd.read_csv(src)
    body = [r"Treatment & Estimator & RR & 95\% CI & $p_{\mathrm{TOST}}$ & "
            r"MDE$_{80}$ (RR) & Countries \\", r"\midrule"]
    for _, r in d.iterrows():
        body.append(" & ".join([
            DEF_LABEL.get(r.definition, r.definition), r.estimator.split(" (")[0],
            f(r.RR), f"[{f(r.RR_ci_lo)}, {f(r.RR_ci_hi)}]", f(r["p_TOST_[2/3,3/2]"], 3),
            f"[{f(r.MDE80_RR_down)}, {f(r.MDE80_RR_up)}]", str(int(r.n_countries))])
            + r" \\")
    note = (r"Poisson models with country and year fixed effects on the 170-economy panel; "
            r"ETWFE estimates a separate effect for each cohort \citep{wooldridge2023} and "
            r"aggregates to a rate ratio over treated observations. Confidence intervals "
            r"from a country-level cluster bootstrap. $p_{\mathrm{TOST}}$ tests equivalence "
            r"to a rate ratio within $[2/3, 3/2]$. Countries with no disputes in any year "
            r"are not identified and are dropped. The panel omits nine respondents "
            r"(57 cases), including Ecuador, the United States, China and the European Union.")
    return table(body, "llrcrcr", "Dispute incidence on the rate-ratio scale.",
                 "tab:uincidence", note)


def t_event(imp: Path, rr: Path) -> str:
    a = pd.read_csv(imp)
    b = pd.read_csv(rr)
    body = [r"Treatment & Outcome & Availability & Imputation ATT & (jackknife) & "
            r"RR CI, $\bar M=0$ & RR CI, $\bar M=0.5$ \\", r"\midrule"]
    for _, r in a.iterrows():
        sub = b[(b.definition == r.definition) & (b.outcome == r.outcome) &
                (b.coding == r.coding)]
        def ci(m):
            s = sub[sub.Mbar == m]
            return "---" if s.empty else f"[{f(s.ci_lo.iloc[0], 1)}, {f(s.ci_hi.iloc[0], 1)}]"
        body.append(" & ".join([
            DEF_LABEL.get(r.definition, r.definition), r.outcome.split(" (")[0],
            "None" if r.coding == "none" else "Three-valued", f(r.att), f"({f(r.se_jack)})",
            ci(0.0), ci(0.5)]) + r" \\")
    note = (r"Imputation estimator of \citet{borusyak2024}: respondent, year (and "
            r"availability) effects estimated on untreated cases only; treated cases of "
            r"respondents with no untreated case cannot be imputed and are excluded. "
            r"Standard errors from a respondent-level jackknife. The last two columns report "
            r"95\% robust confidence intervals for the average post-onset event-study effect "
            r"under the relative-magnitude restriction of \citet{rambachan2023}.")
    return table(body, "lllrrcc",
                 "Heterogeneity-robust estimates and sensitivity to parallel trends.",
                 "tab:uevent", note)


def t_ovb(src: Path) -> str:
    d = pd.read_csv(src)
    body = [r"Treatment & Outcome & $\hat\beta$ & $R^2_{S\sim Z}$ & $R^2_{Y\sim Z}$ & "
            r"$k=0.5$ & $k=1$ & $k=2$ \\", r"\midrule"]
    for _, r in d.iterrows():
        body.append(" & ".join([
            DEF_LABEL.get(r.definition, r.definition), r.outcome.split(" (")[0],
            f(r.beta_with_Z), f(r.partial_r2_S_on_Z, 3), f(r.partial_r2_Y_on_Z, 3),
            f(r.get("k0.5_beta_adj")), f(r.get("k1.0_beta_adj")), f(r.get("k2.0_beta_adj"))])
            + r" \\")
    note = (r"Sensitivity of the eligibility-conditioned estimate to an omitted legal "
            r"factor $k$ times as strong as the three-valued eligibility indicators $Z$ "
            r"\citep{cinelli2020}. $R^2_{S\sim Z}$ and $R^2_{Y\sim Z}$ are partial $R^2$ "
            r"after respondent and year fixed effects. The last three columns report the "
            r"bias-adjusted estimate.")
    return table(body, "llrrrrrr",
                 "Omitted-variable sensitivity benchmarked on Convention-status eligibility.",
                 "tab:uovb", note)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="paper_tables_upgrade")
    ap.add_argument("--out", dest="dst", default="paper/tables")
    ap.add_argument("--primary", default="bloc1_k2")
    ap.add_argument("--legacy", default="idx10")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    jobs = [
        (f"tab_u_core_{a.primary}.tex", ["tab_core_inference.csv"],
         lambda: t_core(src / "tab_core_inference.csv", a.primary)),
        (f"tab_u_core_{a.legacy}.tex", ["tab_core_inference.csv"],
         lambda: t_core(src / "tab_core_inference.csv", a.legacy)),
        ("tab_u_bounds.tex", ["tab_interpretive_bounds.csv"],
         lambda: t_bounds(src / "tab_interpretive_bounds.csv")),
        ("tab_u_family.tex", ["tab_treatment_family.csv"],
         lambda: t_family(src / "tab_treatment_family.csv")),
        ("tab_u_specjoint.tex", ["tab_spec_curve_joint.csv"],
         lambda: t_specjoint(src / "tab_spec_curve_joint.csv")),
        ("tab_u_incidence.tex", ["tab_incidence_etwfe.csv"],
         lambda: t_incidence(src / "tab_incidence_etwfe.csv")),
        ("tab_u_eventstudy.tex", ["tab_imputation.csv", "tab_honest_did_forum.csv"],
         lambda: t_event(src / "tab_imputation.csv", src / "tab_honest_did_forum.csv")),
        ("tab_u_ovb.tex", ["tab_ovb_sensitivity.csv"],
         lambda: t_ovb(src / "tab_ovb_sensitivity.csv")),
    ]
    for name, needs, fn in jobs:
        if not all((src / n).exists() for n in needs):
            print(f"  [skip] {name}: missing {needs}")
            continue
        tex = fn()
        if tex:
            (dst / name).write_text(tex, encoding="utf-8")
            print(f"  \\input{{tables/{name[:-4]}}}")


if __name__ == "__main__":
    main()
