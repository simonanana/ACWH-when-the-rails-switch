#!/usr/bin/env python3
"""
build_exhibits.py -- single entry point for every generated exhibit in
"When the Rails Switch".

    python build_exhibits.py                # build everything
    python build_exhibits.py --list         # print the manifest and exit
    python build_exhibits.py --only F3 T2   # build selected exhibits

Outputs
    figures/*.pdf     vector, print-ready, CVD- and greyscale-safe
    tables/*.tex      booktabs fragments, \input{} from main.tex
    EXHIBITS.tsv      machine-readable manifest of what was written

Inputs (set with the flags below if your layout differs)
    --results   directory holding the upgrade toolkit's CSV output
                (tab_core_inference.csv, tab_event_study.csv, ...)
    --cases     cases_with_navigator.parquet   (UNCTAD Navigator merge)
    --cohorts   gsdb_treatment_cohorts.csv     (treatment onset by ISO3)
    --icsid     icsid_convention_status_icsid3.csv (Contracting-State dates)
    --src       the project's src/ directory, for legal_eligibility.py

Every number that is typed in rather than read from a file appears in a
PROVENANCE comment directly above the function that uses it, so the whole
file can be audited without running anything.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ===========================================================================
# 0.  HOUSE STYLE
# ===========================================================================
# Two hues, chosen for print: CVD separation dE 19.3 (protan) / 26.8 (tritan),
# normal-vision dE 26.7, both >= 3:1 against white. Every coloured mark also
# carries a hatch or a distinct marker, so nothing depends on colour alone.
BLUE, ORANGE = "#1A6B8F", "#C4741A"
BLUE_L, ORANGE_L = "#9DC0D2", "#E4B87E"
INK, MUTED, GRID = "#1a1a1a", "#5a6066", "#d8d8d4"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False,
    "pdf.fonttype": 42,
})

FIGDIR = Path("figures")
TABDIR = Path("tables")


def _save(fig, name: str) -> str:
    FIGDIR.mkdir(exist_ok=True)
    out = FIGDIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return str(out)


def _write_tex(body: str, name: str) -> str:
    TABDIR.mkdir(exist_ok=True)
    out = TABDIR / name
    out.write_text(body, encoding="utf-8")
    print(f"  wrote {out}")
    return str(out)


def _grid(ax, axis="y"):
    getattr(ax, f"{axis}axis").grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ===========================================================================
# 1.  DATA ACCESS
# ===========================================================================
class Data:
    """Lazily loads and caches the four inputs."""

    def __init__(self, args):
        self.results = Path(args.results)
        self.cases_path = Path(args.cases)
        self.cohorts_path = Path(args.cohorts)
        self.icsid_path = Path(args.icsid)
        self.src = Path(args.src)
        self._cases = None

    def csv(self, stem: str) -> pd.DataFrame:
        p = self.results / f"{stem}.csv"
        if not p.exists():
            raise SystemExit(f"missing input: {p}")
        return pd.read_csv(p)

    @property
    def cases(self) -> pd.DataFrame:
        """958 cases + bloc1_k2 treatment + Convention-status eligibility."""
        if self._cases is not None:
            return self._cases
        d = pd.read_parquet(self.cases_path)

        coh = pd.read_csv(self.cohorts_path).set_index("iso3")["bloc1_k2"]
        d["onset"] = d["respondent_iso3"].map(coh)
        d["S"] = ((d["onset"].notna()) & (d["year"] >= d["onset"])).astype(int)

        sys.path.insert(0, str(self.src))
        from legal_eligibility import EligibilityTable, add_forum_availability
        et = EligibilityTable.load(self.icsid_path)
        d = add_forum_availability(d, et)

        d["nonconv"] = 1 - d["is_icsid_conv"].astype(int)
        d["conv_open"] = (d["choice_set_type"] == "both_parties").astype(int)
        self._cases = d
        return d

    def switchers(self) -> list[str]:
        d = self.cases
        return sorted(s for s in d["respondent_iso3"].dropna().unique()
                      if d.loc[d.respondent_iso3 == s, "S"].nunique() > 1)


# ===========================================================================
# 2.  FIGURES
# ===========================================================================

# --- F1 -------------------------------------------------------------------
# PROVENANCE (tab_core_inference.csv, definition=bloc1_k2, outcome=Non-Convention):
#   coding (0) no control        33.284 (13.534)
#   coding (2) three-valued      11.578 ( 8.994)
#   attenuation 21.71, paired cluster-bootstrap CI [3.7, 28.5]  (tab_attenuation_paired.csv)
#   "+ eligibility and treaty menu" 10.79 (9.12): two-layer decomposition row (3),
#   reproduced by exhibits_twolayer() below -- see note in EXHIBITS.md.
#   descriptive panel: genuine-choice subsample, UNCITRAL share
#   treated 23.1% (N=26) vs other 22.0% (N=255)
def fig_cascade(D: Data) -> str:
    core = D.csv("tab_core_inference")
    q = core[(core.definition == "bloc1_k2")
             & core.outcome.str.startswith("Non-Convention")]
    b0 = q[q.coding.str.startswith("(0)")].iloc[0]
    b2 = q[q.coding.str.startswith("(2)")].iloc[0]

    vals = [b0.effect, b2.effect, 10.79]
    ses = [b0.se_cr1, b0.se_cr1 * 0 + b2.se_cr1, 9.12]
    labels = ["Raw\ntwo-way FE", "+ Convention-status\neligibility",
              "+ eligibility and\ntreaty forum menu"]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(7.4, 3.5),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.30})

    x = np.arange(3)
    ax.axhline(0, color=MUTED, lw=0.7, zorder=1)
    bars = ax.bar(x, vals, width=0.56, color=[BLUE, BLUE_L, BLUE_L],
                  edgecolor=BLUE, lw=0.9, zorder=2)
    for b in bars[1:]:
        b.set_hatch("///")
    ax.errorbar(x, vals, yerr=[1.96 * s for s in ses], fmt="none",
                ecolor=INK, elinewidth=1.0, capsize=3.5, capthick=1.0, zorder=3)
    for xi, v in zip(x, vals):
        ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color=INK)

    ax.annotate("", xy=(0.30, vals[0]), xytext=(0.70, vals[1]),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.3))
    ax.text(0.50, (vals[0] + vals[1]) / 2 + 0.5,
            "attenuation\n21.71 pp\n[3.7, 28.5]", ha="center", va="center",
            fontsize=8, color=ORANGE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=ORANGE, lw=0.7))

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Percentage points outside the ICSID Convention")
    ax.set_ylim(-16, 64); _grid(ax)
    ax.set_title("(a)  What survives each layer of legal control",
                 fontsize=9.5, loc="left", pad=8)

    ax2.axhline(0, color=MUTED, lw=0.7, zorder=1)
    ax2.bar([0], [1.1], width=0.5, color="white", edgecolor=ORANGE,
            lw=1.4, hatch="...", zorder=2)
    ax2.annotate("1.1", (0, 1.1), xytext=(0, 5), textcoords="offset points",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                 color=INK)
    ax2.set_xticks([0])
    ax2.set_xticklabels(["cases where the\nclaimant had\nboth rails\n(N = 281)"],
                        fontsize=8)
    ax2.set_ylim(-16, 64); ax2.set_yticklabels([]); _grid(ax2)
    ax2.set_title("(b)  Descriptive", fontsize=9.5, loc="left", pad=8)
    ax2.set_xlabel("descriptive;\nnot an estimate", fontsize=7.5,
                   color=MUTED, style="italic", labelpad=6)
    return _save(fig, "fig_cascade_bloc1_k2.pdf")


# --- F2 -------------------------------------------------------------------
def fig_postaward(D: Data) -> str:
    d = D.cases
    conv, nonc = d[d.is_icsid_conv == 1], d[d.is_icsid_conv == 0]
    n_ann, n_sa = int(d.fo_annulment.sum()), int(d.fo_setaside.sum())
    r_ann = 100 * conv.fo_annulment.mean()
    r_sa = 100 * nonc.fo_setaside.mean()
    full_ann = int(d.loc[d.fo_annulment == 1, "fo_annulled_full"].sum())
    full_sa = int(d.loc[d.fo_setaside == 1, "fo_annulled_full"].sum())

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.6, 3.3),
                                   gridspec_kw={"wspace": 0.55})
    y, h = np.arange(2), 0.34
    axL.barh(y + h / 2, [r_ann, 0], height=h, color=BLUE, edgecolor=BLUE,
             lw=0.8, label="ICSID annulment (Art. 52)")
    axL.barh(y - h / 2, [0, r_sa], height=h, color="white", edgecolor=ORANGE,
             lw=1.2, hatch="///", label="National-court set-aside")
    for yi, v, n in [(y[0] + h / 2, r_ann, n_ann), (y[1] - h / 2, r_sa, n_sa)]:
        axL.annotate(f"{v:.1f}%  (n = {n})", (v, yi), xytext=(4, 0),
                     textcoords="offset points", va="center",
                     fontsize=8.5, color=INK)
    for yi in [y[0] - h / 2, y[1] + h / 2]:
        axL.annotate("0", (0, yi), xytext=(4, 0), textcoords="offset points",
                     va="center", fontsize=8.5, color=MUTED)
    axL.set_yticks(y)
    axL.set_yticklabels([f"ICSID Convention\n({len(conv)} cases)",
                         f"Outside the Convention\n({len(nonc)} cases)"],
                        fontsize=8.5)
    axL.invert_yaxis(); axL.set_xlim(0, 33)
    axL.set_xlabel("Share of cases in the group (%)"); _grid(axL, "x")
    axL.set_title("(a)  Which mechanism reviews the award",
                  fontsize=9.5, loc="left", pad=8)

    vals = [100 * full_ann / n_ann, 100 * full_sa / n_sa]
    bars = axR.barh(np.arange(2), vals, height=0.42, color="white",
                    edgecolor=[BLUE, ORANGE], lw=1.3)
    bars[1].set_hatch("///")
    for i, (v, n) in enumerate(zip(vals, [full_ann, full_sa])):
        axR.annotate(f"{v:.1f}%  (n = {n})", (v, i), xytext=(4, 0),
                     textcoords="offset points", va="center",
                     fontsize=8.5, color=INK)
    axR.set_yticks(np.arange(2))
    axR.set_yticklabels([f"ICSID annulment\n({n_ann} proceedings)",
                         f"National-court set-aside\n({n_sa} proceedings)"],
                        fontsize=8.5)
    axR.invert_yaxis(); axR.set_xlim(0, 24)
    axR.set_xlabel("Award set aside or annulled in full (%)"); _grid(axR, "x")
    axR.set_title("(b)  What the challenge does to the award",
                  fontsize=9.5, loc="left", pad=8)

    hs, ls = axL.get_legend_handles_labels()
    fig.legend(hs, ls, fontsize=8, loc="lower center",
               bbox_to_anchor=(0.5, -0.17), ncol=2,
               handlelength=1.8, columnspacing=2.0)
    return _save(fig, "fig_postaward_tracks.pdf")


# --- F3  (upgrade 4) ------------------------------------------------------
def fig_raw_trend(D: Data) -> str:
    """Annual non-Convention share, sanctioned vs not, with a count panel.

    Two stacked panels share the x-axis rather than a second y-axis: the
    shares are unreadable without the counts, and a dual-axis chart would
    invite exactly the wrong comparison.
    """
    d = D.cases
    a = (d.groupby(["year", "S"])
           .agg(n=("nonconv", "size"), share=("nonconv", "mean"))
           .reset_index())
    a["share"] *= 100
    yrs = np.arange(int(d.year.min()), int(d.year.max()) + 1)

    fig, (ax, axn) = plt.subplots(
        2, 1, figsize=(7.2, 4.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})

    for s, col, mk, lab, ls in [
            (0, BLUE, "o", "Respondent not under financial sanctions", "-"),
            (1, ORANGE, "s", "Respondent under financial sanctions", "--")]:
        q = a[a.S == s].set_index("year").reindex(yrs)
        ax.plot(yrs, q["share"], color=col, lw=1.8, ls=ls, zorder=3, label=lab)
        ax.scatter(yrs, q["share"], s=np.clip(q["n"].fillna(0) * 1.6, 12, 90),
                   color=col, edgecolor="white", lw=0.8, zorder=4)

    # onset markers for the two largest switchers
    for yr, txt in [(2014, "RUS, TZA onset"), (2017, "VEN onset")]:
        ax.axvline(yr, color=MUTED, lw=0.7, ls=":", zorder=1)
        ax.annotate(txt, (yr, 97), rotation=90, fontsize=7, color=MUTED,
                    ha="right", va="top")

    ax.set_ylabel("Cases conducted outside the\nICSID Convention (%)")
    ax.set_ylim(0, 100); _grid(ax)
    ax.legend(fontsize=8, loc="lower left", ncol=1)
    ax.set_title("Marker area is proportional to the number of cases that year",
                 fontsize=8, loc="left", color=MUTED, pad=6)

    w = 0.4
    for s, col, off, hatch in [(0, BLUE_L, -w / 2, ""), (1, ORANGE_L, w / 2, "///")]:
        q = a[a.S == s].set_index("year").reindex(yrs)
        axn.bar(yrs + off, q["n"].fillna(0), width=w, color=col,
                edgecolor=col if s == 0 else ORANGE, lw=0.6, hatch=hatch)
    axn.set_ylabel("Cases"); axn.set_xlabel("Year of initiation")
    axn.set_xticks(yrs); axn.tick_params(axis="x", rotation=0)
    _grid(axn)
    return _save(fig, "fig_raw_share_trend.pdf")


# --- F4  (upgrade 5) ------------------------------------------------------
# PROVENANCE -- every date and citation below is the one already footnoted in
# section 6 of main.tex; nothing new is asserted here.
# (date, seat, label, colour, marker kind, label offset in points)
CRIMEA = [
    ("2018-04", "Paris",     "Award\n(Oschadbank)",                        BLUE,   "award",    +20),
    ("2018-10", "Geneva",    "Swiss Federal Supreme Court\n4A_396/2017",   BLUE,   "upheld",   -20),
    ("2019-12", "Geneva",    "Challenges dismissed\n4A_244/2019, 4A_246/2019", BLUE, "upheld", +20),
    ("2021-03", "Paris",     "Award set aside\nCA Paris 19/04161",         ORANGE, "setaside", +20),
    ("2022-12", "Paris",     "Cassation quashes\nCass. 1re civ. 21-15.390", BLUE,  "upheld",   +20),
    ("2024-12", "The Hague", "Awards confirmed\nHR 2024:1807/1810/1813",   BLUE,   "upheld",   +20),
    ("2025-06", "Paris",     "Application dismissed\non remand, RG 24/05336", BLUE, "upheld",  -20),
]


def fig_crimea(D: Data) -> str:
    dates = [pd.Period(d, freq="M").to_timestamp() for d, *_ in CRIMEA]
    seats = ["Paris", "Geneva", "The Hague"]
    ypos = {s: len(seats) - 1 - i for i, s in enumerate(seats)}

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for s in seats:
        ax.axhline(ypos[s], color=GRID, lw=1.0, zorder=1)

    for dt, (_, seat, label, col, kind, dy) in zip(dates, CRIMEA):
        y = ypos[seat]
        mk = "v" if kind == "setaside" else ("D" if kind == "award" else "o")
        ax.scatter([dt], [y], s=62, marker=mk, color="white",
                   edgecolor=col, lw=1.6, zorder=4)
        ax.annotate(label, (dt, y), xytext=(0, dy),
                    textcoords="offset points", ha="center",
                    va="bottom" if dy > 0 else "top", fontsize=7.2,
                    color=INK, linespacing=1.3,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5,
                                    shrinkA=0, shrinkB=6))

    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels(list(ypos.keys()), fontsize=9)
    ax.set_ylim(-1.0, len(seats) - 0.05)
    ax.set_xlim(pd.Timestamp("2017-09"), pd.Timestamp("2026-03"))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _grid(ax, "x")
    ax.set_xlabel("")
    ax.legend(handles=[
        Line2D([], [], marker="D", ls="", mfc="white", mec=BLUE, mew=1.5,
               label="award rendered"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=BLUE, mew=1.5,
               label="award upheld / challenge dismissed"),
        Line2D([], [], marker="v", ls="", mfc="white", mec=ORANGE, mew=1.5,
               label="award set aside"),
    ], fontsize=7.6, loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3)
    ax.set_title("Seven years of national-court review, three seats, one dispute group",
                 fontsize=9, loc="left", color=MUTED, pad=10)
    return _save(fig, "fig_crimea_timeline.pdf")


# --- F5  (upgrade 6) ------------------------------------------------------
def fig_influence_event(D: Data) -> str:
    """fig_influence + fig_event_study, one figure, shared legend."""
    loo = D.csv("tab_influence_loo")
    es = D.csv("tab_event_study")
    q = loo[(loo.definition == "bloc1_k2")
            & loo.outcome.str.startswith("Non-Convention")].copy()
    base = q[q.dropped == "(none)"].iloc[0]
    q = q[q.dropped != "(none)"].sort_values("beta_none")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.6, 3.6),
                                   gridspec_kw={"wspace": 0.28})

    # (a) leave-one-out
    x = np.arange(len(q))
    axA.axhline(base.beta_none, color=BLUE, lw=0.9, ls="--", zorder=1)
    axA.axhline(base.beta_three, color=ORANGE, lw=0.9, ls=":", zorder=1)
    axA.axhline(0, color=MUTED, lw=0.7, zorder=1)
    axA.scatter(x, q.beta_none, s=26, color=BLUE, marker="o",
                edgecolor="white", lw=0.6, zorder=3, label="No eligibility control")
    axA.scatter(x, q.beta_three, s=30, color="white", marker="s",
                edgecolor=ORANGE, lw=1.2, zorder=3,
                label="Three-valued eligibility control")
    axA.set_xticks(x)
    axA.set_xticklabels(q.dropped, rotation=90, fontsize=6.6)
    axA.set_ylabel("Estimate after dropping the State (pp)")
    axA.set_xlabel("Respondent State dropped", fontsize=8.5, labelpad=2)
    _grid(axA)
    # label the two most influential drops
    for col, colr in [("beta_none", BLUE), ("beta_three", ORANGE)]:
        j = int(np.argmin(q[col].values))
        axA.annotate(q.dropped.values[j], (j, q[col].values[j]),
                     xytext=(9, 3 if col == "beta_none" else -11),
                     textcoords="offset points",
                     fontsize=7.5, color=colr, fontweight="bold")
    axA.set_title("(a)  Leave one respondent State out",
                  fontsize=9.5, loc="left", pad=8)

    # (b) event study
    for coding, col, mk, off, lab in [
            ("none", BLUE, "o", -0.07, "No eligibility control"),
            ("three", ORANGE, "s", 0.07, "Three-valued eligibility control")]:
        e = es[(es.definition == "bloc1_k2")
               & es.outcome.str.startswith("Non-Convention")
               & (es.coding == coding)].sort_values("event_time")
        if e.empty:
            continue
        axB.errorbar(e.event_time + off, e.beta, yerr=1.96 * e.se, fmt=mk,
                     ms=4.6, mfc="white" if coding != "none" else col,
                     mec=col, mew=1.2, color=col, elinewidth=0.9, capsize=2.4,
                     label=lab, zorder=3)
    axB.axhline(0, color=MUTED, lw=0.7)
    axB.axvline(-0.5, color=MUTED, lw=0.8, ls=":")
    axB.set_xlabel(r"Years relative to sanctions onset ($\tau$)", fontsize=8.5)
    axB.set_ylabel("Estimate (pp)")
    _grid(axB)
    axB.set_title(r"(b)  Event study (reference $\tau=-1$)",
                  fontsize=9.5, loc="left", pad=8)

    hs, ls = axB.get_legend_handles_labels()
    fig.legend(hs, ls, fontsize=8, loc="lower center",
               bbox_to_anchor=(0.5, -0.13), ncol=2,
               handlelength=1.8, columnspacing=2.0)
    return _save(fig, "fig_influence_event.pdf")


# ===========================================================================
# 3.  TABLES
# ===========================================================================
TEX_HEAD = ("%% generated by build_exhibits.py -- do not edit by hand\n")


# --- T1  (upgrade 1) ------------------------------------------------------
# PROVENANCE
#   incidence RR 1.13 [0.64, 2.84]          tab_incidence_etwfe.csv
#   raw shares 60.1 / 45.2 / 31.3           section 5.2, computed from cases
#   33.28, 11.58 and their p-values         tab_core_inference.csv
#   attenuation 21.71 [3.7, 28.5] p .004    tab_attenuation_paired.csv
#   genuine-choice 23.1 vs 22.0 (N=281)     two-layer merge, section 5.10
def tab_headline(D: Data) -> str:
    core = D.csv("tab_core_inference")
    q = core[(core.definition == "bloc1_k2")
             & core.outcome.str.startswith("Non-Convention")]
    b0 = q[q.coding.str.startswith("(0)")].iloc[0]
    b2 = q[q.coding.str.startswith("(2)")].iloc[0]

    rows = [
        ("Do sanctions generate more disputes?",
         r"Incidence rate ratio $1.13$",
         r"$[0.64,\ 2.84]$",
         "No detectable change; the interval is too wide to establish equivalence."),
        ("Are sanctioned respondents' disputes elsewhere?",
         r"$60.1\%$ vs $45.2\%$ outside the Convention",
         r"---",
         r"Yes, descriptively; the pre-onset share is already $31.3\%$."),
        ("Is that a sanctions effect?",
         rf"$+{b0.effect:.2f}$ pp "
         rf"($p_{{\mathrm{{CR1}}}}={b0.p_cr1:.3f}$)",
         rf"$p_{{\mathrm{{CV3}}}}={b0.p_cv3:.3f}$",
         "Not robustly: the primary small-cluster procedures do not confirm it."),
        ("What is left after Convention-status eligibility?",
         rf"$+{b2.effect:.2f}$ pp "
         rf"($p_{{\mathrm{{CR1}}}}={b2.p_cr1:.3f}$)",
         r"---",
         "Not distinguishable from zero under any procedure."),
        (r"\textbf{What the design does identify}",
         r"\textbf{Attenuation $21.71$ pp}",
         r"\textbf{$[3.7,\ 28.5]$}",
         r"\textbf{Consistent with substantial legal predetermination.}"),
        ("Where the claimant had both rails",
         r"$23.1\%$ vs $22.0\%$ ($N=281$)",
         r"---",
         "The gap disappears. Descriptive; two switching States only."),
    ]
    body = [TEX_HEAD, r"\begin{table}[t]", r"\centering",
            r"\caption{The paper in one table. Reference treatment definition "
            r"($\geq$2 Western sanctioning jurisdictions); \Ncases{} disputes, "
            r"2010--2023; \Nswitch{} respondent States change status inside "
            r"the window.}",
            r"\label{tab:headline}", r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{p{3.9cm}p{4.0cm}p{2.0cm}p{5.2cm}}",
            r"\toprule",
            r"Question & Estimate & Check & Reading \\",
            r"\midrule"]
    for i, r in enumerate(rows):
        if i == 4:
            body.append(r"\addlinespace")
        body.append(" & ".join(r) + r" \\")
        if i == 4:
            body.append(r"\addlinespace")
    body += [r"\bottomrule", r"\end{tabular}", r"\par\medskip",
             r"\begin{minipage}{0.97\linewidth}\footnotesize",
             r"\emph{Notes:} Rows 3--5 are case-level linear probability models "
             r"with respondent and year fixed effects, outcome in percentage "
             r"points. The interval in row 5 is a paired cluster bootstrap over "
             r"respondent States. Rows 2 and 6 are raw shares and carry no "
             r"standard error. Full results, and the five inference procedures "
             r"behind rows 3 and 4, are in Table~\ref{tab:core}.",
             r"\end{minipage}", r"\end{table}", ""]
    return _write_tex("\n".join(body), "tab_headline_summary.tex")


# --- T2  (upgrade 2) ------------------------------------------------------
def tab_switchers(D: Data) -> str:
    """Twelve switching States, with the column that makes the point:
    does Convention-status eligibility actually vary inside the State?"""
    d = D.cases
    rows = []
    for s in D.switchers():
        x = d[d.respondent_iso3 == s]
        pre, post = x[x.S == 0], x[x.S == 1]
        rows.append(dict(
            iso=s, onset=int(x.onset.iloc[0]),
            npre=len(pre), npost=len(post),
            nc_pre=f"{int(pre.nonconv.sum())}/{len(pre)}",
            nc_post=f"{int(post.nonconv.sum())}/{len(post)}",
            op_pre=f"{int(pre.conv_open.sum())}/{len(pre)}",
            op_post=f"{int(post.conv_open.sum())}/{len(post)}",
            varies=x.conv_open.nunique() > 1))
    t = pd.DataFrame(rows).sort_values("npost", ascending=False)

    names = {"VEN": "Venezuela", "RUS": "Russian Federation", "TZA": "Tanzania",
             "SAU": "Saudi Arabia", "TUR": r"T\"urkiye", "IRQ": "Iraq",
             "GTM": "Guatemala", "ETH": "Ethiopia", "CMR": "Cameroon",
             "NIC": "Nicaragua", "MDA": "Moldova", "SRB": "Serbia"}
    body = [TEX_HEAD, r"\begin{table}[t]", r"\centering",
            r"\caption{The \Nswitch{} respondent States whose sanctions status "
            r"changes inside the sample (reference definition). The last column "
            r"is the one that matters for identification: it marks the States "
            r"in which Convention-status eligibility itself varies, and so the "
            r"States that can contribute to the attenuation in "
            r"Table~\ref{tab:core}.}",
            r"\label{tab:switchers}", r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{lcrrcccc c}", r"\toprule",
            r" & Onset & \multicolumn{2}{c}{Cases} "
            r"& \multicolumn{2}{c}{Non-Convention} "
            r"& \multicolumn{2}{c}{Convention open} & Eligibility \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
            r"State & year & pre & post & pre & post & pre & post & varies? \\",
            r"\midrule"]
    for _, r in t.iterrows():
        body.append(
            f"{names.get(r.iso, r.iso):<20} & {r.onset} & {r.npre:>2} & "
            f"{r.npost:>2} & {r.nc_pre} & {r.nc_post} & {r.op_pre} & "
            f"{r.op_post} & " + (r"$\checkmark$" if r.varies else r"---") + r" \\")
    n_var = int(t.varies.sum())
    body += [r"\bottomrule", r"\end{tabular}", r"\par\medskip",
             r"\begin{minipage}{0.93\linewidth}\footnotesize",
             r"\emph{Notes:} ``Convention open'' counts cases in which both the "
             r"respondent and the claimant's home State were ICSID Contracting "
             r"Parties at filing. ``Eligibility varies'' marks States in which "
             r"that indicator is not constant across the State's own cases; "
             rf"there are {n_var} such States. In the remaining "
             rf"{len(t) - n_var}, respondent fixed effects absorb eligibility "
             r"entirely, so those States contribute nothing to the difference "
             r"between the controlled and uncontrolled estimates.",
             r"\end{minipage}", r"\end{table}", ""]
    return _write_tex("\n".join(body), "tab_switchers.tex")


# --- T3  (upgrade 3) ------------------------------------------------------
def tab_core_two_panel(D: Data) -> str:
    """tab_u_core_bloc1_k2 + tab_u_core_idx10 merged into two panels."""
    core = D.csv("tab_core_inference")
    short = {"(0)": "None", "(1)": "Strict", "(2)": "Three-valued",
             "(3)": "Permissive"}
    panels = [("A", "bloc1_k2",
               r"Reference definition: $\geq$2 Western sanctioning jurisdictions "
               r"(12 switching States, 146 identifying cases)"),
              ("B", "idx10",
               r"Legacy index, retained for comparison "
               r"(8 switching States, 113 identifying cases)")]

    body = [TEX_HEAD, r"\begin{table}[t]", r"\centering",
            r"\caption{Sanctions onset and forum: five inference procedures, "
            r"both treatment definitions. $p_{\mathrm{CV3}}$ and "
            r"$p_{\mathrm{WCB}}$ are the primary small-cluster procedures; "
            r"$p_{\mathrm{CR1}}$ is reported for comparability with the "
            r"literature; the two permutation columns are design diagnostics "
            r"rather than randomisation-based tests.}",
            r"\label{tab:core}", r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{llrrrrrrr}", r"\toprule",
            r"Outcome & Eligibility & Effect & (CR1) & $p_{\mathrm{CR1}}$ & "
            r"$p_{\mathrm{CV3}}$ & $p_{\mathrm{WCB}}$ & $p_{\mathrm{restr}}$ & "
            r"$p_{\mathrm{timing}}$ \\", r"\midrule"]

    for pi, (letter, defn, desc) in enumerate(panels):
        if pi:
            body.append(r"\addlinespace")
        body.append(rf"\multicolumn{{9}}{{l}}{{\emph{{Panel {letter}. {desc}}}}}\\")
        sub = core[core.definition == defn]
        for oi, out in enumerate(["Non-Convention (primary)", "UNCITRAL (secondary)"]):
            q = sub[sub.outcome == out]
            if oi:
                body.append(r"\addlinespace")
            for ri, (_, r) in enumerate(q.iterrows()):
                name = out.split(" (")[0] if ri == 0 else ""
                body.append(
                    f"{name} & {short[r.coding[:3]]} & {r.effect:.2f} & "
                    f"({r.se_cr1:.2f}) & {r.p_cr1:.3f} & {r.p_cv3:.3f} & "
                    f"{r.p_wcb_webb:.3f} & {r.p_perm_restricted:.3f} & "
                    f"{r.p_perm_timing:.3f}" + r" \\")

    body += [r"\bottomrule", r"\end{tabular}", r"\par\medskip",
             r"\begin{minipage}{0.97\linewidth}\footnotesize",
             r"\emph{Notes:} \Ncases{} cases. Case-level linear probability "
             r"models with respondent and year fixed effects, outcomes in "
             r"percentage points. Eligibility codings: none; strict "
             r"(denunciation ends eligibility); three-valued (contested as its "
             r"own category); permissive (Article~72 consent survives). "
             r"$p_{\mathrm{CR1}}$ cluster-robust by respondent; "
             r"$p_{\mathrm{CV3}}$ cluster jackknife; $p_{\mathrm{WCB}}$ "
             r"restricted wild cluster bootstrap-$t$ with Webb six-point "
             r"weights ($B=1999$); $p_{\mathrm{restr}}$ permutation of cohort "
             r"assignment among treated States and untreated States with cases "
             r"in at least two years ($B=999$); $p_{\mathrm{timing}}$ "
             r"permutation of onset years among treated States only.",
             r"\end{minipage}", r"\end{table}", ""]
    return _write_tex("\n".join(body), "tab_core_two_panel.tex")


# ===========================================================================
# 4.  REGISTRY
# ===========================================================================
EXHIBITS = {
    "F1": ("fig_cascade_bloc1_k2.pdf", "The paper's argument in one figure",
           "tab_core_inference.csv + two-layer merge", fig_cascade),
    "F2": ("fig_postaward_tracks.pdf", "Two tracks of post-award review",
           "cases_with_navigator.parquet", fig_postaward),
    "F3": ("fig_raw_share_trend.pdf", "Non-Convention share by year and group",
           "cases_with_navigator.parquet + cohorts", fig_raw_trend),
    "F4": ("fig_crimea_timeline.pdf", "Crimea awards: seven years of review",
           "hand-coded from the judgments cited in section 6", fig_crimea),
    "F5": ("fig_influence_event.pdf", "Leave-one-out and event study",
           "tab_influence_loo.csv + tab_event_study.csv", fig_influence_event),
    "T1": ("tab_headline_summary.tex", "The paper in one table",
           "tab_core_inference.csv + tab_attenuation_paired.csv", tab_headline),
    "T2": ("tab_switchers.tex", "The twelve switching States",
           "cases_with_navigator.parquet + cohorts + ICSID reference",
           tab_switchers),
    "T3": ("tab_core_two_panel.tex", "Five inference procedures, both definitions",
           "tab_core_inference.csv", tab_core_two_panel),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="../upgrade/results")
    ap.add_argument("--cases", default="../proj/paper_tables/cases_with_navigator.parquet")
    ap.add_argument("--cohorts", default="../proj/data/derived/gsdb_treatment_cohorts.csv")
    ap.add_argument("--icsid", default="../proj/data/reference/icsid_convention_status_icsid3.csv")
    ap.add_argument("--src", default="../proj/src")
    ap.add_argument("--only", nargs="*", default=None, metavar="ID")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print(f"{'ID':<4}{'output':<34}{'source':<52}what")
        for k, (out, what, src, _) in EXHIBITS.items():
            print(f"{k:<4}{out:<34}{src:<52}{what}")
        return

    D = Data(args)
    todo = args.only or list(EXHIBITS)
    written = []
    for k in todo:
        if k not in EXHIBITS:
            raise SystemExit(f"unknown exhibit {k!r}; try --list")
        out, what, src, fn = EXHIBITS[k]
        print(f"[{k}] {what}")
        written.append((k, fn(D), src, what))

    Path("EXHIBITS.tsv").write_text(
        "id\tpath\tsource\tdescription\n"
        + "\n".join("\t".join(r) for r in written) + "\n", encoding="utf-8")
    print(f"\n{len(written)} exhibits written; manifest in EXHIBITS.tsv")


if __name__ == "__main__":
    main()
