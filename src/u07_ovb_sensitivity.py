"""
u07_ovb_sensitivity.py — Omitted-variable sensitivity benchmarked on Convention-status
eligibility (M5)

Cinelli & Hazlett (2020, JRSS-B). Model (everything residualised on respondent State +
year FE, FWL):
    Y = β·S + Z'δ + ε,  Z = three-valued eligibility dummies (the observed legal confounder)
Question: how far would an unobserved legal factor U that is **as strong as** Z (k=1),
    half as strong (k=0.5) or twice as strong (k=2), entering the model orthogonally
    to Z, move β?

Partial-R² benchmark (formulae of sensemakr::ovb_partial_r2_bound, with Z as a group
of covariates):
    r2dxj = R²_{S~Z | FE}
    r2yxj = R²_{Y~Z | S, FE}
    r2dz  = k · r2dxj / (1 − r2dxj)
    r2zxj = k · r2dxj² / ((1 − k·r2dxj)(1 − r2dxj))
    r2yz  = ((√k + √r2zxj) / √(1 − r2zxj))² · r2yxj / (1 − r2yxj)
Bias magnitude (independent of degrees of freedom):
    |bias| = sqrt(r2yz · r2dz / (1 − r2dz)) · sqrt(RSS_Y|S,Z,FE / RSS_S|Z,FE)
Adjusted estimate = β̂ − sign(β̂)·|bias| (adjusted towards zero);
adjusted standard error = se_CR1 · sqrt((1 − r2yz)/(1 − r2dz)).

**Limitation (must be stated in the paper)**: the Cinelli–Hazlett inference formulae
assume classical OLS, whereas inference in this paper is clustered. Only the adjusted
point estimate is therefore treated as the main output; the adjusted t-value is for
reference only. The robustness value depends on the degrees of freedom; it is computed
here with N−k and labelled accordingly.

Usage
-----
    python src/u07_ovb_sensitivity.py --defs bloc1_k2 idx10 --out paper_tables
"""
from __future__ import annotations

import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from u00_common import (OUTCOMES, PP, TIME, UNIT, Projector, _dummies, attach,  # noqa: E402
                        load_base, load_cohorts, outdir, save)


def partial_r2_group(target: np.ndarray, Z: np.ndarray, others: np.ndarray | None) -> float:
    """R² of target on Z, partialling out `others` (target, Z already FE-residualised)."""
    def rss(v, M):
        if M is None or M.shape[1] == 0:
            return v @ v
        b = np.linalg.lstsq(M, v, rcond=None)[0]
        e = v - M @ b
        return e @ e
    base = rss(target, others)
    full = rss(target, Z if others is None else np.column_stack([others, Z]))
    return 1 - full / base


def bounds(k, r2dxj, r2yxj):
    r2dz = k * r2dxj / (1 - r2dxj)
    r2zxj = k * r2dxj ** 2 / ((1 - k * r2dxj) * (1 - r2dxj))
    r2yz = ((math.sqrt(k) + math.sqrt(r2zxj)) / math.sqrt(1 - r2zxj)) ** 2 * r2yxj / (1 - r2yxj)
    return r2dz, min(r2yz, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defs", nargs="+", default=["bloc1_k2", "idx10"])
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", default="paper_tables")
    a = ap.parse_args()
    out = outdir(a.out)
    base = load_base(a.ref)
    coh = load_cohorts()

    rows = []
    for dname in a.defs:
        d = attach(base, coh[dname].dropna().to_dict()).reset_index(drop=True)
        P = Projector(d, [UNIT, TIME])
        Zraw = _dummies(d, ["avail_three"])[:, 1:]          # drop the constant
        Z = P.resid(Zraw)
        S = P.resid(d["S"].to_numpy(float))
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for ax, (y, ylab) in zip(axes, OUTCOMES.items()):
            Y = P.resid(d[y].to_numpy(float) * PP)
            # "Full" model including Z
            M = np.column_stack([S, Z])
            coef = np.linalg.lstsq(M, Y, rcond=None)[0]
            b = coef[0]
            e = Y - M @ coef
            rss_y = e @ e
            bS = np.linalg.lstsq(Z, S, rcond=None)[0]
            eS = S - Z @ bS
            rss_s = eS @ eS
            # CR1 (same adjustment as u00)
            sc = np.bincount(P.cl, weights=eS * e, minlength=P.G)
            k_par = P.k_fe + 1 + Z.shape[1]
            adj = (P.G / (P.G - 1)) * ((P.n - 1) / max(P.n - k_par, 1))
            se = math.sqrt(adj * (sc @ sc)) / rss_s
            dof = P.n - k_par
            t = b / se
            f = (b / math.sqrt(rss_y / dof / rss_s)) / math.sqrt(dof)   # classical t / √dof
            rv = 0.5 * (math.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)
            r2dxj = partial_r2_group(S, Z, None)
            r2yxj = partial_r2_group(Y, Z, S[:, None])
            rec = {"definition": dname, "outcome": ylab, "beta_with_Z": b,
                   "se_cr1": se, "t_cr1": t, "partial_r2_S_on_Z": r2dxj,
                   "partial_r2_Y_on_Z": r2yxj, "RV_q1_classical_dof": rv}
            for kk in (0.5, 1.0, 2.0):
                if kk * r2dxj >= 1:
                    continue
                r2dz, r2yz = bounds(kk, r2dxj, r2yxj)
                bias = math.sqrt(r2yz * r2dz / (1 - r2dz)) * math.sqrt(rss_y / rss_s)
                adj_b = b - math.copysign(bias, b)
                adj_se = se * math.sqrt(max(1 - r2yz, 1e-12) / (1 - r2dz))
                rec[f"k{kk}_r2dz"] = r2dz
                rec[f"k{kk}_r2yz"] = r2yz
                rec[f"k{kk}_beta_adj"] = adj_b
                rec[f"k{kk}_t_adj_ref"] = adj_b / adj_se
            rows.append(rec)
            # Contour plot: adjusted β
            gx = np.linspace(0, 0.3, 61)
            gy = np.linspace(0, 0.3, 61)
            XX, YY = np.meshgrid(gx, gy)
            B = b - np.sign(b) * np.sqrt(YY * XX / (1 - XX)) * math.sqrt(rss_y / rss_s)
            cs = ax.contour(XX, YY, B, levels=10, colors="0.4", linewidths=0.6)
            ax.clabel(cs, fontsize=6, fmt="%.0f")
            ax.contour(XX, YY, B, levels=[0], colors="tab:red", linewidths=1.4)
            for kk in (0.5, 1.0, 2.0):
                if f"k{kk}_r2dz" in rec:
                    ax.scatter(rec[f"k{kk}_r2dz"], rec[f"k{kk}_r2yz"], color="k", s=14)
                    ax.annotate(f"{kk}× availability", (rec[f"k{kk}_r2dz"], rec[f"k{kk}_r2yz"]),
                                fontsize=6, xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel("partial R² of confounder with treatment")
            ax.set_ylabel("partial R² of confounder with outcome")
            ax.set_title(f"{ylab}: β = {b:.1f} (red: β = 0)", fontsize=8)
        fig.suptitle(f"Omitted-variable sensitivity benchmarked on legal availability ({dname})",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"fig_ovb_{dname}.pdf")
        print(f"  -> {out / f'fig_ovb_{dname}.pdf'}")

    print("\n[u07] Cinelli–Hazlett sensitivity (benchmark = three-valued eligibility)")
    save(pd.DataFrame(rows), out, "tab_ovb_sensitivity.csv")


if __name__ == "__main__":
    main()
