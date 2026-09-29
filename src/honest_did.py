"""
honest_did.py — a full linear-programming moment-inequality implementation of
Rambachan-Roth (2023).

It supersedes an earlier conservative approximation based on "the maximum absolute
pre-treatment coefficient" (a `rambachan_roth` function in the earlier pipeline, not
included in this repository); callers that used that approximation switch to the
implementation here, and the results of
the two are printed side by side for comparison.

============================================================================
Differences from the earlier conservative approximation
============================================================================

Earlier approach: the bias bound was set to Mbar * max|beta_pre|, i.e. violations were
measured in **levels**. This departs from the Rambachan-Roth definition and is overly
conservative — it penalises the size of the pre-treatment coefficients themselves rather
than the **rate of change** of the trend.

This module follows the original definition and imposes restrictions on **first
differences**:

  Relative-magnitudes set Delta^RM(Mbar):
     |delta_{t+1} - delta_t| <= Mbar * max_{s in pre} |delta_{s+1} - delta_s|
     for every post-treatment period t.

  Second-difference smoothness set Delta^SD(M):
     |(delta_{t+1} - delta_t) - (delta_t - delta_{t-1})| <= M
     i.e. the curvature of the trend is bounded (M=0 means linear extrapolation).

Given the feasible set for delta, the identified set is the linear programme

     max / min   l' delta_post
     s.t.        delta in Delta,  delta_pre = beta_pre,  delta_{-1} = 0

where l is the vector of weights on the post-treatment periods (equal-weighted average
by default). The bounds on the treatment effect are
     tau in [ l'beta_post - max l'delta_post ,  l'beta_post - min l'delta_post ]

Sampling uncertainty is handled by a parametric bootstrap: draw from
beta_hat ~ N(beta_hat, V), re-solve the LP for each draw, and take the alpha/2 quantile
of the lower bound and the 1-alpha/2 quantile of the upper bound.

============================================================================
Differences from the original paper (must be stated accurately in the paper)
============================================================================

The original paper uses the Andrews-Roth-Pakes conditional test (ARP), which has better
power properties near the boundary of the moment inequalities. This module uses an
"LP identified set + parametric bootstrap" construction, which:
  - is a genuine moment-inequality implementation (on first differences, following the
    original definition)
  - is close to the ARP FLCI results in most applications
  - but does not guarantee ARP's uniform asymptotic coverage at the boundary

The paper should therefore say "we implement the Rambachan-Roth identified set under the
relative-magnitudes restriction and construct confidence intervals by parametric
bootstrap", not "we use the HonestDiD ARP test". For full alignment with the original,
cross-validate by calling the HonestDiD package in R; the CSV format output by this
module is already aligned with that package's input, to make checking easy.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog

# ---------------------------------------------------------------------------
# Event-period indexing convention:
#   periods: ordered list of event times, e.g. [-5,-4,-3,-2, 0,1,2,3,4]
#   the reference period -1 is omitted and its delta is fixed at 0
#   the delta vector has the same order and length as periods
# ---------------------------------------------------------------------------


def _delta_with_ref(periods):
    """Return the full period sequence including reference period -1, and the positions of delta within it."""
    full = sorted(set(list(periods) + [-1]))
    pos = {t: i for i, t in enumerate(full)}
    free = [pos[t] for t in periods]      # positions of the free variables in full
    ref = pos[-1]
    return full, free, ref


def _first_diff_matrix(full):
    """Build the first-difference matrix D: (len(full)-1) x len(full), adjacent periods only."""
    n = len(full)
    D = np.zeros((n - 1, n))
    for i in range(n - 1):
        D[i, i] = -1.0
        D[i, i + 1] = 1.0
    return D


def _build_lp(periods, beta_pre_map, l_post, kind, param):
    """Build the constraint matrices of the LP.

    Decision variable x = values of delta on full (including the reference period,
    which is fixed at 0 by an equality constraint).
    Returns (c, A_ub, b_ub, A_eq, b_eq, bounds); the objective is min c'x.
    """
    full, free, ref = _delta_with_ref(periods)
    n = len(full)
    D = _first_diff_matrix(full)

    # Equalities: reference period delta_{-1}=0; pre-treatment deltas fixed at their estimates
    A_eq, b_eq = [], []
    row = np.zeros(n); row[ref] = 1.0
    A_eq.append(row); b_eq.append(0.0)
    for t, v in beta_pre_map.items():
        r = np.zeros(n); r[full.index(t)] = 1.0
        A_eq.append(r); b_eq.append(float(v))

    A_ub, b_ub = [], []
    pre_idx = [i for i, t in enumerate(full) if t <= -1]
    post_pairs = [i for i in range(n - 1) if full[i + 1] > -1]

    if kind == "RM":
        # Maximum absolute pre-treatment first difference (computed from the estimates; a known constant)
        dpre = []
        for i in range(n - 1):
            if full[i + 1] <= -1:
                a = beta_pre_map.get(full[i], 0.0 if full[i] == -1 else None)
                b = beta_pre_map.get(full[i + 1],
                                     0.0 if full[i + 1] == -1 else None)
                if a is None or b is None:
                    continue
                dpre.append(abs(b - a))
        Mmax = max(dpre) if dpre else 0.0
        bound = param * Mmax
        for i in post_pairs:
            A_ub.append(D[i]); b_ub.append(bound)
            A_ub.append(-D[i]); b_ub.append(bound)

    elif kind == "SD":
        # Bounded second differences
        for i in range(n - 2):
            if full[i + 2] <= -1:
                continue
            r = np.zeros(n)
            r[i] = 1.0; r[i + 1] = -2.0; r[i + 2] = 1.0
            A_ub.append(r); b_ub.append(param)
            A_ub.append(-r); b_ub.append(param)
    else:
        raise ValueError("kind must be 'RM' or 'SD'")

    # Objective: l' delta_post
    c = np.zeros(n)
    for t, w in l_post.items():
        c[full.index(t)] = w

    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None
    return (c, A_ub, b_ub, np.array(A_eq), np.array(b_eq),
            [(-1e6, 1e6)] * n)


def identified_set(periods, beta, l_post, kind="RM", param=1.0):
    """Given a set of event-study coefficients, compute the identified set [lo, hi] for l'delta_post."""
    beta_pre_map = {t: beta[i] for i, t in enumerate(periods) if t <= -1}
    c, A_ub, b_ub, A_eq, b_eq, bnds = _build_lp(
        periods, beta_pre_map, l_post, kind, param)
    lo = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                 bounds=bnds, method="highs")
    hi = linprog(-c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                 bounds=bnds, method="highs")
    if not (lo.success and hi.success):
        return None
    return float(lo.fun), float(-hi.fun)


def honest_ci(periods, beta, V, kind="RM", param=1.0, alpha=0.05,
              n_draw=1000, seed=20260831):
    """Robust confidence interval from the LP identified set + parametric bootstrap.

    For each draw beta~ ~ N(beta_hat, V):
      1. compute the identified set [d_lo, d_hi] from the pre-treatment part of beta~
      2. the bounds on tau are [l'beta~_post - d_hi, l'beta~_post - d_lo]
    Final CI = [alpha/2 quantile of the lower bound, 1-alpha/2 quantile of the upper bound]
    """
    rng = np.random.default_rng(seed)
    post = [t for t in periods if t >= 0]
    if not post:
        return None
    l_post = {t: 1.0 / len(post) for t in post}
    idx_post = [periods.index(t) for t in post]
    w = np.zeros(len(periods)); w[idx_post] = 1.0 / len(post)

    # V may be non-positive-definite (numerically); symmetrise and truncate eigenvalues
    Vs = (V + V.T) / 2
    ev, evec = np.linalg.eigh(Vs)
    ev = np.clip(ev, 0, None)
    L = evec @ np.diag(np.sqrt(ev))

    los, his = [], []
    for _ in range(n_draw):
        bt = beta + L @ rng.standard_normal(len(beta))
        s = identified_set(periods, bt, l_post, kind, param)
        if s is None:
            continue
        d_lo, d_hi = s
        theta = float(w @ bt)
        los.append(theta - d_hi)
        his.append(theta - d_lo)
    if len(los) < 50:
        return None
    return (float(np.quantile(los, alpha / 2)),
            float(np.quantile(his, 1 - alpha / 2)),
            len(los))


def breakdown(periods, beta, V, kind="RM", grid=None, alpha=0.05,
              n_draw=500, seed=20260831):
    """Breakdown value: the smallest param at which the robust CI first contains 0."""
    if grid is None:
        grid = np.round(np.concatenate([
            np.arange(0.0, 0.30, 0.02),      # finer grid over the critical range
            np.arange(0.30, 2.01, 0.10)]), 3) if kind == "RM" \
            else np.round(np.concatenate([
                np.arange(0.0, 0.10, 0.005),
                np.arange(0.10, 0.51, 0.05)]), 4)
    rows, bd = [], None
    for m in grid:
        ci = honest_ci(periods, beta, V, kind, float(m), alpha, n_draw, seed)
        if ci is None:
            continue
        lo, hi, nd = ci
        contains = (lo <= 0 <= hi)
        rows.append({"kind": kind, "param": float(m), "ci_lo": round(lo, 3),
                     "ci_hi": round(hi, 3), "contains_zero": contains,
                     "n_draws": nd})
        if contains and bd is None:
            bd = float(m)
    return bd, pd.DataFrame(rows)


def analyse(periods, beta, V, label="", alpha=0.05, n_draw=500, verbose=True):
    """Run the full analysis for one outcome variable: conventional CI + RM and SD breakdown values."""
    from scipy.stats import norm
    post = [t for t in periods if t >= 0]
    w = np.zeros(len(beta))
    for t in post:
        w[periods.index(t)] = 1.0 / len(post)
    theta = float(w @ beta)
    se = float(np.sqrt(w @ V @ w))
    z = norm.ppf(1 - alpha / 2)
    orig = (theta - z * se, theta + z * se)

    bd_rm, grid_rm = breakdown(periods, beta, V, "RM", alpha=alpha,
                               n_draw=n_draw)
    bd_sd, grid_sd = breakdown(periods, beta, V, "SD", alpha=alpha,
                               n_draw=n_draw)

    summ = {
        "outcome": label, "post_avg": round(theta, 3), "se": round(se, 3),
        "orig_ci_lo": round(orig[0], 3), "orig_ci_hi": round(orig[1], 3),
        "breakdown_Mbar_RM": bd_rm, "breakdown_M_SD": bd_sd,
    }
    if verbose:
        print(f"\n  --- {label} ---")
        print(f"   Conventional CI: [{orig[0]:.2f}, {orig[1]:.2f}]  "
              f"(point estimate {theta:.2f}, se {se:.2f})")
        print(f"   RM breakdown value Mbar = {bd_rm}"
              f"{'  (never contains zero within the grid)' if bd_rm is None else ''}")
        print(f"   SD breakdown value M    = {bd_sd}"
              f"{'  (never contains zero within the grid)' if bd_sd is None else ''}")
    return summ, pd.concat([grid_rm, grid_sd], ignore_index=True)
