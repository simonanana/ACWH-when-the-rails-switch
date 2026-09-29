"""
_compat.py — self-contained estimation and inference layer (drop-in).

Design principles
-----------------
1. This module **first tries** to import the functions of the same name from the existing
   `estimators.py`; if the import fails, it uses the pure numpy/pandas implementations in
   this file. This way r22, r23 and r25 can both plug into the existing pipeline and
   run standalone.
2. Every function returns a dict with fixed key names, so results can be written to
   canonical_numbers.csv.
3. No dependency on statsmodels / linearmodels / pyfixest is introduced (consistent with
   the project's established "self-contained toolkit" principle), but a parity-test hook
   is provided (see `PARITY_NOTE`).

Interface contract (for replacing these with another implementation)
--------------------------------------------------------------------
feols(df, y, x, absorb=[...], cluster="...") ->
    {"coef": float, "se": float, "t": float, "p": float,
     "n": int, "n_clusters": int, "resid": np.ndarray, "fitted": np.ndarray}

Author's note: the CR1 finite-sample correction uses G/(G-1) * (N-1)/(N-K), where K
includes the absorbed fixed-effect parameters (approximation: sum_f (levels_f) - (F-1)).
This matches the reghdfe/fixest defaults.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PARITY_NOTE = (
    "parity targets: R fixest::feols(cluster=~id) / Stata reghdfe, "
    "fwildclusterboot::boottest(type='rademacher', impose_null=TRUE)"
)

# --------------------------------------------------------------------------
# 0. Try to reuse the existing in-project implementation
# --------------------------------------------------------------------------
_USING_PROJECT_ESTIMATORS = False
try:  # pragma: no cover
    from estimators import feols as _project_feols  # type: ignore

    _USING_PROJECT_ESTIMATORS = True
except Exception:  # noqa: BLE001
    _project_feols = None


# --------------------------------------------------------------------------
# 1. Multi-way fixed-effects demeaning (alternating projections / Frisch–Waugh–Lovell)
# --------------------------------------------------------------------------
def _demean(mat: np.ndarray, fe_codes: list[np.ndarray], tol: float = 1e-10,
            maxiter: int = 200) -> np.ndarray:
    """Apply the multi-way fixed-effects within transformation to each column of mat. fe_codes is a list of integer-coded arrays."""
    out = mat.astype(float).copy()
    if not fe_codes:
        return out - out.mean(axis=0, keepdims=True)
    for _ in range(maxiter):
        prev = out.copy()
        for codes in fe_codes:
            n_lev = codes.max() + 1
            for j in range(out.shape[1]):
                sums = np.bincount(codes, weights=out[:, j], minlength=n_lev)
                cnts = np.bincount(codes, minlength=n_lev)
                out[:, j] -= (sums / np.maximum(cnts, 1))[codes]
        if np.max(np.abs(out - prev)) < tol:
            break
    return out


def _codes(s: pd.Series) -> np.ndarray:
    return pd.Categorical(s.astype(str)).codes.astype(np.int64)


class _Residualiser:
    """Fixed-effects residualiser: dummy-variable projection when n is small (caching (D'D)^-1), alternating projections otherwise."""

    def __init__(self, fe_codes: list[np.ndarray], n: int, dense_max: int = 20000):
        self.fe_codes = fe_codes
        self.n = n
        self.dense = (n <= dense_max)
        if self.dense:
            blocks = [np.ones((n, 1))]
            for c in fe_codes:
                k = c.max() + 1
                if k > 1:
                    Dm = np.zeros((n, k - 1))
                    m = c > 0
                    Dm[np.where(m)[0], c[m] - 1] = 1.0
                    blocks.append(Dm)
            self.D = np.hstack(blocks)
            self.DtDi = np.linalg.pinv(self.D.T @ self.D)

    def __call__(self, mat: np.ndarray) -> np.ndarray:
        mat = np.asarray(mat, float)
        one_d = mat.ndim == 1
        M = mat[:, None] if one_d else mat
        if self.dense:
            R = M - self.D @ (self.DtDi @ (self.D.T @ M))
        else:
            R = _demean(M, self.fe_codes)
        return R.ravel() if one_d else R


def _absorbed_params(fe_codes: list[np.ndarray]) -> int:
    if not fe_codes:
        return 1
    return int(sum(c.max() + 1 for c in fe_codes) - (len(fe_codes) - 1))


# --------------------------------------------------------------------------
# 2. FE-OLS + cluster-robust (CR1)
# --------------------------------------------------------------------------
def feols(df: pd.DataFrame, y: str, x: list[str] | str,
          absorb: list[str] | None = None, cluster: str | None = None,
          target: str | None = None, use_project: bool = True) -> dict:
    """Single-equation FE-OLS. target specifies the regressor to report (default: the first element of x)."""
    if use_project and _USING_PROJECT_ESTIMATORS:  # pragma: no cover
        try:
            return _project_feols(df, y, x, absorb=absorb, cluster=cluster)
        except Exception:  # noqa: BLE001
            pass

    xs = [x] if isinstance(x, str) else list(x)
    absorb = list(absorb or [])
    cols = [y] + xs + absorb + ([cluster] if cluster else [])
    d = df.dropna(subset=[c for c in cols if c is not None]).copy()

    Y = d[[y]].to_numpy(float)
    X = d[xs].to_numpy(float)
    fe_codes = [_codes(d[a]) for a in absorb]

    rz = _Residualiser(fe_codes, len(d))
    Yt = rz(Y)
    Xt = rz(X)

    XtX = Xt.T @ Xt
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (Xt.T @ Yt.ravel())
    resid = Yt.ravel() - Xt @ beta

    n = len(d)
    k = len(xs) + _absorbed_params(fe_codes)

    if cluster is not None:
        g_codes = _codes(d[cluster])
        G = int(g_codes.max() + 1)
        meat = np.zeros_like(XtX)
        for g in range(G):
            m = g_codes == g
            u = Xt[m].T @ resid[m]
            meat += np.outer(u, u)
        c = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
        V = c * XtX_inv @ meat @ XtX_inv
        dof = G - 1
    else:
        s2 = resid @ resid / max(n - k, 1)
        V = s2 * XtX_inv
        G = n
        dof = n - k

    idx = xs.index(target) if target else 0
    se = float(np.sqrt(np.diag(V))[idx])
    b = float(beta[idx])
    t = b / se if se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), df=max(dof, 1))) if np.isfinite(t) else np.nan

    return {
        "coef": b, "se": se, "t": t, "p": p, "n": n, "n_clusters": G,
        "dof": dof, "term": xs[idx], "resid": resid,
        "fitted": Yt.ravel() - resid, "beta_all": beta, "vcov": V, "xnames": xs,
    }


# --------------------------------------------------------------------------
# 3. Restricted wild cluster bootstrap-t (Rademacher, null imposed)
# --------------------------------------------------------------------------
def wild_cluster_bootstrap(df: pd.DataFrame, y: str, x: list[str] | str,
                           absorb: list[str] | None = None,
                           cluster: str = "respondent_iso3",
                           target: str | None = None, B: int = 1999,
                           seed: int = 20261020, weights: str = "rademacher") -> dict:
    """Restricted wild cluster bootstrap-t (imposing H0: beta_target = 0).

    weights: "rademacher" (default, consistent with the project's
             estimators.wild_cluster_boot) or "webb" (six-point distribution).
             **With only 8 switching clusters, Webb should be used instead**:
             Rademacher can generate only 2^8 = 256 distinct weight combinations,
             so the resolution of the bootstrap p-value is locked at 1/256 ≈ 0.004,
             and the distribution is too discrete to give reliable tails
             (Webb 2014; MacKinnon & Webb 2018).
    """
    xs = [x] if isinstance(x, str) else list(x)
    tgt = target or xs[0]
    ti = xs.index(tgt)
    cols = [y] + xs + list(absorb or []) + [cluster]
    d = df.dropna(subset=cols).copy()
    n = len(d)
    fe_codes = [_codes(d[a]) for a in (absorb or [])]
    rz = _Residualiser(fe_codes, n)
    Yt = rz(d[y].to_numpy(float))
    Xt = rz(d[xs].to_numpy(float))
    g_codes = _codes(d[cluster])
    G = int(g_codes.max() + 1)
    k = len(xs) + _absorbed_params(fe_codes)
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    XtXi = np.linalg.pinv(Xt.T @ Xt)

    def _cr1(yv):
        b = XtXi @ (Xt.T @ yv)
        e = yv - Xt @ b
        U = np.zeros((G, Xt.shape[1]))
        np.add.at(U, g_codes, Xt * e[:, None])
        V = XtXi @ (U.T @ U) @ XtXi * adj
        return b, V

    b, V = _cr1(Yt)
    t_obs = b[ti] / np.sqrt(V[ti, ti])

    keep = [i for i in range(Xt.shape[1]) if i != ti]
    if keep:
        Xr = Xt[:, keep]
        br = np.linalg.pinv(Xr.T @ Xr) @ (Xr.T @ Yt)
        fit_r = Xr @ br
    else:
        fit_r = np.zeros(n)
    u_r = Yt - fit_r

    rng = np.random.default_rng(seed)
    if weights == "webb":
        pool = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5),
                         np.sqrt(0.5), 1.0, np.sqrt(1.5)])
    elif weights == "rademacher":
        pool = np.array([-1.0, 1.0])
    else:
        raise ValueError("weights must be 'rademacher' or 'webb'")
    t_star = np.empty(B)
    for j in range(B):
        w = rng.choice(pool, size=G)[g_codes]
        # u_r*w is no longer orthogonal to the fixed effects and must be re-residualised (otherwise the p-value is too small)
        bb, VV = _cr1(fit_r + rz(u_r * w))
        t_star[j] = bb[ti] / np.sqrt(VV[ti, ti])
    t_star = t_star[np.isfinite(t_star)]
    p = float((np.sum(np.abs(t_star) >= abs(t_obs)) + 1) / (len(t_star) + 1))
    return {"coef": float(b[ti]), "se": float(np.sqrt(V[ti, ti])),
            "t_obs": float(t_obs), "p_wcb": p, "B": len(t_star),
            "n_clusters": G, "weights": weights}


# --------------------------------------------------------------------------
# 4. Permutation inference (optionally stratified; restricted permutation space)
# --------------------------------------------------------------------------
def permutation_test(df: pd.DataFrame, y: str, absorb: list[str],
                     cluster: str, unit: str, time: str,
                     cohort_map: dict, controls: list[str] | None = None,
                     strata: dict | None = None, B: int = 999,
                     seed: int = 20261020, absorbing: bool = True) -> dict:
    """
    State-level permutation test (to be described as permutation inference).

    cohort_map : {unit -> onset_year or NaN}. Permutation pool = the keys of cohort_map:
                 passing all respondent States (untreated as NaN) = the project's
                 randomisation_test convention;
                 passing only ever-treated States = a stricter "timing-only" permutation.
    strata     : {unit -> stratum}; when given, permute only within strata.
    Like the project implementation, uses FWL: y and S are each residualised on the
    fixed effects (and controls).
    """
    controls = list(controls or [])
    units = list(cohort_map.keys())
    cohorts = np.array([cohort_map[u] for u in units], dtype=float)
    d = df.dropna(subset=[y, unit, time] + controls).reset_index(drop=True)
    n = len(d)
    fe_codes = [_codes(d[a]) for a in absorb]
    rz = _Residualiser(fe_codes, n)
    yv = rz(d[y].to_numpy(float))
    if controls:
        C = rz(d[controls].to_numpy(float))
        Pc = C @ np.linalg.pinv(C.T @ C) @ C.T
        yv = yv - Pc @ yv
    yrs = d[time].to_numpy(float)
    uidx = d[unit].map({u: i for i, u in enumerate(units)}).to_numpy()
    valid = ~pd.isna(uidx)
    uidx = np.where(valid, uidx, -1).astype(int)

    def _S(cvec):
        gv = np.where(uidx >= 0, cvec[np.maximum(uidx, 0)], np.nan)
        if absorbing:
            return ((~np.isnan(gv)) & (yrs >= gv)).astype(float)
        return ((~np.isnan(gv)) & (yrs == gv)).astype(float)

    def _beta(S):
        Sr = rz(S)
        if controls:
            Sr = Sr - Pc @ Sr
        den = Sr @ Sr
        return np.nan if den <= 1e-10 else float((Sr @ yv) / den)

    b_obs = _beta(_S(cohorts))
    rng = np.random.default_rng(seed)
    strat = (np.zeros(len(units)) if strata is None
             else np.array([str(strata.get(u, "_NA")) for u in units]))
    b_star = np.empty(B)
    for j in range(B):
        perm = cohorts.copy()
        for lab in np.unique(strat):
            m = strat == lab
            perm[m] = rng.permutation(cohorts[m])
        b_star[j] = _beta(_S(perm))
    b_star = b_star[np.isfinite(b_star)]
    p = float((np.sum(np.abs(b_star) >= abs(b_obs)) + 1) / (len(b_star) + 1))
    return {"coef": b_obs, "p_perm": p, "B": len(b_star),
            "stratified": strata is not None,
            "null_sd": float(np.std(b_star)) if len(b_star) else np.nan}


# --------------------------------------------------------------------------
# 5. Bounding null results: MDE and TOST (consistent with the §4 framework)
# --------------------------------------------------------------------------
def mde(se: float, power: float = 0.80, alpha: float = 0.05) -> float:
    """Minimum detectable effect for a two-sided test at the given power."""
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se)


def tost(coef: float, se: float, bound: float) -> dict:
    """Two one-sided tests; H0: |beta| >= bound."""
    p_low = float(stats.norm.cdf((coef + bound) / se))
    p_high = float(stats.norm.sf((coef - bound) / se))
    p = max(p_low, p_high)
    return {"coef": coef, "se": se, "bound": bound, "p_tost": p,
            "equivalent_5pct": bool(p < 0.05)}


# --------------------------------------------------------------------------
# 6. Utilities
# --------------------------------------------------------------------------
def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.10 else ""


def fmt(v: float, d: int = 2) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def effective_identifying_sample(df: pd.DataFrame, treat: str, unit: str) -> dict:
    """Report the identifying sample rather than the nominal sample (project design principle #5)."""
    g = df.groupby(unit)[treat]
    switchers = [u for u, s in g if s.nunique() > 1]
    any_treated = [u for u, s in g if s.max() > 0]
    ident = df[df[unit].isin(switchers)]
    return {"n_clusters_nominal": int(df[unit].nunique()),
            "n_clusters_any_treated": len(any_treated),
            "n_clusters_switching": len(switchers),
            "switchers": sorted(switchers),
            "n_cases_nominal": int(len(df)),
            "n_cases_identifying": int(len(ident))}
