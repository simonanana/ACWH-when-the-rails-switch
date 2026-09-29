"""
estimators.py — self-contained estimation and inference tools, with no dependency on
linearmodels / did / fixest.

Contents:
  feols            fixed-effects OLS + CR1 cluster-robust standard errors
  wild_cluster_boot restricted wild cluster bootstrap-t (Cameron-Gelbach-Miller)
  randomisation_test State-level permutation test (more conservative than the WCB with
                    few switching clusters)
  ordered_logit     proportional-odds ordered logit, for the delocalisation ladder
  romano_wolf       stepdown correction for multiple testing
  ppml              Poisson pseudo-maximum likelihood with high-dimensional fixed effects
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import N_BOOT_WCB, N_PERM_RI, SEED


# ---------------------------------------------------------------- FE OLS ----
def _design(df, x_cols, fe_cols):
    X = df[x_cols].astype(float).reset_index(drop=True)
    if fe_cols:
        D = pd.get_dummies(df[list(fe_cols)].astype(str), drop_first=True,
                           dtype=float).reset_index(drop=True)
        X = pd.concat([X, D], axis=1)
    return sm.add_constant(X, has_constant="add")


def feols(df, y, x_cols, fe_cols, cluster):
    """Fixed-effects OLS with CR1 cluster-robust standard errors. Returns a dict of (b, se, t, p) for each x."""
    X = _design(df, x_cols, fe_cols)
    yy = df[y].astype(float).values
    r = sm.OLS(yy, X.values).fit(cov_type="cluster",
                                 cov_kwds={"groups": df[cluster].values},
                                 use_t=True)
    cols = list(X.columns)
    out = {}
    for k in x_cols:
        i = cols.index(k)
        out[k] = {"b": float(r.params[i]), "se": float(r.bse[i]),
                  "t": float(r.tvalues[i]), "p": float(r.pvalues[i])}
    out["_nobs"] = int(r.nobs)
    out["_r2"] = float(r.rsquared)
    return out


# ------------------------------------------------- wild cluster bootstrap ---
def _cr1(X, y, cl, XtXi=None, G=None):
    """CR1 cluster-robust variance. The meat is vectorised with np.add.at to avoid looping over clusters."""
    if G is None:
        G = cl.max() + 1
    if XtXi is None:
        XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ (X.T @ y)
    e = y - X @ b
    Xe = X * e[:, None]
    U = np.zeros((G, X.shape[1]))
    np.add.at(U, cl, Xe)          # score sum for each cluster
    meat = U.T @ U
    n, k = X.shape
    adj = (G / (G - 1)) * ((n - 1) / (n - k))
    V = XtXi @ meat @ XtXi * adj
    return b, V, e


def wild_cluster_boot(df, y, x, fe_cols, cluster, B=N_BOOT_WCB, seed=SEED):
    """Restricted wild cluster bootstrap-t. Resamples under H0: beta_x = 0.

    Better than CR1 with few treated clusters, but MacKinnon-Nielsen-Webb (2023) show
    that it can still be distorted when there are very few switching clusters — hence
    this project also reports randomisation_test.
    """
    rng = np.random.default_rng(seed)
    X = _design(df, [x], fe_cols).values.astype(float)
    yy = df[y].astype(float).values
    cl = pd.factorize(df[cluster].values)[0]
    G = cl.max() + 1
    ix = 1  # const in column 0, x in column 1

    XtXi = np.linalg.pinv(X.T @ X)          # X is unchanged across resamples; cache
    b, V, _ = _cr1(X, yy, cl, XtXi, G)
    t_obs = b[ix] / np.sqrt(V[ix, ix])

    keep = [i for i in range(X.shape[1]) if i != ix]
    Xr = X[:, keep]
    br = np.linalg.pinv(Xr.T @ Xr) @ (Xr.T @ yy)
    er = yy - Xr @ br
    fit = Xr @ br

    ts = np.empty(B)
    for j in range(B):
        w = rng.choice([-1.0, 1.0], size=G)[cl]
        bb, VV, _ = _cr1(X, fit + er * w, cl, XtXi, G)
        ts[j] = bb[ix] / np.sqrt(VV[ix, ix])
    p = (np.sum(np.abs(ts) >= abs(t_obs)) + 1) / (B + 1)
    return {"b": float(b[ix]), "se": float(np.sqrt(V[ix, ix])),
            "t": float(t_obs), "p_wcb": float(p), "B": B}


# ---------------------------------------------------- randomisation test ----
def randomisation_test(df, y, fe_cols, cluster, cohort_map, unit_col,
                       year_col="year", B=N_PERM_RI, seed=SEED):
    """Permute the cohort assignment vector (at the unit level) and recompute the null
    distribution of beta.

    This is the most conservative test with few switching clusters: it assumes no
    cluster asymptotics and asks only "if sanction years were randomly assigned to
    States, how often would a coefficient this large still be observed".
    """
    rng = np.random.default_rng(seed)
    units = sorted(df[unit_col].unique())
    gvals = np.array([cohort_map.get(u, np.nan) for u in units], dtype=float)

    d = df.reset_index(drop=True)
    yy = d[y].astype(float).values

    # The fixed-effects design matrix is unchanged across permutations, so its orthogonal
    # projection is built and cached in advance.
    # After residualising y and S on the FE separately, beta is a simple univariate
    # regression coefficient, and the cost per permutation falls to O(n).
    D = sm.add_constant(pd.get_dummies(d[list(fe_cols)].astype(str),
                                       drop_first=True, dtype=float),
                        has_constant="add").values.astype(float)
    DtDi = np.linalg.pinv(D.T @ D)

    def resid(v):
        return v - D @ (DtDi @ (D.T @ v))

    y_r = resid(yy)
    gser = d[unit_col].map(cohort_map)
    S_obs = ((gser.notna()) & (d[year_col] >= gser)).astype(float).values
    S_r = resid(S_obs)
    obs = float((S_r @ y_r) / (S_r @ S_r))

    yrs = d[year_col].values
    uidx = pd.factorize(d[unit_col].values)[0]
    draws = []
    for _ in range(B):
        shuffled = rng.permutation(gvals)
        gv = shuffled[uidx]
        Sp = ((~np.isnan(gv)) & (yrs >= gv)).astype(float)
        if Sp.std() == 0:
            continue
        Sp_r = resid(Sp)
        den = Sp_r @ Sp_r
        if den <= 1e-10:
            continue
        draws.append(float((Sp_r @ y_r) / den))
    draws = np.array(draws)
    p = (np.sum(np.abs(draws) >= abs(obs)) + 1) / (len(draws) + 1)
    return {"b_obs": float(obs), "p_ri": float(p), "B_effective": len(draws)}


# ----------------------------------------------------------- ordered logit --
def ordered_logit(df, y, x_cols, fe_cols=None, cluster=None, boot=0, seed=None):
    """Proportional-odds ordered logit, for the delocalisation ladder (0..3).

    With many fixed-effect dummies, inverting the analytic Hessian often fails
    (statsmodels raises HessianInversionWarning and bse is unavailable).
    Passing boot>0 switches to cluster-bootstrap standard errors: resample by cluster,
    re-estimate on each resample, and use the empirical standard deviation of the
    coefficient as the se. In finite samples this is more robust than relying on the
    analytic Hessian, and it never silently returns an unreliable se.
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    def _fit(d):
        X = d[x_cols].astype(float).reset_index(drop=True)
        if fe_cols:
            D = pd.get_dummies(d[list(fe_cols)].astype(str), drop_first=True,
                               dtype=float).reset_index(drop=True)
            X = pd.concat([X, D], axis=1)
        # drop columns that become constant after resampling; otherwise the Hessian is necessarily singular
        X = X.loc[:, X.std(axis=0) > 0]
        yy = d[y].astype(int).reset_index(drop=True)
        m = OrderedModel(yy, X, distr="logit")
        r = m.fit(method="bfgs", disp=False, maxiter=3000)
        return r, list(X.columns)

    r0, cols0 = _fit(df)
    out = {}
    for k in x_cols:
        i = cols0.index(k)
        out[k] = {"b": float(r0.params.iloc[i]), "odds_ratio":
                  float(np.exp(r0.params.iloc[i]))}

    if boot and cluster is not None:
        rng = np.random.default_rng(seed if seed is not None else SEED)
        groups = df[cluster].unique()
        draws = {k: [] for k in x_cols}
        ok = 0
        for _ in range(boot):
            pick = rng.choice(groups, size=len(groups), replace=True)
            d = pd.concat([df[df[cluster] == gk] for gk in pick],
                          ignore_index=True)
            if d[y].nunique() < 2:
                continue
            try:
                rb, colsb = _fit(d)
            except Exception:
                continue
            try:
                for k in x_cols:
                    draws[k].append(float(rb.params.iloc[colsb.index(k)]))
                ok += 1
            except ValueError:
                continue
        for k in x_cols:
            arr = np.array(draws[k])
            if len(arr) >= 20:
                se = float(arr.std(ddof=1))
                out[k]["se"] = se
                out[k]["p"] = float(2 * (1 - _norm_cdf(abs(out[k]["b"]) / se))) \
                    if se > 0 else np.nan
                out[k]["se_method"] = f"cluster bootstrap (B_ok={ok})"
                out[k]["ci95"] = (round(float(np.percentile(arr, 2.5)), 4),
                                  round(float(np.percentile(arr, 97.5)), 4))
            else:
                out[k]["se"] = np.nan
                out[k]["p"] = np.nan
                out[k]["se_method"] = f"bootstrap failed (B_ok={ok})"
    else:
        # analytic se: may be unavailable, in which case flag it explicitly rather than silently returning a wrong number
        try:
            for k in x_cols:
                i = cols0.index(k)
                se = float(r0.bse.iloc[i])
                out[k]["se"] = se
                out[k]["p"] = float(r0.pvalues.iloc[i])
                out[k]["se_method"] = "analytic Hessian"
        except Exception:
            for k in x_cols:
                out[k]["se"] = np.nan
                out[k]["p"] = np.nan
                out[k]["se_method"] = "analytic Hessian FAILED — use boot>0 instead"

    out["_llf"] = float(r0.llf)
    out["_nobs"] = int(len(df))
    return out


def _norm_cdf(z):
    from scipy.stats import norm
    return float(norm.cdf(z))


# --------------------------------------------------------- multinomial ------
def multinomial_logit(df, y, x_cols, fe_cols=None, base=None):
    """Multinomial logit over the full set of forums. Avoids the loss of SCC/ICC/AF information in a binary LPM."""
    X = df[x_cols].astype(float).reset_index(drop=True)
    if fe_cols:
        D = pd.get_dummies(df[list(fe_cols)].astype(str), drop_first=True,
                           dtype=float).reset_index(drop=True)
        X = pd.concat([X, D], axis=1)
    X = sm.add_constant(X, has_constant="add")
    cats = sorted(df[y].unique())
    if base is not None and base in cats:
        cats = [base] + [c for c in cats if c != base]
    codes = pd.Categorical(df[y], categories=cats, ordered=True).codes
    r = sm.MNLogit(codes, X.values.astype(float)).fit(
        method="bfgs", disp=False, maxiter=3000)
    cols = list(X.columns)
    # statsmodels MNLogit: params has shape (k_exog, J-1); column j corresponds to cats[j+1]
    par = np.asarray(r.params)
    bse = np.asarray(r.bse)
    pv = np.asarray(r.pvalues)
    out = {}
    for k in x_cols:
        i = cols.index(k)
        out[k] = {cats[j + 1]: {"b": float(par[i, j]),
                                "se": float(bse[i, j]),
                                "p": float(pv[i, j]),
                                "rrr": float(np.exp(par[i, j]))}
                  for j in range(par.shape[1])}
    out["_base"] = cats[0]
    out["_nobs"] = int(len(codes))
    return out


# ------------------------------------------------------------- Romano-Wolf --
def romano_wolf(tstats, boot_tstats):
    """Multiple-testing p-values with the Romano-Wolf stepdown correction.

    tstats: (K,) observed t-statistics
    boot_tstats: (B, K) bootstrap t-statistics under the null
    """
    t = np.abs(np.asarray(tstats, dtype=float))
    Bt = np.abs(np.asarray(boot_tstats, dtype=float))
    K = len(t)
    order = np.argsort(-t)
    p_adj = np.empty(K)
    running = 0.0
    for rank, idx in enumerate(order):
        remaining = order[rank:]
        maxT = Bt[:, remaining].max(axis=1)
        p = (np.sum(maxT >= t[idx]) + 1) / (Bt.shape[0] + 1)
        running = max(running, p)
        p_adj[idx] = running
    return p_adj


# -------------------------------------------------------------------- PPML --
def ppml(df, y, x_cols, fe_cols, cluster, drop_separated=True, tol=1e-9,
         maxiter=200):
    """Poisson pseudo-maximum likelihood with high-dimensional fixed effects (Santos Silva-Tenreyro).

    The design matrix is built in scipy sparse format and IRLS is implemented directly.
    With bilateral gravity data (about 140,000 pair-years x thousands of
    origin-year/dest-year dummies) a dense matrix would exhaust memory; the sparse
    implementation reduces memory to the k x k of X'WX.
    Standard errors are cluster-robust (CR1-type, without small-sample adjustment).
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    d = df.copy()
    if drop_separated:
        # Separated observations must be removed iteratively until a fixed point is reached:
        # after dropping one fixed-effect group, new all-zero groups may appear in another
        # dimension. A single pass leaves a huge number of zero observations, enough to
        # exhaust memory in the origin-year x dest-year specification.
        for _ in range(50):
            n0 = len(d)
            for f in fe_cols:
                d = d[d.groupby(f)[y].transform("sum") > 0]
            if len(d) == n0:
                break

    yv = d[y].astype(float).values
    n = len(d)

    # Sparse design: constant + continuous regressors + dummies for each fixed-effect dimension (one dropped from each)
    blocks = [sp.csr_matrix(np.ones((n, 1)))]
    names = ["const"]
    for k in x_cols:
        blocks.append(sp.csr_matrix(d[k].astype(float).values.reshape(-1, 1)))
        names.append(k)
    for f in fe_cols:
        codes, uniq = pd.factorize(d[f].astype(str).values)
        keep = codes > 0                      # drop the first category as the base
        rows = np.arange(n)[keep]
        cols_ = codes[keep] - 1
        M = sp.csr_matrix((np.ones(keep.sum()), (rows, cols_)),
                          shape=(n, max(len(uniq) - 1, 1)))
        blocks.append(M)
        names += [f"{f}_{u}" for u in uniq[1:]]
    X = sp.hstack(blocks).tocsr()
    k_par = X.shape[1]

    beta = np.zeros(k_par)
    beta[0] = np.log(max(yv.mean(), 1e-8))
    for _ in range(maxiter):
        eta = np.clip(X @ beta, -30, 30)
        mu = np.exp(eta)
        W = sp.diags(mu)
        z = eta + (yv - mu) / np.maximum(mu, 1e-10)
        A = (X.T @ W @ X).toarray()
        b_rhs = X.T @ (mu * z)
        new = np.linalg.solve(A + 1e-10 * np.eye(k_par), b_rhs)
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new

    mu = np.exp(np.clip(X @ beta, -30, 30))
    A = (X.T @ sp.diags(mu) @ X).toarray()
    Ainv = np.linalg.pinv(A)
    resid = yv - mu
    cl = pd.factorize(d[cluster].astype(str).values)[0]
    G = cl.max() + 1
    Xr = X.multiply(resid[:, None]).tocsr()
    U = np.zeros((G, k_par))
    for gg in range(G):
        idx_g = np.where(cl == gg)[0]
        if len(idx_g):
            U[gg] = np.asarray(Xr[idx_g].sum(axis=0)).ravel()
    meat = U.T @ U
    V = Ainv @ meat @ Ainv * (G / max(G - 1, 1))
    se = np.sqrt(np.maximum(np.diag(V), 0))

    from scipy.stats import norm
    out = {}
    for k in x_cols:
        i = names.index(k)
        z_ = beta[i] / se[i] if se[i] > 0 else np.nan
        out[k] = {"b": float(beta[i]), "se": float(se[i]),
                  "p": float(2 * (1 - norm.cdf(abs(z_)))) if se[i] > 0 else np.nan,
                  "pct_effect": float(100 * (np.exp(beta[i]) - 1))}
    out["_nobs"] = int(n)
    out["_nclusters"] = int(G)
    return out
