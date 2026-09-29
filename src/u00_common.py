"""
u00_common.py — shared layer for the upgrade modules (u01–u09)

Responsibilities
----------------
1. **Data loading and variable assembly**: reuses the project pipeline
   (config / io_load / legal_eligibility / r26), so that S, the outcome variables
   and the Convention-status eligibility coding come from the same source as every
   other estimate in the paper.
2. **Fast fixed-effects engine**: the projection is computed only once for a given
   (sample, fixed effects, weights) combination; thereafter the coefficient and CR1
   variance for any treatment variable are O(n). The joint permutation in the
   specification curve and the random-assignment simulation for the interpretive
   range both rely on this.
3. **Inference**: CR1, CV3 (cluster jackknife), restricted WCB (Webb six-point
   weights), three permutation pools (full / restricted / timing-only), and the
   Imbens–Manski confidence set for an interval.

All coefficients are reported in **percentage points** (binary outcome × 100).

Note (correction relative to r27)
---------------------------------
The "restricted pool" in r27.permutation_diagnostics placed only countries spanning
more than one year into cohort_map; _compat.permutation_test sets S=0 for every
country not in cohort_map, so excluded **treated** countries were also treated as
untreated in the **observed statistic**. The `permutation_pvalue` in this module
always keeps all treated countries in the pool; the restriction applies only to
untreated countries, and the observed statistic matches the core table.
"""
from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

PP = 100.0
UNIT, TIME = "respondent_iso3", "year"
OUTCOMES = {"natcourt_exposed": "Non-Convention (primary)",
            "is_uncitral": "UNCITRAL (secondary)"}
AVAIL_CODINGS = {
    "none": "(0) no availability control",
    "strict": "(1) strict: denunciation ends availability",
    "three": "(2) three-valued: contested as own category",
    "lax": "(3) permissive: Art. 72 consent survives",
}
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
DEFAULT_REF = ROOT / "data" / "reference" / "icsid_convention_status_icsid3.csv"
TREAT_FILE = ROOT / "data" / "derived" / "gsdb_treatment_cohorts.csv"


# ==========================================================================
# 1. Data loading
# ==========================================================================
def load_base(ref_path: str | Path | None = None) -> pd.DataFrame:
    """Case-level analysis frame: outcomes + three-valued Convention-status eligibility
    + claim groups for parallel claims. Does not include the treatment variable."""
    warnings.filterwarnings("ignore", category=UserWarning)
    from io_load import load_cases
    from legal_eligibility import add_enforcement_regime
    from r26_eligibility_audit import classify, load_ref

    d = load_cases().reset_index(drop=True)
    d = add_enforcement_regime(d)
    ref = load_ref(str(ref_path or DEFAULT_REF))
    d = classify(d, ref)
    d["year"] = d["year"].astype(int)
    for c in ("is_uncitral", "natcourt_exposed"):
        d[c] = d[c].astype(float)
    # Convention-status eligibility codings
    d["avail_none"] = "all"
    d["avail_strict"] = np.where(d.convention_status == "available", "open", "closed")
    d["avail_three"] = d.convention_status.astype(str)
    d["avail_lax"] = np.where(d.convention_status.isin(["available", "contested"]),
                              "open", "closed")
    # Crimea claim group (parallel claims; matches the 10 cases in claim_groups_manual.csv:
    # respondent State RUS, claimant's home State UKR, 2015–2019, UNCITRAL rules)
    crimea = ((d[UNIT] == "RUS") & (d.claimant_iso3 == "UKR")
              & d.year.between(2015, 2019) & (d.rules == "UNCITRAL"))
    d["claim_group"] = np.where(crimea, "CRIMEA", d.case_id.astype(str))
    d["claim_group_size"] = d.groupby("claim_group").case_id.transform("size")
    d["w_group"] = 1.0 / d["claim_group_size"]
    return d


def load_cohorts(path: str | Path = TREAT_FILE) -> pd.DataFrame:
    """Cohort table produced by u02: iso3 × onset year for each treatment definition
    (NaN = untreated within the window)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found: run u02_gsdb_treatment.py first")
    return pd.read_csv(path).set_index("iso3")


def attach(d: pd.DataFrame, g: pd.Series | dict, name: str = "S") -> pd.DataFrame:
    x = d.copy()
    gm = x[UNIT].map(dict(g) if not isinstance(g, dict) else g)
    x["g"] = gm
    x[name] = ((gm.notna()) & (x[TIME] >= gm)).astype(float)
    x["et"] = x[TIME] - gm
    return x


def identification(x: pd.DataFrame, treat: str = "S") -> dict:
    s = x.groupby(UNIT)[treat]
    sw = sorted(s.nunique().pipe(lambda v: v[v > 1]).index)
    return {"n": len(x), "clusters_treated": int(s.max().sum()),
            "switchers": len(sw), "switcher_list": " ".join(sw),
            "treated_cases": int(x[treat].sum()),
            "identifying_cases": int(x[UNIT].isin(sw).sum())}


# ==========================================================================
# 2. Fast fixed-effects engine
# ==========================================================================
def _dummies(frame: pd.DataFrame, absorb: list[str]) -> np.ndarray:
    blocks = [np.ones((len(frame), 1))]
    for a in absorb:
        codes = pd.Categorical(frame[a].astype(str)).codes
        k = codes.max() + 1
        if k > 1:
            M = np.zeros((len(frame), k - 1))
            m = codes > 0
            M[np.where(m)[0], codes[m] - 1] = 1.0
            blocks.append(M)
    return np.hstack(blocks)


@dataclass
class Projector:
    """Caches the residualising matrix for a given sample, set of fixed effects and weights.

    Weighted regression: OLS on variables scaled by sqrt(w) (the standard WLS
    transformation); CR1 scores are aggregated as sqrt(w)·x̃·ẽ, equivalent to an
    aweight in Stata/fixest.
    """
    frame: pd.DataFrame
    absorb: list[str]
    weights: np.ndarray | None = None

    def __post_init__(self):
        n = len(self.frame)
        self.sw = np.ones(n) if self.weights is None else np.sqrt(self.weights)
        D = _dummies(self.frame, self.absorb) * self.sw[:, None]
        Q, R = np.linalg.qr(D, mode="reduced")
        keep = np.abs(np.diag(R)) > 1e-9 * np.abs(R).max()
        self.Q = Q[:, keep]
        self.k_fe = int(keep.sum())
        self.cl = pd.Categorical(self.frame[UNIT].astype(str)).codes
        self.G = int(self.cl.max() + 1)
        self.n = n

    def resid(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, float) * (self.sw if v.ndim == 1 else self.sw[:, None])
        return v - self.Q @ (self.Q.T @ v)

    def fit(self, y: np.ndarray, s: np.ndarray, yr: np.ndarray | None = None) -> dict:
        """Single-regressor coefficient + CR1. yr may be passed as a pre-residualised y."""
        yt = self.resid(y) if yr is None else yr
        st = self.resid(s)
        den = st @ st
        if den < 1e-10:
            return {"coef": np.nan, "se": np.nan, "p": np.nan, "t": np.nan}
        b = (st @ yt) / den
        e = yt - b * st
        sc = np.bincount(self.cl, weights=st * e, minlength=self.G)
        k = self.k_fe + 1
        adj = (self.G / (self.G - 1)) * ((self.n - 1) / max(self.n - k, 1))
        se = math.sqrt(adj * (sc @ sc)) / den
        t = b / se if se > 0 else np.nan
        p = 2 * stats.t.sf(abs(t), df=self.G - 1) if np.isfinite(t) else np.nan
        return {"coef": float(b), "se": float(se), "t": float(t), "p": float(p)}


def design_frame(d: pd.DataFrame, coding: str = "none", drop_unit: str | None = None,
                 sample: str = "all") -> tuple[pd.DataFrame, list[str]]:
    x = d if drop_unit is None else d[d[UNIT] != drop_unit]
    if sample == "excl_VEN":
        x = x[x[UNIT] != "VEN"]
    absorb = [UNIT, TIME] + ([] if coding == "none" else [f"avail_{coding}"])
    return x.reset_index(drop=True), absorb


def estimate(d: pd.DataFrame, y: str, treat: str = "S", coding: str = "none",
             weights: str | None = None, sample: str = "all") -> dict:
    x, absorb = design_frame(d, coding, sample=sample)
    w = None if weights is None else x[weights].to_numpy(float)
    P = Projector(x, absorb, w)
    r = P.fit(x[y].to_numpy(float) * PP, x[treat].to_numpy(float))
    r.update(identification(x, treat))
    return r


# ==========================================================================
# 3. Inference
# ==========================================================================
def cv3(d: pd.DataFrame, y: str, treat: str = "S", coding: str = "none",
        sample: str = "all") -> dict:
    """Cluster jackknife variance (CV3 of MacKinnon, Nielsen & Webb 2023).

    V = (G-1)/G · Σ_g (β_(-g) - β̂)²; t distribution with G-1 degrees of freedom.
    Also returns β_(-g) for each cluster, for use in influence diagnostics.
    """
    x, absorb = design_frame(d, coding, sample=sample)
    full = Projector(x, absorb).fit(x[y].to_numpy(float) * PP, x[treat].to_numpy(float))
    units = sorted(x[UNIT].unique())
    betas = {}
    for u in units:
        xu = x[x[UNIT] != u].reset_index(drop=True)
        if xu[treat].nunique() < 2:
            continue
        betas[u] = Projector(xu, absorb).fit(xu[y].to_numpy(float) * PP,
                                             xu[treat].to_numpy(float))["coef"]
    bj = np.array([v for v in betas.values() if np.isfinite(v)])
    G = len(bj)
    se = math.sqrt((G - 1) / G * np.sum((bj - full["coef"]) ** 2))
    t = full["coef"] / se if se > 0 else np.nan
    return {"coef": full["coef"], "se_cv3": se,
            "p_cv3": float(2 * stats.t.sf(abs(t), df=G - 1)),
            "loo": betas}


def wcb_webb(d: pd.DataFrame, y: str, treat: str = "S", coding: str = "none",
             B: int = 1999, seed: int = 20260917, sample: str = "all") -> float:
    """Restricted WCB-t with Webb six-point weights (imposes H0: β=0)."""
    x, absorb = design_frame(d, coding, sample=sample)
    P = Projector(x, absorb)
    yt = P.resid(x[y].to_numpy(float) * PP)
    st = P.resid(x[treat].to_numpy(float))
    obs = P.fit(x[y].to_numpy(float) * PP, x[treat].to_numpy(float), yr=yt)
    if not np.isfinite(obs["t"]):
        return np.nan
    # Under H0 the restricted residuals are yt itself (the sole regressor is constrained to 0)
    rng = np.random.default_rng(seed)
    tstar = np.empty(B)
    den = st @ st
    k = P.k_fe + 1
    adj = (P.G / (P.G - 1)) * ((P.n - 1) / max(P.n - k, 1))
    for j in range(B):
        w = rng.choice(WEBB, size=P.G)[P.cl]
        ys = yt * w
        ys = ys - P.Q @ (P.Q.T @ ys)          # re-residualise to stay orthogonal to the FE
        b = (st @ ys) / den
        e = ys - b * st
        sc = np.bincount(P.cl, weights=st * e, minlength=P.G)
        se = math.sqrt(adj * (sc @ sc)) / den
        tstar[j] = b / se
    return float((np.sum(np.abs(tstar) >= abs(obs["t"])) + 1) / (B + 1))


def _S_from_cohort(x: pd.DataFrame, cmap: dict) -> np.ndarray:
    g = x[UNIT].map(cmap).to_numpy(float)
    return ((~np.isnan(g)) & (x[TIME].to_numpy(float) >= g)).astype(float)


def permutation_pvalue(d: pd.DataFrame, y: str, cohort: dict, pool: str = "timing",
                       coding: str = "none", B: int = 999, seed: int = 20260917,
                       sample: str = "all") -> dict:
    """Country-level permutation test.

    pool = "all"        permute the cohort vector (including NaN) across all respondent States
           "restricted" all treated States + untreated States whose cases span at least two years
           "timing"     permute onset years among treated States only (a direct test of
                        "not a response to timing")
    """
    x, absorb = design_frame(d, coding, sample=sample)
    P = Projector(x, absorb)
    yv = x[y].to_numpy(float) * PP
    yt = P.resid(yv)
    units = sorted(x[UNIT].unique())
    treated = [u for u in units if np.isfinite(cohort.get(u, np.nan))]
    if pool == "timing":
        members = treated
    elif pool == "all":
        members = units
    elif pool == "restricted":
        span = x.groupby(UNIT)[TIME].nunique()
        members = sorted(set(treated) | set(span[span >= 2].index))
    else:
        raise ValueError(pool)
    base = {u: cohort.get(u, np.nan) for u in units}
    vals = np.array([base[u] for u in members], float)
    b_obs = P.fit(yv, _S_from_cohort(x, base), yr=yt)["coef"]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        perm = dict(base)
        perm.update(dict(zip(members, rng.permutation(vals))))
        bs.append(P.fit(yv, _S_from_cohort(x, perm), yr=yt)["coef"])
    bs = np.array([b for b in bs if np.isfinite(b)])
    p = (np.sum(np.abs(bs) >= abs(b_obs)) + 1) / (len(bs) + 1)
    n_distinct = _n_distinct_perms(vals)
    return {"coef": b_obs, "p": float(p), "pool_size": len(members),
            "B_eff": len(bs), "n_distinct_assignments": n_distinct,
            "min_attainable_p": 1.0 / min(n_distinct, len(bs) + 1)}


def _n_distinct_perms(vals: np.ndarray) -> int:
    lab = pd.Series(vals).fillna(-1).value_counts().to_numpy()
    n = math.factorial(len(vals))
    for c in lab:
        n //= math.factorial(int(c))
    return int(min(n, 10**12))


def imbens_manski(b_lo: float, se_lo: float, b_hi: float, se_hi: float,
                  level: float = 0.95) -> tuple[float, float, float]:
    """Imbens & Manski (2004) confidence interval for a partially identified parameter."""
    from scipy.optimize import brentq
    if b_lo > b_hi:
        b_lo, se_lo, b_hi, se_hi = b_hi, se_hi, b_lo, se_lo
    delta, s = b_hi - b_lo, max(se_lo, se_hi)
    c = brentq(lambda c: stats.norm.cdf(c + delta / s) - stats.norm.cdf(-c) - level,
               0.0, 10.0)
    return b_lo - c * se_lo, b_hi + c * se_hi, c


# ==========================================================================
# 4. Output
# ==========================================================================
def outdir(p: str | Path) -> Path:
    q = Path(p)
    q.mkdir(parents=True, exist_ok=True)
    return q


def save(df: pd.DataFrame, out: Path, name: str, show: bool = True) -> None:
    df.to_csv(out / name, index=False)
    if show:
        with pd.option_context("display.width", 200, "display.max_columns", 40):
            print(df.round(3).to_string(index=False))
    print(f"  -> {out / name}")


def append_canonical(rows: list[tuple[str, float]], out: Path,
                     name: str = "canonical_numbers_upgrade.csv") -> None:
    f = out / name
    new = pd.DataFrame(rows, columns=["quantity", "value"])
    if f.exists():
        old = pd.read_csv(f)
        old = old[~old.quantity.isin(new.quantity)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(f, index=False)
