"""Helpers for the unsupervised pipeline (RQ2): grid cells, cell features, cluster validation, rule mining."""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

KM_PER_DEG = 111.32
GRID_REF_LAT = 37.5            # fixed latitude for the east-west scale, so cells keep the same index everywhere
CONUS_BOX = (-125.0, -66.0, 24.0, 50.0)
FED_OWNERS = {'USFS', 'BLM', 'NPS', 'FWS', 'BOR', 'OTHER FEDERAL', 'DOD', 'DOE', 'TVA', 'UNDEFINED FEDERAL'}
PRIVATE_OWNERS = {'PRIVATE', 'STATE OR PRIVATE'}
STATE_LOCAL_OWNERS = {'STATE', 'COUNTY', 'MUNICIPAL/LOCAL', 'OTHER NON-FEDERAL'}
TRIBAL_OWNERS = {'TRIBAL', 'BIA'}   # BIA manages tribal trust land
OWNER_GROUPS = ('federal', 'private', 'state_local', 'tribal', 'missing')


def owner_group(owner: pd.Series) -> pd.Series:
    """Collapse the land-owner codes into five groups (high-cardinality categorical -> a few shares)."""
    owner = owner.astype(str).str.upper()   # one record says 'Private'
    return pd.Series(np.select([owner.isin(FED_OWNERS), owner.isin(PRIVATE_OWNERS), owner.isin(STATE_LOCAL_OWNERS),
                                owner.isin(TRIBAL_OWNERS)],
                               ['federal', 'private', 'state_local', 'tribal'], 'missing'), index=owner.index)


def grid_index(lat: np.ndarray, lon: np.ndarray, cell_km: float) -> tuple[np.ndarray, np.ndarray]:
    """Integer (row, col) of a roughly cell_km x cell_km grid cell."""
    cy = np.floor(np.asarray(lat) * KM_PER_DEG / cell_km).astype(int)
    cx = np.floor(np.asarray(lon) * KM_PER_DEG * np.cos(np.radians(GRID_REF_LAT)) / cell_km).astype(int)
    return cy, cx


def hour_angle_columns(disc_hour: pd.Series) -> pd.DataFrame:
    """Discovery hour on the 24-hour circle (bin centre), NaN when the report has no time."""
    angle = 2 * np.pi * (disc_hour + 0.5) / 24
    return pd.DataFrame({'HOUR_SIN': np.sin(angle), 'HOUR_COS': np.cos(angle)}, index=disc_hour.index)


def cell_features(fires: pd.DataFrame, n_years: int) -> pd.DataFrame:
    """Describe every grid cell by how its fires behave. `fires` needs CY, CX and the per-fire columns used below.

    Inputs to clustering: fire rate, seasonal and diurnal circular means, weekend share, station access and land
    owner shares. Held-out columns (cause, size, reporting) are computed too, but only for interpretation.
    """
    g = fires.groupby(['CY', 'CX'], sort=True)
    owner_shares = pd.crosstab([fires['CY'], fires['CX']], fires['OWNER_GRP'], normalize='index')
    out = pd.DataFrame({
        'n_fires': g.size(),
        'lat': g['LATITUDE'].mean(), 'lon': g['LONGITUDE'].mean(),
        'log_fire_rate': np.log1p(g.size() / n_years),
        'season_sin': g['DOY_SIN'].mean(), 'season_cos': g['DOY_COS'].mean(),
        'hour_sin': g['HOUR_SIN'].mean(), 'hour_cos': g['HOUR_COS'].mean(),   # NaN-skipping means
        'weekend_share': g['WEEKEND'].mean(),
        'log_median_dist': np.log(g['D1_KM'].median()),
        'log1p_stations_25km': np.log1p(g['STATIONS_25KM'].median()),
    })
    for grp in OWNER_GROUPS:
        out[f'{grp}_share'] = owner_shares[grp] if grp in owner_shares else 0.0
    # held out: never used to build the clusters
    out['lightning_share'] = g['NATURAL'].mean()
    out['cause_missing_share'] = g['CAUSE_MISSING'].mean()
    out['hour_missing_share'] = g['HOUR_MISSING'].mean()
    out['share_large'] = g['LARGE'].mean()
    out['median_log_size'] = g['LOG_SIZE'].median()
    return out


def hopkins(X: np.ndarray, frac: float = 0.1, seed: int = 0) -> float:
    """Hopkins statistic: about 0.5 for uniform noise, close to 1 for strong cluster structure."""
    rng = np.random.default_rng(seed)
    m = max(1, int(frac * len(X)))
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    uniform = rng.uniform(X.min(axis=0), X.max(axis=0), size=(m, X.shape[1]))
    u = nn.kneighbors(uniform, n_neighbors=1)[0][:, 0]
    w = nn.kneighbors(X[rng.choice(len(X), m, replace=False)], n_neighbors=2)[0][:, 1]
    return float(u.sum() / (u.sum() + w.sum()))


def kmeans_sweep(X: np.ndarray, ks: range, seed: int, n_init: int = 20) -> pd.DataFrame:
    """Inertia, silhouette, Davies-Bouldin and Calinski-Harabasz for K-Means at each k."""
    rows = []
    for k in ks:
        fit = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)
        rows.append({'k': k, 'inertia': fit.inertia_, 'silhouette': silhouette_score(X, fit.labels_),
                     'davies_bouldin': davies_bouldin_score(X, fit.labels_),
                     'calinski_harabasz': calinski_harabasz_score(X, fit.labels_)})
    return pd.DataFrame(rows).set_index('k')


def bootstrap_stability(X: np.ndarray, k: int, n_boot: int, seed: int, n_init: int = 10) -> np.ndarray:
    """ARI between pairs of K-Means fits on independent bootstrap resamples, both scored on all of X.

    Comparing two resampled fits (not one fit against the full-data fit) measures how reproducible the partition
    is when the data change, which is what k should be chosen for.
    """
    rng = np.random.default_rng(seed)
    scores = np.empty(n_boot)
    for b in range(n_boot):
        fits = [KMeans(n_clusters=k, n_init=n_init, random_state=int(rng.integers(1 << 31)))
                .fit(X[rng.choice(len(X), len(X), replace=True)]) for _ in range(2)]
        scores[b] = adjusted_rand_score(fits[0].predict(X), fits[1].predict(X))
    return scores


def gap_statistic(X: np.ndarray, ks: range, n_ref: int, seed: int) -> pd.DataFrame:
    """Tibshirani gap statistic with uniform reference data in the bounding box of X."""
    rng = np.random.default_rng(seed)
    lo, hi = X.min(axis=0), X.max(axis=0)
    rows = []
    for k in ks:
        log_w = np.log(KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X).inertia_)
        ref = np.array([np.log(KMeans(n_clusters=k, n_init=3, random_state=seed)
                               .fit(rng.uniform(lo, hi, size=X.shape)).inertia_) for _ in range(n_ref)])
        rows.append({'k': k, 'gap': ref.mean() - log_w, 'sk': ref.std() * np.sqrt(1 + 1 / n_ref)})
    return pd.DataFrame(rows).set_index('k')


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Bias-uncorrected Cramer's V between two categorical series."""
    table = pd.crosstab(a, b).to_numpy(dtype=float)
    n = table.sum()
    expected = table.sum(1, keepdims=True) @ table.sum(0, keepdims=True) / n
    chi2 = ((table - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    """Share of the variance of `values` explained by `groups` (one-way ANOVA effect size)."""
    grand = values.mean()
    between = values.groupby(groups).agg(['size', 'mean'])
    ss_between = (between['size'] * (between['mean'] - grand) ** 2).sum()
    return float(ss_between / ((values - grand) ** 2).sum())


def wilson_bounds(k: np.ndarray, n: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for k successes out of n."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    p = k / n
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return centre - half, centre + half


def min_support_count(base_rate: float, rel_precision: float, z: float = 1.96) -> int:
    """Smallest antecedent count whose confidence CI half-width is at most rel_precision * base_rate.

    Solves z * sqrt(p0 (1 - p0) / n) <= rel_precision * p0 for n at the base rate p0, the least favourable case
    for a rule whose lift is near 1.
    """
    return int(np.ceil(z ** 2 * (1 - base_rate) / (rel_precision ** 2 * base_rate)))


def mine_rules(items: pd.DataFrame, target: pd.Series, max_len: int, min_count: int) -> pd.DataFrame:
    """All rules {attr=value, ...} -> target with up to max_len antecedent items and at least min_count matches.

    Every transaction has exactly one value per attribute, so enumerating attribute combinations with a group-by
    returns exactly the frequent itemsets Apriori would find, with exact counts.
    """
    y = pd.Series(target.to_numpy().astype(int))
    rows = []
    for length in range(1, max_len + 1):
        for attrs in combinations(items.columns, length):
            grp = y.groupby([items[a].to_numpy() for a in attrs]).agg(['size', 'sum'])
            grp = grp[grp['size'] >= min_count]
            for key, (n, k) in zip(grp.index, grp.to_numpy()):
                key = key if isinstance(key, tuple) else (key,)
                rows.append({'antecedent': tuple(f'{a}={v}' for a, v in zip(attrs, key)), 'length': length,
                             'matches': int(n), 'large': int(k)})
    rules = pd.DataFrame(rows)
    base = y.mean()
    rules['support'] = rules['matches'] / len(y)
    rules['joint_support'] = rules['large'] / len(y)
    rules['confidence'] = rules['large'] / rules['matches']
    rules['lift'] = rules['confidence'] / base
    return rules


def rule_significance(rules: pd.DataFrame, base_rate: float, n_tests: int, alpha: float = 0.05) -> pd.DataFrame:
    """Bonferroni-corrected (one-sided) Wilson lower bound of confidence and the matching lower-bound lift."""
    z = norm.ppf(1 - alpha / n_tests)
    lo, _ = wilson_bounds(rules['large'], rules['matches'], z)
    out = rules.copy()
    out['conf_lower'] = lo
    out['lift_lower'] = lo / base_rate
    return out


def add_improvement(rules: pd.DataFrame) -> pd.DataFrame:
    """Minimum improvement (Bayardo et al.): confidence minus the best confidence of any proper sub-rule."""
    conf = dict(zip(rules['antecedent'], rules['confidence']))
    best_parent = []
    for ante in rules['antecedent']:
        parents = [conf[p] for r in range(1, len(ante)) for p in combinations(ante, r) if p in conf]
        best_parent.append(max(parents) if parents else np.nan)
    out = rules.copy()
    out['best_subrule_conf'] = best_parent
    out['improvement'] = out['confidence'] - out['best_subrule_conf']
    return out
