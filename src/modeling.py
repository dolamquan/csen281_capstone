"""Evaluation and imbalance helpers for the supervised pipeline (RQ1)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def undersample(frame: pd.DataFrame, target: str, neg_per_pos: int, seed: int) -> pd.DataFrame:
    """Keep every positive and a random `neg_per_pos` negatives per positive (training rows only)."""
    pos = frame[frame[target] == 1]
    neg = frame[frame[target] == 0]
    n_neg = min(len(neg), neg_per_pos * len(pos))
    return pd.concat([pos, neg.sample(n_neg, random_state=seed)]).sample(frac=1, random_state=seed)


def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Probability cut-off that maximises F1 (chosen on validation years, then frozen)."""
    prec, rec, thr = precision_recall_curve(y_true, prob)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.argmax(f1)])


def score(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    """Ranking metrics plus F1 / precision / recall at a fixed threshold."""
    pred = (prob >= threshold).astype(int)
    return {'PR-AUC': average_precision_score(y_true, prob), 'ROC-AUC': roc_auc_score(y_true, prob),
            'F1': f1_score(y_true, pred, zero_division=0), 'precision': precision_score(y_true, pred, zero_division=0),
            'recall': recall_score(y_true, pred, zero_division=0), 'threshold': threshold}


def normalized_confusion(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> np.ndarray:
    """Confusion matrix with each row (true class) summing to 1."""
    return confusion_matrix(y_true, (prob >= threshold).astype(int), normalize='true')


def block_bootstrap_gain(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray, blocks: np.ndarray,
                         n_boot: int, seed: int) -> np.ndarray:
    """PR-AUC(b) - PR-AUC(a) over bootstrap resamples of whole blocks (e.g. state x year).

    Resampling blocks rather than single fires respects the spatial and temporal correlation between
    fires in the same place and season, so the interval is not over-confident.
    """
    rng = np.random.default_rng(seed)
    codes, uniques = pd.factorize(blocks)
    members = [np.flatnonzero(codes == k) for k in range(len(uniques))]
    gains = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        gains[b] = average_precision_score(y_true[idx], prob_b[idx]) - average_precision_score(y_true[idx], prob_a[idx])
    return gains


def odds_ratio_per_doubling(frame: pd.DataFrame, target: str, dist_col: str, cats: list, nums: list,
                            n: int, seed: int) -> dict:
    """Unpenalised logistic regression; odds ratio of `dist_col` (log2 km) with a 95% Wald interval."""
    s = frame.sample(min(n, len(frame)), random_state=seed)
    parts = [s[[dist_col]] - s[dist_col].mean(), s[nums] - s[nums].mean()]
    if cats:
        parts.append(pd.get_dummies(s[cats], drop_first=True))
    X = pd.concat(parts, axis=1).to_numpy(dtype=float)
    y = s[target].to_numpy()
    model = LogisticRegression(penalty=None, max_iter=3000).fit(X, y)
    p = model.predict_proba(X)[:, 1]
    Xd = np.column_stack([np.ones(len(X)), X])
    cov = np.linalg.pinv((Xd * (p * (1 - p))[:, None]).T @ Xd)
    b, se = model.coef_[0, 0], np.sqrt(cov[1, 1])
    return {'odds_ratio': np.exp(b), 'ci_low': np.exp(b - 1.96 * se), 'ci_high': np.exp(b + 1.96 * se), 'n': len(y)}
