"""
Shared evaluation utilities — discrimination, calibration, clinical utility, and the bootstrap
machinery used to put confidence intervals around every headline number.

The AUC implementation here is rank-based rather than a call into scikit-learn. That matters
only because the paired bootstrap runs a thousand resamples across three models on a hundred
thousand rows, and the rank-based form is fast enough to make that practical on a laptop.
"""

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def fast_auc(y_true, scores):
    """Rank-based AUC, equivalent to roc_auc_score but cheap enough to bootstrap."""
    ranks = rankdata(scores)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def summarise(y_true, scores):
    """The three numbers reported for every model, in one place so they cannot drift apart."""
    return {
        "auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "brier": brier_score_loss(y_true, scores),
    }


def paired_bootstrap(y_true, score_a, score_b=None, n_resamples=1000, seed=0):
    """Bootstrap the AUC of one model, or the paired difference between two.

    Both models are scored on the same resampled indices, which is what makes the difference
    paired — the resampling noise largely cancels, and the interval on the difference comes out
    far tighter than the intervals on the two models separately would suggest.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    draws = rng.integers(0, n, (n_resamples, n))

    values = []
    for index in draws:
        sample_y = y_true[index]
        if sample_y.sum() < 10:
            continue
        value = fast_auc(sample_y, score_a[index])
        if score_b is not None:
            value -= fast_auc(sample_y, score_b[index])
        values.append(value)

    values = np.array(values)
    lower, upper = np.percentile(values, [2.5, 97.5])
    # Two-sided bootstrap p-value for a difference, obtained from the proportion of resamples
    # falling on the wrong side of zero.
    p_value = 2 * min((values <= 0).mean(), (values >= 0).mean()) if score_b is not None else np.nan
    return {"mean": values.mean(), "lo": lower, "hi": upper, "p": p_value}


def threshold_table(y_true, scores, thresholds=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30)):
    """Sensitivity, specificity, and predictive values across candidate operating points.

    Worth stating plainly: this must be given out-of-fold predictions. Handing it predictions
    from a model that has already seen the rows produces a table that looks far better than the
    model is, which is exactly the failure mode documented in docs/PRIOR_WORK_AUDIT.md.
    """
    rows = []
    for threshold in thresholds:
        flagged = scores >= threshold
        tp = int((flagged & (y_true == 1)).sum())
        fp = int((flagged & (y_true == 0)).sum())
        fn = int((~flagged & (y_true == 1)).sum())
        tn = int((~flagged & (y_true == 0)).sum())
        rows.append({
            "threshold": threshold,
            "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "ppv": tp / (tp + fp) if tp + fp else 0.0,
            "npv": tn / (tn + fn) if tn + fn else 0.0,
            "flagged_per_1000": flagged.mean() * 1000,
        })
    return rows


def net_benefit(y_true, scores, threshold):
    """Net benefit at one threshold, following Vickers and Elkin.

    The weight term pt/(1-pt) is the exchange rate between a false positive and a true positive
    that the threshold itself implies: a clinician willing to act at a 10% predicted risk is
    saying nine unnecessary interventions are worth one prevented event.
    """
    n = len(y_true)
    flagged = scores >= threshold
    tp = (flagged & (y_true == 1)).sum()
    fp = (flagged & (y_true == 0)).sum()
    return tp / n - (fp / n) * (threshold / (1 - threshold))


def decision_curve(y_true, scores, thresholds=None):
    """Net benefit for the model, treat-all, and treat-none across a threshold range."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.30, 60)

    n = len(y_true)
    prevalence_count = y_true.sum()
    model, treat_all = [], []
    for threshold in thresholds:
        model.append(net_benefit(y_true, scores, threshold))
        weight = threshold / (1 - threshold)
        treat_all.append(prevalence_count / n - ((n - prevalence_count) / n) * weight)

    return {
        "thresholds": np.asarray(thresholds),
        "model": np.asarray(model),
        "treat_all": np.asarray(treat_all),
        "treat_none": np.zeros(len(thresholds)),
    }


def capacity_table(y_true, scores, fractions=(0.05, 0.10, 0.15, 0.20, 0.30, 0.50)):
    """What a fixed intervention budget buys, which is the question a hospital actually asks.

    Ranking by predicted risk and taking the top k is a different framing from thresholding on
    a probability, and it is the more useful one when capacity — not risk tolerance — is the
    binding constraint.
    """
    order = np.argsort(-scores)
    base_rate = y_true.mean()
    rows = []
    for fraction in fractions:
        k = int(len(y_true) * fraction)
        captured = int(y_true[order[:k]].sum())
        rows.append({
            "target_fraction": fraction,
            "n_flagged": k,
            "captured": captured,
            "recall": captured / y_true.sum(),
            "precision": captured / k,
            "lift": (captured / k) / base_rate,
        })
    return rows


def calibration_in_the_large(y_true, scores):
    """Observed events divided by predicted events. One means calibrated on average."""
    return y_true.mean() / scores.mean()
