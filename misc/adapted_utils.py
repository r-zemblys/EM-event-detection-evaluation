import numpy as np


def balanced_accuracy(c):
    """Calculates balanced accuracy. Adapted from sklearn implementation to ommit warning
    https://github.com/scikit-learn/scikit-learn/blob/fd237278e/sklearn/metrics/_classification.py#L1745
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.diag(c) / c.sum(axis=1)
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)

    return score


def cohen_kappa_score(c, zero_division=None):
    """Calculates Cohen's kappa. Adapted from sklearn implementation to handle division by 0
    https://github.com/scikit-learn/scikit-learn/blob/fd237278e895b42abe8d8d09105cbb82dc2cbba7/sklearn/metrics/_classification.py#L54
    """
    n_classes, _ = c.shape

    sum0 = np.sum(c, axis=0)
    sum1 = np.sum(c, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    w_mat = np.ones([n_classes, n_classes], dtype=np.int)
    w_mat.flat[:: n_classes + 1] = 0

    _denom = np.sum(w_mat * expected)
    k = zero_division if _denom == 0 else 1 - np.sum(w_mat * c) / _denom

    return k


def matthews_corrcoef(c, zero_division=None):
    """Compute the Matthews correlation coefficient (MCC). Adapted from sklearn implementation to handle division by 0
    https://github.com/scikit-learn/scikit-learn/blob/fd237278e/sklearn/metrics/_classification.py#L764
    """
    t_sum = c.sum(axis=1, dtype=np.float64)
    p_sum = c.sum(axis=0, dtype=np.float64)
    n_correct = np.trace(c, dtype=np.float64)
    n_samples = p_sum.sum()
    cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
    cov_ypyp = n_samples ** 2 - np.dot(p_sum, p_sum)
    cov_ytyt = n_samples ** 2 - np.dot(t_sum, t_sum)

    _denom = cov_ytyt * cov_ypyp
    mcc = zero_division if _denom == 0 else cov_ytyp / np.sqrt(_denom)

    return mcc
