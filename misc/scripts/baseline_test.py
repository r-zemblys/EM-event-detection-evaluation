import os
import random

import numpy as np
import pandas as pd
from scipy import stats

try:
    from __init__ import ROOT
except ImportError:
    ROOT = None

from misc import utils
from misc import eval_utils
from misc.constants import metric_map


##
def cat(arr):
    return "".join(map(str, arr))


def make_gt(n_samples, n_pairs, prob=0.9):
    _n_positive = np.round(prob * n_samples)
    _n_negative = np.round((1 - prob) * n_samples)
    _positive = np.ones(int(_n_positive), dtype=int)
    _negative = np.zeros(int(_n_negative), dtype=int)
    sig = np.concatenate((_positive, _negative) * n_pairs)
    return sig


##
# config
random.seed(0x062217)
np.random.seed(0x062217)
n_samples_base = 100
n_repeats = 100
n = n_samples_base * n_repeats

# data
gt = make_gt(n_samples_base, n_repeats, prob=0.9)

# tests
gt_random = gt.copy()
np.random.shuffle(gt_random)
all_but_one_majority = [0, *np.ones(n - 1, dtype=int)]
all_but_one_minority = [*np.zeros(n - 1, dtype=int), 1]

# random event shuffle
events = utils.aggr_events(gt)
random.shuffle(events)
dur, evt = zip(*[(e - s, evt) for s, e, evt in events])
random_event_shuffle = utils.evt_unfold(evt, dur)

predictions = {
    "All majority": np.ones(n, dtype=int),
    "All but one\nmajority": all_but_one_majority,
    "All minority": np.zeros(n, dtype=int),
    "All but one\nminority": all_but_one_minority,
    "Random": np.random.choice([0, 1], n),
    "Shuffle": gt_random,
    "Opposite": 1 - gt,
    "Random\nevent shuffle": random_event_shuffle,
}

##
# calc measures

score_accum = []
for k, pr in predictions.items():
    # # add probabilities to predictions
    # _pr = np.array(pr, dtype=float)
    # mask = _pr == 1
    # _pr[mask] = np.random.uniform(low=0.5, high=1.0, size=np.sum(mask))
    # _pr[~mask] = np.random.uniform(low=0.0, high=0.5, size=np.sum(~mask))
    # auc = metrics.roc_auc_score(gt, _pr)

    scores = eval_utils.calc_binary_metrics(gt, pr, zero_division=None)

    pearsonr, _ = stats.pearsonr(gt, pr)
    spearmanr, _ = stats.spearmanr(gt, pr)
    kendalltau, _ = stats.kendalltau(gt, pr)
    nld = eval_utils._calc_nld(gt, pr)
    scores_aux = {
        "pearsonr": pearsonr,
        "spearmanr": spearmanr,
        "kendalltau": kendalltau,
        "_nld": nld,
    }

    scores = utils.merge_dicts([scores, scores_aux])
    score_accum.append((k, scores))

scores = pd.DataFrame(dict(score_accum)).reset_index()
# sort
mapper = {name: order for order, name in enumerate(metric_map)}
scores.sort_values(by="index", key=lambda x: x.map(mapper), inplace=True)
scores.replace({"index": metric_map}, inplace=True)

# remove nan
scores.fillna("-", inplace=True)

# save
scores.to_csv(
    os.path.join(ROOT, "results", "score_baselines.csv"),
    float_format="%.2f",
    index=False,
)
