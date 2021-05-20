import copy
import math

import numpy as np
from sklearn import metrics
import Levenshtein

from misc import matching
from misc import adapted_utils
from misc.constants import UU_FILTER, MU_FILTER, UNC_FILTER, UNC_MAPPING
from misc.utils import BColors
from misc import utils


##
# evaluation utils
def calc_scores(
    event_matcher,
    matcher,
    matching_kwargs,
    labels,
    multiclass_strategy,
    binary_strategy,
    meta=(),
    unittest=None,
):
    """Calculates multiclass and binary scores for a given matcher"""
    # multiclass evaluation scores
    result_multiclass = multiclass_eval(
        event_matcher=event_matcher,
        matcher=matcher,
        matching_kwargs=matching_kwargs,
        labels=labels,
        strategy=multiclass_strategy,
        unittest=unittest,
    )

    result_binary = []
    if len(binary_strategy):
        result_binary = binary_eval(
            event_matcher=event_matcher,
            matcher=matcher,
            matching_kwargs=matching_kwargs,
            labels=labels,
            strategy=binary_strategy,
            unittest=unittest,
        )

    result = [*result_multiclass, *result_binary]

    # Add meta data
    result = [utils.merge_dicts([*meta, _result]) for _result in result]

    return result


def calc_nld(event_matcher, method):
    """Interface for Normalized Levenshtein distance"""
    if method in ["sample"]:
        # sample level nld
        gt = event_matcher.data_gt["evt"].values
        pr = event_matcher.data_pr["evt"].values
    else:
        gt = event_matcher.evt_gt["evt"].values
        pr = event_matcher.evt_pr["evt"].values

    return _calc_nld(gt, pr)


def _calc_nld(gt, pr):
    """Calculates Normalized Levenshtein distance."""
    gt_chr = utils.seq2chr(gt)
    pr_chr = utils.seq2chr(pr)
    _l = len(gt)
    _check = (_l == len(gt_chr), len(pr) == len(pr_chr))
    assert all(_check)
    nld = Levenshtein.distance(gt_chr, pr_chr) / _l
    # nld = np.clip(nld, 0.0, 1.0)
    return nld


def calc_multiclass_metrics(gt, pr, labels=None, zero_division=None):
    """Calculates multiclass metrics"""
    _metrics = ("accuracy", "accuracy_balanced", "kappa", "mcc")

    if len(gt):
        c = metrics.confusion_matrix(gt, pr, labels=labels)
        scores = (
            metrics.accuracy_score(gt, pr),
            adapted_utils.balanced_accuracy(c),
            adapted_utils.cohen_kappa_score(c, zero_division=zero_division),
            adapted_utils.matthews_corrcoef(c, zero_division=zero_division),
        )
    else:
        scores = [None] * len(_metrics)

    result = {m: s for m, s in zip(_metrics, scores)}

    return result


def calc_binary_metrics(gt, pr, zero_division=None):
    """Calculates binary metrics and explicitly handles cases where metrics are undefined.
    Metrics:
        Accuracy, Balanced accuracy
        Precision, Sensitivity (Recall), Specificity, F1-score
        IoU (Jaccard Index)
        Cohen's Kappa
        MCC
        ROC AUC
        Normalized Levenshtein distance
    """
    c = metrics.confusion_matrix(gt, pr, labels=[0, 1])
    tn, fp, fn, tp = c.ravel()

    # accuracy
    accuracy = (tp + tn) / sum((tn, fp, fn, tp))
    accuracy_balanced = adapted_utils.balanced_accuracy(c)

    # precision, sensitivity, specificity, f1_score
    _denom = tp + fp
    precision = tp / _denom if _denom > 0 else zero_division

    _denom = tp + fn
    sensitivity = tp / _denom if _denom > 0 else zero_division

    _denom = tn + fp
    specificity = tn / _denom if _denom > 0 else zero_division

    _denom = 2 * tp + fp + fn
    f1_score = 2 * tp / _denom if _denom > 0 else zero_division

    # IoU
    _denom = tp + fp + fn
    iou = tp / _denom if _denom > 0 else zero_division

    # Kappa
    kappa = adapted_utils.cohen_kappa_score(c, zero_division=zero_division)

    # Matthews Correlation Coefficient
    _denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / math.sqrt(_denom) if _denom > 0 else zero_division

    # ROC AUC
    _auc_check = len(np.unique(gt)) == 2
    auc = metrics.roc_auc_score(gt, pr) if _auc_check else None

    # concat result
    result = {
        "accuracy": accuracy,
        "accuracy_balanced": accuracy_balanced,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1_score,
        "iou": iou,
        "kappa": kappa,
        "mcc": mcc,
        "auc": auc,
    }

    return result


def multiclass_eval(
    event_matcher, matcher, matching_kwargs, labels, strategy=None, unittest=None
):
    """Multiclass evaluation"""
    strategy = ["all"] if strategy is None else strategy

    # calc additional scores
    result_aux = {"nld": calc_nld(event_matcher, matcher)}

    # run matching
    output, events = event_matcher.run_matching(matcher, **matching_kwargs)

    # convert unmatched to a dedicated class
    reserved = 255
    _mask = output == -1
    output[_mask] = reserved
    _labels = [reserved, *labels] if not (reserved in labels) else labels
    assert all(np.in1d(np.unique(output), _labels))

    result_accum = []
    for _strategy in strategy:
        _meta = f"multiclass/{_strategy}" if not (_strategy == "all") else "multiclass"
        meta = {"eval": _meta, "event": "all"}

        # handle undefined events
        mask_remove = np.zeros(len(output), dtype=np.bool)
        if _strategy in ["ignore_unmatched_undef", "ignore_undef"]:
            _mask = get_filter_mask(output, UU_FILTER)
            mask_remove = np.logical_or(mask_remove, _mask)
        if _strategy in ["ignore_matched_undef", "ignore_undef"]:
            _mask = get_filter_mask(output, MU_FILTER)
            mask_remove = np.logical_or(mask_remove, _mask)

        _output = copy.deepcopy(output)
        _output = _output[~mask_remove]
        _output = _output.T

        # unittest
        if unittest is not None:
            bcolors = BColors()
            _unittest = unittest.get(_meta, None)
            uresult = bcolors("Test missing", "WARNING")
            if _unittest is not None:
                _unittest = np.array((_unittest["gt"], _unittest["pr"]))
                _unittest[_unittest == -1] = 255
                if _unittest.shape == _output.shape:
                    _uresult = _unittest == _output

                    if np.all(_uresult):
                        uresult = bcolors("Pass", "OKGREEN")
                    else:

                        uresult = bcolors("Fail", "FAIL")
                        _fail_idx = np.where(~_uresult.all(axis=0))[0]
                        print(_fail_idx)
                        print(_unittest[:, _fail_idx])
                        print(_output[:, _fail_idx])
                else:
                    uresult = bcolors("Fail (dimension mismatch)", "FAIL")
            print(f"Matcher: {matcher}, eval: {_meta}. {uresult}")

        # calculate and store result
        result = calc_multiclass_metrics(*_output, labels=_labels, zero_division=None)
        result_accum.append(utils.merge_dicts([meta, result, result_aux]))

    return result_accum


def binary_eval(
    event_matcher, matcher, matching_kwargs, labels, strategy, unittest=None
):
    """Binary evaluation"""
    data_gt = event_matcher.data_gt
    data_pr = event_matcher.data_pr

    data_gt_bin = data_gt.copy()
    data_pr_bin = data_pr.copy()
    result_accum = []
    for _class in labels:
        data_gt_bin["evt"] = data_gt["evt"] == _class
        data_pr_bin["evt"] = data_pr["evt"] == _class
        event_matcher_bin = matching.EventMatcher(gt=data_gt_bin, pr=data_pr_bin)
        # calc additional scores
        result_aux = {"nld": calc_nld(event_matcher_bin, matcher)}

        output_bin, events_bin = event_matcher_bin.run_matching(
            matcher, **matching_kwargs
        )
        eval_result_bin = _binary_eval(output_bin, strategy, matcher)

        for result, _binary_strategy in eval_result_bin:
            meta = {"eval": f"binary/{_binary_strategy}", "event": _class}
            result_accum.append(utils.merge_dicts([meta, result, result_aux]))
    return result_accum


def _binary_eval(output, binary_strategy, matcher=None):
    # find unmatched negative class
    mask_unc = get_filter_mask(output, UNC_FILTER)
    _mask_unc = mask_unc.reshape((-1, 1))

    # convert unmatched positives to errors
    mask_upc = np.logical_and(output == -1, ~_mask_unc)
    output[mask_upc] = 0

    result_accum = []
    mask_unc_idx = np.logical_and(output == -1, _mask_unc)
    for _binary_strategy in binary_strategy:
        _output = copy.deepcopy(output)

        unc_value = UNC_MAPPING.get(_binary_strategy, None)
        if unc_value is not None:
            _output[mask_unc_idx] = unc_value
        else:
            _output = _output[~mask_unc]
        if len(_output):
            result_bin = calc_binary_metrics(*_output.T, zero_division=None)
            if not (matcher in ["sample"]):
                # remove event level iou to avoid confusion
                _ = result_bin.pop("iou")
        else:
            result_bin = {}
        result_accum.append((result_bin, _binary_strategy))

    return result_accum


##
# misc
def get_filter_mask(output, filt):
    mask = [np.all((output - _f) == 0, axis=1) for _f in filt]
    mask = np.any(mask, axis=0)
    return mask
