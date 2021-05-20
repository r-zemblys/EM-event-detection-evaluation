import os
from distutils.dir_util import mkpath

import numpy as np
import pandas as pd

import itertools
import matplotlib.pyplot as plt
from sklearn import metrics

try:
    from __init__ import ROOT
except ImportError:
    ROOT = None
from misc import matching
from misc import utils

plt.ion()
plt.rc("axes.spines", top=False, right=False)

error_mapping = {
    "D": 0,
    "I": 0,
    "u": 1,
    "o": 1,
    "F": 2,
    "M": 2,
    "DN": 3,
    "IN": 3,
    "uN": 3,
    "oN": 3,
    "FN": 3,
    "MN": 3,
}

# error_mapping = {
#     'D': 0,
#     'I': 0,
#     'u': 1,
#     'o': 1,
#     'F': 2,
#     'M': 2,
#     'DN': 3,
#     'IN': 3,
#     'uN': 4,
#     'oN': 4,
#     'FN': 5,
#     'MN': 5
# }


def unmatched_at_edge(matches):
    """
    Helper function for EDEC event scoring
    """
    return all(matches.values[[0, -1]])


def between_matches(matches):
    """
    Helper function for EDEC segment scoring
    """
    idx = None
    if len(matches[matches]) > 1:
        idx = matches[matches].index
        s = idx[0]
        e = idx[-1]
        if s + 1 < e:
            idx = (s + 1, e)
    return idx


def edec_no_event_label(
    err, seg_idx, mask_match, mask_one_match_both, ground_truth, dev
):
    """
    Helper function for EDEC event scoring
    """
    correct_labels = ["C", "Cu"] if ground_truth else ["C", "Co"]
    err_seg = err[seg_idx]

    mask_no_label = np.vstack(
        (np.in1d(err_seg, correct_labels), mask_match, ~mask_one_match_both)
    )

    event_idx = seg_idx[np.all(mask_no_label, axis=0)]
    if dev:
        # preserves o and u in no label events
        err[event_idx] = [
            "" if len(label) == 1 else label[1:] for label in err[event_idx]
        ]
    else:
        err[event_idx] = ""

    return err


def edec_event_scoring(evt, events, k, ground_truth=False):
    err = np.full(len(evt), "", dtype="|U2")

    _agg = events.groupby(k)["segment_match"].agg(
        ["sum", "count", unmatched_at_edge, between_matches]
    )
    nmatch, nseg, edges, ind = _agg.values.T

    # D (I)
    err[nmatch == 0] = "D" if ground_truth else "I"

    # F (M) and U (O)
    mask = nmatch > 1
    edges = ~edges.astype(np.bool)

    err[mask & edges] = "Fu" if ground_truth else "Mo"
    err[mask & ~edges] = "F" if ground_truth else "M"

    # C and U (O)
    mask = nmatch == 1
    mask_nseg = nseg == 1

    err[mask & mask_nseg] = "C"
    err[mask & ~mask_nseg] = "Cu" if ground_truth else "Co"

    return err, nmatch, nseg, ind


def edec(evt_gt, evt_pr, input_events, dev=False):
    # Event-driven error characterization
    events = input_events.copy()
    gt_idx, pr_idx = (events["gt_idx"].values, events["pr_idx"].values)
    mask_segment_match = events["gt"].values == events["pr"].values

    events["segment_match"] = mask_segment_match
    output_gt = edec_event_scoring(evt_gt, events, "gt_idx", True)
    output_pr = edec_event_scoring(evt_pr, events, "pr_idx", False)
    err_gt, nmatch_gt, nseg_gt, ind_gt = output_gt
    err_pr, nmatch_pr, nseg_pr, ind_pr = output_pr

    # segment and timing scoring
    seg_gt, seg_t_gt = zip(*[list(label.ljust(2)) for label in err_gt[gt_idx]])
    seg_pr, seg_t_pr = zip(*[list(label.ljust(2)) for label in err_pr[pr_idx]])

    seg = {
        "seg_gt": seg_gt,
        "seg_pr": seg_pr,
        "seg_t_gt": seg_t_gt,
        "seg_t_pr": seg_t_pr,
        "between_matches_gt": False,
        "between_matches_pr": False,
    }
    seg = pd.DataFrame.from_dict(seg)
    events = pd.merge(events, seg, left_index=True, right_index=True)

    # in between matches
    between_matches_gt = [i for ind in ind_gt if ind is not None for i in range(*ind)]
    between_matches_pr = [i for ind in ind_pr if ind is not None for i in range(*ind)]
    events.loc[between_matches_gt, "between_matches_gt"] = True
    events.loc[between_matches_pr, "between_matches_pr"] = True

    seg_label = map("".join, zip(seg["seg_gt"], seg["seg_pr"]))
    mask_id = [_seg in ["DI"] for _seg in seg_label]
    mask_class_err = np.vstack(
        (
            mask_segment_match,
            events["between_matches_gt"],
            events["between_matches_pr"],
            mask_id,
        )
    )
    mask_class_err = mask_class_err.any(axis=0)

    mask_u = np.logical_and(~mask_class_err, events["seg_t_gt"] == "u")
    mask_o = np.logical_and(~mask_class_err, events["seg_t_pr"] == "o")
    events.loc[mask_u, "seg_gt"] = "u"
    events.loc[mask_o, "seg_pr"] = "o"

    # Handle no label; operates on segment level for 1-to-1 matches
    _nmatch_gt, _nmatch_pr = (nmatch_gt[gt_idx], nmatch_pr[pr_idx])
    mask_one_match_both = np.logical_and(_nmatch_gt == 1, _nmatch_pr == 1)
    err_gt = edec_no_event_label(
        err_gt,
        gt_idx,
        mask_segment_match,
        mask_one_match_both,
        ground_truth=True,
        dev=dev,
    )
    err_pr = edec_no_event_label(
        err_pr,
        pr_idx,
        mask_segment_match,
        mask_one_match_both,
        ground_truth=False,
        dev=dev,
    )

    # transform output to event matches
    events["match"] = events["segment_match"]

    # match deletions-insertions, i.e. misclassifications
    mask_match_di = np.logical_and(events["seg_gt"] == "D", events["seg_pr"] == "I")
    if not dev:
        events.loc[mask_match_di, "match"] = True
    else:
        _events = events.sort_values(
            ["It", "gt_idx", "pr_idx"], ascending=[False, True, True]
        )
        mask_evt_gt = np.ones(len(evt_gt), dtype=np.bool)
        mask_evt_pr = np.ones(len(evt_pr), dtype=np.bool)
        mask_evt_gt[events.loc[mask_match_di, "gt_idx"].values] = False
        mask_evt_pr[events.loc[mask_match_di, "pr_idx"].values] = False

        match_idx = matching._solve_match_conflicts(
            _events, evt_gt, evt_pr, mask_evt_gt, mask_evt_pr
        )
        events.loc[match_idx, "match"] = True

    return (err_gt, err_pr), events


def run_edec_examples(data_gt, data_pr):
    """
    Replicates results for example events (Ward et al., 2006, Fig. 1).
    Results are expected to match Table 3 and Table 4.

    TODO: in examples b and c there is a mismatch of #segments:
    edec_b: IN=10, IN'=9 (Ward et al., 2006, Table 4, middle)
    edec_c: NF=5, NF'=1 (Ward et al., 2006, Table 4, right)


    1. Ward, J. A., Lukowicz, P., & Tröster, G. (2006, May). Evaluating performance in continuous context
    recognition using event-driven error characterisation. In International Symposium on Location-and
    Context-Awareness (pp. 239-255).
    """
    # event matching
    event_matcher = matching.EventMatcher(gt=data_gt, pr=data_pr, mode="raw")

    (err_gt, err_pr), events = edec(
        evt_gt=event_matcher.evt_gt,
        evt_pr=event_matcher.evt_pr,
        input_events=event_matcher.events,
        dev=True,
    )

    # debug edec
    evt_gt = event_matcher.evt_gt
    evt_pr = event_matcher.evt_pr
    mask_gt = evt_gt["evt"].values > 0
    mask_pr = evt_pr["evt"].values > 0

    # event debug
    insertions = np.in1d(err_pr[mask_pr], ["I"]).sum()
    deletions = np.in1d(err_gt[mask_gt], ["D"]).sum()
    merges = np.in1d(err_pr[mask_pr], ["M", "Mo"]).sum()
    fragmentations = np.in1d(err_gt[mask_gt], ["F", "Fu"]).sum()

    print(
        "Event errors (for Positive, non-NULL classes only)\n"
        f"I: {insertions}\nD: {deletions}\nM: {merges}\nF: {fragmentations}"
    )

    # segment debug
    _mask_gt = events["gt"] > 0
    _mask_pr = events["pr"] > 0
    u = np.logical_and(_mask_gt, events["seg_gt"] == "u").sum()
    o = np.logical_and(_mask_pr, events["seg_pr"] == "o").sum()
    print(f"O: {o}\nU: {u}")

    _events = events.copy()
    _events = _events[~_events["segment_match"]].reset_index(drop=True)
    _mask_gt = _events["gt"] > 0
    _mask_pr = _events["pr"] > 0
    _events.loc[~_mask_gt, "seg_gt"] = _events.loc[~_mask_gt, "seg_gt"] + "N"
    _events.loc[~_mask_pr, "seg_pr"] = _events.loc[~_mask_pr, "seg_pr"] + "N"

    _events[["seg_gt", "seg_pr"]] = _events[["seg_gt", "seg_pr"]].replace(error_mapping)

    # ground-truth on top
    cm = metrics.confusion_matrix(_events["seg_pr"], _events["seg_gt"])
    print("SETs for positive classes (P) vs. NULL\n", cm)

    it = itertools.product(range(4), repeat=2)
    n_frames = [
        _events.query("seg_gt==@gt and seg_pr==@pr")["I"].sum() for gt, pr in it
    ]
    n_frames = np.reshape(n_frames, (4, 4), order="F")
    print("Counts of segment errors and corresponding number of frames\n", n_frames)


def plot_edec(events, evt_gt, evt_pr, labels=(), xlim=None):
    fig, ax = plt.subplots(figsize=(3.2, 1.2))
    ax = [None, ax]
    kwargs = {
        "wspace": 0,
        "hspace": 0.3,
        "top": 0.8,
        "bottom": 0,
        "left": 0.2,
        "right": 0.98,
    }
    fig.subplots_adjust(**kwargs)
    if xlim is not None:
        ax[1].set_xlim(*xlim)

    # scarf plots
    scarf_plot = utils.ScarfPlot(ax[1])
    scarf_plot.plot(evt_gt, label="Ground\ntruth")
    scarf_plot.plot(evt_pr, label="Prediction")

    # match plots
    plot_kwargs = {
        "ax": ax[1],
        "ids": (0, 1),
        "enum": True,
        "exclude_unmatched": True,
        "labels": labels,
    }
    utils.match_plot(events, evt_gt, evt_pr, plot_kwargs=plot_kwargs)

    return fig


def main():
    print("Event-driven error characterization test")
    root = os.path.join(ROOT, "assets")
    sdir = os.path.join(ROOT, "results", "match-plots", "edec")
    mkpath(sdir)
    event_map = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 255: 255}

    jobs = [
        {
            "gt": "edec/edec_a.csv",
            "pr": "edec_test/edec_a.csv",
            "debug": True,
        },
        {
            "gt": "edec/edec_b.csv",
            "pr": "edec_test/edec_b.csv",
            "debug": True,
        },
        {
            "gt": "edec/edec_c.csv",
            "pr": "edec_test/edec_c.csv",
            "debug": True,
        },
        {
            "gt": "edec/edec.csv",
            "pr": "edec_test/edec.csv",
            "plot": True,
        },
        {
            "gt": "edec/edec_steil.csv",
            "pr": "edec_test/edec_steil.csv",
            "plot": True,
        },
        {
            "gt": "edec/edec_fail.csv",
            "pr": "edec_test/edec_fail.csv",
            "plot": True,
        },
        {
            "gt": "edec/edec_di.csv",
            "pr": "edec_test/edec_di.csv",
            "plot": True,
            "dev": False,
        },
        {
            "gt": "edec/edec_di.csv",
            "pr": "edec_test/edec_di.csv",
            "plot": True,
            "dev": True,
        },
    ]

    for job in jobs:
        fpath_gt = os.path.join(root, job["gt"])
        fpath_pr = os.path.join(root, job["pr"])
        _, fname = utils.split_path(fpath_gt)
        print(f"Calculating EDEC for {fname}")
        data_gt, data_pr = utils.load_data(fpath_gt, fpath_pr, event_map)

        if job.get("debug", False):
            run_edec_examples(data_gt, data_pr)

        if job.get("plot", False):
            # event matching
            event_matcher = matching.EventMatcher(gt=data_gt, pr=data_pr, mode="raw")
            dev = job.get("dev", False)
            (err_gt, err_pr), events = edec(
                evt_gt=event_matcher.evt_gt,
                evt_pr=event_matcher.evt_pr,
                input_events=event_matcher.events,
                dev=dev,
            )

            _e = utils.idx2timestamp(event_matcher.evt_gt.iloc[-1]["e"] + 1, data_gt)
            fig = plot_edec(
                events,
                event_matcher.evt_gt,
                event_matcher.evt_pr,
                labels=(err_gt, err_pr),
                xlim=(data_gt["t"].min(), _e),
            )
            sname = f"{fname}" if not dev else f"{fname}-dev{dev}"
            fig.savefig(os.path.join(sdir, f"{sname}.eps"))
            fig.savefig(os.path.join(sdir, f"{sname}.png"))


if __name__ == "__main__":
    main()
