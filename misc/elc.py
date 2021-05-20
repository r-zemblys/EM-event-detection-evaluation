import os
from operator import attrgetter
from distutils.dir_util import mkpath

import pandas as pd
import matplotlib.pyplot as plt

try:
    from __init__ import ROOT
except ImportError:
    ROOT = None

from misc import matching
from misc import utils

plt.ion()
plt.rc("axes.spines", top=False, right=False)

GIW_EVENT_MAPPING = {1: 1, 2: 4, 3: 2, 4: 5, 5: 12}
GIW_EVENT_MAPPING_REV = {v: k for k, v in GIW_EVENT_MAPPING.items()}


##
# ELC
def elc(evt_gt, evt_pr, ws, data_gt, data_pr, fpath_elc_mat, **kwargs):
    """Interface to matching framework"""
    # TODO: simply parse copnfusion matrix from ELCCM.m
    events_match = elc_tp_matching(evt_gt, evt_pr, ws)
    events_match = events_match.query("match==True")

    # load ELC alignment
    if not os.path.exists(fpath_elc_mat):
        print("Run ELCCM.m")
        _, fname = os.path.split(fpath_elc_mat)
        prepare_for_elc_mat(data_gt.copy(), data_pr.copy(), fname)

    data_mat_gt, data_mat_pr, data_mat = parse_elc_mat(fpath_elc_mat)
    matcher_elc_mat = get_elc_mat_events(data_mat, [["ref_aligned"], ["test_aligned"]])
    events_err = matcher_elc_mat.events
    events_err = events_err.query("match==True")

    events = pd.concat((events_match, events_err))
    events.reset_index(drop=True, inplace=True)

    output = events[["gt", "pr"]].values

    return output, events


def elc_tp_matching(evt_gt, evt_pr, ws, eps=1e-5):
    """Transition point matching for ELC visualization"""
    _ws = ws / 2 + eps

    # ELC transition points and matches
    accum_tp = []
    ag = attrgetter("st", "et", "evt")
    _evt_pr = evt_pr["evt"].values
    # find matching transition points
    for evt in evt_gt.itertuples():
        st, et, _evt = ag(evt)
        pr_onset = evt_pr.query("st>@st-@_ws and st<@st+@_ws and evt==@_evt")
        pr_offset = evt_pr.query("et>@et-@_ws and et<@et+@_ws and evt==@_evt")
        pr_onset = pr_onset.index.values[0] if len(pr_onset) else -1
        pr_offset = pr_offset.index.values[0] if len(pr_offset) else -1
        if -1 not in [pr_onset, pr_offset]:
            match = _evt_pr[pr_onset] == _evt_pr[pr_offset]
        else:
            match = False
        accum_tp.append([pr_onset, pr_offset, match])
    tp_onset, tp_offset, mask_match = zip(*accum_tp)

    # gather data
    gt_idx = list(range(len(evt_gt)))
    tp_onset = zip(gt_idx, tp_onset)
    match_tp_onset = list(filter(lambda x: -1 not in x, tp_onset))
    tp_offset = zip(gt_idx, tp_offset)
    match_tp_offset = list(filter(lambda x: -1 not in x, tp_offset))

    match_tp_onset = pd.DataFrame(match_tp_onset, columns=["gt_idx", "pr_idx"])
    match_tp_onset["onset"] = True
    match_tp_offset = pd.DataFrame(match_tp_offset, columns=["gt_idx", "pr_idx"])
    match_tp_offset["offset"] = True
    events_match_tp = pd.merge(
        match_tp_onset, match_tp_offset, on=["gt_idx", "pr_idx"], how="outer"
    )
    events_match_tp.fillna(False, inplace=True)

    gt_idx_match = [_m for _f, _m in zip(mask_match, gt_idx) if _f]
    events_match_tp["match"] = events_match_tp["gt_idx"].isin(gt_idx_match)

    gt_idx = events_match_tp["gt_idx"]
    pr_idx = events_match_tp["pr_idx"]
    events_match_tp["gt"] = evt_gt.loc[gt_idx, "evt"].values
    events_match_tp["pr"] = evt_pr.loc[pr_idx, "evt"].values

    return events_match_tp


# mat functions
def prepare_for_elc_mat(data_gt, data_pr, name, sdir):
    """Prepares data to be imported to matlab"""

    data_gt.replace({"evt": GIW_EVENT_MAPPING_REV}, inplace=True)
    data_pr.replace({"evt": GIW_EVENT_MAPPING_REV}, inplace=True)
    data_gt["evt-test"] = data_pr["evt"]
    data_gt.to_csv(os.path.join(sdir, f"{name}-data.csv"), index=False)
    for d, tag in zip([data_gt, data_pr], ["gt", "pr"]):
        evt = utils.aggr_events_df(d["evt"])
        # get timestamps for event bounds; adjust for matlab's 1-based indexing
        evt["st"] = d.iloc[evt["s"].values]["t"].values
        evt["et"] = d.iloc[evt["e"].values - 1]["t"].values
        evt["s"] += 1
        evt.to_csv(os.path.join(sdir, f"{name}-{tag}.csv"), index=False)


def parse_elc_mat(fpath):
    data = utils.loadmat(fpath)

    labels_ref = utils.get_nested_item(data, ["label_ref", "Labels"])
    labels_test = utils.get_nested_item(data, ["label_test", "Labels"])

    timestamps = utils.get_nested_item(data, ["label_test", "T"])
    data_gt = pd.DataFrame.from_dict({"t": timestamps, "evt": labels_ref})
    data_pr = pd.DataFrame.from_dict({"t": timestamps, "evt": labels_test})

    return data_gt, data_pr, data


def get_elc_mat_events(data_mat, sources):
    _source_gt, _source_pr = sources
    labels_ref = utils.get_nested_item(data_mat, _source_gt)
    labels_test = utils.get_nested_item(data_mat, _source_pr)
    timestamps = utils.get_nested_item(data_mat, ["label_test", "T"])
    _gt = pd.DataFrame.from_dict({"t": timestamps, "evt": labels_ref})
    _pr = pd.DataFrame.from_dict({"t": timestamps, "evt": labels_test})
    _gt.replace({"evt": GIW_EVENT_MAPPING}, inplace=True)
    _pr.replace({"evt": GIW_EVENT_MAPPING}, inplace=True)
    matcher = matching.EventMatcher(gt=_gt, pr=_pr, mode="raw")
    match_mask = matcher.events["gt"] == matcher.events["pr"]
    matcher.events["match"] = ~match_mask
    return matcher


# plotting functions
def plot_elc(events, matcher, matcher_mat, plot_transitions=False, xlim=None):
    kwargs = {
        "wspace": 0,
        "hspace": 0.3,
        "top": 0.9,
        "bottom": 0,
        "left": 0.2,
        "right": 0.98,
    }

    fig, ax = plt.subplots(figsize=(3.2, 1.2 * 1.5))
    ax = [None, ax]
    fig.subplots_adjust(**kwargs)
    if xlim is not None:
        ax[1].set_xlim(*xlim)

    # scarf plots
    scarf_plot = utils.ScarfPlot(ax[1])
    scarf_plot.plot(matcher.evt_gt, label="Ground\ntruth")
    scarf_plot.plot(matcher.evt_pr, label="Prediction")

    # match plots
    plot_kwargs = {"ax": ax[1], "ids": (0, 1), "enum": True, "exclude_null": True}
    utils.match_plot(events, matcher.evt_gt, matcher.evt_pr, plot_kwargs=plot_kwargs)

    # plot aligned events
    scarf_plot.plot(matcher_mat.evt_gt, label="Aligned\nground\ntruth")
    scarf_plot.plot(matcher_mat.evt_pr, label="Aligned\nprediction")
    plot_kwargs = {
        "ax": ax[1],
        "ids": (2, 3),
        "enum": False,
        "linestyle": "--",
        "match-point": "mid",
    }
    utils.match_plot(matcher_mat.events, plot_kwargs=plot_kwargs)

    # visualize transition points
    if plot_transitions:
        match_plot_kwargs = {
            "ids": (0, 1),
            "enum": False,
            "exclude_null": True,
            "linestyle": "--",
            "linewidth": 0.75,
            "color": "green",
            "match-point": "s",
        }
        plot_kwargs.update(match_plot_kwargs)
        utils.match_plot(
            events,
            matcher.evt_gt,
            matcher.evt_pr,
            col_match="onset",
            plot_kwargs=plot_kwargs,
        )

        match_plot_kwargs = {
            "enum": False,
            "exclude_null": True,
            "linestyle": "--",
            "linewidth": 0.75,
            "color": "red",
            "match-point": "e",
        }
        plot_kwargs.update(match_plot_kwargs)
        utils.match_plot(
            events,
            matcher.evt_gt,
            matcher.evt_pr,
            col_match="offset",
            plot_kwargs=plot_kwargs,
        )

    return fig


##
def main():
    root = os.path.join(ROOT, "assets")
    dataset = "elc"
    sdir = os.path.join(ROOT, "results", "match-plots", "elc")
    mkpath(sdir)

    jobs = [
        {"fname": "elc", "plot_transitions": True},
        {"fname": "elc_fail", "plot_transitions": False},
        {"fname": "elc_fail_timing", "plot_transitions": False},
    ]

    for job in jobs:
        fname = job["fname"]
        data_gt = pd.read_csv(os.path.join(root, dataset, f"{fname}.csv"))
        data_pr = pd.read_csv(os.path.join(root, f"{dataset}_test", f"{fname}.csv"))

        event_matcher = matching.EventMatcher(gt=data_gt, pr=data_pr, mode="raw")
        events = elc_tp_matching(
            event_matcher.evt_gt, event_matcher.evt_pr, ws=10 / 300.0
        )

        # analyse data from matlab implementation
        fpath_elc_mat = os.path.join(
            root, f"{dataset}_test", "elccm", f"{fname}-{fname}_test.mat"
        )
        if not os.path.exists(fpath_elc_mat):
            print(f"Run ELCCM.m for {fname} and copy to {fpath_elc_mat}")
            prepare_for_elc_mat(data_gt, data_pr, fname, sdir)
            continue

        data_mat_gt, data_mat_pr, data_mat = parse_elc_mat(fpath_elc_mat)
        matcher_mat = get_elc_mat_events(data_mat, [["ref_aligned"], ["test_aligned"]])

        # plot
        _e = utils.idx2timestamp(event_matcher.evt_gt.iloc[-1]["e"] + 1, data_gt)
        plot_transitions = job.get("plot_transitions", False)
        fig = plot_elc(
            events,
            event_matcher,
            matcher_mat,
            plot_transitions=plot_transitions,
            xlim=(data_gt["t"].min(), _e),
        )
        fig.savefig(os.path.join(sdir, f"{fname}.eps"))
        fig.savefig(os.path.join(sdir, f"{fname}.png"))


if __name__ == "__main__":
    main()

# debug ELCCM.m output
# fig, ax = plt.subplots(figsize=(3.2, 2.4))
# ax = [None, ax]
# fig.subplots_adjust(**kwargs)
# _e = utils.idx2timestamp(event_matcher.evt_gt.iloc[-1]['e'] + 1, data_gt)
# ax[1].set_xlim(data_gt['t'].min(), _e)
#
# # scarf plots
# scarf_plot = utils.ScarfPlot(ax[1])
# scarf_plot.plot(event_matcher.evt_gt, label='Ground\ntruth')
# scarf_plot.plot(event_matcher.evt_pr, label='Test')
# sources = [
#     [['label_ref', 'Labels'], ['label_test', 'Labels']],
#     # [['ref_fu'], ['test_fu']],
#     [['ref_aligned'], ['test_aligned']],
# ]
# for source in sources:
#     matcher_elc = get_elc_mat_events(data_mat, source)
#     _source_gt, _source_pr = source
#     scarf_plot.plot(matcher_elc.evt_gt, label='-'.join(_source_gt))
#     scarf_plot.plot(matcher_elc.evt_pr, label='-'.join(_source_pr))
