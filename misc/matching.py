import itertools
from operator import attrgetter
from collections import Counter

import numpy as np
import pandas as pd

# from misc.edec import edec
# from misc.elc import elc
from misc import utils


##
def sample(evt_gt, evt_pr):
    output = np.vstack((evt_gt, evt_pr)).T
    output = output.astype(int)
    return output


def majority_voting(evt_gt, evt_pr, input_events, **kwargs):
    """Implements Majority Voting
    Hoppe, S., & Bulling, A. (2016). End-to-end eye movement detection using
    convolutional neural networks. arXiv preprint arXiv:1609.02452.
    """
    events = input_events.copy()

    pr = utils.evt_unfold(evt_pr["evt"], evt_pr["dur"])
    ag = attrgetter("s", "e")
    majority_count = [
        Counter(pr[slice(*ag(r))]) for r in evt_gt[["s", "e"]].itertuples()
    ]
    majority = [c.most_common(1) for c in majority_count]
    majority = np.array(majority).squeeze(axis=1)
    majority_event, majority_count = zip(*majority[events["gt_idx"]])
    events["majority"] = majority_event
    events["majority_count"] = majority_count

    evt_gt_dur = evt_gt["dur"].values[events["gt_idx"]]
    events["majority_proportion"] = np.array(majority_count) / evt_gt_dur

    mask_match = (events["pr"] == events["majority"]) & (
        events["majority_proportion"] > 0.5
    )
    events["match"] = mask_match

    if not kwargs.get("plot-mode", False):
        # include only events from ground-truth
        events = events[mask_match].groupby("gt_idx").head(1)
        events.reset_index(drop=True, inplace=True)

        # match mask
        events["match"] = True
        mask_match = events["match"].values

    return mask_match, events


def plurality_voting(evt_gt, evt_pr, input_events, **kwargs):
    """Implements Plurality Voting"""
    events = input_events.copy()

    pr = utils.evt_unfold(evt_pr["evt"], evt_pr["dur"])
    ag = attrgetter("s", "e")
    majority_count = [
        Counter(pr[slice(*ag(r))]) for r in evt_gt[["s", "e"]].itertuples()
    ]
    majority = np.array([c.most_common(1)[0][0] for c in majority_count])
    events["majority"] = majority[events["gt_idx"]]
    mask_match = events["pr"] == events["majority"]
    events["match"] = mask_match

    if not kwargs.get("plot-mode", False):
        # include only events from ground-truth
        events = events[mask_match].groupby("gt_idx").head(1)
        events.reset_index(drop=True, inplace=True)

        # match mask
        events["match"] = True
        mask_match = events["match"].values

    return mask_match, events


def overlap(evt_gt, evt_pr, input_events, **kwargs):
    """Implements Overlap Matching
    Hauperich, A.-K., Young, L., & Smithson, H. (2020). What makes a microsaccade?
    A review of 70 years research prompts a new detection method.
    Journal of Eye Movement Research, 12(6). https://doi.org/10.16910/jemr.12.6.13
    """
    events = input_events.copy()

    # only match positive events
    _match = np.logical_and(events["gt"] == events["pr"], events["gt"] > 0)
    _events = events[_match]
    match_idx = _solve_match_conflicts(_events, evt_gt, evt_pr)

    # match mask
    mask_match = np.zeros(len(events), dtype=np.bool)
    mask_match[match_idx] = True
    events["match"] = mask_match

    if kwargs.get("one-match-only", False):
        # find unmatched events that are part of a merge or fragmentation
        mask_rm = np.logical_and(_match, ~mask_match)
        gt_idx_rm = set(events.loc[mask_rm, "gt_idx"]) - set(
            events.loc[mask_match, "gt_idx"]
        )
        pr_idx_rm = set(events.loc[mask_rm, "pr_idx"]) - set(
            events.loc[mask_match, "pr_idx"]
        )

        # set these events to -1
        gt_mask_rm = events["gt_idx"].isin(gt_idx_rm)
        events.loc[gt_mask_rm, "gt_idx"] = -1
        pr_mask_rm = events["pr_idx"].isin(pr_idx_rm)
        events.loc[pr_mask_rm, "pr_idx"] = -1

    return mask_match, events


def earliest_overlap(evt_gt, evt_pr, input_events, backward=False, **kwargs):
    """Implements Earliest Overlap matching from
    Hooge, I. T., Niehorster, D. C., Nyström, M., Andersson, R., & Hessels, R. S. (2018).
    Is human classification by experienced untrained observers a gold standard in fixation
    detection?. Behavior Research Methods, 50(5), 1864-1881.
    """
    events = input_events.copy()

    if not backward:
        matching_pt = "st"
        sort_order = [True, True, True]
    else:
        matching_pt = "et"
        sort_order = [False, True, True]

    overlap_onset = np.vstack(
        (
            evt_gt.loc[events["gt_idx"], matching_pt],
            evt_pr.loc[events["pr_idx"], matching_pt],
        )
    )
    events["overlap_onset"] = np.max(overlap_onset, axis=0)
    mask = np.any(events[["gt", "pr"]] == 0, axis=1)

    _events = events[~mask].sort_values(
        ["overlap_onset", "gt_idx", "pr_idx"], ascending=sort_order
    )
    match_idx = _solve_match_conflicts(_events, evt_gt, evt_pr)

    mask_match = np.zeros(len(events), dtype=np.bool)
    mask_match[match_idx] = True
    events["match"] = mask_match

    return mask_match, events


def maximum_overlap(evt_gt, evt_pr, input_events, **kwargs):
    """Implements Maximum Overlap matching from
    Zemblys, R., Niehorster, D. C., & Holmqvist, K. (2019). gazeNet: End-to-end
    eye-movement event detection with deep neural networks.
    Behavior research methods, 51(2), 840-864.
    """
    events = input_events.copy()
    _events = events.sort_values(
        ["It", "gt_idx", "pr_idx"], ascending=[False, True, True]
    )
    match_idx = _solve_match_conflicts(_events, evt_gt, evt_pr)

    mask_match = np.zeros(len(events), dtype=np.bool)
    mask_match[match_idx] = True
    events["match"] = mask_match

    return mask_match, events


def maximum_iou(evt_gt, evt_pr, input_events, **kwargs):
    """Implements Maximum Intersection over Union from
    Startsev, M., Agtzidis, I., & Dorr, M. (2019). 1D CNN with BLSTM for automated
    classification of fixations, saccades, and smooth pursuits.
    Behavior research methods, 51(2), 556-572.
    """
    events = input_events.copy()
    iou_threshold = kwargs.get("iou_threshold", 0.0)

    # get unions
    gt_idx = events["gt_idx"]
    pr_idx = events["pr_idx"]
    _s = map(lambda x: min(x), zip(evt_gt.loc[gt_idx, "st"], evt_pr.loc[pr_idx, "st"]))
    _e = map(lambda x: max(x), zip(evt_gt.loc[gt_idx, "et"], evt_pr.loc[pr_idx, "et"]))
    events["U"] = list(map(lambda s, e: e - s, _s, _e))
    events["IoU"] = events["It"] / events["U"]

    # match events
    mask_match = (events["IoU"] > 0.5).values  # ensures 1-to-1 match
    _gt_idx = events.loc[mask_match, "gt_idx"]
    _pr_idx = events.loc[mask_match, "pr_idx"]
    mask_evt_gt = np.zeros(len(evt_gt), dtype=np.bool)
    mask_evt_pr = np.zeros(len(evt_pr), dtype=np.bool)
    mask_evt_gt[_gt_idx] = True
    mask_evt_pr[_pr_idx] = True

    mask = events["IoU"] > iou_threshold
    mask = mask & ~mask_match
    _events = events[mask].sort_values(
        ["IoU", "gt_idx", "pr_idx"], ascending=[False, True, True]
    )
    # solve match conflicts
    match_idx = _solve_match_conflicts(
        _events, evt_gt, evt_pr, mask_evt_gt, mask_evt_pr
    )

    # simple version; takes longer
    # mask = events['IoU'] > iou_threshold
    # _events = events[mask].sort_values(['IoU', 'gt_idx', 'pr_idx'],
    #                                    ascending=[False, True, True])
    # match_idx = solve_match_conflicts(_events, evt_gt, evt_pr)
    # mask_match = np.zeros(len(events), dtype=np.bool)

    mask_match[match_idx] = True
    events["match"] = mask_match

    return mask_match, events


def _solve_match_conflicts(events, evt_gt, evt_pr, mask_evt_gt=None, mask_evt_pr=None):
    """Makes all matches one-to-one"""
    match_idx = []
    if mask_evt_gt is None:
        mask_evt_gt = np.zeros(len(evt_gt), dtype=np.bool)
    if mask_evt_pr is None:
        mask_evt_pr = np.zeros(len(evt_pr), dtype=np.bool)

    ag = attrgetter("gt_idx", "pr_idx", "Index")
    for evt in events.itertuples():
        gt_idx, pr_idx, idx = ag(evt)
        if not any((mask_evt_gt[gt_idx], mask_evt_pr[pr_idx])):
            mask_evt_gt[gt_idx] = True
            mask_evt_pr[pr_idx] = True
            match_idx.append(idx)

    return match_idx


class EventMatcher(object):
    matching_methods = {
        "sample": sample,
        "majority-voting": majority_voting,
        "plurality-voting": plurality_voting,
        "overlap": overlap,
        "earliest-overlap": earliest_overlap,
        "maximum-overlap": maximum_overlap,
        "iou": maximum_iou,
        # 'edec': edec,
        # 'elc': elc,
    }

    def __init__(self, gt, pr, mode="raw"):
        if mode == "raw":
            _lgt, _lpr = len(gt), len(pr)
            assert _lgt == _lpr

            evt_gt = utils.aggr_events_df(gt["evt"])
            evt_pr = utils.aggr_events_df(pr["evt"])

            evt_gt["st"] = gt.loc[evt_gt["s"], "t"].values
            evt_pr["st"] = pr.loc[evt_pr["s"], "t"].values

            # last offset index is out of range. Estimate timestamp
            evt_gt["et"] = utils.idx2timestamp(evt_gt["e"], gt)
            evt_pr["et"] = utils.idx2timestamp(evt_pr["e"], pr)

            # store sample level events
            data_gt = gt
            data_pr = pr

        else:
            evt_gt = gt
            evt_pr = pr

            # TODO: construct sample level data
            data_gt = None
            data_pr = None

        evt_gt["dur_t"] = evt_gt["et"] - evt_gt["st"]
        evt_pr["dur_t"] = evt_pr["et"] - evt_pr["st"]

        # get intersections
        gt_idx = utils.evt_unfold(range(len(evt_gt)), evt_gt["dur"])
        pr_idx = utils.evt_unfold(range(len(evt_pr)), evt_pr["dur"])
        _overlap = zip(gt_idx, pr_idx)
        events = [_k + tuple([len([*_g])]) for _k, _g in itertools.groupby(_overlap)]

        events = pd.DataFrame(events, columns=["gt_idx", "pr_idx", "I"])
        gt_idx = events["gt_idx"]
        pr_idx = events["pr_idx"]

        # timestamp based intersection
        _s = map(
            lambda x: max(x), zip(evt_gt.loc[gt_idx, "st"], evt_pr.loc[pr_idx, "st"])
        )
        _e = map(
            lambda x: min(x), zip(evt_gt.loc[gt_idx, "et"], evt_pr.loc[pr_idx, "et"])
        )
        st, et, it = zip(*map(lambda s, e: (s, e, e - s), _s, _e))
        events["It"] = it
        events["st"] = st
        events["et"] = et

        events["gt"] = evt_gt.loc[gt_idx, "evt"].values
        events["pr"] = evt_pr.loc[pr_idx, "evt"].values

        # expose variables
        self.events = events
        self.evt_gt = evt_gt
        self.evt_pr = evt_pr
        self.data_gt = data_gt
        self.data_pr = data_pr

    def run_matching(self, matcher_label, **kwargs):
        method, *_ = matcher_label.split("/")
        matching_func = EventMatcher.matching_methods.get(method, None)
        if matching_func is None:
            print(f"Warning: {method} matcher does not exist")
            return None, None

        if method == "sample":
            evt_gt_sample = self.data_gt["evt"].values
            evt_pr_sample = self.data_pr["evt"].values
            output = matching_func(evt_gt=evt_gt_sample, evt_pr=evt_pr_sample)
            events = None
        else:
            # event level matching
            result = matching_func(
                evt_gt=self.evt_gt,
                evt_pr=self.evt_pr,
                input_events=self.events,
                **kwargs,
            )
            mask_match, events = result
            output, events = EventMatcher.format_output(events, mask_match)

        return output, events

    @staticmethod
    def format_output(events, mask_match):
        """Format output"""
        # include all events
        match = events.loc[mask_match, ["gt", "pr"]].values

        # handle unmatched events
        gt_umatch = EventMatcher.add_unmatched(events, mask_match, "gt")
        pr_umatch = EventMatcher.add_unmatched(events, mask_match, "pr")

        # cat all events
        output = np.concatenate((match, gt_umatch, pr_umatch))
        output = output.astype(int)

        return output, events

    @staticmethod
    def add_unmatched(events, mask_match, stream, ulabel=-1):
        """Finds unmatched events and constructs output array"""
        assert stream in ["gt", "pr"]
        _label_idx = f"{stream}_idx"
        event_ids = events[_label_idx].values

        # finds events that are already accounted for
        _mask = np.logical_or(mask_match, event_ids == -1)
        match_ids = event_ids[_mask]
        _mask = np.in1d(event_ids, match_ids)

        # find remaining events
        events_umatch = events[~_mask].groupby(_label_idx).head(1)

        _l = len(events_umatch)
        if not _l:
            umatch = np.empty((0, 2))
        else:
            umatch = events_umatch[stream].values
            _fill = np.full(_l, ulabel)
            if stream == "gt":
                umatch = np.array((umatch, _fill))
            else:
                umatch = np.array((_fill, umatch))

            umatch = umatch.T

        return umatch
