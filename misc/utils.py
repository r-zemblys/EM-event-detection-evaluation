import os
import re
import copy
import fnmatch
import itertools
from collections import Counter
from functools import reduce
from operator import getitem
from distutils.dir_util import mkpath

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
import scipy.io as scio

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from misc.constants import event_mapping, matcher_map


##
# data utils
def get_file_list(job, root):
    fpaths = job.get("files", [])
    path_gt = path2abs(job.get("gt", None), root)
    path_pr = path2abs(job.get("pr", None), root)

    if len(fpaths):
        # use only specified files
        fpaths_gt = [path2abs(fpath, path_gt) for fpath in fpaths]
    else:
        # use all files in the directory
        fpaths_gt = dir_walk(path_gt, "csv")

    # format paths of prediction files
    fpaths_pr = [fpath.replace(path_gt, path_pr) for fpath in fpaths_gt]

    # check if prediction files exist
    fpaths = [
        (_gt, _pr) for _gt, _pr in zip(fpaths_gt, fpaths_pr) if os.path.exists(_pr)
    ]
    n_gt, n_pr = len(fpaths_gt), len(fpaths)
    if not (n_gt == n_pr):
        print(f"{n_pr} prediction files exist out of {n_gt}")

    fpaths_gt, fpaths_pr = zip(*fpaths)
    return fpaths_gt, fpaths_pr


def dir_walk(path, ext):
    fpaths = [
        os.path.join(_root, _file)
        for _root, _dir, _files in os.walk(path)
        for _file in fnmatch.filter(_files, f"*.{ext}")
    ]
    return sorted(fpaths)


def load_data(fpath_gt, fpath_pr, event_map):
    data_gt = pd.read_csv(fpath_gt)
    data_pr = pd.read_csv(fpath_pr)

    # construct event map
    event_set = set(data_gt["evt"]) | set(data_pr["evt"])
    _undef = event_map.get(0, 0)
    _event_map = {_evt: _undef for _evt in event_set}
    _event_map.update(event_map)

    # event mapping
    data_gt.replace({"evt": _event_map}, inplace=True)
    data_pr.replace({"evt": _event_map}, inplace=True)

    return data_gt, data_pr


##
# ET utils
def get_et_stats(etdata, fs=None):
    n_samples = len(etdata)
    counts = {_evt: 0 for _evt in event_mapping.keys()}
    counts.update(Counter(etdata["evt"]))

    fs_est = estimate_fs(etdata["t"])
    fs = fs if fs is not None else fs_est
    et_stats = {"fs": fs, "fs_est": fs_est, "n-samples": n_samples}
    et_stats.update(counts)

    return et_stats


def estimate_fs(t):
    """Estimates data sampling rate"""
    sampling_rates = [
        2000,
        1250,
        1000,
        600,
        500,
        300,
        250,
        240,
        200,
        120,
        75,
        60,
        50,
        30,
        25,
    ]

    fs_est = np.median(1 / np.diff(t))
    fs = min(sampling_rates, key=lambda x: abs(x - fs_est))

    return fs


##
# event utils
def aggr_events(events_raw):
    """Aggregates event vector to the list of compact event vectors.
    Parameters:
        events_raw  --  vector of raw events
    Returns:
        events_aggr --  list of compact event vectors ([onset, offset, event])
    """

    events_aggr = []
    s = 0
    for bit, group in itertools.groupby(events_raw):
        event_length = len(list(group))
        e = s + event_length
        events_aggr.append([s, e, bit])
        s = e
    return events_aggr


def aggr_events_df(events_raw):
    evt = pd.DataFrame(aggr_events(events_raw), columns=["s", "e", "evt"])
    evt["dur"] = evt["e"] - evt["s"]
    evt = evt.astype(np.int64)
    return evt


def evt_unfold_repeat(args):
    return itertools.repeat(args[0], int(args[1]))


def evt_unfold(fill, n):
    _evt_unfolded = map(evt_unfold_repeat, zip(fill, n))
    _evt_unfold = itertools.chain.from_iterable(_evt_unfolded)
    return [*_evt_unfold]


def get_evt_mid(evt):
    s = evt["s"].values
    e = evt["e"].values
    idx_mid = np.round(s + (e - s) / 2).astype(int)

    return idx_mid


def idx2timestamp(idx, data):
    """
    Helper function to convert from sample index to timestamp
    """
    f = interp(data.index, data["t"], "linear")
    return f(idx)


##
# mat utils
def mat_check_keys(mat):
    """
    Checks if entries in dictionary are mat-objects
    and converts them to nested dictionaries
    """
    for key in mat:
        if isinstance(mat[key], scio.matlab.mio5_params.mat_struct):
            mat[key] = mat2dict(mat[key])
    return mat


def mat2dict(matobj):
    """
    A recursive function that constructs nested dictionaries from matobjects
    """
    dictionary = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scio.matlab.mio5_params.mat_struct):
            dictionary[strg] = mat2dict(elem)
        else:
            dictionary[strg] = elem
    return dictionary


def loadmat(filename):
    """
    Replaces spio.loadmat
    """
    data = scio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return mat_check_keys(data)


##
# plotting utils
def plot_job(
    matcher,
    events,
    spath=None,
    odir=None,
    match_plot_kwargs=None,
    matcher_label=None,
    data=None,
):
    # TODO: handle plotting in the background
    # plot limits
    plot_kwargs = {}
    if data is not None:
        _e = matcher.evt_gt.iloc[-1]["e"]
        xlim = (data["t"].min(), idx2timestamp(_e, data))
        plot_kwargs = {"xlim": xlim}

    # plot
    fig = plot_matching(matcher, events, plot_kwargs, match_plot_kwargs)

    # save
    _save = (spath is not None, odir is not None)
    if all(_save):
        sdir, sname = split_path(spath)
        _label = valid_fname(matcher_label, "-")
        sname = "-".join(filter(None, (sname, _label)))
        sdir = os.path.join(odir, "match-plots", sdir)
        mkpath(sdir)

        fig.savefig(os.path.join(sdir, f"{sname}.eps"))
        fig.savefig(os.path.join(sdir, f"{sname}.png"))


def get_figure(n=2, figsize=(3.2, 1.2), label=None):
    gs_kw = dict(height_ratios=[16] * n + [4])
    figure, axes = plt.subplots(
        ncols=1,
        nrows=n + 1,
        sharex=True,
        constrained_layout=False,
        gridspec_kw=gs_kw,
        figsize=figsize,
    )

    if n > 0:
        axes[n - 1].xaxis.set_tick_params(which="both", labelbottom=True)
        axes[n - 1].set_xlabel("Time, sec")
        if label is not None:
            axes[n - 1].set_ylabel(label)
    else:
        axes = [None, axes]

    kwargs = {
        "wspace": 0,
        "hspace": 0.3,
        "top": 0.8,
        "bottom": 0,
        "left": 0.2,
        "right": 0.98,
    }
    figure.subplots_adjust(**kwargs)

    return figure, axes


def plot_matching(matcher, events, kwargs, match_plot_kwargs=None):
    fig, ax = get_figure(n=0)

    xlim = kwargs.get("xlim", None)
    if xlim is not None:
        ax[1].set_xlim(*xlim)

    # scarf plots
    scarf_plot = ScarfPlot(ax[-1])
    scarf_plot.plot(matcher.evt_gt, label="Ground\ntruth")
    scarf_plot.plot(matcher.evt_pr, label="Prediction")

    # match plots
    if not isinstance(events, tuple):
        plot_kwargs = {"ax": ax[-1], "ids": (0, 1), "enum": True}
        if match_plot_kwargs is not None:
            plot_kwargs.update(match_plot_kwargs)
        match_plot(events, matcher.evt_gt, matcher.evt_pr, plot_kwargs=plot_kwargs)
    else:
        # used for custom plotting
        if match_plot_kwargs is None:
            match_plot_kwargs = [{} for _ in range(len(events))]
        match_plot_kwargs = (
            match_plot_kwargs
            if isinstance(match_plot_kwargs, list)
            else [match_plot_kwargs]
        )
        for _events, _plot_kwargs in zip(events, match_plot_kwargs):
            plot_kwargs = {"ax": ax[-1], "ids": (0, 1), "enum": True}
            plot_kwargs.update(_plot_kwargs)
            match_plot(_events, matcher.evt_gt, matcher.evt_pr, plot_kwargs=plot_kwargs)

    return fig


class ScarfPlot(object):
    def __init__(self, ax):
        self.n = 0
        self.labels = []
        self.ax = ax
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    def plot(self, events, label=[]):
        _events = copy.deepcopy(events)
        # seg is array identifying event boundaries
        seg = _events[["st", "et"]].values

        self.labels.append(label)
        colors = [
            get_nested_item(event_mapping, [_evt, "color"], "#ffffff")
            for _evt in _events["evt"]
        ]

        patches = []
        for s, e in seg:
            rect = Rectangle((s, -self.n * 2), e - s, 1)
            patches.append(rect)

        # Create patch collection with specified colour
        pc = PatchCollection(patches, facecolor=colors)
        self.ax.add_collection(pc)

        self.ax.set_ylim(-self.n * 2 - 1, 1)
        self.ax.set_yticks(np.arange(self.n + 1) * -2 + 0.5)
        self.ax.set_yticklabels(self.labels, fontsize="x-small")

        self.n += 1


def match_plot(
    events_input, evt_gt=None, evt_pr=None, col_match="match", plot_kwargs={}
):

    events = events_input.copy()
    ax = plot_kwargs.get("ax", plt.gca())
    ids = np.array(plot_kwargs.get("ids", (0, 1)))
    ids = ids * -2 + 0.5  # center of the plot

    # plot segment matches:
    if any((evt_gt is None, evt_pr is None)):
        events["gt_idx"] = events["pr_idx"] = np.arange(len(events))
        evt_gt = evt_pr = events

    # select matches to plot
    mask_gt = [np.ones(len(evt_gt), dtype=np.bool)]
    mask_pr = [np.ones(len(evt_pr), dtype=np.bool)]
    _query = [f"{col_match}>0"]
    labels_gt, labels_pr = plot_kwargs.get("labels", ([], []))

    # exclude class 0
    if plot_kwargs.get("exclude_null", False):
        mask_gt.append(evt_gt["evt"].values > 0)
        mask_pr.append(evt_pr["evt"].values > 0)
        _query.append("gt>0 and pr>0")

    # exclude class 255
    if plot_kwargs.get("exclude_unmatched", False):
        mask_gt.append(evt_gt["evt"].values < 255)
        mask_pr.append(evt_pr["evt"].values < 255)
        _query.append("gt<255 and pr<255")

    mask_gt = np.all(mask_gt, axis=0)
    mask_pr = np.all(mask_pr, axis=0)
    _query = " and ".join(_query)
    labels_gt = labels_gt[mask_gt] if len(labels_gt) else []
    labels_pr = labels_pr[mask_pr] if len(labels_gt) else []

    # calculate timestamps for match plots
    match_point = plot_kwargs.get("match-point", "mid")
    if match_point == "mid":
        t_gt = evt_gt["st"] + (evt_gt["et"] - evt_gt["st"]) / 2
        t_pr = evt_pr["st"] + (evt_pr["et"] - evt_pr["st"]) / 2
    else:
        t_gt = evt_gt[f"{match_point}t"]
        t_pr = evt_pr[f"{match_point}t"]
    t_gt = t_gt[mask_gt]
    t_pr = t_pr[mask_pr]

    # get timestamps of matched event mid points
    matches = events.query(_query)
    t_gt_match = t_gt[matches["gt_idx"]].values
    t_pr_match = t_pr[matches["pr_idx"]].values

    # number events
    if plot_kwargs.get("enum", True):
        kwargs_gt = {
            "ha": "center",
            "va": "baseline",
            "fontsize": "xx-small",
            "rotation": 90,
        }
        kwargs_pr = {"ha": "center", "va": "top", "fontsize": "x-small", "rotation": 90}
        for n, _t in enumerate(t_gt):
            ax.text(_t, ids[0] + 0.7, "GT%s" % (n + 1), **kwargs_gt)
        for n, _t in enumerate(t_pr):
            ax.text(_t, ids[1] - 0.6, "P%s" % (n + 1), **kwargs_pr)

        # annotate event labels
        kwargs = {"ha": "center", "va": "center", "fontsize": "xx-small"}
        for _t, label in zip(t_gt, labels_gt):
            ax.text(_t, ids[0], label, **kwargs)
        for _t, label in zip(t_pr, labels_pr):
            ax.text(_t, ids[1], label, **kwargs)

    # plot matches
    segs = np.zeros((len(matches), 2, 2))
    segs[:, :, 0] = np.array((t_gt_match, t_pr_match)).T
    segs[:, :, 1] = ids + [-0.25, 0.25]

    line_kwargs = {
        "linestyle": plot_kwargs.get("linestyle", "-"),
        "linewidth": plot_kwargs.get("linewidth", 0.5),
        "color": plot_kwargs.get("color", "black"),
    }
    lc = LineCollection(segs, **line_kwargs)
    ax.add_collection(lc)


def plot_metrics_vs_matchers(
    df,
    odir,
    _plot_kwargs,
    _kwargs_legend,
    ylim=(0, 1),
    sname=None,
    title=None,
    palette=None,
):
    _matchers = set(df["matcher"])
    _order = [
        matcher_map.get(_matcher) for _matcher in matcher_map if _matcher in _matchers
    ]

    # plot results
    sns.catplot(
        x="matcher_label",
        y="score",
        hue="metric_label",
        order=_order,
        data=df,
        palette=palette,
        **_plot_kwargs,
    )
    plt.legend(**_kwargs_legend)

    if title is not None:
        plt.title(title, loc="left", pad=25)

    # separate NLD and sample level scores
    sep = [
        ["nld-sample", "nld-event"],
        ["sample", "plurality-voting", "majority-voting"],
    ]
    _kwargs_sep = {"colors": "darkgray", "linestyles": "dashed"}
    sep = [[matcher_map.get(_matcher) for _matcher in _sep] for _sep in sep]
    pos = map(lambda x: [_order.index(_x) for _x in x if _x in _order], sep)
    for _pos in pos:
        if len(_pos):
            plt.vlines(max(_pos) + 0.5, *ylim, **_kwargs_sep)

    _format_matcher_metric_plot(ylim=ylim)

    if sname is not None:
        # save plots
        plt.savefig(os.path.join(odir, f"{sname}.eps"))

        # save png
        odir_png = os.path.join(odir, "png")
        mkpath(odir_png)
        plt.savefig(os.path.join(odir_png, f"{sname}.png"))

        plt.close()

        # save results
        odir_data = os.path.join(odir, "data")
        mkpath(odir_data)

        df_aggr = df.groupby(["metric_label", "matcher_label"], as_index=False).mean()
        df_aggr.to_csv(os.path.join(odir_data, f"{sname}.csv"), index=False)


def _format_matcher_metric_plot(ylim=None):
    plt.xticks(rotation=90, fontsize="small")

    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
    plt.xlabel(None)
    plt.ylabel("Score")
    plt.subplots_adjust(top=0.9)

    plt.tight_layout()


##
# helper functions
def get_event_map(job, event_map_default):
    event_map = job.get("event_map", event_map_default)
    event_map = keys2num(event_map)

    event_labels = event_map.values()
    if 0 not in event_map.keys():
        # add undef label
        event_labels = [0, *event_labels]

    event_labels = list(set(event_labels))

    return event_map, event_labels


def format_perclass_scores(scores, labels, metric):
    assert len(scores) == len(labels)
    result = {f"{metric}/{label}": score for score, label in zip(scores, labels)}
    return result


##
# misc
class BColors:
    colors = {
        "HEADER": "\033[95m",
        "OKBLUE": "\033[94m",
        "OKGREEN": "\033[92m",
        "WARNING": "\033[93m",
        "FAIL": "\033[91m",
        "ENDC": "\033[0m",
        "BOLD": "\033[1m",
        "UNDERLINE": "\033[4m",
    }

    def __init__(self):
        self.colors = BColors.colors
        self.eol = self.colors.get("ENDC")

    def __call__(self, msg, msg_type):
        color = self.colors.get(msg_type, self.eol)
        return f"{color}{msg}{self.eol}"


def interp(x, y, kind="nearest"):
    f = interp1d(x, y, kind=kind, bounds_error=False, fill_value="extrapolate")
    return f


def get_nested_item(dictionary, keys, default=None):
    try:
        return reduce(getitem, keys, dictionary)
    except (KeyError, IndexError):
        return default


def merge_dicts(dictionaries):
    merged = dict(itertools.chain(*map(dict.items, dictionaries)))
    return merged


def keys2num(dictonary):
    return {str2int(k): v for k, v in dictonary.items()}


def path2abs(path, root):
    path = path if os.path.isabs(path) else os.path.join(root, path)
    return path


def split_path(fpath):
    fdir, fname = os.path.split(os.path.splitext(fpath)[0])
    return fdir, fname


def valid_fname(fname, replace=""):
    valid_chars = r"[^-a-zA-Z0-9_.() ]+"
    valid_file_name = re.sub(valid_chars, replace, fname)
    return valid_file_name


def multiple_replace(string, rep):
    pattern = re.compile(
        "|".join([re.escape(k) for k in sorted(rep, key=len, reverse=True)]),
        flags=re.DOTALL,
    )
    return pattern.sub(lambda x: rep[x.group(0)], string)


def seq2chr(seq):
    return "".join([chr(s) for s in seq])


def str2int(s):
    return int(s) if s.isnumeric() else s
