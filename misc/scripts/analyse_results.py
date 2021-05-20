import os
import itertools
from distutils.dir_util import mkpath

import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

try:
    from __init__ import ROOT as PACKAGE_ROOT
except ImportError:
    PACKAGE_ROOT = "."

from misc.constants import (
    event_mapping,
    matcher_map,
    metric_map,
    alg_map,
    exclude_map,
    plot_kwargs,
    kwargs_legend,
)
from misc import utils

plt.ion()
plt.rc("axes.spines", top=False, right=False)
sns.set_context(rc={"lines.linewidth": 1})


##
# helpers
def plot_aggr(result_df, sname=None, suffix=None, colors=None):
    if sname is not None:
        sname = sname if suffix is None else "_".join((sname, suffix))
    utils.plot_metrics_vs_matchers(
        result_df,
        odir,
        plot_kwargs,
        kwargs_legend,
        ylim=(0, 1),
        sname=sname,
        palette=colors,
    )


def plot_per_alg(result_df, save=False, suffix=None, colors=None):
    sname = None
    for (pr, alg), d in result_df.groupby(["pr", "alg_label"]):
        if save:
            sname = pr if suffix is None else "_".join((pr, suffix))
        utils.plot_metrics_vs_matchers(
            d,
            odir,
            plot_kwargs,
            kwargs_legend,
            ylim=(0, 1),
            sname=sname,
            title=alg,
            palette=colors,
        )


def plot_per_event(result_df, sname=None, suffix=None, colors=None):
    if sname is not None:
        sname = sname if suffix is None else "_".join((sname, suffix))

    for evt, d in result_df.groupby("event"):
        _evt = utils.get_nested_item(event_mapping, [int(evt), "label"])
        sname_evt = f"{sname}_{_evt}" if sname is not None else None
        utils.plot_metrics_vs_matchers(
            d,
            odir,
            plot_kwargs,
            kwargs_legend,
            ylim=(0, 1),
            sname=sname_evt,
            title=_evt,
            palette=colors,
        )


def calc_nld_corr(result_df, label=None):
    # NLD correlation analysis
    cols_nld = ["nld-event-nld", "nld-sample-nld"]

    _result_df = result_df.copy()
    _result_df["matcher_metric"] = _result_df["matcher"] + "-" + _result_df["metric"]
    _result_df = _result_df.pivot_table(
        values="score", index="alg_label", columns="matcher_metric"
    )

    cols = _result_df.columns
    cols_sample = cols.str.contains("sample")
    result_accum = []
    for col_nld, cols_data in zip(cols_nld, (cols[~cols_sample], cols[cols_sample])):

        result = [
            [c1, c2, *stats.linregress(*_result_df[[c1, c2]].values.T)]
            for c1, c2 in itertools.product([col_nld], cols_data)
            if not (c1 == c2)
        ]
        result_accum.extend(result)

    rcols = ["slope", "intercept", "r_value", "p_value", "std_err"]
    result = pd.DataFrame(result_accum, columns=["c1", "c2", *rcols])
    result["exp"] = label

    return result


##
# config
fdir = os.path.join(PACKAGE_ROOT, "results")
fname = "job_hollywood2"

save = True
indv_plots = True
clip = True

exclude_pr = utils.get_nested_item(exclude_map, [fname, "pr"], [])

odir = os.path.join(fdir, fname, "result-plots")
mkpath(odir)

_sname = fname if save else None

# sort mapper
mapper = {name: order for order, name in enumerate(metric_map)}

##
# load data
fpath = os.path.join(fdir, f"{fname}.csv")
data = pd.read_csv(fpath)

# exclude recordings
_exclude = utils.get_nested_item(exclude_map, [fname, "fname"], [])
mask = data["fname"].str.contains("|".join(_exclude))
data = data[~mask].reset_index(drop=True)

# convert nld to one-endian, i.e. so 1 is the highest score
data["nld"] = 1 - data["nld"]

# analyse results
_metrics = data.columns.intersection(metric_map.keys())
results = pd.melt(
    data,
    id_vars=["gt", "pr", "matcher", "fname", "eval", "event"],
    value_vars=_metrics,
    var_name="metric",
    value_name="score",
)
if clip:
    results["score"].clip(0, 1, inplace=True)

# treat nld as a separate matcher
_mask_nld = results["metric"] == "nld"
_mask_sample = results["matcher"] == "sample"
results.loc[_mask_nld & _mask_sample, "matcher"] = "nld-sample"
results.loc[_mask_nld & ~_mask_sample, "matcher"] = "nld-event"

# camera-ready labels
results["matcher_label"] = results["matcher"].replace(matcher_map)
results["alg_label"] = results["pr"].replace(alg_map)
results["metric_label"] = results["metric"].replace(metric_map)

_metrics = results["metric_label"].unique()
metric_colors = {
    metric: color
    for metric, color in zip(_metrics, sns.color_palette(n_colors=len(_metrics)))
}

# sort
results.sort_values(by="metric", key=lambda x: x.map(mapper), inplace=True)

# average over the recordings
cols = [
    "gt",
    "pr",
    "matcher",
    "eval",
    "event",
    "metric",
    "matcher_label",
    "alg_label",
    "metric_label",
]
results_avg = results.groupby(cols, as_index=False).mean()

# sort
results_avg.sort_values(by="metric", key=lambda x: x.map(mapper), inplace=True)

# container for nld correlation analysis
result_nld = []

##
# multiclass setting plots
_filter = (
    'eval=="multiclass"',
    'event=="all"',
    'metric in ["accuracy", "accuracy_balanced", "mcc", "kappa", "nld"]',
)
_filter = " and ".join(_filter)

# per algorithm plots
if indv_plots:
    results_part = results.query(_filter)
    plot_per_alg(results_part, save=save, colors=metric_colors)

# aggr plot
results_part = results_avg.query(_filter)
results_aggr = results_part[~results_part["pr"].isin(exclude_pr)]
plot_aggr(results_aggr, sname=_sname, colors=metric_colors)

# NLD correlation; multiclass setting
result_nld.append(calc_nld_corr(results_aggr, label="multiclass"))

##
# # multiclass setting plots; ignore_undef
# _filter = (
#     'eval=="multiclass/ignore_undef"',
#     'event=="all"',
#     'metric in ["accuracy", "accuracy_balanced", "mcc", "kappa", "nld"]',
# )
# _filter = ' and '.join(_filter)
#
# # per algorithm plots
# if indv_plots:
#     results_part = results.query(_filter)
#     plot_per_alg(results_part, save=save, suffix='ignore_undef')
#
# # aggr plot
# results_part = results_avg.query(_filter)
# results_aggr = results_part[~results_part['pr'].isin(exclude_pr)]
# plot_aggr(results_aggr, sname=_sname, suffix='ignore_undef', colors=metric_colors)

##
# binary setting plots; multiclass metrics
_filter = (
    'eval=="binary/ignore"',
    'event in ["1", "2", "4"]',
    'metric in ["accuracy", "accuracy_balanced", "mcc", "kappa", "nld"]',
)
_filter = " and ".join(_filter)

results_part = results_avg.query(_filter)
results_aggr = results_part[~results_part["pr"].isin(exclude_pr)]

# per event plots
plot_per_event(results_aggr, _sname, colors=metric_colors)

# aggr plot
_aggr_cols = ["alg_label", "matcher_label", "metric_label", "matcher", "metric"]
results_aggr_event = results_aggr.groupby(_aggr_cols, as_index=False).mean()
results_aggr_event.sort_values(by="metric", key=lambda x: x.map(mapper), inplace=True)
plot_aggr(results_aggr_event, sname=_sname, suffix="bin", colors=metric_colors)

##
# binary setting plots; multiclass metrics, unmatched negatives as true negatives
_filter = (
    'eval=="binary/tn"',
    'event in ["1", "2", "4"]',
    'metric in ["accuracy", "accuracy_balanced", "mcc", "kappa", "nld"]',
    'matcher not in ["nld-sample", "nld-event"]',
)
_filter = " and ".join(_filter)
results_part = results_avg.query(_filter)
results_aggr = results_part[~results_part["pr"].isin(exclude_pr)]

_aggr_cols = ["alg_label", "matcher_label", "metric_label", "matcher", "metric"]
results_aggr_event = results_aggr.groupby(_aggr_cols, as_index=False).mean()
results_aggr_event.sort_values(by="metric", key=lambda x: x.map(mapper), inplace=True)
plot_aggr(results_aggr_event, sname=_sname, suffix="bin_tn", colors=metric_colors)

##
# binary setting plots; multiclass metrics, unmatched negatives as error
_filter = (
    'eval=="binary/error"',
    'event in ["1", "2", "4"]',
    'metric in ["accuracy", "accuracy_balanced", "mcc", "kappa", "nld"]',
    'matcher not in ["nld-sample", "nld-event"]',
)
_filter = " and ".join(_filter)
results_part = results_avg.query(_filter)
results_aggr = results_part[~results_part["pr"].isin(exclude_pr)]

_aggr_cols = ["alg_label", "matcher_label", "metric_label", "matcher", "metric"]
results_aggr_event = results_aggr.groupby(_aggr_cols, as_index=False).mean()
results_aggr_event.sort_values(by="metric", key=lambda x: x.map(mapper), inplace=True)
plot_aggr(results_aggr_event, sname=_sname, suffix="bin_error", colors=metric_colors)

##
# binary setting plots; binary metrics
_filter = (
    'eval=="binary/ignore"',
    'event in ["1", "2", "4"]',
    'metric in ["precision", "sensitivity", "specificity", "f1_score", "auc", "nld", "iou"]',
    'matcher not in ["nld-sample", "nld-event"]',
)
_filter = " and ".join(_filter)

# per algorithm plots
if indv_plots:
    results_part = results.query(_filter)

    # per event plots
    for (pr, alg), d in results_part.groupby(["pr", "alg_label"]):
        sname_evt = f"{pr}_binary_metrics" if save else None
        plot_per_event(d, sname_evt, colors=metric_colors)

    # aggr per algorithm plots
    _aggr_cols = [
        "pr",
        "alg_label",
        "matcher_label",
        "metric_label",
        "matcher",
        "metric",
        "fname",
    ]
    results_aggr_event = results_part.groupby(_aggr_cols, as_index=False).mean()
    results_aggr_event.sort_values(
        by="metric", key=lambda x: x.map(mapper), inplace=True
    )
    plot_per_alg(
        results_aggr_event, save=save, suffix="binary_metrics", colors=metric_colors
    )

# per event plots
results_part = results_avg.query(_filter)
results_aggr = results_part[~results_part["pr"].isin(exclude_pr)]
plot_per_event(results_aggr, _sname, suffix="binary_metrics", colors=metric_colors)

# aggr plot
_aggr_cols = ["alg_label", "matcher_label", "metric_label", "matcher", "metric"]
results_aggr_event = results_aggr.groupby(_aggr_cols, as_index=False).mean()
results_aggr_event.sort_values(by="metric", key=lambda x: x.map(mapper), inplace=True)
plot_aggr(
    results_aggr_event, sname=_sname, suffix="binary_metrics", colors=metric_colors
)

##
# NLD correlation; binary setting, all metrics
_filter = (
    'eval=="binary/ignore"',
    'event in ["1", "2", "4"]',
    'metric in ["accuracy", "accuracy_balanced", "mcc", "kappa", "nld", '
    '"precision", "sensitivity", "specificity", "f1_score", "auc", "nld", "iou"]',
)
_filter = " and ".join(_filter)

results_part = results_avg.query(_filter)
results_aggr = results_part[~results_part["pr"].isin(exclude_pr)]
_aggr_cols = ["alg_label", "matcher_label", "metric_label", "matcher", "metric"]
results_aggr_event = results_aggr.groupby(_aggr_cols, as_index=False).mean()
result_nld.append(calc_nld_corr(results_aggr_event, label="binary"))

result_nld_df = pd.concat(result_nld).reset_index(drop=True)
result_nld_df.sort_values(["c1", "exp", "r_value"], ascending=False, inplace=True)
# save results
odir_nld_corr = os.path.join(odir, "data")
mkpath(odir_nld_corr)
result_nld_df.to_csv(os.path.join(odir_nld_corr, "nld_corr.csv"), index=False)
