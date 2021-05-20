import os
import argparse
from distutils.dir_util import mkpath

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from __init__ import ROOT
except ImportError:
    ROOT = None

from misc.constants import event_mapping
from misc import utils
from misc.data_parsers.lund2013 import parse_lund2013
from misc.data_parsers.irf import parse_irf
from misc.data_parsers.humanFixationClassification import parse_hfc
from misc.data_parsers.tum import (
    parse_gazecom,
    parse_hollywood2_em,
    parse_360em,
    parse_360em_secondary,
)
from misc.data_parsers.giw import parse_giw
from misc.data_parsers.mp import parse_egofix

plt.ioff()
plt.rc("axes.spines", top=False, right=False)


def get_arguments():
    parser = argparse.ArgumentParser(description="Parse datasets")
    parser.add_argument(
        "-root", type=str, default=None, required=True, help="Path to dataset"
    )
    parser.add_argument(
        "-dataset", type=str, choices=data_parsers.keys(), required=True, help="Dataset"
    )
    # TODO: description how to get dataset
    parser.add_argument(
        "--coder",
        type=str,
        default=None,
        choices=["expert", "alg"],
        help="Select coder set. Only applies to TUM datasets",
    )
    parser.add_argument(
        "--events",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 11, 12, 13, 14],
        help="Select events for stat calculation",
    )
    parser.add_argument(
        "--output", type=str, default="results/etdata", help="Output directory"
    )
    parser.add_argument(
        "--plot", action="store_true", default=False, help="Plot trials"
    )
    return parser.parse_args()


data_parsers = {
    "lund2013": parse_lund2013,
    "irf": parse_irf,
    "humanFixationClassification": parse_hfc,
    "gazecom": parse_gazecom,
    "hollywood2em": parse_hollywood2_em,
    "360em": parse_360em,
    "360em-secondary": parse_360em_secondary,
    "giw": parse_giw,
    "egofix": parse_egofix,
}

##
args = get_arguments()
root = args.root
output = args.output if os.path.isabs(args.output) else os.path.join(ROOT, args.output)
dataset = args.dataset
kwargs = {"coder": args.coder} if args.coder is not None else {}
data_parser = data_parsers.get(dataset)

data_stats = []
data = data_parser(root, **kwargs)
for etdata, spath in tqdm(data):
    fdir, fname = os.path.split(spath)

    # calculate data stats
    _data_stats = utils.get_et_stats(etdata)
    _data_stats["fname"] = fname
    data_stats.append(_data_stats)

    # save
    sdir = os.path.join(output, fdir)
    mkpath(sdir)
    etdata.to_csv(os.path.join(sdir, f"{fname}.csv"), index=False)

    # plot
    if args.plot:
        status = etdata["status"].values
        etdata.loc[~status, ["x", "y"]] = np.nan

        fig, ax = utils.get_figure(1, figsize=(19.2, 10.8))
        scarfplot = utils.ScarfPlot(ax[-1])
        # ylim = max((screen_w, screen_h))

        ax[0].plot(etdata["t"], etdata["x"], ".-", label="Horizontal")
        ax[0].plot(etdata["t"], etdata["y"], ".-", label="Vertical")
        ax[0].set_ylabel("Gaze position, px")
        # ax[0].set_ylim(0, 1980)
        _title = ", ".join([f"{k}: {v}" for k, v in _data_stats.items()])
        ax[0].set_title(_title, fontsize="medium")
        ax[0].legend(
            loc="lower left", bbox_to_anchor=(0.05, 1.0), frameon=False, ncol=5
        )

        evt = utils.aggr_events_df(etdata["evt"])
        f = utils.interp(range(len(etdata)), etdata["t"])
        evt["st"] = f(evt["s"])
        evt["et"] = f(evt["e"])
        scarfplot.plot(evt, label="Events")

        plt.tight_layout()
        fig.savefig(os.path.join(sdir, f"{fname}.png"))
        plt.close(fig)

# data stats
data_stats = pd.DataFrame(data_stats)
# duration in mins
data_stats["duration"] = data_stats["n-samples"] / data_stats["fs_est"] / 60

# aggregate stats
summary = data_stats.sum(axis=0)
# duration of unique filenames
summary["dur_unique"] = np.sum(data_stats.drop_duplicates("fname")["duration"])

rename = {k: v.get("label") for k, v in event_mapping.items()}
_all_events = list(rename.keys())
summary.loc[_all_events] /= summary["n-samples"] / 100

# select only events of interest
_other = list(set(_all_events) - set(args.events))
summary["other"] = np.sum(summary.loc[_other])

# format event stat report
_event_names = {evt: event_mapping[evt].get("label") for evt in args.events}
result = (
    f'Duration: {summary["duration"]:.1f} min',
    f'Duration unique: {summary["dur_unique"]:.1f} min',
    *[f"{summary[evt]:.2f}% {_event_names[evt]}" for evt in args.events],
    f'{summary["other"]:.2f}% Other',
)
result = "\n".join(result)
print(result)

# check sampling rate
print(f'Sampling rate: {set(data_stats["fs"])}, {set(data_stats["fs_est"])}')
