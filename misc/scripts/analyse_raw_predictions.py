import os
import json
from operator import attrgetter
from distutils.dir_util import mkpath

from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

try:
    from __init__ import ROOT as PACKAGE_ROOT
except ImportError:
    PACKAGE_ROOT = "."

from misc.constants import exclude_map, event_mapping
from misc import utils

plt.ion()
plt.rc("axes.spines", top=False, right=False)
sns.set_context(rc={"lines.linewidth": 1})

##
# config
fdir = os.path.join(PACKAGE_ROOT, "assets")
jname = "job_hollywood2"

job = os.path.join(fdir, f"{jname}.json")
jpath = utils.path2abs(job, PACKAGE_ROOT)
jdir, jname = utils.split_path(jpath)

with open(jpath, "r") as f:
    jobs = json.load(f)
_root, _ = utils.split_path(jpath)
root = jobs.get("root", _root)

event_map, _ = utils.get_event_map({}, jobs.get("event_map", {}))

odir = os.path.join(PACKAGE_ROOT, "results", jname, "result-plots")
mkpath(odir)

##
fpath_samples = os.path.join(odir, "data")
mkpath(fpath_samples)
fname_samples = os.path.join(fpath_samples, f"{jname}-samples.npy")
if not os.path.exists(fname_samples):
    print("Running sample extraction")
    meta_accum = []
    event_accum = []
    for job in jobs.get("jobs", []):
        files = utils.get_file_list(job, root)
        alg = job.get("pr")
        for fpath_gt, fpath_pr in zip(*files):
            # init
            fname = os.path.relpath(fpath_gt, os.path.join(root))
            _gt, _pr = job.get("gt"), job.get("pr")
            meta = {"gt": _gt, "pr": _pr, "fpath_gt": fpath_gt, "fpath_pr": fpath_pr}
            meta_accum.append(meta)

    meta = pd.DataFrame(meta_accum)

    exclude_pr = utils.get_nested_item(exclude_map, [jname, "pr"], [])
    exclude_fp = utils.get_nested_item(exclude_map, [jname, "fname"], [])
    _check = (
        meta["pr"].isin(exclude_pr),
        meta["fpath_gt"].str.contains("|".join(exclude_fp)),
    )
    mask = np.any(_check, axis=0)
    meta = meta[~mask]
    meta = meta.groupby(["gt", "pr", "fpath_gt", "fpath_pr"]).head(1)
    meta.reset_index(drop=True, inplace=True)

    # load data
    ag = attrgetter("gt", "pr", "fpath_gt", "fpath_pr")
    event_accum = []
    for _meta in tqdm(meta.itertuples(), total=len(meta)):
        gt, pr, fpath_gt, fpath_pr = ag(_meta)

        data_gt, data_pr = utils.load_data(fpath_gt, fpath_pr, event_map)
        event_accum.append((data_gt["evt"].values, data_pr["evt"].values))
    events = np.concatenate(event_accum, axis=1)
    np.save(fname_samples, events)

##
# confusion matrix
print("Loading sample data...")
events = np.load(fname_samples)

print("Computing confusion matrix...")
labels = sorted(event_map.values())
label_names = [utils.get_nested_item(event_mapping, [_l, "label"]) for _l in labels]

for _evt, evt_name in zip(labels, label_names):
    _events = events == _evt
    acc = metrics.accuracy_score(*_events)
    bacc = metrics.balanced_accuracy_score(*_events, adjusted=False)
    print(f"{evt_name}. Accuracy: {acc}, Balanced accuracy: {bacc}")

cm = metrics.confusion_matrix(*events, labels=labels)
# normalize
cm_norm = cm / cm.sum(axis=1, keepdims=True)

cm_kwargs = {
    "vmin": 0,
    "vmax": 1,
    "cmap": "Blues",
    "cbar": False,
    "annot": True,
    "fmt": ".2f",
}
plot_kwargs = {"rotation": 0, "fontsize": "x-small"}

plt.figure(figsize=(2.8, 2.4))
sns.heatmap(cm_norm, **cm_kwargs)

_idx = np.arange(len(labels)) + 0.5

plt.xticks(_idx, label_names, **plot_kwargs)
plt.yticks(_idx, label_names, **plot_kwargs)
plt.title("Predicted", fontsize="small")
plt.ylabel("Ground truth", fontsize="small")
plt.tight_layout()

sname = f"CM-{jname}"
plt.savefig(os.path.join(odir, f"{sname}.eps"))

odir_png = os.path.join(odir, "png")
mkpath(odir_png)
plt.savefig(os.path.join(odir_png, f"{sname}.png"))
