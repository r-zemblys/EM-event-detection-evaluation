import os
import copy
import itertools

import numpy as np
import pandas as pd

try:
    from __init__ import ROOT
except ImportError:
    ROOT = None

from misc import utils

data_path = os.path.normpath("humanFixationClassification/data")


def _parse_hfc(trial, coder, trial_accum, coder_accum):
    data = copy.deepcopy(trial_accum.get(trial), None)
    events = coder_accum.get(coder).query("Trial==@trial")
    if data is None:
        return None

    _l = len(data)
    evt = np.zeros(_l, dtype=int)

    if len(events):
        f = utils.interp(data["t"], range(_l))
        fixation_samples = itertools.chain(
            *[
                range(int(s), int(e + 1))
                for s, e in zip(f(events["FixStart"]), f(events["FixEnd"]))
            ]
        )
        evt[list(fixation_samples)] = 1

    data["evt"] = evt
    data["t"] /= 1000  # convert time to seconds

    return data, _get_spath(coder, trial)


def _get_spath(coder, trial):
    spath = os.path.join(f"humanFixationClassification_{coder}", trial)
    return spath


def parse_hfc(root, **kwargs):
    data_dir = os.path.join(root, data_path)
    print(f"Parsing humanFixationClassification from {data_dir}")

    # get ET data
    files = utils.dir_walk(os.path.join(data_dir, "ETdata"), "txt")
    trial_accum = []
    for fpath in files:
        _, fname = utils.split_path(fpath)
        data = pd.read_csv(fpath, sep="\t", usecols=["time", "x", "y"])
        data.rename(columns={"time": "t"}, inplace=True)
        data["status"] = np.all(np.isfinite(data[["x", "y"]]).values, axis=1)
        trial_accum.append((fname, data))
    trial_accum = dict(trial_accum)

    # parse coder data
    files = utils.dir_walk(os.path.join(data_dir, "coderSettings"), "txt")
    _, coders = zip(*map(lambda x: utils.split_path(x), files))
    coder_accum = dict(
        [(coder, pd.read_csv(fpath, sep="\t")) for coder, fpath in zip(coders, files)]
    )

    data_accum = [
        _parse_hfc(trial, coder, trial_accum, coder_accum)
        for trial, coder in itertools.product(trial_accum, coder_accum)
    ]
    data_accum = list(filter(None, data_accum))

    return data_accum
