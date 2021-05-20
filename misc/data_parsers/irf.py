import os

import numpy as np
import pandas as pd

try:
    from __init__ import ROOT
except ImportError:
    ROOT = None

from misc import utils

data_path = os.path.normpath("irf/etdata/lookAtPoint_EL")


def parse_irf(root, **kwargs):
    data_dir = os.path.join(root, data_path)
    print(f"Parsing IRF from {data_dir}")

    files = utils.dir_walk(data_dir, "npy")
    data_accum = []
    for fpath in files:
        fdir, fname = utils.split_path(fpath)
        rdir = os.path.relpath(fdir, data_dir)
        spath = os.path.join("irf_RZ", rdir, fname)

        etdata = pd.DataFrame(np.load(fpath))

        # use only tags 1, 2 and 3
        mask = etdata["evt"].isin([1, 2, 3])
        etdata.loc[~mask, "evt"] = 0

        data_accum.append((etdata, spath))

    return data_accum
