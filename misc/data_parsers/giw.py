import os

from tqdm import tqdm
import parse

import numpy as np
import pandas as pd

try:
    from __init__ import ROOT
except ImportError:
    ROOT = None

from misc.elc import GIW_EVENT_MAPPING
from misc import utils

data_path_giw = "Gaze-In-Wild/LabelData"
data_path_giw_data = "Gaze-In-Wild/ProcessData"


def parse_giw(root, **kwargs):
    dataset_name = "giw"
    fname_fmt = "{fname}_Lbr_{coder:d}"
    data_dir = os.path.join(root, data_path_giw)
    print(f"Parsing {dataset_name} from {data_dir}...")

    files = utils.dir_walk(data_dir, "mat")
    data_accum = []
    for fpath in tqdm(files):
        fdir, fname = utils.split_path(fpath)
        _p = parse.parse(fname_fmt, fname)
        coder = _p["coder"]
        data = utils.loadmat(fpath)["LabelData"]
        assert coder == data["LbrIdx"]

        # try to load gaze data
        _fname = _p["fname"]
        fpath_gaze = os.path.join(root, data_path_giw_data, f"{_fname}.mat")
        if os.path.exists(fpath_gaze):
            data_gaze = utils.loadmat(fpath_gaze)["ProcessData"]
            x, y = data_gaze["ETG"]["POR"].T

            # label data with confidence < 0.3 as trackloss (Kothari et al. 2020, pp.6)
            trackloss = data_gaze["ETG"]["Confidence"] < 0.3

            x[trackloss] = np.nan
            y[trackloss] = np.nan
        else:
            x = y = np.zeros(_l, dtype=np.float32)
            trackloss = np.ones(_l, dtype=np.bool)

        _l = len(data["Labels"])
        etdata = pd.DataFrame(
            {
                "t": data["T"],
                "x": x,
                "y": y,
                "status": ~trackloss,
                "evt": data["Labels"],
            }
        )
        etdata.replace({"evt": GIW_EVENT_MAPPING}, inplace=True)

        rdir = os.path.relpath(fdir, data_dir)
        spath = os.path.join(f"{dataset_name}_{coder}", rdir, _fname)

        data_accum.append((etdata, spath))

    return data_accum
