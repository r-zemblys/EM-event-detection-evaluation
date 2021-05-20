import os

import numpy as np
import pandas as pd

EGOFIX_EVENT_MAPPING = {"fixation": 1, "blink": 5, "0": 0}

data_path_egofix = "MPIIEgoFixation"
dataset_name = "egofix"
data_cols = [
    "frame_scene",
    "t_scene",
    "frame",
    "t",
    "px",
    "py",
    "x",
    "y",
    "evt",
    "patch",
]


def parse_egofix(root, **kwargs):
    data_dir = os.path.join(root, data_path_egofix)
    print(f"Parsing {dataset_name} from {data_dir}...")

    files = [
        os.path.join(_root, _dir)
        for _root, _dirs, _ in os.walk(data_dir)
        for _dir in _dirs
    ]
    data_accum = []
    for fpath in files:
        data = pd.read_csv(os.path.join(fpath, "data.csv"), sep=";", names=data_cols)
        # get trackloss
        trackloss = np.any(data[["x", "y"]] == -1, axis=1)
        data.loc[trackloss, ["x", "y"]] = np.nan

        # remove time offset
        data["t"] -= data["t"].values[0]

        # map events to internal codes
        data.replace({"evt": EGOFIX_EVENT_MAPPING}, inplace=True)

        # store
        etdata = pd.DataFrame(
            {
                "t": data["t"],
                "x": data["x"],
                "y": data["y"],
                "status": ~trackloss,
                "evt": data["evt"],
            }
        )

        rdir = os.path.relpath(fpath, data_dir)
        spath = os.path.join(f"{dataset_name}", rdir)

        data_accum.append((etdata, spath))

    return data_accum
