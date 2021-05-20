import os
import parse

import numpy as np
import pandas as pd

try:
    from __init__ import ROOT
except ImportError:
    ROOT = None

from misc import utils

LUND2013_EVENT_MAPPING = {
    1: 1,  # fixation
    2: 2,  # saccade
    3: 3,  # PSO
    4: 4,  # Pursuit
    5: 5,  # Blink
    6: 0,  # undefinded
}
LUND2013_TRACKLOSS = 0
LUND2013_TOLLERANCE = 100

data_path = os.path.normpath(
    "EyeMovementDetectorEvaluation/annotated_data/originally uploaded data/"
)
annotation_fixes = {
    "UH29_img_Europe_labelled_MN": "EyeMovementDetectorEvaluation/annotated_data/fix_by_Zemblys2018/UH29_img_Europe_labelled_FIX_MN.mat"
}
fmt_fname = "{fname}_labelled_{coder}"


def parse_lund2013(root, **kwargs):
    data_dir = os.path.join(root, data_path)
    print(f"Parsing Lund2013 from {data_dir}")

    files = utils.dir_walk(data_dir, "mat")
    data_accum = []
    for fpath in files:
        fdir, fname = utils.split_path(fpath)
        etdata, sname, coder = _parse_lund2013(fpath)

        # include Zemblys et al. 2019 fix
        if fname in annotation_fixes:
            _fix_path = os.path.normpath(annotation_fixes.get(fname))
            _fix_path = os.path.join(root, _fix_path)
            etdata, _, _ = _parse_lund2013(_fix_path)

        rdir = os.path.relpath(fdir, data_dir)
        spath = os.path.join(f"lund2013_{coder}", rdir, sname)
        data_accum.append((etdata, spath))

    return data_accum


def _parse_lund2013(fpath):
    fdir, fname = utils.split_path(fpath)
    _coder = parse.parse(fmt_fname, fname).named

    # read mat file
    mat = utils.loadmat(fpath)["ETdata"]
    data = mat["pos"]

    # read meta data
    screen_w, screen_h = mat["screenRes"]
    fs = mat["sampFreq"]

    # parse data
    timestamps = data[:, 0].astype(np.float64)
    if np.all(np.isfinite(timestamps)):
        mask = timestamps == 0
        timestamps = timestamps[~mask]
        timestamps = (timestamps - timestamps[0]) / 1e6
    else:
        # some files do not have timestamps. Construct based on sampling rate
        timestamps = np.arange(len(timestamps)) / fs
        mask = np.zeros(len(timestamps), dtype=np.bool)

    x, y, evt = data[~mask, 3:].T

    # get tracking status
    status = (
        np.logical_or(x == LUND2013_TRACKLOSS, y == LUND2013_TRACKLOSS),
        np.logical_or(x < -LUND2013_TOLLERANCE, x > screen_w + LUND2013_TOLLERANCE),
        np.logical_or(y < -LUND2013_TOLLERANCE, y > screen_h + LUND2013_TOLLERANCE),
    )
    status = np.any(status, axis=0)

    # cat data
    etdata = pd.DataFrame(
        {"t": timestamps, "x": x, "y": y, "status": ~status, "evt": evt}
    )

    # map events to our scheme
    etdata.replace({"evt": LUND2013_EVENT_MAPPING}, inplace=True)

    return etdata, _coder.get("fname"), _coder.get("coder")
